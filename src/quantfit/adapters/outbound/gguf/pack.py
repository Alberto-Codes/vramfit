"""llama.cpp pack adapter: convert and quantize by subprocess.

Implements the `RecipePacker` port for the GGUF serving path
(ADR-0010, ADR-0012). `convert` runs ``convert_hf_to_gguf.py`` under
a caller-supplied interpreter — that interpreter carries torch, this
package never imports it (ADR-0005). `pack` runs ``llama-quantize``
with the recipe's type mapping from
[quantfit.adapters.outbound.gguf.types][]. Tool failures translate to
`PackError` at this boundary (ADR-0011), carrying the tool's last
output lines.

Examples:
    Pack a recipe with a local llama.cpp checkout:

    ```python
    packer = LlamaCppPacker(
        model_dir=Path("model"),
        base_gguf=Path("model-f16.gguf"),
        out_path=Path("packed.gguf"),
        convert_script=Path("llama.cpp/convert_hf_to_gguf.py"),
        quantize_bin=Path("llama.cpp/build/bin/llama-quantize"),
        python_bin=Path(sys.executable),
    )
    packer.convert()
    result = packer.pack(recipe)
    ```

See Also:
    - [quantfit.ports.outbound][]: `RecipePacker`, which this
      satisfies.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from quantfit.adapters.outbound.gguf.types import (
    PackError,
    base_type,
    tensor_overrides,
    token_embedding_type,
)
from quantfit.domain.model import Recipe
from quantfit.domain.pack import PackResult

_TAIL_LINES: Final[int] = 15


def _run_tool(command: list[str], stage: str) -> None:
    """Run one toolchain subprocess, translating failure to `PackError`.

    Args:
        command: Argument vector. The first element is the tool path.
        stage: Stage name for the error message (``convert`` or
            ``quantize``).

    Raises:
        PackError: If the tool cannot start or exits nonzero. The
            message carries the last `_TAIL_LINES` lines the tool
            wrote.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed tool paths from the composition root, no shell
            command, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise PackError(f"{stage}: cannot run {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        output = completed.stderr or completed.stdout or ""
        tail = "\n".join(output.splitlines()[-_TAIL_LINES:])
        raise PackError(
            f"{stage} failed with exit code {completed.returncode}:\n{tail}"
        )


@dataclass(frozen=True, slots=True)
class LlamaCppPacker:
    """`RecipePacker` adapter driving the llama.cpp toolchain.

    Attributes:
        model_dir (Path): Hugging Face checkpoint directory.
        base_gguf (Path): Full-precision base GGUF. `convert` creates
            it when absent and reuses it when present.
        out_path (Path): Packed model destination.
        convert_script (Path): ``convert_hf_to_gguf.py`` path.
        quantize_bin (Path): ``llama-quantize`` path.
        python_bin (Path): Interpreter for the convert script. Must
            import torch.
        threads (int): Quantizer thread count.

    Examples:
        The composition root wires the paths:

        ```python
        packer: RecipePacker = LlamaCppPacker(
            model_dir,
            base_gguf,
            out_path,
            convert_script,
            quantize_bin,
            python_bin,
        )
        ```
    """

    model_dir: Path
    base_gguf: Path
    out_path: Path
    convert_script: Path
    quantize_bin: Path
    python_bin: Path
    threads: int = 8

    def convert(self) -> int:
        """Materialize the f16 base GGUF, reusing any existing file.

        Returns:
            Size of the base GGUF in bytes.

        Raises:
            PackError: If the convert tool fails or writes no file.
        """
        if self.base_gguf.exists():
            return self.base_gguf.stat().st_size
        _run_tool(
            [
                str(self.python_bin),
                str(self.convert_script),
                str(self.model_dir),
                "--outfile",
                str(self.base_gguf),
                "--outtype",
                "f16",
            ],
            stage="convert",
        )
        if not self.base_gguf.exists():
            raise PackError(f"convert exited 0 but wrote no file at {self.base_gguf}")
        return self.base_gguf.stat().st_size

    def pack(self, recipe: Recipe) -> PackResult:
        """Quantize the base GGUF into the recipe's packed model.

        Args:
            recipe: The recipe to apply.

        Returns:
            The accounting record, with the real packed size.

        Raises:
            PackError: If the base GGUF is missing, the recipe cannot
                be mapped (ADR-0012), or the quantizer fails.
        """
        if not self.base_gguf.exists():
            raise PackError(
                f"base GGUF {self.base_gguf} does not exist — run convert first"
            )
        base = base_type(recipe)
        embedding = token_embedding_type(recipe)
        overrides = tensor_overrides(recipe)
        command = [str(self.quantize_bin), "--pure"]
        if embedding is not None:
            command += ["--token-embedding-type", embedding]
        for override in overrides:
            command += ["--tensor-type", f"{override.pattern}={override.ggml_type}"]
        command += [str(self.base_gguf), str(self.out_path), base, str(self.threads)]
        _run_tool(command, stage="quantize")
        if not self.out_path.exists():
            raise PackError(f"quantize exited 0 but wrote no file at {self.out_path}")
        return PackResult(
            packed_bytes=self.out_path.stat().st_size,
            base_type=base,
            token_embedding_type=embedding,
            overrides=overrides,
        )
