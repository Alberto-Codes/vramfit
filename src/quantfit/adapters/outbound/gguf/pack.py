"""llama.cpp pack adapter: convert and quantize by subprocess.

Implements the `RecipePacker` port for the GGUF serving path
(ADR-0010, ADR-0012). `convert` runs ``convert_hf_to_gguf.py`` under
a caller-supplied interpreter — that interpreter carries torch, this
package never imports it (ADR-0005). `pack` first rejects a recipe
recorded for a foreign runtime (ADR-0013), then runs
``llama-quantize`` with the recipe's type mapping from
[quantfit.adapters.outbound.gguf.types][] — pattern overrides per
layer group plus dedicated embedding and output-head flags. Every failure — a tool
that cannot start, exits nonzero, dies to a signal, or leaves no
usable file — translates to `PackError` at this boundary (ADR-0011),
carrying the tool's last output lines.

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

import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from quantfit.adapters.outbound.gguf.types import (
    PackError,
    base_type,
    check_runtime,
    output_tensor_type,
    tensor_overrides,
    token_embedding_type,
)
from quantfit.domain.model import Recipe
from quantfit.domain.pack import PackResult

_TAIL_LINES: Final[int] = 15


def _sized_file(path: Path, stage: str) -> int:
    """Measure a tool's output file, rejecting absent or empty results.

    Args:
        path: The file the tool reported writing.
        stage: Stage name for the error message.

    Returns:
        The file size in bytes, always positive.

    Raises:
        PackError: If the file is absent, empty, or unreadable — a
            zero-exit tool that wrote nothing must not pass as
            success.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PackError(f"{stage} output {path} cannot be inspected: {exc}") from exc
    if size == 0:
        raise PackError(f"{stage} exited 0 but wrote an empty file at {path}")
    return size


def _run_tool(command: list[str], stage: str) -> None:
    """Run one toolchain subprocess, translating failure to `PackError`.

    stderr merges into stdout so the failure tail is the tool's real
    last words, whichever stream carried them.

    Args:
        command: Argument vector. The first element is the tool path.
        stage: Stage name for the error message (``convert`` or
            ``quantize``).

    Raises:
        PackError: If the tool cannot start, exits nonzero, or dies
            to a signal (named in the message). The message carries
            the last `_TAIL_LINES` lines the tool wrote.
    """
    try:
        # No timeout on purpose: a 49B conversion legitimately runs for
        # a long time, and a wrong guess kills real work. stderr merges
        # into stdout so the failure tail never hides the wrong stream.
        completed = subprocess.run(  # noqa: S603 - fixed tool paths from the composition root, no shell
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise PackError(f"{stage}: cannot run {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-_TAIL_LINES:])
        code = completed.returncode
        died = (
            f"killed by signal {signal.Signals(-code).name}"
            if code < 0
            else f"failed with exit code {code}"
        )
        raise PackError(f"{stage} {died}:\n{tail or '(no output captured)'}")


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
            PackError: If the convert tool fails, writes no usable
                file, or the file cannot be inspected.
        """
        if not self.base_gguf.exists():
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
        return _sized_file(self.base_gguf, stage="convert")

    def pack(self, recipe: Recipe) -> PackResult:
        """Quantize the base GGUF into the recipe's packed model.

        The embedding and output-head flags resolve independently: an
        ``lm_head`` group drives the output flag with its own
        assignment, and the embedding assignment stands in when the
        scan measured no head (ADR-0012).

        Args:
            recipe: The recipe to apply.

        Returns:
            The accounting record, with the real packed size and the
            resolved flag types.

        Raises:
            PackError: If the recipe targets another runtime
                (ADR-0013), the base GGUF is missing, the recipe
                cannot be mapped (ADR-0012), the quantizer fails, or
                it writes no usable file.
        """
        check_runtime(recipe)
        if not self.base_gguf.exists():
            raise PackError(
                f"base GGUF {self.base_gguf} does not exist — run convert first"
            )
        base = base_type(recipe)
        embedding = token_embedding_type(recipe)
        output = output_tensor_type(recipe)
        overrides = tensor_overrides(recipe)
        command = [str(self.quantize_bin), "--pure"]
        if embedding is not None:
            command += ["--token-embedding-type", embedding]
        if output is not None:
            # Without the flag an untied output head would fall to the
            # --pure base type — the recipe's floor (ADR-0012).
            command += ["--output-tensor-type", output]
        for override in overrides:
            command += ["--tensor-type", f"{override.pattern}={override.quant_type}"]
        command += [str(self.base_gguf), str(self.out_path), base, str(self.threads)]
        _run_tool(command, stage="quantize")
        return PackResult(
            packed_bytes=_sized_file(self.out_path, stage="quantize"),
            base_type=base,
            token_embedding_type=embedding,
            output_tensor_type=output,
            overrides=overrides,
        )
