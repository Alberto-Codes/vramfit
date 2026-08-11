"""llama.cpp pack adapter: convert and quantize by subprocess.

Implements the `RecipePacker` port for the GGUF serving path
(ADR-0010, ADR-0012). `convert` runs ``convert_hf_to_gguf.py`` under
a caller-supplied interpreter — that interpreter carries torch, this
package never imports it (ADR-0005). `pack` first rejects a recipe
recorded for a foreign runtime (ADR-0013), then runs
``llama-quantize`` with the recipe's type mapping from
[vramfit.adapters.outbound.gguf.types][]: protection overrides
first (ADR-0022 — the quantizer applies the first matching
pattern), then pattern overrides per
layer group, plus dedicated embedding and output-head flags. With an
importance matrix (ADR-0016) the adapter also scans the quantizer's
zero-exit output for tensors the matrix did not cover — the
quantizer only warns, and a silently unassisted tensor must not
pass unrecorded. The recipe's imatrix exclusions become
``--exclude-weights`` flags, and their intentional misses stay out
of that coverage record (ADR-0023). Every failure — a tool that cannot start, exits
nonzero, dies to a signal, or leaves no usable file — translates to
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
    - [vramfit.ports.outbound][]: `RecipePacker`, which this
      satisfies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.gguf.toolrun import run_tool, sized_file
from vramfit.adapters.outbound.gguf.types import (
    PackError,
    base_type,
    check_runtime,
    imatrix_exclusion_names,
    output_tensor_type,
    protection_overrides,
    tensor_overrides,
    token_embedding_type,
)
from vramfit.domain.model import Recipe
from vramfit.domain.pack import PackResult

# The quantizer's zero-exit warning for a tensor the importance
# matrix does not cover (llama.cpp src/llama-quant.cpp). The tensor
# is then quantized without importance data.
_IMATRIX_MISS: Final[re.Pattern[str]] = re.compile(r"did not find weights for (\S+)")


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
        imatrix (Path | None): Importance matrix file for the
            quantizer (ADR-0016). None packs without one.

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
    imatrix: Path | None = None

    def convert(self) -> int:
        """Materialize the f16 base GGUF, reusing any existing file.

        The convert tool runs through the shared toolchain plumbing
        ([vramfit.adapters.outbound.gguf.toolrun][]).

        Returns:
            Size of the base GGUF in bytes.

        Raises:
            PackError: If the convert tool fails, writes no usable
                file, or the file cannot be inspected.
        """
        if not self.base_gguf.exists():
            run_tool(
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
        return sized_file(self.base_gguf, stage="convert")

    def pack(self, recipe: Recipe) -> PackResult:
        """Quantize the base GGUF into the recipe's packed model.

        The embedding and output-head flags resolve independently: an
        ``lm_head`` group drives the output flag with its own
        assignment, and the embedding assignment stands in when the
        scan measured no head (ADR-0012). A protected recipe's
        resolved pairs become the leading overrides (ADR-0022). A configured importance
        matrix reaches the quantizer as ``--imatrix``, lands in the
        result's provenance, and the quantizer's output is scanned
        for tensors the matrix did not cover (ADR-0016). The
        recipe's imatrix exclusions become ``--exclude-weights``
        flags (ADR-0023) — the quantizer drops those rows, so the
        coverage scan discounts them as intentional. Without an
        imatrix the exclusions are no-ops and stay unemitted.

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
        # Protection overrides first (ADR-0022): the quantizer applies
        # the first matching pattern, so a protected tensor must match
        # its own pattern before its group's.
        overrides = protection_overrides(recipe) + tensor_overrides(recipe)
        excluded = imatrix_exclusion_names(recipe) if self.imatrix is not None else ()
        command = [str(self.quantize_bin), "--pure"]
        if self.imatrix is not None:
            command += ["--imatrix", str(self.imatrix)]
        for name in excluded:
            command += ["--exclude-weights", name]
        if embedding is not None:
            command += ["--token-embedding-type", embedding]
        if output is not None:
            # Without the flag an untied output head would fall to the
            # --pure base type — the recipe's floor (ADR-0012).
            command += ["--output-tensor-type", output]
        for override in overrides:
            command += ["--tensor-type", f"{override.pattern}={override.quant_type}"]
        command += [str(self.base_gguf), str(self.out_path), base, str(self.threads)]
        quantize_output = run_tool(command, stage="quantize")
        # An excluded tensor's row is gone from the loaded matrix, so
        # the quantizer reports it as a miss — an intentional one,
        # recorded in imatrix_excluded instead (ADR-0023).
        uncovered = (
            tuple(
                name
                for name in dict.fromkeys(_IMATRIX_MISS.findall(quantize_output))
                if name not in excluded
            )
            if self.imatrix is not None
            else ()
        )
        return PackResult(
            packed_bytes=sized_file(self.out_path, stage="quantize"),
            base_type=base,
            token_embedding_type=embedding,
            output_tensor_type=output,
            overrides=overrides,
            imatrix_path=None if self.imatrix is None else str(self.imatrix),
            imatrix_uncovered=uncovered,
            imatrix_excluded=excluded,
        )
