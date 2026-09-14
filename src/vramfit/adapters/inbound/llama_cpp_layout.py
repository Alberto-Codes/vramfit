"""Where a llama.cpp checkout keeps the tools vramfit runs.

One spelling of the layout, read by every composition root that
drives the toolchain. `pack` and `refine` both wire the convert
script and the quantizer, and both reach `llama-perplexity` — one
through the smoke test (ADR-0017), the other through the divergence
meter (ADR-0031). A second spelling would let a pre-flight check one
file while the run executes another, which is the whole reason a
pre-flight exists.

Examples:
    Resolve a checkout's tools:

    ```python
    from pathlib import Path

    from vramfit.adapters.inbound.llama_cpp_layout import LlamaCppTools

    tools = LlamaCppTools.under(Path("~/llama.cpp"))
    ```

See Also:
    - [vramfit.adapters.inbound.cli_refine][]: Pre-flights all three
      and wires all three.
    - [vramfit.adapters.inbound.cli_pack_smoke][]: Pre-flights the
      convert script and the quantizer, and the perplexity binary
      when ``--smoke-text`` is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LlamaCppTools:
    """The tools a checkout carries, named once.

    A pre-flight checks these paths and the wiring runs them, so
    neither can mean a different file from the other. A second
    spelling would let the check pass while the run dies after the
    convert, which is the cost this record exists to prevent.

    Attributes:
        convert_script (Path): ``convert_hf_to_gguf.py``, which
            writes the f16 base.
        quantize_bin (Path): ``llama-quantize``, which packs a
            recipe.
        perplexity_bin (Path): ``llama-perplexity``, which measures
            perplexity for the smoke test and per-chunk divergence
            for the refinement pass.

    Examples:
        Read the quantizer's path:

        ```python
        print(LlamaCppTools.under(checkout).quantize_bin)
        ```
    """

    convert_script: Path
    quantize_bin: Path
    perplexity_bin: Path

    @classmethod
    def under(cls, llama_cpp: Path) -> LlamaCppTools:
        """Name the tools a checkout carries.

        Args:
            llama_cpp: The llama.cpp checkout.

        Returns:
            The three paths, built once for every caller.
        """
        built = llama_cpp / "build" / "bin"
        return cls(
            convert_script=llama_cpp / "convert_hf_to_gguf.py",
            quantize_bin=built / "llama-quantize",
            perplexity_bin=built / "llama-perplexity",
        )
