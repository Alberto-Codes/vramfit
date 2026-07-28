"""GGUF pack backend: the llama.cpp outbound adapter (ADR-0012).

Two modules split the backend at the IO boundary, and neither imports
torch — the conversion script's interpreter carries the heavy
dependencies (ADR-0005).

Attributes:
    types: Submodule holding the pure type mapping — nominal bits to
        K-quant types, layer groups to tensor patterns (ADR-0012).
    pack: Submodule driving ``convert_hf_to_gguf.py`` and
        ``llama-quantize`` as subprocesses.

Examples:
    Wire the packer from the composition root:

    ```python
    from quantfit.adapters.outbound.gguf.pack import LlamaCppPacker
    ```

See Also:
    - [quantfit.ports.outbound][]: `RecipePacker`, which
      `LlamaCppPacker` satisfies.
"""

from __future__ import annotations
