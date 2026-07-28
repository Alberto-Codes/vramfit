"""Torch-backed scan adapters, installed via the ``scan`` extra.

This package is the one place quantfit touches torch and transformers
(ADR-0005, ADR-0008). Nothing here is imported at CLI startup — the
scan command imports [quantfit.adapters.outbound.scan.meter][] lazily,
so the base install stays torch-free.

Attributes:
    meter: Submodule holding `TorchDamageMeter`, the `DamageMeter`
        port implementation. Set only after an explicit import.
    quantize: Submodule with round-to-nearest quantize-dequantize
        (the ADR-0006 v1 method).
    kl: Submodule computing mean KL between reference and perturbed
        logits.
    calibration: Submodule for calibration text loading and batching.

Examples:
    Import lazily, at the point of use:

    ```python
    from quantfit.adapters.outbound.scan.meter import TorchDamageMeter
    ```

See Also:
    - [quantfit.ports.outbound][]: `DamageMeter`, the port these
      adapters implement.
    - [quantfit.adapters.inbound.cli_scan][]: The command that drives
      them.
"""

from __future__ import annotations
