"""Adapters: where vramfit touches the outside world.

Split by direction: [vramfit.adapters.inbound][] drives the domain (the
CLI); [vramfit.adapters.outbound][] implements the driven ports (JSON
artifact files, Hugging Face configs). All IO, serialization, and schema
validation lives here — the domain stays pure (ADR-0008).

Attributes:
    inbound: Driving adapters (the Typer CLI).
    outbound: Driven adapters (JSON artifacts, Hugging Face configs).

Examples:
    Load a map through the outbound JSON adapter:

    ```python
    from vramfit.adapters.outbound.sensitivity_map_json import (
        load_sensitivity_map,
    )

    map_ = load_sensitivity_map(path)
    ```

See Also:
    - [vramfit.ports][]: The protocols outbound adapters satisfy.
"""

from __future__ import annotations
