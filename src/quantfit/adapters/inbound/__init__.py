"""Inbound (driving) adapters: entry points that call into the domain.

The Typer CLI is the only inbound adapter today. It is also the
composition root — the place that instantiates outbound adapters and
hands them to orchestration code typed against ports.

Attributes:
    cli: The Typer application and the ``quantfit`` console script.

Examples:
    Run the CLI programmatically:

    ```python
    from quantfit.adapters.inbound.cli import app

    app(["version"])
    ```

See Also:
    - [quantfit.adapters.outbound][]: The driven side.
"""

from __future__ import annotations
