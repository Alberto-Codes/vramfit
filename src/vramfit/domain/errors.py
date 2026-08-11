"""The error root every vramfit exception inherits (ADR-0011).

One root gives the CLI a single honest catch: anything that is a
`VramfitError` carries a user-facing message and maps to a clean
``error:`` line. The adapters translate foreign exceptions (torch,
transformers, the OS) into `VramfitError` subclasses at the boundary.

Examples:
    Catch every vramfit failure at the composition root:

    ```python
    try:
        run()
    except VramfitError as exc:
        print(f"error: {exc}")
    ```

See Also:
    - [vramfit.domain.solver][]: `PinError`, `InfeasibleBudgetError`.
    - [vramfit.adapters.outbound.json_common][]: `ArtifactError`.
"""

from __future__ import annotations


class VramfitError(Exception):
    """Root of every exception vramfit raises on purpose.

    Subclasses keep their historical bases (`ValueError`,
    `RuntimeError`) so existing callers' catches stay valid.

    Examples:
        Every vramfit exception satisfies one isinstance check:

        ```python
        from vramfit.adapters.outbound.json_common import ArtifactError

        assert issubclass(ArtifactError, VramfitError)
        ```
    """
