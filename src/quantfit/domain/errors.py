"""The error root every quantfit exception inherits (ADR-0011).

One root gives the CLI a single honest catch: anything that is a
`QuantfitError` carries a user-facing message and maps to a clean
``error:`` line. The adapters translate foreign exceptions (torch,
transformers, the OS) into `QuantfitError` subclasses at the boundary.

Examples:
    Catch every quantfit failure at the composition root:

    ```python
    try:
        run()
    except QuantfitError as exc:
        print(f"error: {exc}")
    ```

See Also:
    - [quantfit.domain.solver][]: `PinError`, `InfeasibleBudgetError`.
    - [quantfit.adapters.outbound.json_common][]: `ArtifactError`.
"""

from __future__ import annotations


class QuantfitError(Exception):
    """Root of every exception quantfit raises on purpose.

    Subclasses keep their historical bases (`ValueError`,
    `RuntimeError`) so existing callers' catches stay valid.

    Examples:
        Every quantfit exception satisfies one isinstance check:

        ```python
        from quantfit.adapters.outbound.json_common import ArtifactError

        assert issubclass(ArtifactError, QuantfitError)
        ```
    """
