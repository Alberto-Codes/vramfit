"""The solver's two refusals, split out to keep `solver` inside the cap.

`PinError` rejects a ``--pin`` the candidate set or the map cannot
satisfy. `InfeasibleBudgetError` reports a budget no assignment
reaches, and it assembles the whole explanation: the gap, the
precisions a target runtime removed, the protections holding a floor,
and the groups a size source holds at reference precision (ADR-0029).

Both sit under the `VramfitError` root (ADR-0011) and carry messages
the CLI prints verbatim. [vramfit.domain.solver][] re-exports them, so
callers keep importing from there.

Examples:
    Report the gap to the user:

    ```python
    from vramfit.domain.solver import InfeasibleBudgetError

    try:
        solve_with_tiny_budget()
    except InfeasibleBudgetError as exc:
        print(f"over budget by {exc.gap_bytes} bytes")
    ```

See Also:
    - [vramfit.domain.solver][]: Raises both.
"""

from __future__ import annotations

from vramfit.domain.budget import format_size
from vramfit.domain.errors import VramfitError


class PinError(VramfitError, ValueError):
    """A ``--pin`` pattern is unusable. Under the `VramfitError` root.

    Raised when a pin names a precision the scan did not measure, or when
    its pattern matches no group (usually a typo).

    Examples:
        A pin against an unscanned precision:

        ```python
        from vramfit.domain.solver import PinError

        try:
            solve_with_pins(pins={"g*": 6})
        except PinError as exc:
            print(exc)
        ```
    """


class InfeasibleBudgetError(VramfitError):
    """No recipe fits the weight budget. Under the `VramfitError` root.

    Attributes:
        gap_bytes (int): How far the best possible total overshoots the
            budget.
        minimum_bytes (int): The smallest achievable total (pins
            respected).
        weight_budget_bytes (int): The budget that could not be met.
        runtime (str | None): Target runtime the solve was constrained
            to, when one was given.
        dropped_precisions (tuple[int, ...]): Scanned precisions the
            runtime cannot serve — the floor the message reports
            excludes them.
        held_count (int): Groups the map does not measure, held at
            reference precision (ADR-0029).
        held_bytes (int): What those groups reserve.

    Examples:
        Report the gap to the user:

        ```python
        from vramfit.domain.solver import InfeasibleBudgetError

        try:
            solve_with_tiny_budget()
        except InfeasibleBudgetError as exc:
            print(f"over budget by {exc.gap_bytes} bytes")
        ```
    """

    def __init__(  # noqa: PLR0913 - one keyword per clause the message can carry
        self,
        gap_bytes: int,
        minimum_bytes: int,
        weight_budget_bytes: int,
        *,
        runtime: str | None = None,
        dropped_precisions: tuple[int, ...] = (),
        protected_count: int = 0,
        held_count: int = 0,
        held_bytes: int = 0,
    ) -> None:
        """Build the error from the budget arithmetic.

        The message renders every size with `format_size`, so the CLI
        can print it verbatim as its ``error:`` line. When a runtime
        filter removed scanned precisions, the message names them —
        the reported floor is higher than the scan alone allows, and
        the user must see why. Protections raise the floor the same
        way (ADR-0022), so the message counts them too.

        Args:
            gap_bytes: Overshoot of the smallest achievable total.
            minimum_bytes: The smallest achievable total in bytes.
            weight_budget_bytes: The budget that could not be met.
            runtime: Target runtime that constrained the solve, when
                one was given.
            dropped_precisions: Scanned precisions the runtime cannot
                serve.
            protected_count: Tensors held at a protection floor.
            held_count: Groups the map does not measure, held at
                reference precision (ADR-0029).
            held_bytes: What those groups reserve.
        """
        message = (
            f"no recipe fits the {format_size(weight_budget_bytes)} weight "
            f"budget — minimum achievable is {format_size(minimum_bytes)} "
            f"({format_size(gap_bytes)} over)"
        )
        if runtime is not None and dropped_precisions:
            message += (
                f'. The target runtime "{runtime}" cannot serve the scanned '
                f"precisions {list(dropped_precisions)}, so the floor sits "
                f"higher than the scan alone allows"
            )
        if protected_count:
            message += (
                f". Protections hold {protected_count} tensors at their "
                "floors, raising the minimum (ADR-0022)"
            )
        if held_count:
            message += (
                f". The checkpoint holds {held_count} groups the map does "
                f"not measure, reserving {format_size(held_bytes)} at "
                "reference precision (ADR-0029). Scan them to spend it"
            )
        super().__init__(message)
        self.gap_bytes = gap_bytes
        self.minimum_bytes = minimum_bytes
        self.weight_budget_bytes = weight_budget_bytes
        self.runtime = runtime
        self.dropped_precisions = dropped_precisions
        self.protected_count = protected_count
        self.held_count = held_count
        self.held_bytes = held_bytes
