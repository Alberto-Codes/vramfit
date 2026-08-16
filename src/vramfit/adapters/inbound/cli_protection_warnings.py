"""Warnings the plan command raises about protection input.

A protection rule or an imatrix-exclusion glob that changed nothing
must not read as protection applied (ADR-0022, ADR-0023, issue #59).
These warnings report the command's own flags, not artifact content.
The CLI's app callback routes the artifact-field reports (#261).

Examples:
    Report the gaps in one solved plan:

    ```python
    from vramfit.adapters.inbound.cli_protection_warnings import (
        warn_protection_gaps,
    )

    warn_protection_gaps(protections, exclusions, map_, state, runtime)
    ```

See Also:
    - [vramfit.adapters.inbound.cli][]: The plan command that calls it.
    - [vramfit.domain.protection][]: The expansion rules it reports on.
"""

from __future__ import annotations

import typer

from vramfit.domain.model import SensitivityMap
from vramfit.domain.protection import (
    expand_exclusions,
    expand_protections,
    noop_protected_tensors,
    noop_protection_patterns,
    overreaching_exclusion_patterns,
)


def warn_protection_gaps(
    protections: dict[str, int],
    exclusions: tuple[str, ...],
    map_: SensitivityMap,
    state: dict[str, int],
    runtime: str,
) -> None:
    """Warn about protection input that changed nothing.

    A protection that changed nothing must not read as protection
    applied (ADR-0022), and an exclusion glob that reaches outside
    the protected set must not read as full coverage (ADR-0023).
    A dead rule warns once per pattern. A dropped no-op pair warns
    per tensor unless its rule already warned as fully dead
    (issue #59). The solver already validated the rules, so
    re-expansion here cannot fail.

    Args:
        protections: The verbatim pattern-to-floor rules.
        exclusions: The verbatim ``--exclude-imatrix`` patterns.
        map_: The solved sensitivity map.
        state: Final assigned precision per group name.
        runtime: The target runtime the floors were validated for.
    """
    floors = expand_protections(protections, map_, runtime)
    for pattern in noop_protection_patterns(protections, map_, state, floors):
        typer.echo(
            f'warning: --protect "{pattern}={protections[pattern]}" is a '
            "no-op — every tensor it governs already meets the floor, "
            "or a later rule overrides it",
            err=True,
        )
    group_of = {t: g.name for g in map_.groups for t in g.tensors}
    excluded = expand_exclusions(exclusions, floors, map_)
    for name in noop_protected_tensors(protections, map_, state, floors):
        group = group_of[name]
        message = (
            f'warning: protection floor {floors[name]} on "{name}" is a '
            f'per-tensor no-op — group "{group}" already packs at '
            f"{state[group]}-bit. The recipe drops the pair."
        )
        if name in excluded:
            message += " Its imatrix exclusion drops with it (ADR-0023)."
        typer.echo(message, err=True)
    overreach = overreaching_exclusion_patterns(exclusions, floors, map_)
    for pattern, outside in overreach.items():
        typer.echo(
            f'warning: --exclude-imatrix "{pattern}" also matches '
            f'{len(outside)} unprotected tensors (first: "{outside[0]}") '
            "— their imatrix rows stay (ADR-0023)",
            err=True,
        )
