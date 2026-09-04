"""Pin resolution for the greedy solver.

Implements the pin surface of ADR-0007, as amended 2026-08-22
(issue #301): a pin may name any width the target runtime serves,
beyond the map's candidate set, and it may land on any
checkpoint-discovered group, beyond the map's groups. A pinned
group never enters the downgrade loop, so a pin orders no groups
the map did not measure. At a width the map never measured the
assignment records 0.0 damage, the way an uncovered held group
does. An unquantizable-class group holds at the F16 passthrough
(ADR-0012, 2026-08-20 amendment). A pattern that resolves to that
one group refuses. A pattern that resolves to more than one group
skips the held group instead, and `held_pin_skips` names each skip
for the caller to warn about (ADR-0007, 2026-09-04 amendment, #371).

Examples:
    Resolve dense pins at nominal 8 beside a stack-keyed map:

    ```python
    from vramfit.domain.pins import resolve_pins

    pinned, uncovered_pins, user_pinned = resolve_pins(
        {"model.layers.*.mixer.in_proj": 8},
        map_,  # a stack-keyed vramfit.domain.model.SensitivityMap
        candidates=(4, 2),
        runtime="llama.cpp",
        discovered_bytes=discovered,  # from discovered_group_bytes
    )
    ```

See Also:
    - [vramfit.domain.solver][]: The caller. `resolve_pins` runs
      before the downgrade loop and `assignment_damage` prices the
      final assignments.
    - [vramfit.domain.solver_errors][]: `PinError`.
"""

from __future__ import annotations

from collections.abc import Mapping

from vramfit.domain.model import LayerGroup, SensitivityMap
from vramfit.domain.pin_skips import HeldPinSkip, match_pattern
from vramfit.domain.runtime import RUNTIME_CAPABILITIES, unquantizable_filter
from vramfit.domain.sizes import REFERENCE_BITS
from vramfit.domain.solver_errors import PinError


def _expand_pins(
    pins: Mapping[str, int],
    sensitivity_map: SensitivityMap,
    candidates: tuple[int, ...],
    runtime: str | None,
    discovered_bytes: Mapping[str, int] | None,
) -> dict[str, int]:
    """Resolve pin patterns to concrete per-group precisions.

    A pin may name any width the target runtime serves, beyond the
    map's candidate set, and it may land on any checkpoint-discovered
    group (the 2026-08-22 ADR-0007 amendment). Without a runtime the
    candidate set still bounds the width, and without a size source
    the map's groups still bound the match. A pattern that resolves
    to more than one group skips the unquantizable-class groups it
    sweeps (ADR-0007, 2026-09-04 amendment, #371).

    Args:
        pins: Ordered mapping of glob pattern to forced precision.
            Later patterns override earlier ones for overlapping
            groups.
        sensitivity_map: The map whose groups are matched.
        candidates: The solver's candidate precisions — the scanned
            set, runtime-filtered when a target runtime is given.
        runtime: Target runtime name, or None. Its capability table
            widens the allowed pin widths.
        discovered_bytes: Bytes per checkpoint-discovered group
            (ADR-0029), or None. Its names widen the match universe.

    Returns:
        Mapping of group name to pinned precision. Empty when the
        caller passed no pins — the match universe is never built.

    Raises:
        PinError: If a pin uses a precision neither scanned nor
            runtime-servable, or matches no group.
    """
    if not pins:
        return {}
    allowed = set(candidates) | RUNTIME_CAPABILITIES.get(runtime, frozenset())
    # Sorted, so the expansion order is structural rather than an
    # accident of set iteration — recipes stay deterministic
    # (ADR-0007).
    names = sorted(
        {g.name for g in sensitivity_map.groups} | set(discovered_bytes or {})
    )
    pinned: dict[str, int] = {}
    for pattern, bits in pins.items():
        if bits not in allowed:
            raise PinError(
                f'pin "{pattern}={bits}": precision {bits} is not in the candidate '
                f"set {sorted(set(candidates), reverse=True)}"
                + (f' and runtime "{runtime}" does not serve it' if runtime else "")
            )
        matched, _skipped = match_pattern(pattern, names, runtime)
        if not matched and not _skipped:
            raise PinError(f'pin "{pattern}={bits}" matches no group')
        for name in matched:
            pinned[name] = bits
    return pinned


def _hold_unquantizable(
    sensitivity_map: SensitivityMap,
    pinned: dict[str, int],
    runtime: str | None,
) -> dict[str, int]:
    """Pin every unquantizable-class group at the F16 passthrough.

    Such a group holds at the passthrough whatever the map measured.
    The runtime's quantizer refuses its tensors through a name
    filter, so a lower width would record a type the artifact cannot
    carry (ADR-0012, 2026-08-20 amendment). The hold enters
    ``pinned``, which the downgrade loop never touches.

    Args:
        sensitivity_map: Damage curves for every group.
        pinned: Resolved user pins, updated in place.
        runtime: Target runtime name, or None — only a runtime with
            a filter table holds anything.

    Returns:
        The same ``pinned`` mapping, holds added.

    Raises:
        PinError: If a user pin lands on such a group. A pin below
            the passthrough asks for what the record refuses, and a
            pin at the passthrough is redundant — the message says
            which.
    """
    for group in sensitivity_map.groups:
        filter_name = unquantizable_filter(group.name, runtime)
        if filter_name is None:
            continue
        if group.name in pinned:
            raise PinError(
                _unquantizable_message(
                    group.name, runtime, filter_name, pinned[group.name]
                )
            )
        pinned[group.name] = REFERENCE_BITS
    return pinned


def _unquantizable_message(
    name: str, runtime: str | None, filter_name: str, bits: int
) -> str:
    """Word the refusal of a pin on an unquantizable-class group.

    Args:
        name: The pinned group.
        runtime: Target runtime name.
        filter_name: The upstream filter that refuses the tensors.
        bits: The pinned width, so a redundant pin reads as one.

    Returns:
        The `PinError` message.
    """
    base = (
        f'group "{name}" holds at the F16 passthrough — '
        f'runtime "{runtime}" refuses its tensors through the '
        f'"{filter_name}" filter (ADR-0012, 2026-08-20 '
        f"amendment)"
    )
    if bits == REFERENCE_BITS:
        return base + ", so the pin is redundant — remove it"
    return base + ", so a pin cannot move it"


def _refuse_unquantizable_pins(
    uncovered_pins: Mapping[str, int], runtime: str | None
) -> None:
    """Refuse a pin on an unquantizable-class uncovered group.

    An uncovered group of such a class holds at the F16 passthrough
    (ADR-0012, 2026-08-20 amendment). The widened pin surface must
    not move it, the way `_hold_unquantizable` refuses the same pin
    on a measured group.

    Args:
        uncovered_pins: Pins that landed on checkpoint-discovered
            groups the map does not carry.
        runtime: Target runtime name, or None.

    Raises:
        PinError: If a pin lands on such a group. A pin at the
            passthrough is redundant, and the message says so.
    """
    for name, bits in uncovered_pins.items():
        filter_name = unquantizable_filter(name, runtime)
        if filter_name is not None:
            raise PinError(_unquantizable_message(name, runtime, filter_name, bits))


def resolve_pins(
    pins: Mapping[str, int],
    sensitivity_map: SensitivityMap,
    candidates: tuple[int, ...],
    runtime: str | None,
    discovered_bytes: Mapping[str, int] | None,
) -> tuple[dict[str, int], dict[str, int], frozenset[str]]:
    """Expand, split, and guard the caller's pins.

    Args:
        pins: Ordered glob-pattern pins.
        sensitivity_map: Damage curves for every measured group.
        candidates: Runtime-filtered candidate precisions.
        runtime: Target runtime name, or None.
        discovered_bytes: Bytes per checkpoint-discovered group, or
            None.

    Returns:
        A triple: measured-group pins with the unquantizable holds
        merged in, uncovered-group pins, and the names the caller's
        pins forced.

    Raises:
        PinError: On a bad width, a matchless pattern, or a pin on an
            unquantizable-class group.
    """
    expanded = _expand_pins(
        pins, sensitivity_map, candidates, runtime, discovered_bytes
    )
    map_names = {g.name for g in sensitivity_map.groups}
    uncovered_pins = {n: b for n, b in expanded.items() if n not in map_names}
    _refuse_unquantizable_pins(uncovered_pins, runtime)
    pinned = _hold_unquantizable(
        sensitivity_map,
        {n: b for n, b in expanded.items() if n in map_names},
        runtime,
    )
    return pinned, uncovered_pins, frozenset(expanded)


def held_pin_skips(
    pins: Mapping[str, int],
    sensitivity_map: SensitivityMap,
    runtime: str | None,
    discovered_bytes: Mapping[str, int] | None,
) -> tuple[HeldPinSkip, ...]:
    """List the held groups the caller's multi-group pins skipped.

    A pattern that resolves to more than one group skips every
    unquantizable-class group it sweeps, because the group holds at
    the F16 passthrough and no pin can move it (ADR-0007, 2026-09-04
    amendment, #371). The caller prints one warning per skip. A
    pattern that resolves to exactly one group never skips: it
    refuses through `resolve_pins` instead.

    Args:
        pins: Ordered glob-pattern pins, as given to `resolve_pins`.
        sensitivity_map: Damage curves for every measured group.
        runtime: Target runtime name, or None.
        discovered_bytes: Bytes per checkpoint-discovered group, or
            None.

    Returns:
        One entry per skipped group, in pattern order and then name
        order. A group two patterns sweep appears once per pattern.
    """
    names = sorted(
        {g.name for g in sensitivity_map.groups} | set(discovered_bytes or {})
    )
    skips: list[HeldPinSkip] = []
    for pattern, bits in pins.items():
        _matched, skipped = match_pattern(pattern, names, runtime)
        skips.extend(
            HeldPinSkip(group=name, pattern=pattern, bits=bits, filter=filter_name)
            for name, filter_name in skipped
        )
    return tuple(skips)


def assignment_damage(
    group: LayerGroup, bits: int, user_pinned: frozenset[str]
) -> float:
    """Record one assignment's marginal damage.

    A reference-held group carries no damage row for the passthrough,
    and reference precision is the zero-damage baseline. A pinned
    group at a width the map never measured records 0.0 the same way
    (the 2026-08-22 ADR-0007 amendment) — `predicted_damage` sums
    measured marginals only. Every other state value is a scanned
    candidate, so any other missing key stays a loud KeyError.

    Args:
        group: The assigned group.
        bits: Its final precision.
        user_pinned: Names the caller's pins forced.

    Returns:
        The marginal damage the recipe records.
    """
    if bits == REFERENCE_BITS:
        return group.sensitivity.get(REFERENCE_BITS, 0.0)
    if group.name in user_pinned and bits not in group.sensitivity:
        return 0.0
    return group.sensitivity[bits]
