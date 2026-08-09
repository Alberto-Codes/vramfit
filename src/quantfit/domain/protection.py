"""Within-layer protections: expansion, validation, and size pricing.

Implements ADR-0022 and the imatrix-exclusion side of ADR-0023. A
protection is an ordered fnmatch pattern over
tensor names plus a protection floor. A protected tensor packs at the
higher of its group's assignment and the floor. An imatrix exclusion
is an fnmatch pattern over *protected* tensor names — a matched
tensor keeps its promotion and quantizes without its imatrix row,
the fit-collapse remedy. The solver prices a
protection by size only — predicted damage stays the group-level sum,
and no per-tensor damage is invented. Validation is total: an
unservable floor, a no-match pattern, a protection on a single-tensor
group, a map without tensor sizes, and an exclusion that matches no
protected tensor each raise `ProtectionError`
before any solving starts. Nothing about a protection is silent —
the dead rule (`noop_protection_patterns`), the dropped per-tensor
no-op pair (`noop_protected_tensors`), and the overreaching
exclusion glob (`overreaching_exclusion_patterns`) surface as CLI
warnings after solving.

Examples:
    Expand protections against a map:

    ```python
    from quantfit.domain.protection import expand_protections

    floors = expand_protections(
        {"*.self_attn.v_proj.weight": 5}, map_, runtime="llama.cpp"
    )
    ```

See Also:
    - [quantfit.domain.solver][]: Prices protected groups through
      `protected_group_bytes`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase

from quantfit.domain.errors import QuantfitError
from quantfit.domain.model import (
    LayerGroup,
    ProtectedTensor,
    SensitivityMap,
)
from quantfit.domain.runtime import RUNTIME_CAPABILITIES


class ProtectionError(QuantfitError, ValueError):
    """A ``--protect`` rule is unusable. Under the `QuantfitError` root.

    Raised when a protection floor is not servable by the target
    runtime, a pattern matches no tensor, a pattern matches a
    single-tensor group (where ``--pin`` is the right tool), or the
    map records no per-tensor sizes for a matched group (ADR-0022).

    Examples:
        A protection against a map without tensor sizes:

        ```python
        from quantfit.domain.protection import ProtectionError

        try:
            solve_with_protections()
        except ProtectionError as exc:
            print(exc)
        ```
    """


def expand_protections(
    protections: Mapping[str, int],
    map_: SensitivityMap,
    runtime: str | None,
) -> dict[str, int]:
    """Resolve protection patterns to concrete per-tensor floors.

    Args:
        protections: Ordered mapping of fnmatch pattern to protection
            floor; later patterns override earlier ones for
            overlapping tensors, like pins.
        map_: The sensitivity map whose group tensors are matched.
        runtime: Target runtime whose capability table gates the
            floors (ADR-0013), or None for an unconstrained plan.

    Returns:
        Mapping of tensor name to protection floor.

    Raises:
        ProtectionError: If a floor is not servable by the runtime, a
            pattern matches no tensor, a pattern matches a tensor in
            a single-tensor group, or a matched group has no
            ``tensor_bytes`` record.
    """
    tensor_group: dict[str, LayerGroup] = {
        tensor: group for group in map_.groups for tensor in group.tensors
    }
    floors: dict[str, int] = {}
    for pattern, floor in protections.items():
        if runtime is not None and floor not in RUNTIME_CAPABILITIES[runtime]:
            raise ProtectionError(
                f'protection "{pattern}={floor}": runtime "{runtime}" cannot '
                f"serve {floor}-bit — it serves "
                f"{sorted(RUNTIME_CAPABILITIES[runtime], reverse=True)} (ADR-0013)"
            )
        matched = [name for name in tensor_group if fnmatchcase(name, pattern)]
        if not matched:
            raise ProtectionError(f'protection "{pattern}={floor}" matches no tensor')
        for name in matched:
            group = tensor_group[name]
            if len(group.tensors) == 1:
                raise ProtectionError(
                    f'protection "{pattern}={floor}" matches "{name}", the only '
                    f'tensor of group "{group.name}" — pin the group instead: '
                    f'--pin "{group.name}={floor}"'
                )
            if not group.tensor_bytes:
                raise ProtectionError(
                    f'protection "{pattern}={floor}": group "{group.name}" has '
                    "no tensor_bytes record — re-scan, or backfill the map "
                    "from the checkpoint's safetensors headers (ADR-0022)"
                )
            floors[name] = floor
    return floors


def expand_exclusions(
    exclusions: tuple[str, ...],
    floors: Mapping[str, int],
    map_: SensitivityMap,
) -> frozenset[str]:
    """Resolve exclusion patterns to concrete protected tensor names.

    An imatrix exclusion rides a protection (ADR-0023): only a
    protected tensor can drop its imatrix row, because the known fit
    collapses live in promotions under an imatrix. A pattern that
    matches no protected tensor refuses loudly. A pattern that
    matches both sides expands to its protected matches only — the
    caller warns about the rest through
    `overreaching_exclusion_patterns`.

    Args:
        exclusions: Exclusion patterns, in rule order.
        floors: The expanded per-tensor protection floors.
        map_: The sensitivity map whose tensors diagnose a miss.

    Returns:
        The excluded protected tensor names.

    Raises:
        ProtectionError: If exclusions are given without protections,
            a pattern matches no protected tensor, or a pattern
            matches only unprotected tensors.
    """
    if exclusions and not floors:
        raise ProtectionError(
            "imatrix exclusions require protections — an exclusion keeps a "
            "protected tensor's promotion and drops its imatrix row (ADR-0023)"
        )
    all_tensors = [name for group in map_.groups for name in group.tensors]
    excluded: set[str] = set()
    for pattern in exclusions:
        matched = [name for name in floors if fnmatchcase(name, pattern)]
        if not matched:
            unprotected = [name for name in all_tensors if fnmatchcase(name, pattern)]
            if unprotected:
                raise ProtectionError(
                    f'imatrix exclusion "{pattern}" matches only unprotected '
                    f'tensors (first: "{unprotected[0]}") — protect them '
                    "first, or narrow the pattern (ADR-0023)"
                )
            raise ProtectionError(f'imatrix exclusion "{pattern}" matches no tensor')
        excluded.update(matched)
    return frozenset(excluded)


def overreaching_exclusion_patterns(
    exclusions: tuple[str, ...],
    floors: Mapping[str, int],
    map_: SensitivityMap,
) -> dict[str, tuple[str, ...]]:
    """Name each exclusion pattern's unprotected matches.

    A pattern that matches protected and unprotected tensors expands
    to the protected subset only — correct, but the truncation must
    not be silent (ADR-0023). The CLI warns with this record, as it
    does for no-op protections.

    Args:
        exclusions: Exclusion patterns, in rule order.
        floors: The expanded per-tensor protection floors.
        map_: The sensitivity map whose tensors are matched.

    Returns:
        Mapping of overreaching pattern to its unprotected matches,
        in rule order. Empty when every pattern stays inside the
        protected set.
    """
    all_tensors = [name for group in map_.groups for name in group.tensors]
    overreach: dict[str, tuple[str, ...]] = {}
    for pattern in exclusions:
        outside = tuple(
            name
            for name in all_tensors
            if name not in floors and fnmatchcase(name, pattern)
        )
        if outside and any(fnmatchcase(name, pattern) for name in floors):
            overreach[pattern] = outside
    return overreach


def protected_group_bytes(
    group: LayerGroup,
    bits: int,
    floors: Mapping[str, int],
    price: Callable[[int, int], int],
) -> int:
    """Predict a group's size with its protected tensors held at floor.

    Each protected tensor prices at the higher of the candidate
    precision and its floor. Unprotected bytes price at the candidate
    precision in one piece, so a group without protections prices
    exactly as before.

    Args:
        group: The group to price.
        bits: Candidate precision for the group.
        floors: Protection floor per tensor name.
        price: Size predictor ``(bytes_fp16, nominal_bits) -> bytes``
            carrying the solver's effective-bits table and overhead.

    Returns:
        Predicted bytes at the candidate precision.
    """
    protected = [name for name in group.tensors if name in floors]
    if not protected:
        return price(group.bytes_fp16, bits)
    plain = group.bytes_fp16 - sum(group.tensor_bytes[name] for name in protected)
    total = price(plain, bits) if plain > 0 else 0
    for name in protected:
        total += price(group.tensor_bytes[name], max(bits, floors[name]))
    return total


def resolve_protected(
    map_: SensitivityMap,
    state: Mapping[str, int],
    floors: Mapping[str, int],
    excluded: frozenset[str] = frozenset(),
) -> tuple[ProtectedTensor, ...]:
    """Record the resolved (tensor, precision) pairs for the recipe.

    A pair exists only where the floor exceeds the group's
    assignment. A floor the assignment already meets is a per-tensor
    no-op: its pair would quantize identically in the protected pack
    and the unprotected reference, and the reconstruction check's
    strict inequality would read the tie as a collapse (ADR-0022).
    `noop_protected_tensors` names the dropped tensors for the CLI
    warning.

    Args:
        map_: The sensitivity map, fixing tensor order.
        state: Final assigned precision per group name.
        floors: Protection floor per tensor name.
        excluded: Protected tensors that quantize without their
            imatrix rows (ADR-0023).

    Returns:
        One `ProtectedTensor` per effective protection, in map
        order, each at its floor, marked ``exclude_imatrix`` when
        excluded. A dropped no-op pair takes its imatrix exclusion
        with it — the exclusion rides the protection (ADR-0023).
    """
    return tuple(
        ProtectedTensor(
            tensor=name,
            bits=floors[name],
            exclude_imatrix=name in excluded,
        )
        for group in map_.groups
        for name in group.tensors
        if name in floors and floors[name] > state[group.name]
    )


def noop_protected_tensors(
    map_: SensitivityMap,
    state: Mapping[str, int],
    floors: Mapping[str, int],
) -> tuple[str, ...]:
    """Name the protected tensors whose floor changed nothing.

    `resolve_protected` drops these pairs from the recipe. The
    per-pattern warning cannot see them: a glob that lifts 47 real
    floors and no-ops on its 48th match is not a dead rule, yet its
    48th pair would falsely fail the reconstruction gate. The CLI
    warns per tensor with this record (ADR-0022).

    Args:
        map_: The sensitivity map, fixing tensor order.
        state: Final assigned precision per group name.
        floors: Protection floor per tensor name.

    Returns:
        The dropped tensor names, in map order.
    """
    return tuple(
        name
        for group in map_.groups
        for name in group.tensors
        if name in floors and floors[name] <= state[group.name]
    )


def noop_protection_patterns(
    protections: Mapping[str, int],
    map_: SensitivityMap,
    state: Mapping[str, int],
    floors: Mapping[str, int],
) -> tuple[str, ...]:
    """Name the protection patterns that changed nothing.

    A pattern is a no-op in two cases: every tensor it governs sits
    in a group whose final assignment already meets the floor, or a
    later rule overrides every tensor it matched — a dead rule, and
    a same-floor override still kills the earlier rule. The
    CLI warns either way — a silent no-op would read as protection
    applied (ADR-0022).

    Args:
        protections: The verbatim pattern-to-floor rules.
        map_: The sensitivity map, for tensor-to-group lookup.
        state: Final assigned precision per group name.
        floors: The expanded per-tensor floors (later patterns
            already override earlier ones).

    Returns:
        The no-op patterns, in rule order.
    """
    group_of: dict[str, str] = {
        tensor: group.name for group in map_.groups for tensor in group.tensors
    }
    # Replay the rules in order so the *last* matching pattern governs
    # each tensor — matching on floor value would credit an earlier
    # rule a same-floor successor overrides.
    governed_by: dict[str, str] = {}
    for pattern in protections:
        for name in floors:
            if fnmatchcase(name, pattern):
                governed_by[name] = pattern
    noop: list[str] = []
    for pattern, floor in protections.items():
        governed = [name for name, p in governed_by.items() if p == pattern]
        if all(state[group_of[name]] >= floor for name in governed):
            noop.append(pattern)
    return tuple(noop)
