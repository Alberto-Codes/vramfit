"""Spread placement rule for cheapest-width expert-stack downgrades.

Implements the ADR-0007 amendment of 2026-08-21 (issue #321, built
under #374). The greedy selection key is unchanged — the rule only
narrows which downgrades to the cheapest in-budget width are
candidates, and it reads on expert-stack groups alone. Dense groups
keep the plain damage-per-byte order. Two clauses, first clause
first:

1. The allocator refuses a second cheapest-width stack in a layer
   while a layer with none remains. A layer remains while it still
   holds a stack the loop can take to the cheapest width — unpinned,
   above that width, and freeing bytes. A layer that cannot take one
   never blocks the others, so every budget the ADR-0007 precheck
   accepts stays solvable.
2. Within the layers clause 1 admits, only the stack the map prices
   cheapest at the candidate width stays a candidate — the
   projection tie-break. Stacks at one price all stay, and the
   selection key's group-name term breaks the remaining tie.

The cheapest in-budget width is the smallest candidate the solver
holds after the ADR-0013 capability filter. The group name carries
the (layer, projection) relation, so the rule derives the layer from
the name ([vramfit.domain.scan.layer_prefix][]) and adds no schema
field. The rule is a deterministic function of the allocation state,
so recipes stay deterministic and input-order invariant.

Examples:
    Clause 1 refuses layer 0's second stack while layer 1 has none:

    ```python
    from vramfit.domain.placement import refused_cheapest_stack_moves

    refused = refused_cheapest_stack_moves(
        map_.groups,
        pinned={},
        state={up0: 2, down0: 8, up1: 8, down1: 8},
        cheapest=2,
        size=size,
    )
    assert down0 in refused
    ```

See Also:
    - [vramfit.domain.solver][]: The greedy loop this rule narrows.
    - [vramfit.domain.scan][]: `is_expert_stack` and `layer_prefix`,
      the naming predicates the rule reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from vramfit.domain.model import LayerGroup
from vramfit.domain.scan import is_expert_stack, layer_prefix


def refused_cheapest_stack_moves(
    groups: Sequence[LayerGroup],
    pinned: Mapping[str, int],
    state: Mapping[str, int],
    cheapest: int,
    size: Callable[[LayerGroup, int], int],
) -> frozenset[str]:
    """Name the stacks whose cheapest-width downgrade the rule refuses.

    A stack pinned at the cheapest width counts toward its layer —
    the clause reads the allocation state, not the move history.

    Args:
        groups: The map's measured groups, any granularity.
        pinned: Group names the downgrade loop never moves.
        state: Current precision per group name.
        cheapest: The smallest candidate width after the ADR-0013
            capability filter.
        size: The solver's group-size predictor, protections and
            effective bits included — eligibility must match the
            downgrade loop's own freeing test.

    Returns:
        Expert-stack group names refused at ``cheapest`` in the
        current state. Other moves are never named here.
    """
    at_cheapest: dict[str, int] = {}
    movers: dict[str, list[LayerGroup]] = {}
    for group in groups:
        if not is_expert_stack(group.name):
            continue
        layer = layer_prefix(group.name)
        if layer is None:  # pragma: no cover - a stack name carries a layer
            continue
        current = state[group.name]
        if current == cheapest:
            at_cheapest[layer] = at_cheapest.get(layer, 0) + 1
        elif (
            group.name not in pinned
            and current > cheapest
            and size(group, current) > size(group, cheapest)
        ):
            movers.setdefault(layer, []).append(group)
    a_layer_remains = any(not at_cheapest.get(layer) for layer in movers)
    refused: set[str] = set()
    for layer, layer_movers in movers.items():
        if a_layer_remains and at_cheapest.get(layer):
            refused.update(g.name for g in layer_movers)
            continue
        floor_price = min(g.sensitivity[cheapest] for g in layer_movers)
        refused.update(
            g.name for g in layer_movers if g.sensitivity[cheapest] > floor_price
        )
    return frozenset(refused)
