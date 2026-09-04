"""Checkpoint tensor sizes: the size source `plan` reads (ADR-0029).

`plan` used to treat its input map's group list as the model, so a
group the map omitted contributed zero bytes (#337). ADR-0029 gives
the plan step a size source independent of the map. This module holds
the pure half of that source.

`TensorSize` is what the port carries: one checkpoint tensor's stored
bytes and the dtype it is stored at (decision 5). The dtype is there
so the domain can recover the element count rather than assume two
bytes per parameter — a checkpoint at fp32 or fp8 counts differently.
`reference_bytes` does that recovery. Reference precision stays
16-bit throughout, per the glossary, because the solver's size model
prices every precision against a 16-bit base.

`reconcile_root` maps a checkpoint tensor name onto the naming root
the maps use, against an explicit root table and never a prefix
wildcard (decision 7). #177 measured what a wildcard costs: it mapped
a vision tower's ``layers.5`` onto the decoder's ``blk.5`` and would
have priced it against the wrong columns. A checkpoint rooted outside
the table refuses.

`discovered_group_bytes` sums the reference bytes of every tensor into
the group the map would name, so the checkpoint's 128 per-expert
tensors become one stack group (decision 6). Group naming stays
`vramfit.domain.scan.group_key`, so one rule serves the meter and the
size source. A tensor of an unquantizable class keys by its own name
under every granularity (#204, #409): the meter skips it, so a layer
group that absorbed its bytes would hide them behind a covered name.

Attributes:
    REFERENCE_BITS (int): Bits per weight at reference precision. The
        solver's size model prices against this base.
    DTYPE_ELEMENT_BYTES (Mapping[str, int]): Bytes per element for
        each safetensors float dtype the source reads. A dtype
        outside the table refuses — an integer checkpoint holds no
        reference precision to price against.
    MAP_ROOT (str): The naming root every sensitivity map this
        project holds emits.
    CHECKPOINT_ROOTS (Mapping[str, str]): Checkpoint naming root to
        the map root it reconciles onto. The explicit table decision
        7 requires. Each new target costs one entry.

Examples:
    Sum a checkpoint's per-expert tensors into stack groups:

    ```python
    from vramfit.domain.sizes import TensorSize, discovered_group_bytes

    sizes = {
        "backbone.layers.1.mixer.experts.0.up_proj.weight": TensorSize(
            dtype="BF16", bytes=8
        ),
        "backbone.layers.1.mixer.experts.1.up_proj.weight": TensorSize(
            dtype="BF16", bytes=8
        ),
    }
    assert discovered_group_bytes(sizes, "stack") == {
        "model.layers.1.mixer.experts.up_proj": 16
    }
    ```

`held_assignments` turns the uncovered set into the recipe rows decision
3 requires. It lives here rather than in the solver, because it states
what the size source implies and not how the budget is spent. It also
refuses a plan with no size source on a map whose family holds a class
the scan skips, because nothing else prices that class (the 2026-09-04
decision 3 amendment). The solver adds its bytes to the total and never
ranks a downgrade for one. The solver's predictor prices each: an
uncovered expert stack through the ADR-0028 table, a layer-class group
whose rows refuse the 256 super-block the same way (the 2026-08-20
amendment), and an unquantizable class at the convert dtype (#409).

See Also:
    - [vramfit.ports.outbound][]: `TensorSizeSource`, the port that
      carries these values.
    - [vramfit.domain.solver][]: Prices every discovered group.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from vramfit.domain.errors import VramfitError
from vramfit.domain.model import Assignment, SensitivityMap
from vramfit.domain.runtime import (
    RUNTIME_CAPABILITIES,
    RuntimeCapabilityError,
    missing_unquantizable_module,
    unquantizable_class,
)
from vramfit.domain.scan import group_key, matches_a_layer

REFERENCE_BITS: Final[int] = 16

# Safetensors spells its dtypes this way. Only float dtypes appear: a
# checkpoint stored at int8 is already quantized, and no reference
# precision exists to price its groups against.
DTYPE_ELEMENT_BYTES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }
)

MAP_ROOT: Final[str] = "model."

# The explicit root table (ADR-0029 decision 7). The 30B target's
# checkpoint roots at `backbone.` and its maps root at `model.`, so
# the two must be reconciled before a name reaches a group. A prefix
# wildcard would do it in one line and would price a vision tower's
# tensors against a decoder group (#177).
CHECKPOINT_ROOTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "backbone.": MAP_ROOT,
        "model.": MAP_ROOT,
    }
)


class SizeSourceError(VramfitError, ValueError):
    """The size source cannot price the checkpoint it was given.

    Raised for a dtype outside `DTYPE_ELEMENT_BYTES`, a stored size
    that is not a whole number of elements, and a layer-bearing
    tensor name rooted outside `CHECKPOINT_ROOTS`. Also raised for a
    plan with no size source on a map whose family holds a class the
    scan skips (#204, #409): nothing else prices that class. Under
    the `VramfitError` root (ADR-0011) with a message the CLI prints
    verbatim. The safetensors adapter raises it for file-level
    refusals too, so one catch covers the whole source.

    Examples:
        An unknown dtype refuses:

        ```python
        from vramfit.domain.sizes import SizeSourceError, TensorSize, reference_bytes

        try:
            reference_bytes("w", TensorSize(dtype="I8", bytes=4))
        except SizeSourceError as exc:
            print(exc)
        ```
    """


@dataclass(frozen=True, slots=True)
class TensorSize:
    """One checkpoint tensor's stored size and dtype (ADR-0029 decision 5).

    Attributes:
        dtype (str): The dtype the checkpoint stores the tensor at, as
            the shard header spells it, e.g. ``BF16``. The adapter
            reports it verbatim and invents no convention.
        bytes (int): The tensor's stored size in bytes.

    Examples:
        One expert of the 30B target's up projection:

        ```python
        from vramfit.domain.sizes import TensorSize

        size = TensorSize(dtype="BF16", bytes=9_977_856)
        ```
    """

    dtype: str
    bytes: int

    def __post_init__(self) -> None:
        """Enforce the record's invariants.

        Raises:
            ValueError: If ``dtype`` is empty or ``bytes`` is not
                positive.
        """
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        if self.bytes <= 0:
            raise ValueError("bytes must be positive")


def reference_bytes(tensor: str, size: TensorSize) -> int:
    """Convert one stored tensor size to bytes at reference precision.

    The stored size divides by the dtype's element size to recover the
    element count, which then prices at `REFERENCE_BITS`. Reading the
    stored bytes as reference bytes would overstate an fp32 checkpoint
    by a factor of two (ADR-0029 decision 5).

    Args:
        tensor: The tensor's name, named in every refusal.
        size: The stored size and dtype the source read.

    Returns:
        The tensor's size in bytes at reference precision.

    Raises:
        SizeSourceError: If the dtype is outside
            `DTYPE_ELEMENT_BYTES`, or the stored size is not a whole
            number of elements of that dtype. `TensorSize` already
            refuses a non-positive size, so a zero element count
            cannot reach here.

    Examples:
        An fp32 tensor prices at half its stored size:

        ```python
        from vramfit.domain.sizes import TensorSize, reference_bytes

        assert reference_bytes("w", TensorSize(dtype="F32", bytes=16)) == 8
        ```
    """
    try:
        element = DTYPE_ELEMENT_BYTES[size.dtype]
    except KeyError:
        raise SizeSourceError(
            f'tensor "{tensor}": dtype "{size.dtype}" has no reference size '
            f"— the table covers {sorted(DTYPE_ELEMENT_BYTES)}"
        ) from None
    count, remainder = divmod(size.bytes, element)
    if remainder:
        raise SizeSourceError(
            f'tensor "{tensor}": {size.bytes} bytes is not a whole number of '
            f'"{size.dtype}" elements of {element} bytes'
        )
    return count * REFERENCE_BITS // 8


def reconcile_root(tensor: str) -> str:
    """Rewrite one checkpoint tensor name onto the map's naming root.

    The table is explicit and the match is a whole root, never a
    prefix wildcard (ADR-0029 decision 7). A name carrying no known
    root passes through when it names no decoder layer, which is how
    ``lm_head.weight`` reaches its own group. A layer-bearing name
    under an unknown root refuses instead: that is the case a
    wildcard would price against the wrong groups (#177).

    Args:
        tensor: A checkpoint tensor name, e.g.
            ``backbone.layers.1.mixer.experts.0.up_proj.weight``.

    Returns:
        The name under `MAP_ROOT`.

    Raises:
        SizeSourceError: If the name carries a decoder-layer prefix
            under a root the table does not name.

    Examples:
        The 30B target's root reconciles onto the map's:

        ```python
        from vramfit.domain.sizes import reconcile_root

        assert reconcile_root("backbone.layers.1.mixer") == "model.layers.1.mixer"
        assert reconcile_root("lm_head.weight") == "lm_head.weight"
        ```
    """
    for root, mapped in CHECKPOINT_ROOTS.items():
        if tensor.startswith(root):
            return mapped + tensor.removeprefix(root)
    if matches_a_layer(tensor):
        raise SizeSourceError(
            f'tensor "{tensor}" names a decoder layer under a root the table '
            f"does not carry — it covers {sorted(CHECKPOINT_ROOTS)} "
            f"(ADR-0029). Add the root rather than a prefix wildcard (#177)"
        )
    return tensor


def discovered_group_bytes(
    sizes: Mapping[str, TensorSize],
    group_by: Literal["layer", "tensor", "stack"],
) -> dict[str, int]:
    """Sum a checkpoint's tensors into the groups a map would name.

    The aggregation lives here rather than in the adapter, because it
    reads model structure and structure is a domain concept (ADR-0029
    decision 6). It reuses `vramfit.domain.scan.group_key`, so the
    size source and the torch meter name a group the same way. Under
    ``stack`` granularity the 128 tensors of one routed-expert
    projection collapse into one group.

    A tensor of a class the quantizer refuses keys by its own name
    whatever the granularity. The meter skips it at discovery (#204),
    so under ``layer`` granularity the map's layer group holds no
    bytes for it. Folding it into that covered group would drop its
    bytes from the plan. Its own name is uncovered, so the solver
    holds it at the convert dtype (#409).

    Args:
        sizes: Stored size per checkpoint tensor name, from a
            `vramfit.ports.outbound.TensorSizeSource`.
        group_by: The granularity of the map being planned against,
            from its ``scan.group_by``.

    Returns:
        Bytes at reference precision per group name, under `MAP_ROOT`.
        Empty when ``sizes`` is empty.

    Raises:
        SizeSourceError: If a dtype has no reference size, or a
            layer-bearing name is rooted outside `CHECKPOINT_ROOTS`.

    Examples:
        ```python
        from vramfit.domain.sizes import TensorSize, discovered_group_bytes

        sizes = {
            "backbone.layers.0.mlp.up_proj.weight": TensorSize("BF16", 8),
            "backbone.layers.0.mixer.conv1d.weight": TensorSize("BF16", 4),
        }
        assert discovered_group_bytes(sizes, "layer") == {
            "model.layers.0": 8,
            "model.layers.0.mixer.conv1d": 4,
        }
        ```
    """
    groups: dict[str, int] = {}
    for tensor, size in sizes.items():
        name = reconcile_root(tensor)
        group = group_key(name, "tensor")
        if unquantizable_class(group) is None:
            group = group_key(name, group_by)
        groups[group] = groups.get(group, 0) + reference_bytes(tensor, size)
    return groups


def uncovered_groups(
    discovered: Mapping[str, int], covered: Collection[str]
) -> tuple[tuple[str, int], ...]:
    """Select the discovered groups the sensitivity map does not carry.

    These are the groups ADR-0029 decision 3 holds at reference
    precision. The order is by name, so a recipe planned twice from
    one checkpoint lists them identically.

    Args:
        discovered: Bytes at reference precision per discovered group,
            from `discovered_group_bytes`.
        covered: Group names the sensitivity map carries.

    Returns:
        ``(group, bytes)`` pairs for every discovered group outside
        ``covered``, sorted by group name.

    Examples:
        ```python
        from vramfit.domain.sizes import uncovered_groups

        assert uncovered_groups({"a": 1, "b": 2}, ["a"]) == (("b", 2),)
        ```
    """
    seen = set(covered)
    return tuple(
        (name, size) for name, size in sorted(discovered.items()) if name not in seen
    )


def _reference_refusal(
    runtime: str, capability: frozenset[int], held: list[str]
) -> RuntimeCapabilityError:
    """Word the refusal of a runtime that serves no reference precision.

    A group of a class the quantizer refuses gets its own wording:
    the scan skips it (#204), so no scan supplies a width, and the
    runtime has none until a pack path for it exists (#409).

    Args:
        runtime: The target runtime.
        capability: The precisions it serves.
        held: The uncovered groups holding at reference precision.

    Returns:
        The error, message built.
    """
    refused = [name for name in held if unquantizable_class(name) is not None]
    if refused:
        classes = sorted({unquantizable_class(name) or "" for name in refused})
        return RuntimeCapabilityError(
            f'runtime "{runtime}" cannot serve reference precision '
            f"{REFERENCE_BITS}, so it cannot hold the {len(refused)} groups of "
            f"a class the quantizer refuses ({', '.join(classes)}). Those "
            f'classes have no "{runtime}" width until a "{runtime}" pack path '
            f"exists (ADR-0013). It serves {sorted(capability, reverse=True)}"
        )
    return RuntimeCapabilityError(
        f'runtime "{runtime}" cannot serve reference precision '
        f"{REFERENCE_BITS}, so it cannot hold the {len(held)} "
        f"groups the map does not measure (ADR-0029). It serves "
        f"{sorted(capability, reverse=True)}. Plan without a size "
        f"source, or scan those groups"
    )


def held_assignments(
    discovered_bytes: Mapping[str, int] | None,
    sensitivity_map: SensitivityMap,
    runtime: str | None,
    price_for: Callable[[str], Callable[[int, int], int]],
    pins: Mapping[str, int] | None = None,
) -> tuple[Assignment, ...]:
    """Assign every discovered group the map does not measure.

    ADR-0029 decision 3. Each such group holds at reference precision:
    no measurement ranks a downgrade for it, so it is a constant in
    the budget and never a move. The recipe still names it, because
    `pack` runs the quantizer at the recipe's floor and would
    otherwise quantize the group the plan just reserved reference
    bytes for. A pinned uncovered group prices at the pinned width
    instead (the 2026-08-22 ADR-0007 amendment), still a constant in
    the budget and never a move.

    Args:
        discovered_bytes: Bytes at reference precision per group the
            checkpoint holds, or None for no size source.
        sensitivity_map: The map, whose groups are the covered set.
        runtime: Target runtime name, or None.
        price_for: The solver's predictor builder: a group name to
            its ``(bytes_fp16, bits) -> bytes`` predictor. The
            builder routes each group to its table, so this function
            reads no model structure of its own.
        pins: Uncovered-group pins the solver resolved and validated
            — name to precision. None means no uncovered group is
            pinned. The parameter is solver-private: this function
            checks no width itself.

    Returns:
        One assignment per uncovered group, in name order. Empty when
        no size source was given, or the map covers every group.

    Raises:
        SizeSourceError: If no size source was given and the map
            names the module of a class the scan skips without the
            class itself (#204, #409). Only a size source prices
            that class, so the plan would drop its bytes.
        RuntimeCapabilityError: If a group holds at reference
            precision and the target runtime cannot serve it. The
            recipe would record an assignment `recipe_json` refuses
            to read back. A pinned group does not trigger this — the
            solver validated its width against the runtime. Also if
            the map names such a module with no size source and the
            runtime serves no reference precision: a size source
            would only reach the same refusal, so this one states
            the runtime's limit instead of naming the flag.
    """
    if discovered_bytes is None:
        module = missing_unquantizable_module(
            tensor for group in sensitivity_map.groups for tensor in group.tensors
        )
        if module is None:
            return ()
        named = (
            f'the map names the "{module}" module and no tensor of a class '
            f"the quantizer refuses. The scan skips that class (#204)"
        )
        if runtime is not None and REFERENCE_BITS not in RUNTIME_CAPABILITIES.get(
            runtime, frozenset()
        ):
            raise RuntimeCapabilityError(
                f'{named}, and it has no "{runtime}" width until a "{runtime}" '
                f"pack path exists (ADR-0013)"
            )
        raise SizeSourceError(
            f"{named}, so only a size source prices it. Plan with --checkpoint "
            f"(ADR-0029 decision 3)"
        )
    pins = dict(pins or {})
    uncovered = uncovered_groups(
        discovered_bytes, [g.name for g in sensitivity_map.groups]
    )
    held_at_reference = [name for name, _ in uncovered if name not in pins]
    if held_at_reference and runtime is not None:
        # `servable_precisions` words its refusal around the scanned
        # set, and reference precision was never scanned.
        capability = RUNTIME_CAPABILITIES.get(runtime, frozenset())
        if REFERENCE_BITS not in capability:
            raise _reference_refusal(runtime, capability, held_at_reference)
    return tuple(
        Assignment(
            group=name,
            bits=pins.get(name, REFERENCE_BITS),
            bytes=price_for(name)(bytes_fp16, pins.get(name, REFERENCE_BITS)),
            damage=0.0,
        )
        for name, bytes_fp16 in uncovered
    )
