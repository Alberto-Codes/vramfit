"""Runtime capability: the precisions a target runtime can serve.

Implements the ADR-0013 capability table and the ADR-0014
effective-bits tables. A recipe is only servable when every assigned
precision has a kernel in the target runtime, so the solver filters
its candidate set through the capability table before any solving
starts. For a runtime with a measured serving path, a second table
records the effective bits per weight each nominal precision spends —
the solver predicts sizes from it instead of a scalar overhead.
Nominal 16 is the F16 passthrough (ADR-0029 decision 4): it holds a
group at reference precision, so a recipe can name a group the scan
never measured. It spends exactly 16.0 bits per weight, in both
tables, because GGUF `F16` carries no block scale. A group of an
unquantizable class is the exception: the quantizer refuses it, so
it holds at the dtype the converter wrote, and the passthrough
prices it from that dtype (#409). `convert_dtype_bits` reads the
convert dtype table for such a class and nothing else, so every
other group keeps its table's 16 row. `missing_unquantizable_module`
reports a map that names such a class's module and not the class,
which only a size source can price.

Attributes:
    LLAMA_CPP (str): The llama.cpp runtime name. Pack backends and
        the CLI reference this constant, never the literal.
    VLLM (str): The vLLM runtime name.
    RUNTIME_CAPABILITIES (Mapping[str, frozenset[int]]): Nominal
        precisions each known target runtime serves.
    EFFECTIVE_BITS (Mapping[str, Mapping[int, float]]): Effective
        bits per weight, per nominal precision, per runtime. Only
        runtimes with a measured pack path have an entry — a runtime
        with a table covers its full capability set.
    EXPERT_STACK_EFFECTIVE_BITS (Mapping[str, Mapping[int, float]]):
        Effective bits per weight for a routed-expert-stack group,
        per nominal precision, per runtime. Expert stacks map
        through their own type table (ADR-0028), so their per-weight
        costs differ from the dense table's at 2, 4, 5, and 6. The
        same table reaches
        a layer-class group whose rows refuse the 256 super-block
        (the 2026-08-20 amendment) — `SUPER_BLOCK_REFUSED_CLASSES`
        names those classes.
    SUPER_BLOCK_REFUSED_CLASSES (frozenset[str]): Layer-class group
        suffixes whose tensor rows refuse the k-quant 256
        super-block. Each maps and prices through the ADR-0028
        table.
    UNQUANTIZABLE_CLASS_FILTERS (Mapping[str, Mapping[str, str]]):
        Per runtime, the layer-class suffixes whose tensors the
        runtime's quantizer refuses, each mapped to the name of the
        upstream filter that refuses it. A group of such a class
        packs at the F16 passthrough and never lower (ADR-0012,
        2026-08-20 amendment).
    CONVERT_DTYPE_BITS (Mapping[str, Mapping[str, float]]): Per
        runtime, the bits per weight the converter stores each
        unquantizable class at, whatever output type the conversion
        asked for. The passthrough prices such a class from this
        table, because the packed file holds it at the convert
        dtype (#409).

Examples:
    Filter a scanned candidate set for vLLM:

    ```python
    from vramfit.domain.runtime import VLLM, servable_precisions

    assert servable_precisions((8, 4, 3, 2), VLLM) == (8, 4)
    ```

See Also:
    - [vramfit.domain.solver][]: Applies the filter when a target
      runtime is given.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final

from vramfit.domain.errors import VramfitError

LLAMA_CPP: Final[str] = "llama.cpp"
VLLM: Final[str] = "vllm"

RUNTIME_CAPABILITIES: Final[Mapping[str, frozenset[int]]] = MappingProxyType(
    {
        LLAMA_CPP: frozenset({16, 8, 6, 5, 4, 3, 2}),
        VLLM: frozenset({8, 4}),
    }
)

# Effective bits per weight for each K-quant type the ADR-0012 mapping
# assigns (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K). Exact block-layout
# constants, verified byte-for-byte against packed files (ADR-0014).
# The 16 row is the F16 passthrough (ADR-0029 decision 4): GGUF `F16`
# stores two bytes per weight with no block overhead, so an
# unquantized group spends exactly its reference bits. vLLM has no
# entry: no measured pack path exists yet, so vLLM plans fall back to
# the scalar overhead.
EFFECTIVE_BITS: Final[Mapping[str, Mapping[int, float]]] = MappingProxyType(
    {
        LLAMA_CPP: MappingProxyType(
            {
                16: 16.0,
                8: 8.5,
                6: 6.5625,
                5: 5.5,
                4: 4.5,
                3: 3.4375,
                2: 2.625,
            }
        ),
    }
)

# Effective bits per weight for the expert-stack type table (Q8_0,
# Q5_1, Q5_0, Q4_0, Q2_0 — ADR-0028 decision 1). Exact block-layout
# constants. The 6 and 5 rows date from the 2026-09-04 amendment
# (#232): Q5_1 at 6.00 and Q5_0 at 5.50, both block 32.
# The table carries no 3-bit row: no GGUF type lands between 2.25 and
# 4.25 bits per weight on the stack rows, and pack refuses nominal 3
# there (ADR-0028 decision 2). The 16 row is the F16 passthrough,
# which costs the same on a stack row as on a dense one — `F16` has
# no super-block to divide (ADR-0029 decision 4).
EXPERT_STACK_EFFECTIVE_BITS: Final[Mapping[str, Mapping[int, float]]] = (
    MappingProxyType(
        {
            LLAMA_CPP: MappingProxyType(
                {
                    16: 16.0,
                    8: 8.5,
                    6: 6.0,
                    5: 5.5,
                    4: 4.5,
                    2: 2.25,
                }
            ),
        }
    )
)


# The Nemotron-H dense classes, by group suffix. Their tensor rows are
# 2688 wide, which no 256-element k-quant super-block divides, so each
# maps and prices through the ADR-0028 table (ADR-0012 and ADR-0028,
# 2026-08-20 amendments). `mixer.gate` and `mixer.conv1d` stay out:
# they pin at the F16 passthrough below and take no table row.
SUPER_BLOCK_REFUSED_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "mixer.in_proj",
        "mixer.out_proj",
        "mixer.shared_experts.up_proj",
        "mixer.shared_experts.down_proj",
        "mixer.q_proj",
        "mixer.k_proj",
        "mixer.v_proj",
        "mixer.o_proj",
    }
)

# The layer-class suffixes llama-quantize refuses to quantize, each
# mapped to the upstream filter that refuses it. The filter list in
# `tensor_allows_quantization` (llama.cpp src/llama-quant.cpp:289-367
# at the pinned instrument, commit 3653e6d6d) is a copied external
# contract: no CLI reaches the predicate, so vramfit copies the
# filters its targets reach (ADR-0012, 2026-08-20 amendment; #305
# carries the residual, #207 how a test pins a copy). The contract's
# rank gate lives in the size source and the meter instead, and the
# `_norm.weight` filter reaches no group — every norm is rank 1.
UNQUANTIZABLE_CLASS_FILTERS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        LLAMA_CPP: MappingProxyType(
            {
                "mixer.gate": "ffn_gate_inp.weight",
                "mixer.conv1d": "ssm_conv1d",
            }
        ),
    }
)

# The bits per weight the converter stores each unquantizable class
# at. `convert_hf_to_gguf.py` writes `FFN_GATE_INP` and `SSM_CONV1D`
# at float32 whatever `--outtype` asks (the always-float32 list in
# `ModelBase.prepare_tensors` at the pinned instrument), and the
# quantizer drops the override, so the packed file holds the class at
# 32 bits (#409). The keys mirror `UNQUANTIZABLE_CLASS_FILTERS`: a
# class one table names, the other names too.
CONVERT_DTYPE_BITS: Final[Mapping[str, Mapping[str, float]]] = MappingProxyType(
    {
        LLAMA_CPP: MappingProxyType(
            {
                "mixer.gate": 32.0,
                "mixer.conv1d": 32.0,
            }
        ),
    }
)

# A layer-class group: a decoder-layer prefix under any naming family,
# then the class suffix. The capture is what the two class tables key
# on.
_CLASS_SUFFIX: Final[re.Pattern[str]] = re.compile(
    r"^.+\.(?:layers|h|blocks)\.\d+\.(.+)$"
)


def rows_refuse_super_block(group: str) -> bool:
    """Report whether a group's rows refuse the k-quant super-block.

    Such a layer-class group maps and prices through the ADR-0028
    table, exactly like a routed-expert stack (ADR-0028, 2026-08-20
    amendment). The class list is `SUPER_BLOCK_REFUSED_CLASSES`.

    Args:
        group: Group name, as `vramfit.domain.scan.group_key`
            produces it.

    Returns:
        True when the group's class suffix is in the list.

    Examples:
        ```python
        from vramfit.domain.runtime import rows_refuse_super_block

        assert rows_refuse_super_block("model.layers.3.mixer.in_proj")
        assert not rows_refuse_super_block("model.layers.3")
        ```
    """
    match = _CLASS_SUFFIX.match(group)
    return match is not None and match.group(1) in SUPER_BLOCK_REFUSED_CLASSES


def unquantizable_filter(group: str, runtime: str | None) -> str | None:
    """Name the upstream filter that refuses this group's tensors.

    A group of an unquantizable class holds at the F16 passthrough:
    the quantizer drops an override on such a tensor and exits 0, so
    any lower width would record a type the artifact cannot carry
    (ADR-0012, 2026-08-20 amendment).

    Args:
        group: Group name, as `vramfit.domain.scan.group_key`
            produces it.
        runtime: Target runtime name, or None for an unconstrained
            plan.

    Returns:
        The upstream filter's name, or None when the runtime carries
        no filter table or no filter refuses the class.

    Examples:
        ```python
        from vramfit.domain.runtime import unquantizable_filter

        group = "model.layers.3.mixer.gate"
        assert unquantizable_filter(group, "llama.cpp") == "ffn_gate_inp.weight"
        assert unquantizable_filter(group, None) is None
        ```
    """
    if runtime is None:
        return None
    table = UNQUANTIZABLE_CLASS_FILTERS.get(runtime)
    if table is None:
        return None
    match = _CLASS_SUFFIX.match(group)
    if match is None:
        return None
    return table.get(match.group(1))


def unquantizable_class(group: str) -> str | None:
    """Name the class suffix a known quantizer refuses.

    The scan carries no target runtime, so discovery skips a class
    that any runtime's filter table refuses (#204). Such a class
    holds at the convert dtype whatever the map measured, so a cell
    the scan prices for it is a cell no recipe can act on. Today one
    table exists, llama.cpp's.

    Args:
        group: Group name, as `vramfit.domain.scan.group_key`
            produces it under ``tensor`` granularity.

    Returns:
        The class suffix, or None when no table refuses the class.

    Examples:
        ```python
        from vramfit.domain.runtime import unquantizable_class

        assert unquantizable_class("model.layers.3.mixer.conv1d") == "mixer.conv1d"
        assert unquantizable_class("model.layers.3.mixer.in_proj") is None
        ```
    """
    match = _CLASS_SUFFIX.match(group)
    if match is None:
        return None
    suffix = match.group(1)
    for table in UNQUANTIZABLE_CLASS_FILTERS.values():
        if suffix in table:
            return suffix
    return None


def missing_unquantizable_module(tensors: Iterable[str]) -> str | None:
    """Name the module whose refused class the map does not carry.

    The scan skips a class the quantizer refuses (#204), so a map
    scanned since then names the class's module through its siblings
    and never the class itself. Only a size source prices that class,
    so a plan without one drops its bytes (#409). A map that carries
    the class predates the skip and prices it itself.

    Args:
        tensors: Every tensor name the map's groups carry.

    Returns:
        The module, e.g. ``mixer``, when the map names a tensor under
        it and none of the module's refused classes. None otherwise.

    Examples:
        ```python
        from vramfit.domain.runtime import missing_unquantizable_module

        assert (
            missing_unquantizable_module(["model.layers.0.mixer.in_proj.weight"])
            == "mixer"
        )
        assert (
            missing_unquantizable_module(
                [
                    "model.layers.0.mixer.in_proj.weight",
                    "model.layers.0.mixer.conv1d.weight",
                ]
            )
            is None
        )
        ```
    """
    refused: dict[str, set[str]] = {}
    for table in UNQUANTIZABLE_CLASS_FILTERS.values():
        for suffix in table:
            refused.setdefault(suffix.rpartition(".")[0], set()).add(suffix)
    named: set[str] = set()
    carried: set[str] = set()
    for tensor in tensors:
        match = _CLASS_SUFFIX.match(tensor.removesuffix(".weight"))
        if match is None:
            continue
        suffix = match.group(1)
        named.add(suffix.partition(".")[0])
        if unquantizable_class(match.string) is not None:
            carried.add(suffix)
    for module, classes in sorted(refused.items()):
        if module in named and not classes & carried:
            return module
    return None


def convert_dtype_bits(group: str, runtime: str | None) -> float | None:
    """Report the bits per weight the converter stores a refused class at.

    A group of an unquantizable class holds at the dtype the
    converter wrote, which the packed file then carries: 32.0 bits
    on both llama.cpp classes (#409). Pricing it at the `f16`
    override's 16.0 under-priced publication #2's recipe by
    16,923,492 B against a 16,874,535 B margin. Every other group
    prices at its effective-bits table, whose 16 row is the
    passthrough (ADR-0029 decision 4).

    Args:
        group: Group name, as `vramfit.domain.scan.group_key`
            produces it.
        runtime: Target runtime name, or None for an unconstrained
            plan.

    Returns:
        The convert dtype's bits per weight, or None when the group
        is not of a refused class or the runtime carries no table.

    Examples:
        ```python
        from vramfit.domain.runtime import convert_dtype_bits

        assert convert_dtype_bits("model.layers.3.mixer.conv1d", "llama.cpp") == 32.0
        assert convert_dtype_bits("model.layers.3.mixer.in_proj", "llama.cpp") is None
        ```
    """
    if runtime is None:
        return None
    table = CONVERT_DTYPE_BITS.get(runtime)
    if table is None:
        return None
    match = _CLASS_SUFFIX.match(group)
    if match is None:
        return None
    return table.get(match.group(1))


class RuntimeCapabilityError(VramfitError, ValueError):
    """A target runtime cannot serve the requested precisions.

    Raised for a runtime name outside the ADR-0013 table, and for a
    scanned candidate set the runtime serves nothing of. Under the
    `VramfitError` root (ADR-0011) with a message the CLI prints
    verbatim.

    Examples:
        An unknown runtime name:

        ```python
        servable_precisions((8, 4), "tgi")  # raises
        ```
    """


def servable_precisions(precisions: tuple[int, ...], runtime: str) -> tuple[int, ...]:
    """Filter candidate precisions to those the runtime can serve.

    Args:
        precisions: Scanned candidate precisions. Order is preserved,
            so a descending input stays descending.
        runtime: Target runtime name from `RUNTIME_CAPABILITIES`.

    Returns:
        The servable precisions, in input order. Never empty.

    Raises:
        RuntimeCapabilityError: If the runtime is unknown, or it
            serves none of the given precisions.

    Examples:
        llama.cpp serves the whole ADR-0010 scan set:

        ```python
        assert servable_precisions((8, 4, 3, 2), "llama.cpp") == (8, 4, 3, 2)
        ```
    """
    try:
        capability = RUNTIME_CAPABILITIES[runtime]
    except KeyError:
        raise RuntimeCapabilityError(
            f'unknown runtime "{runtime}" — the ADR-0013 table covers '
            f"{sorted(RUNTIME_CAPABILITIES)}"
        ) from None
    filtered = tuple(bits for bits in precisions if bits in capability)
    if not filtered:
        raise RuntimeCapabilityError(
            f'runtime "{runtime}" serves none of the scanned precisions '
            f"{list(precisions)} — it serves "
            f"{sorted(capability, reverse=True)}"
        )
    return filtered


def effective_bits(runtime: str | None) -> Mapping[int, float] | None:
    """Look up a runtime's effective-bits table.

    Args:
        runtime: Target runtime name, or None for an unconstrained
            plan.

    Returns:
        The nominal-to-effective bits mapping, or None when the
        runtime is None or has no measured table (ADR-0014). Runtime
        name validation stays with `servable_precisions` — an unknown
        name returns None here.

    Examples:
        llama.cpp spends 4.5 effective bits on a 4-bit assignment:

        ```python
        from vramfit.domain.runtime import effective_bits

        assert effective_bits("llama.cpp")[4] == 4.5
        assert effective_bits("vllm") is None
        ```
    """
    if runtime is None:
        return None
    return EFFECTIVE_BITS.get(runtime)


def expert_stack_effective_bits(runtime: str | None) -> Mapping[int, float] | None:
    """Look up a runtime's expert-stack effective-bits table.

    A routed-expert-stack group maps through its own type table
    (ADR-0028), so the plan step prices it at that table's per-weight
    costs — 2.25 bits at nominal 2, not Q2_K's 2.625.

    Args:
        runtime: Target runtime name, or None for an unconstrained
            plan.

    Returns:
        The nominal-to-effective bits mapping for expert-stack
        groups, or None when the runtime is None or has no table.
        Runtime name validation stays with `servable_precisions`.

    Examples:
        llama.cpp spends 2.25 bits on a 2-bit expert stack:

        ```python
        from vramfit.domain.runtime import expert_stack_effective_bits

        assert expert_stack_effective_bits("llama.cpp")[2] == 2.25
        assert expert_stack_effective_bits(None) is None
        ```
    """
    if runtime is None:
        return None
    return EXPERT_STACK_EFFECTIVE_BITS.get(runtime)
