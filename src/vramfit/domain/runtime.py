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
tables, because GGUF `F16` carries no block scale.

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
        costs differ from the dense table's.

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

from collections.abc import Mapping
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
# Q4_0, Q2_0 — ADR-0028 decision 1). Exact block-layout constants.
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
                    4: 4.5,
                    2: 2.25,
                }
            ),
        }
    )
)


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
