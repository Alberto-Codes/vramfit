"""Runtime capability: the precisions a target runtime can serve.

Implements the ADR-0013 capability table and the ADR-0014
effective-bits tables. A recipe is only servable when every assigned
precision has a kernel in the target runtime, so the solver filters
its candidate set through the capability table before any solving
starts. For a runtime with a measured serving path, a second table
records the effective bits per weight each nominal precision spends —
the solver predicts sizes from it instead of a scalar overhead.

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

Examples:
    Filter a scanned candidate set for vLLM:

    ```python
    from quantfit.domain.runtime import VLLM, servable_precisions

    assert servable_precisions((8, 4, 3, 2), VLLM) == (8, 4)
    ```

See Also:
    - [quantfit.domain.solver][]: Applies the filter when a target
      runtime is given.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from quantfit.domain.errors import QuantfitError

LLAMA_CPP: Final[str] = "llama.cpp"
VLLM: Final[str] = "vllm"

RUNTIME_CAPABILITIES: Final[Mapping[str, frozenset[int]]] = MappingProxyType(
    {
        LLAMA_CPP: frozenset({8, 6, 5, 4, 3, 2}),
        VLLM: frozenset({8, 4}),
    }
)

# Effective bits per weight for each K-quant type the ADR-0012 mapping
# assigns (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K). Exact block-layout
# constants, verified byte-for-byte against packed files (ADR-0014).
# vLLM has no entry: no measured pack path exists yet, so vLLM plans
# fall back to the scalar overhead.
EFFECTIVE_BITS: Final[Mapping[str, Mapping[int, float]]] = MappingProxyType(
    {
        LLAMA_CPP: MappingProxyType(
            {
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


class RuntimeCapabilityError(QuantfitError, ValueError):
    """A target runtime cannot serve the requested precisions.

    Raised for a runtime name outside the ADR-0013 table, and for a
    scanned candidate set the runtime serves nothing of. Under the
    `QuantfitError` root (ADR-0011) with a message the CLI prints
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
        from quantfit.domain.runtime import effective_bits

        assert effective_bits("llama.cpp")[4] == 4.5
        assert effective_bits("vllm") is None
        ```
    """
    if runtime is None:
        return None
    return EFFECTIVE_BITS.get(runtime)
