"""Runtime capability: the precisions a target runtime can serve.

Implements the ADR-0013 capability table. A recipe is only servable
when every assigned precision has a kernel in the target runtime, so
the solver filters its candidate set through this table before any
solving starts. The table maps runtime names to nominal bit-widths —
what each runtime's quantization types cover, not how they spend
their effective bits (that stays a pack concern, ADR-0012).

Attributes:
    LLAMA_CPP (str): The llama.cpp runtime name. Pack backends and
        the CLI reference this constant, never the literal.
    VLLM (str): The vLLM runtime name.
    RUNTIME_CAPABILITIES (Mapping[str, frozenset[int]]): Nominal
        precisions each known target runtime serves.

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
