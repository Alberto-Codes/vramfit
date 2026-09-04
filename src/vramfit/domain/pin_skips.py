"""Pattern matching for pins, with the held-group skip.

A pin pattern that resolves to more than one group skips every
unquantizable-class group it sweeps. Such a group holds at the F16
passthrough, and no pin can move it (ADR-0012, 2026-08-20
amendment). A pattern that resolves to exactly one group keeps that
group, so the caller refuses a literal pin on a held class the way
it always did (ADR-0007, 2026-09-04 amendment, #371).

Examples:
    Split one pattern's matches into kept and skipped groups:

    ```python
    from vramfit.domain.pin_skips import match_pattern

    names = ["model.layers.0.mixer.gate", "model.layers.0.mixer.in_proj"]
    kept, skipped = match_pattern("model.layers.0.*", names, "llama.cpp")
    assert kept == ["model.layers.0.mixer.in_proj"]
    assert skipped == [("model.layers.0.mixer.gate", "ffn_gate_inp.weight")]
    ```

See Also:
    - [vramfit.domain.pins][]: The caller. `resolve_pins` pins the
      kept groups and `held_pin_skips` reports the skipped ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

from vramfit.domain.runtime import unquantizable_filter


@dataclass(frozen=True)
class HeldPinSkip:
    """One held group a multi-group pin pattern skipped.

    Attributes:
        group (str): The skipped group's name.
        pattern (str): The pin pattern that swept it.
        bits (int): The width the pattern asked for.
        filter (str): The upstream filter that refuses the group's
            tensors.

    Examples:
        ```python
        from vramfit.domain.pin_skips import HeldPinSkip

        skip = HeldPinSkip(
            group="model.layers.0.mixer.gate",
            pattern="*",
            bits=8,
            filter="ffn_gate_inp.weight",
        )
        assert skip.filter == "ffn_gate_inp.weight"
        ```
    """

    group: str
    pattern: str
    bits: int
    filter: str


def match_pattern(
    pattern: str, names: list[str], runtime: str | None
) -> tuple[list[str], list[tuple[str, str]]]:
    """Resolve one pin pattern, skipping held groups under a sweep.

    Args:
        pattern: A case-sensitive `fnmatch` glob.
        names: The sorted match universe.
        runtime: Target runtime name, or None. Only a runtime with a
            filter table holds anything.

    Returns:
        A pair: the group names the pattern pins, and the
        ``(group, filter)`` pairs it skipped. The second list is empty
        unless the pattern resolves to more than one group.
    """
    matched = [name for name in names if fnmatchcase(name, pattern)]
    if len(matched) < 2:  # noqa: PLR2004 - one match is a literal pin, which refuses
        return matched, []
    kept: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name in matched:
        filter_name = unquantizable_filter(name, runtime)
        if filter_name is None:
            kept.append(name)
        else:
            skipped.append((name, filter_name))
    return kept, skipped
