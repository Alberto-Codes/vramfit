"""Shared fixtures for the port contract suites.

`LlamaCppPacker.pack` reads the base GGUF's tensor names to refuse an
override that matches nothing (ADR-0012 as amended 2026-08-16, #303).
The `RecipePacker` suite stubs the base GGUF as opaque bytes, so the
read has no file to open. `base_gguf_names` serves a decoder's names
instead, which keeps that suite passing the check unchanged and free
of gguf-py.

`imatrix_entry_names` does the same for the exclusion check (#309).
That suite stubs the imatrix as a path to no file, so the read has
nothing to open either. An imatrix prices tensors the base GGUF
carries, so one decoder's names serve both stubs.

The fixtures are not autouse. A contract suite for another port must
not run against a patched adapter, and a pack suite that means to
exercise a refusal must say so. `TestRecipePackerContract` and
`TestLlamaCppCommandLines` request both by name.

`tests/unit/adapters/test_override_match_gguf.py` and
`tests/unit/adapters/test_exclusion_match_gguf.py` cover the real
reads against written files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import decoder_tensor_names
from vramfit.adapters.outbound.gguf import exclusion_match, override_match


@pytest.fixture
def base_gguf_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a decoder's tensor names in place of a base-GGUF read."""

    def names(_: Path) -> tuple[str, ...]:
        return decoder_tensor_names()

    monkeypatch.setattr(override_match, "base_tensor_names", names)


@pytest.fixture
def imatrix_entry_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a decoder's tensor names in place of an imatrix read."""

    def names(_: Path) -> tuple[str, ...]:
        return decoder_tensor_names()

    monkeypatch.setattr(exclusion_match, "imatrix_entry_names", names)
