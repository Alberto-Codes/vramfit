"""Shared fixtures for the port contract suites.

`LlamaCppPacker.pack` reads the base GGUF's tensor names to refuse an
override that matches nothing (ADR-0012 as amended 2026-08-16, #303).
The `RecipePacker` suite stubs the base GGUF as opaque bytes, so the
read has no file to open. The fixture below serves a decoder's names
instead, which keeps every recipe in that suite passing the check
unchanged and keeps the suite free of gguf-py.

`tests/unit/adapters/test_override_match_gguf.py` covers the real
read against a written file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import decoder_tensor_names
from vramfit.adapters.outbound.gguf import override_match


@pytest.fixture(autouse=True)
def base_gguf_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a decoder's tensor names in place of a base-GGUF read."""

    def names(_: Path) -> tuple[str, ...]:
        return decoder_tensor_names()

    monkeypatch.setattr(override_match, "base_tensor_names", names)
