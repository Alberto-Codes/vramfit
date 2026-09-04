"""Shared fixtures for the port contract suites.

`LlamaCppPacker.pack` reads the base GGUF's tensor names to refuse an
override that matches nothing (ADR-0012 as amended 2026-08-16, #303).
The `RecipePacker` suite stubs the base GGUF as opaque bytes, so the
read has no file to open. `base_gguf_names` serves a decoder's names
instead, which keeps that suite passing the check unchanged and free
of gguf-py.

`imatrix_entry_names` does the same for the exclusion check (#309).
That suite stubs the imatrix as a path to no file, so the read has
nothing to open either. It serves a strict subset of the base
GGUF's names, because a matrix prices fewer tensors than the file
carries. Serving one list for both would pass a check handed the
wrong file's names.

`packed_layout` does the same for the file-type relabel (#414).
That suite's quantize stub writes opaque bytes, so the header parse
has no GGUF to read. It serves the published 30B pack's composition
in tenths of a percent (#413) and turns the in-place write into a
no-op, so the real adapter still composes the modal rule over the
same table the fake receives.

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

from tests.fakes import decoder_imatrix_entry_names, decoder_tensor_names
from vramfit.adapters.outbound.gguf import exclusion_match, file_type, override_match
from vramfit.adapters.outbound.gguf.file_type import PackedLayout

# The published 30B pack's composition by bytes, in tenths of a
# percent (#413). Q4_0 is the modal type.
PACKED_TYPE_BYTES = {"Q4_0": 743, "Q8_0": 138, "Q2_0": 117, "F32": 2}


@pytest.fixture
def base_gguf_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a decoder's tensor names in place of a base-GGUF read."""

    def names(_: Path) -> tuple[str, ...]:
        return decoder_tensor_names()

    monkeypatch.setattr(override_match, "base_tensor_names", names)


@pytest.fixture
def imatrix_entry_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a decoder matrix's entry names in place of an imatrix read."""

    def names(_: Path) -> tuple[str, ...]:
        return decoder_imatrix_entry_names()

    monkeypatch.setattr(exclusion_match, "imatrix_entry_names", names)


@pytest.fixture
def packed_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the 30B composition in place of a packed-GGUF header parse."""

    def layout(_: Path) -> PackedLayout:
        return PackedLayout(dict(PACKED_TYPE_BYTES), file_type_offset=0, file_type=10)

    def write(_: Path, offset: int, value: int) -> None:
        del offset, value

    monkeypatch.setattr(file_type, "read_layout", layout)
    monkeypatch.setattr(file_type, "write_file_type", write)
