from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import check_banned_terms as gate  # noqa: E402

pytestmark = pytest.mark.unit

# The fixtures build the banned term from the gate's own constant, so
# this file carries no occurrence of its own and needs no allowlist
# entry.
TERM = gate.BANNED_TERM


def test_report_clean_file_reports_no_failures(tmp_path) -> None:
    path = tmp_path / "clean.md"
    path.write_text("The reader accepts only vramfit_schema.\n")

    failures, allowed, allowed_files = gate.report(tmp_path, [path], {})

    assert failures == []
    assert (allowed, allowed_files) == (0, 0)


def test_report_term_outside_allowlist_fails_with_line_number(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(f"first line\nthe {TERM}_schema envelope\n")

    failures, _, _ = gate.report(tmp_path, [path], {})

    assert len(failures) == 1
    assert failures[0].startswith("FAIL doc.md:2:")


def test_report_uppercase_term_fails(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(f"{TERM.capitalize()}Error was the pre-rename root.\n")

    failures, _, _ = gate.report(tmp_path, [path], {})

    assert len(failures) == 1


def test_report_allowlisted_file_counts_every_occurrence(tmp_path) -> None:
    path = tmp_path / "guard.py"
    path.write_text(f'"{TERM}_schema" not in obj  # rejects {TERM}\n')

    failures, allowed, allowed_files = gate.report(
        tmp_path, [path], {"guard.py": "the guard"}
    )

    assert failures == []
    assert (allowed, allowed_files) == (2, 1)


def test_report_stale_allowlist_entry_fails(tmp_path) -> None:
    path = tmp_path / "clean.md"
    path.write_text("vramfit only.\n")

    failures, _, _ = gate.report(tmp_path, [path], {"clean.md": "no longer true"})

    assert len(failures) == 1
    assert "delete the entry" in failures[0]


def test_report_empty_file_list_fails(tmp_path) -> None:
    failures, _, _ = gate.report(tmp_path, [], {})

    assert failures == ["FAIL: the gate scanned no files"]


def test_occurrences_binary_file_yields_nothing(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"\xff\xfe" + TERM.encode() + b"\x00\x80")

    assert gate.occurrences(path) == []


def test_check_this_repository_passes() -> None:
    root = _SCRIPTS.parent

    failures, allowed, _ = gate.check(root)

    assert failures == []
    assert allowed > 0
