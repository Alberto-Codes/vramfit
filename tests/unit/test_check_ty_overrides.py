from __future__ import annotations

import check_ty_overrides as gate
import pytest

pytestmark = pytest.mark.unit

COVERS_ALL = gate.Override(patterns=("**",), suppresses=True)


def write(tmp_path, name: str, body: str):
    """Write a Python file under a temporary root, parents included."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_report_file_without_optional_import_reports_no_failures(tmp_path) -> None:
    path = write(tmp_path, "src/plain.py", "from __future__ import annotations\n")

    failures, covered = gate.report(tmp_path, [path], [COVERS_ALL])

    assert failures == []
    assert covered == 0


def test_report_uncovered_optional_import_fails_with_line_number(tmp_path) -> None:
    path = write(tmp_path, "src/meter.py", "import json\nimport torch\n")

    failures, covered = gate.report(tmp_path, [path], [])

    assert len(failures) == 1
    assert failures[0].startswith("FAIL src/meter.py:2:")
    assert "torch" in failures[0]
    assert covered == 0


def test_report_covered_optional_import_passes(tmp_path) -> None:
    path = write(tmp_path, "src/scan/meter.py", "import torch\n")
    override = gate.Override(patterns=("src/scan/**",), suppresses=True)

    failures, covered = gate.report(tmp_path, [path], [override])

    assert failures == []
    assert covered == 1


def test_report_override_that_does_not_silence_the_rule_fails(tmp_path) -> None:
    # A block may cover a file and set some other rule. Ty still
    # reports unresolved-import, so the gate must not accept it.
    path = write(tmp_path, "src/scan/meter.py", "import torch\n")
    override = gate.Override(patterns=("src/scan/**",), suppresses=False)

    failures, _ = gate.report(tmp_path, [path], [override])

    assert len(failures) == 1
    assert "src/scan/meter.py:1:" in failures[0]


@pytest.mark.parametrize(
    ("body", "line"),
    [
        ("import torch\n", 1),
        ("from torch import nn\n", 1),
        ("import torch.nn.functional as F\n", 1),
        ("def load():\n    import torch\n", 2),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n", 3),
        ("try:\n    import torch\nexcept ImportError:\n    torch = None\n", 2),
    ],
    ids=[
        "module-level",
        "from-import",
        "dotted-alias",
        "function-level",
        "type-checking",
        "guarded",
    ],
)
def test_report_finds_an_optional_import_at_any_nesting(tmp_path, body, line) -> None:
    # Ty resolves every one of these, so every one needs the override.
    path = write(tmp_path, "src/meter.py", body)

    failures, _ = gate.report(tmp_path, [path], [])

    assert len(failures) == 1
    assert f"src/meter.py:{line}:" in failures[0]


def test_report_ignores_a_module_named_only_in_prose(tmp_path) -> None:
    # `LlamaCppPacker`'s docstring says the interpreter "Must import
    # torch". Prose is not an import.
    path = write(tmp_path, "src/pack.py", '"""The interpreter must import torch."""\n')

    failures, covered = gate.report(tmp_path, [path], [])

    assert failures == []
    assert covered == 0


def test_report_ignores_a_file_that_does_not_parse(tmp_path) -> None:
    # ruff already fails a syntax error. Two reports on one cause help
    # nobody.
    path = write(tmp_path, "src/broken.py", "import torch\ndef (:\n")

    failures, _ = gate.report(tmp_path, [path], [])

    assert failures == []


def test_report_stale_include_pattern_fails(tmp_path) -> None:
    path = write(tmp_path, "src/scan/meter.py", "import torch\n")
    override = gate.Override(
        patterns=("src/scan/**", "src/removed/**"), suppresses=True
    )

    failures, _ = gate.report(tmp_path, [path], [override])

    assert len(failures) == 1
    assert failures[0].startswith("FAIL 'src/removed/**':")


def test_report_empty_file_list_fails_loudly(tmp_path) -> None:
    # A gate that scans nothing must not pass silently.
    failures, _ = gate.report(tmp_path, [], [COVERS_ALL])

    assert failures == ["FAIL: the gate scanned no files"]


@pytest.mark.parametrize("root", sorted(gate.OPTIONAL_ROOTS))
def test_report_every_declared_optional_root_is_detected(tmp_path, root) -> None:
    path = write(tmp_path, "src/mod.py", f"import {root}\n")

    failures, _ = gate.report(tmp_path, [path], [])

    assert len(failures) == 1
    assert repr(root) in failures[0]


@pytest.mark.parametrize(
    ("pattern", "name", "matches"),
    [
        ("src/scan/**", "src/scan/meter.py", True),
        ("src/scan/**", "src/scan/deep/meter.py", True),
        ("src/scan/**", "src/other/meter.py", False),
        ("tests/conftest.py", "tests/conftest.py", True),
        ("tests/conftest.py", "tests/unit/conftest.py", False),
        ("src/*.py", "src/meter.py", True),
        ("src/*.py", "src/scan/meter.py", False),
    ],
    ids=[
        "recursive-direct",
        "recursive-nested",
        "recursive-sibling",
        "literal-hit",
        "literal-miss",
        "star-stops-at-separator",
        "star-does-not-cross-separator",
    ],
)
def test_glob_to_regex_matches_ty_include_semantics(pattern, name, matches) -> None:
    assert bool(gate.glob_to_regex(pattern).match(name)) is matches


def test_read_overrides_reads_patterns_and_the_silenced_rule(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[[tool.ty.overrides]]\n"
        'include = ["src/scan/**"]\n'
        "[tool.ty.overrides.rules]\n"
        'unresolved-import = "ignore"\n'
    )

    overrides = gate.read_overrides(pyproject)

    assert overrides == [gate.Override(patterns=("src/scan/**",), suppresses=True)]


def test_read_overrides_without_the_rule_reports_no_suppression(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[[tool.ty.overrides]]\ninclude = ["src/scan/**"]\n')

    overrides = gate.read_overrides(pyproject)

    assert overrides == [gate.Override(patterns=("src/scan/**",), suppresses=False)]


def test_read_overrides_with_no_block_reports_none(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "vramfit"\n')

    assert gate.read_overrides(pyproject) == []


def test_check_passes_on_this_repository() -> None:
    # The gate must agree with CI, which type-checks with no extras.
    from pathlib import Path

    root = Path(gate.__file__).resolve().parent.parent
    failures, covered = gate.check(root)

    assert failures == []
    assert covered > 0
