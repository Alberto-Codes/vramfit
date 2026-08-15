from __future__ import annotations

from pathlib import Path

import check_ty_overrides as gate
import pytest

pytestmark = pytest.mark.unit

COVERS_ALL = gate.TyOverride(patterns=("**",), suppresses=True)

# The roots a base install cannot resolve, stated as a literal so a
# deletion from the gate's own table fails here. Measured 2026-08-15
# against `uv sync --locked --dev` with no extras.
EXPECTED_ROOTS = {
    "gguf",
    "numpy",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
}


def write(tmp_path: Path, name: str, body: str) -> Path:
    """Write a Python file under a temporary root, parents included."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_optional_roots_match_the_measured_set() -> None:
    # A root dropped from the table would silently stop being gated.
    assert set(gate.OPTIONAL_ROOTS) == EXPECTED_ROOTS


def test_report_file_without_optional_import_reports_no_failures(tmp_path) -> None:
    path = write(tmp_path, "src/plain.py", "from __future__ import annotations\n")

    failures, covered = gate.report(tmp_path, [path], [])

    assert failures == []
    assert covered == 0


def test_report_uncovered_optional_import_fails_with_line_number(tmp_path) -> None:
    path = write(tmp_path, "src/meter.py", "import json\nimport torch\n")

    failures, covered = gate.report(tmp_path, [path], [])

    assert len(failures) == 1
    assert failures[0].startswith("FAIL src/meter.py:2:")
    assert "torch" in failures[0]
    assert covered == 0


def test_report_covered_optional_import_reports_no_failures(tmp_path) -> None:
    path = write(tmp_path, "src/scan/meter.py", "import torch\n")
    override = gate.TyOverride(patterns=("src/scan/**",), suppresses=True)

    failures, covered = gate.report(tmp_path, [path], [override])

    assert failures == []
    assert covered == 1


def test_report_override_that_does_not_silence_the_rule_fails(tmp_path) -> None:
    # A block may cover a file and set some other rule. Ty still
    # reports unresolved-import, so the gate must not accept it.
    path = write(tmp_path, "src/scan/meter.py", "import torch\n")
    override = gate.TyOverride(patterns=("src/scan/**",), suppresses=False)

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
def test_report_nested_optional_import_fails_with_line_number(
    tmp_path, body, line
) -> None:
    # Ty resolves every one of these, so every one needs the override.
    path = write(tmp_path, "src/meter.py", body)

    failures, _ = gate.report(tmp_path, [path], [])

    assert len(failures) == 1
    assert f"src/meter.py:{line}:" in failures[0]


def test_report_module_named_only_in_prose_reports_no_failures(tmp_path) -> None:
    # `LlamaCppPacker`'s docstring says the interpreter "Must import
    # torch". Prose is not an import.
    path = write(tmp_path, "src/pack.py", '"""The interpreter must import torch."""\n')

    failures, covered = gate.report(tmp_path, [path], [])

    assert failures == []
    assert covered == 0


def test_report_importorskip_string_reports_no_failures(tmp_path) -> None:
    # `pytest.importorskip` names its module in a string. Ty never
    # reports it, so the gate must not either.
    path = write(tmp_path, "src/t.py", 'import pytest\npytest.importorskip("torch")\n')

    failures, covered = gate.report(tmp_path, [path], [])

    assert failures == []
    assert covered == 0


def test_report_unparsable_file_reports_no_failures(tmp_path) -> None:
    # ruff already fails a syntax error. Two reports on one cause help
    # nobody.
    path = write(tmp_path, "src/broken.py", "import torch\ndef (:\n")

    failures, _ = gate.report(tmp_path, [path], [])

    assert failures == []


def test_report_file_missing_from_the_worktree_reports_no_failures(tmp_path) -> None:
    # `git ls-files --cached` lists a file removed with plain `rm`.
    tmp_path.joinpath("src").mkdir()
    missing = tmp_path / "src" / "gone.py"

    failures, _ = gate.report(tmp_path, [missing], [])

    assert failures == []


def test_report_pattern_earning_nothing_fails(tmp_path) -> None:
    path = write(tmp_path, "src/scan/meter.py", "import torch\n")
    override = gate.TyOverride(
        patterns=("src/scan/**", "src/removed/**"), suppresses=True
    )

    failures, _ = gate.report(tmp_path, [path], [override])

    assert len(failures) == 1
    assert failures[0].startswith("FAIL 'src/removed/**':")


def test_report_pattern_matching_only_import_free_files_fails(tmp_path) -> None:
    # The stale rule matches check_banned_terms.py: an entry earns its
    # place only against a file that imports an optional root.
    write(tmp_path, "src/scan/meter.py", "import torch\n")
    plain = write(tmp_path, "tests/plain.py", "import json\n")
    covered_file = tmp_path / "src" / "scan" / "meter.py"
    override = gate.TyOverride(
        patterns=("src/scan/**", "tests/plain.py"), suppresses=True
    )

    failures, _ = gate.report(tmp_path, [covered_file, plain], [override])

    assert len(failures) == 1
    assert failures[0].startswith("FAIL 'tests/plain.py':")


def test_report_duplicate_pattern_in_a_second_block_still_fails(tmp_path) -> None:
    # Keying by text alone would let the dead second entry hide behind
    # the live first one.
    path = write(tmp_path, "src/scan/meter.py", "import torch\n")
    live = gate.TyOverride(patterns=("src/scan/**",), suppresses=True)
    dead = gate.TyOverride(patterns=("src/gone/**",), suppresses=True)

    failures, _ = gate.report(tmp_path, [path], [live, dead])

    assert len(failures) == 1
    assert failures[0].startswith("FAIL 'src/gone/**':")


def test_report_empty_file_list_fails_loudly(tmp_path) -> None:
    # A gate that scans nothing must not pass silently.
    failures, _ = gate.report(tmp_path, [], [COVERS_ALL])

    assert failures == ["FAIL: the gate scanned no files"]


@pytest.mark.parametrize("root", sorted(EXPECTED_ROOTS))
def test_report_declared_optional_root_fails_without_override(tmp_path, root) -> None:
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
    ],
    ids=[
        "recursive-direct",
        "recursive-nested",
        "recursive-sibling",
        "literal-hit",
        "literal-miss",
    ],
)
def test_glob_to_regex_supported_pattern_matches_expected_path(
    pattern, name, matches
) -> None:
    # Verified against ty 0.0.64 directly: these two shapes agree.
    assert bool(gate.glob_to_regex(pattern).match(name)) is matches


@pytest.mark.parametrize(
    ("pattern", "phrase"),
    [
        ("src/scan/", "trailing '/'"),
        ("src/*", "wildcard outside"),
        ("src/*.py", "wildcard outside"),
        ("src/**/top.py", "wildcard outside"),
        ("**/conftest.py", "wildcard outside"),
    ],
    ids=[
        "trailing-slash",
        "directory-glob",
        "suffix-glob",
        "mid-recursive",
        "leading-recursive",
    ],
)
def test_unsupported_reason_unmodelled_shape_names_the_shape(
    tmp_path, pattern, phrase
) -> None:
    # ty reads each of these differently than a naive translation, so
    # the gate refuses rather than guesses.
    reason = gate.unsupported_reason(pattern, tmp_path)

    assert reason is not None
    assert phrase in reason


def test_unsupported_reason_bare_directory_names_the_fix(tmp_path) -> None:
    tmp_path.joinpath("src", "scan").mkdir(parents=True)

    reason = gate.unsupported_reason("src/scan", tmp_path)

    assert reason is not None
    assert "src/scan/**" in reason


@pytest.mark.parametrize(
    "pattern",
    ["src/scan/**", "tests/conftest.py"],
    ids=["recursive", "literal"],
)
def test_unsupported_reason_supported_shape_reports_none(tmp_path, pattern) -> None:
    assert gate.unsupported_reason(pattern, tmp_path) is None


def test_report_unsupported_pattern_stops_before_judging_files(tmp_path) -> None:
    # Every later judgement would rest on a pattern the gate reads
    # differently than ty does.
    path = write(tmp_path, "src/meter.py", "import torch\n")
    override = gate.TyOverride(patterns=("src/",), suppresses=True)

    failures, covered = gate.report(tmp_path, [path], [override])

    assert len(failures) == 1
    assert "trailing '/'" in failures[0]
    assert covered == 0


def test_read_overrides_reads_patterns_and_the_silenced_rule(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[[tool.ty.overrides]]\n"
        'include = ["src/scan/**"]\n'
        "[tool.ty.overrides.rules]\n"
        'unresolved-import = "ignore"\n'
    )

    overrides = gate.read_overrides(pyproject)

    assert overrides == [gate.TyOverride(patterns=("src/scan/**",), suppresses=True)]


def test_read_overrides_without_the_rule_reports_no_suppression(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[[tool.ty.overrides]]\ninclude = ["src/scan/**"]\n')

    overrides = gate.read_overrides(pyproject)

    assert overrides == [gate.TyOverride(patterns=("src/scan/**",), suppresses=False)]


def test_read_overrides_with_no_block_reports_none(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "vramfit"\n')

    assert gate.read_overrides(pyproject) == []


def test_read_scanned_roots_reads_the_configured_include(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.ty.src]\ninclude = ["src", "tests", "docs"]\n')

    assert gate.read_scanned_roots(pyproject) == ("src", "tests", "docs")


def test_read_scanned_roots_without_the_key_reports_the_default(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "vramfit"\n')

    assert gate.read_scanned_roots(pyproject) == gate._DEFAULT_SCANNED_ROOTS


@pytest.mark.parametrize(
    ("failures", "phrase"),
    [
        (["FAIL 'x': ... — delete the entry"], "Delete the entry"),
        (["FAIL src/a.py:1: imports 'torch' ..."], "ADR-0005"),
        (
            ["FAIL 'x': ... — delete the entry", "FAIL src/a.py:1: imports 'torch'"],
            "ADR-0005",
        ),
    ],
    ids=["only-stale", "only-missing", "mixed"],
)
def test_guidance_names_the_action_the_failures_call_for(failures, phrase) -> None:
    assert phrase in gate.guidance(failures)


def test_check_on_this_repository_reports_no_failures() -> None:
    # The gate must agree with CI, which type-checks with no extras.
    root = Path(gate.__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"

    failures, covered = gate.check(root)

    assert failures == []
    # Cross-check the count independently rather than pinning a number.
    files = gate.scanned_files(root, gate.read_scanned_roots(pyproject))
    assert covered == sum(1 for path in files if gate.optional_imports(path))
