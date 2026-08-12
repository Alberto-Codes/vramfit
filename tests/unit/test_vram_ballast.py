"""Hermetic tests for ``scripts/vram_ballast.py`` (#172).

The script's CUDA path needs a device, so these tests cover the parts
that decide behaviour without one: the sizing rule, the refusal, the
argument split, and the wrapper's exit-code contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

MIB = 1024 * 1024


def _load_script() -> ModuleType:
    """Import the script by path — ``scripts/`` is not an installed package."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "vram_ballast.py"
    spec = importlib.util.spec_from_file_location("vram_ballast", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ballast = _load_script()

pytestmark = pytest.mark.unit


def test_size_ballast_sizes_from_free_not_total():
    assert ballast.size_ballast(22504 * MIB, 12288 * MIB) == 10216 * MIB


def test_size_ballast_at_exactly_target_holds_nothing():
    assert ballast.size_ballast(12288 * MIB, 12288 * MIB) == 0


def test_size_ballast_below_target_refuses():
    with pytest.raises(ValueError, match="below the 12288 MiB target"):
        ballast.size_ballast(8000 * MIB, 12288 * MIB)


def test_size_ballast_nonpositive_target_refuses():
    with pytest.raises(ValueError, match="must exceed 0 MiB"):
        ballast.size_ballast(22504 * MIB, 0)


def test_parse_args_strips_the_separator_from_the_command():
    args = ballast.parse_args(
        ["--target-mib", "12288", "--", "llama-cli", "--list-devices"]
    )
    assert args.target_mib == 12288
    assert args.command == ["llama-cli", "--list-devices"]


def test_parse_args_without_a_command_holds():
    args = ballast.parse_args(["--target-mib", "4096"])
    assert args.command == []


def test_parse_args_defaults_to_the_twelve_gib_target():
    assert ballast.parse_args([]).target_mib == 12288


def test_run_command_returns_the_child_exit_code():
    assert ballast.run_command([sys.executable, "-c", "raise SystemExit(42)"]) == 42


def test_run_command_missing_binary_returns_127(capsys):
    assert ballast.run_command(["/no/such/binary"]) == ballast.EXIT_COMMAND_NOT_FOUND
    assert "cannot run /no/such/binary" in capsys.readouterr().err
