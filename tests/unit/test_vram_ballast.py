"""Hermetic tests for ``scripts/vram_ballast.py`` (#172).

The script's CUDA path needs a device. These tests cover the parts that
decide behavior without one: the sizing rule, the refusal, the argument
split, the wrapper's exit-status contract, and the driver binding
against a fake ``libcuda``.
"""

from __future__ import annotations

import ctypes
import importlib.util
import signal
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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


FAKE_FREE_BYTES = 22504 * MIB
FAKE_TOTAL_BYTES = 24106 * MIB


class FakeEntry:
    """One recording stub for a driver entry point.

    It carries ``argtypes`` and ``restype``, so the binding code
    exercises the same assignment it makes against the real driver.
    """

    def __init__(self, driver: FakeDriver, name: str) -> None:
        """Bind the stub to its driver and entry-point name."""
        self._driver = driver
        self._name = name
        self.argtypes: list[Any] = []
        self.restype: Any = None

    def __call__(self, *args: Any) -> int:
        """Record the call and report the queued CUresult."""
        self._driver.calls.append(self._name)
        if self._name == "cuMemGetInfo_v2":
            args[0]._obj.value = FAKE_FREE_BYTES
            args[1]._obj.value = FAKE_TOTAL_BYTES
        return self._driver.results.get(self._name, ballast.CUDA_SUCCESS)


class FakeDriver:
    """A stand-in for ``libcuda.so.1`` that records calls and returns codes."""

    def __init__(self, results: dict[str, int] | None = None) -> None:
        """Queue a CUresult per entry point. Absent names succeed."""
        self.results = results or {}
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> FakeEntry:
        """Yield a recording stub for one driver entry point."""
        entry = FakeEntry(self, name)
        self.__dict__[name] = entry
        return entry


@pytest.fixture
def fake_ballast(monkeypatch: pytest.MonkeyPatch):
    """Build a CudaBallast over a fake driver."""

    def build(results: dict[str, int] | None = None):
        driver = FakeDriver(results)
        monkeypatch.setattr(ctypes, "CDLL", lambda _name: driver)
        return ballast.CudaBallast(), driver

    return build


def test_size_ballast_from_free_memory_returns_the_difference() -> None:
    assert ballast.size_ballast(FAKE_FREE_BYTES, 12288 * MIB) == 10216 * MIB


def test_size_ballast_at_exactly_target_holds_nothing() -> None:
    assert ballast.size_ballast(12288 * MIB, 12288 * MIB) == 0


def test_size_ballast_below_target_refuses() -> None:
    with pytest.raises(ValueError, match="below the 12288 MiB target"):
        ballast.size_ballast(8000 * MIB, 12288 * MIB)


def test_size_ballast_nonpositive_target_refuses() -> None:
    with pytest.raises(ValueError, match="must exceed 0 MiB"):
        ballast.size_ballast(FAKE_FREE_BYTES, 0)


def test_parse_args_with_a_separator_strips_it_from_the_command() -> None:
    args = ballast.parse_args(
        ["--target-mib", "12288", "--", "llama-cli", "--list-devices"]
    )
    assert args.target_mib == 12288
    assert args.command == ["llama-cli", "--list-devices"]


def test_parse_args_without_a_command_holds() -> None:
    args = ballast.parse_args(["--target-mib", "4096"])
    assert args.command == []


def test_parse_args_without_a_target_defaults_to_twelve_gib() -> None:
    assert ballast.parse_args([]).target_mib == 12288


def test_exit_status_for_a_clean_exit_passes_the_code_through() -> None:
    assert ballast.exit_status(42) == 42


def test_exit_status_for_a_signalled_child_returns_the_shell_convention() -> None:
    assert ballast.exit_status(-signal.SIGTERM) == 128 + int(signal.SIGTERM)
    assert ballast.exit_status(-signal.SIGINT) == 128 + int(signal.SIGINT)


def test_run_command_with_a_clean_child_returns_the_child_exit_code() -> None:
    assert ballast.run_command([sys.executable, "-c", "raise SystemExit(42)"]) == 42


def test_run_command_with_a_signalled_child_returns_128_plus_the_signal() -> None:
    killer = "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
    assert ballast.run_command([sys.executable, "-c", killer]) == 143


def test_run_command_restores_the_previous_signal_handlers() -> None:
    before = {number: signal.getsignal(number) for number in ballast._WATCHED_SIGNALS}
    ballast.run_command([sys.executable, "-c", ""])
    after = {number: signal.getsignal(number) for number in ballast._WATCHED_SIGNALS}
    assert after == before


def test_run_command_with_an_empty_command_returns_127(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ballast.run_command([]) == ballast.EXIT_COMMAND_NOT_FOUND
    assert "no command to run" in capsys.readouterr().err


def test_run_command_with_a_missing_binary_returns_127(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ballast.run_command(["/no/such/binary"]) == ballast.EXIT_COMMAND_NOT_FOUND
    assert "cannot run /no/such/binary" in capsys.readouterr().err


def test_hold_at_zero_bytes_skips_the_allocation(fake_ballast) -> None:
    holder, driver = fake_ballast()
    holder.hold(0)
    assert "cuMemAlloc_v2" not in driver.calls


def test_repeated_release_frees_and_destroys_once(fake_ballast) -> None:
    holder, driver = fake_ballast()
    holder.open_device(0)
    holder.hold(10216 * MIB)
    holder.release()
    holder.release()
    assert driver.calls.count("cuMemFree_v2") == 1
    assert driver.calls.count("cuCtxDestroy_v2") == 1


def test_release_without_a_hold_frees_nothing(fake_ballast) -> None:
    holder, driver = fake_ballast()
    holder.release()
    assert "cuMemFree_v2" not in driver.calls


def test_release_reports_a_driver_failure_without_raising(
    fake_ballast, capsys: pytest.CaptureFixture[str]
) -> None:
    holder, _ = fake_ballast({"cuMemFree_v2": 999})
    holder.hold(10216 * MIB)
    holder.release()
    err = capsys.readouterr().err
    assert "cuMemFree failed with CUresult 999" in err
    assert "process exit reclaims" in err


def test_driver_error_on_allocation_raises(fake_ballast) -> None:
    holder, _ = fake_ballast({"cuMemAlloc_v2": 2})
    with pytest.raises(ballast.CudaDriverError, match="cuMemAlloc failed"):
        holder.hold(10216 * MIB)


def test_missing_driver_library_raises_a_named_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_name: str) -> None:
        raise OSError("no such file")

    monkeypatch.setattr(ctypes, "CDLL", refuse)
    with pytest.raises(ballast.CudaDriverError, match=r"cannot load libcuda\.so\.1"):
        ballast.CudaBallast()


def test_memory_info_reports_the_drivers_free_and_total(fake_ballast) -> None:
    holder, _ = fake_ballast()
    assert holder.memory_info() == (FAKE_FREE_BYTES, FAKE_TOTAL_BYTES)
