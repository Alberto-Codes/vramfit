"""Generate assisted `Q4_0` golden fixtures from libggml (ADR-0018).

Drives ``ggml_quantize_chunk`` for ``Q4_0`` with a non-NULL imatrix
pointer and records the dequantized outputs the torch port of
``quantize_row_q4_0_impl`` must reproduce. Needs a llama.cpp CPU
build at the campaign instrument, 4801e3c56 (b10362) —
``ggml/src/ggml-quants.c`` is byte-identical to 3653e6d6d, so a
b10326 build serves too. The script first replays one committed
`q0-ref` fixture and proves the assisted branch engages, so a wrong
library fails loudly before it writes anything.

Usage:
    uv run python scripts/gen_q0_assisted_goldens.py <build/bin dir>
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

ROWS = 4
N_PER_ROW = 512
GGML_TYPE_Q4_0 = 2
OUT = Path(__file__).parent.parent / "tests" / "data" / "q0_ref"


def _load(bin_dir: Path) -> ctypes.CDLL:
    """Load libggml-base and declare the entry points.

    Args:
        bin_dir: The llama.cpp ``build/bin`` directory.

    Returns:
        The loaded library with argtypes set.
    """
    lib = ctypes.CDLL(str(bin_dir / "libggml-base.so"))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.dequantize_row_q4_0.restype = None
    lib.dequantize_row_q4_0.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ]
    return lib


def _round_trip(lib: ctypes.CDLL, x: np.ndarray, qw: np.ndarray | None) -> np.ndarray:
    """Quantize rows through libggml and dequantize them back.

    Args:
        lib: The loaded library.
        x: Input rows, shape ``(rows, n_per_row)``, float32.
        qw: Imatrix column weights, shape ``(n_per_row,)``, or None
            for the unassisted reference path.

    Returns:
        Dequantized values, shape of ``x``.

    Raises:
        RuntimeError: If the quantizer wrote an unexpected size.
    """
    rows, n_per_row = x.shape
    src = np.ascontiguousarray(x, dtype=np.float32)
    row_size = lib.ggml_row_size(GGML_TYPE_Q4_0, n_per_row)
    buf = ctypes.create_string_buffer(row_size * rows)
    qw_ptr = (
        np.ascontiguousarray(qw, dtype=np.float32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        if qw is not None
        else None
    )
    written = lib.ggml_quantize_chunk(
        GGML_TYPE_Q4_0,
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        buf,
        0,
        rows,
        n_per_row,
        qw_ptr,
    )
    if written != row_size * rows:
        raise RuntimeError(
            f"quantize wrote {written} bytes, expected {row_size * rows}"
        )
    out = np.empty(rows * n_per_row, dtype=np.float32)
    lib.dequantize_row_q4_0(
        buf, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), rows * n_per_row
    )
    return out.reshape(rows, n_per_row)


def _cases(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the (x, qw) fixture cases.

    ADR-0018 decision 4's shapes with the assisted-fixture weight
    patterns: a lognormal activation profile, flat ones, one spiked
    column, and zeroed holes — plus the value patterns the reference
    suite pins (outliers, positive, constant, zeros, tiny).

    Args:
        rng: Seeded generator, so fixtures regenerate identically.

    Returns:
        ``(x, qw)`` pairs keyed by case name.
    """
    shape = (ROWS, N_PER_ROW)
    gauss = rng.standard_normal(shape).astype(np.float32) * 0.05
    outliers = gauss.copy()
    outliers[:, rng.integers(0, N_PER_ROW, 8)] *= 40.0
    act = rng.lognormal(mean=0.0, sigma=1.5, size=N_PER_ROW).astype(np.float32)
    spike = act.copy()
    spike[N_PER_ROW // 3] *= 1000.0
    holes = act.copy()
    holes[::7] = 0.0
    return {
        "gauss_act": (gauss, act),
        "gauss_ones": (gauss, np.ones(N_PER_ROW, dtype=np.float32)),
        "gauss_spike": (gauss, spike),
        "gauss_holes": (gauss, holes),
        "outliers_act": (outliers, act),
        "positive_act": (np.abs(gauss) + 0.01, act),
        "constant_act": (np.full(shape, 0.25, dtype=np.float32), act),
        "zeros_act": (np.zeros(shape, dtype=np.float32), act),
        "tiny_act": (gauss * np.float32(1e-20), act),
    }


def _check_library(lib: ctypes.CDLL) -> None:
    """Replay committed fixtures to prove the library matches.

    Args:
        lib: The loaded library.

    Raises:
        SystemExit: If the unassisted ``Q4_0`` replay diverges from
            the committed golden, the assisted branch does not
            engage, or an existing assisted fixture fails to replay.
    """
    with np.load(OUT / "golden.npz") as committed:
        x = committed["x_gauss"]
        expected = committed["q4_0_gauss"]
    if not np.array_equal(_round_trip(lib, x, None), expected):
        sys.exit("Q4_0 replay diverges from golden.npz — wrong llama.cpp build?")
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((ROWS, N_PER_ROW)).astype(np.float32)
    weights = rng.random(N_PER_ROW).astype(np.float32) + 0.1
    if np.array_equal(_round_trip(lib, probe, weights), _round_trip(lib, probe, None)):
        sys.exit("assisted output equals unassisted — imatrix pointer not marshaled?")
    existing = OUT / "golden-assisted.npz"
    if not existing.exists():
        return
    with np.load(existing) as committed:
        replay = _round_trip(lib, committed["x_gauss_act"], committed["qw_gauss_act"])
        if not np.array_equal(replay, committed["q4_0_gauss_act"]):
            sys.exit("assisted replay diverges from golden-assisted.npz — wrong build?")


def main() -> None:
    """Generate ``tests/data/q0_ref/golden-assisted.npz``."""
    bin_dir = Path(sys.argv[1]).expanduser()
    lib = _load(bin_dir)
    _check_library(lib)
    arrays: dict[str, np.ndarray] = {}
    for case, (x, qw) in _cases(np.random.default_rng(20260821)).items():
        arrays[f"x_{case}"] = x
        arrays[f"qw_{case}"] = qw
        arrays[f"q4_0_{case}"] = _round_trip(lib, x, qw)
    # A NULL-pointer marshaling bug would record unassisted outputs
    # under the assisted keys — prove the weights reached the C.
    unassisted = _round_trip(lib, arrays["x_gauss_act"], None)
    if np.array_equal(arrays["q4_0_gauss_act"], unassisted):
        sys.exit("assisted output equals unassisted — imatrix pointer not marshaled?")
    np.savez_compressed(OUT / "golden-assisted.npz", **arrays)
    print(f"wrote {OUT / 'golden-assisted.npz'} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
