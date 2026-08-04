"""Generate assisted K-quant golden fixtures from libggml (ADR-0020).

Drives ``ggml_quantize_chunk`` through ctypes with a non-NULL imatrix
pointer and records the dequantized outputs the torch port must
reproduce. Needs a llama.cpp CPU build at the pinned checkout
(e9fa078) — pass its ``build/bin`` directory. The script first
replays one unassisted case against the committed ADR-0018 fixtures,
so a wrong library version fails loudly before it writes anything.

Usage:
    uv run python scripts/gen_kquant_assisted_goldens.py ~/llama.cpp/build/bin
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

QK_K = 256
ROWS = 4
N_PER_ROW = 512
TYPES = {"q2": 10, "q3": 11, "q4": 12}
OUT = Path(__file__).parent.parent / "tests" / "data" / "kquant"


def _load(bin_dir: Path) -> ctypes.CDLL:
    """Load libggml-base and declare the three entry points.

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
    for name in ("dequantize_row_q2_K", "dequantize_row_q3_K", "dequantize_row_q4_K"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
    return lib


def _round_trip(
    lib: ctypes.CDLL, type_id: int, x: np.ndarray, qw: np.ndarray | None
) -> np.ndarray:
    """Quantize rows through libggml and dequantize them back.

    Args:
        lib: The loaded library.
        type_id: The ggml type enum value.
        x: Input rows, shape ``(rows, n_per_row)``, float32.
        qw: Imatrix column weights, shape ``(n_per_row,)``, or None
            for the unassisted reference path.

    Returns:
        Dequantized values, shape of ``x``.
    """
    rows, n_per_row = x.shape
    src = np.ascontiguousarray(x, dtype=np.float32)
    row_size = lib.ggml_row_size(type_id, n_per_row)
    buf = ctypes.create_string_buffer(row_size * rows)
    qw_ptr = (
        np.ascontiguousarray(qw, dtype=np.float32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        if qw is not None
        else None
    )
    written = lib.ggml_quantize_chunk(
        type_id,
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
    dequantize = {
        10: lib.dequantize_row_q2_K,
        11: lib.dequantize_row_q3_K,
        12: lib.dequantize_row_q4_K,
    }[type_id]
    dequantize(
        buf, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), rows * n_per_row
    )
    return out.reshape(rows, n_per_row)


def _cases(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the (x, qw) fixture cases.

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
        "tiny_act": (gauss * 1e-20, act),
    }


def _check_library(lib: ctypes.CDLL) -> None:
    """Replay one ADR-0018 fixture to prove the library matches.

    Args:
        lib: The loaded library.

    Raises:
        SystemExit: If the unassisted Q3_K replay diverges from the
            committed golden — wrong checkout or wrong build.
    """
    with np.load(OUT / "golden.npz") as committed:
        x = committed["x_gauss"]
        expected = committed["q3_gauss"]
    replay = _round_trip(lib, TYPES["q3"], x.reshape(ROWS, -1), None)
    if not np.array_equal(replay.reshape(x.shape), expected):
        sys.exit("library replay diverges from golden.npz — wrong llama.cpp build?")
    # The assisted kernels changed across llama.cpp history while the
    # unassisted ones stayed put — replay one assisted case too.
    assisted_path = OUT / "golden-assisted.npz"
    if assisted_path.exists():
        with np.load(assisted_path) as committed:
            xa = committed["x_gauss_act"]
            qw = committed["qw_gauss_act"]
            expected_a = committed["q2_gauss_act"]
        replay_a = _round_trip(lib, TYPES["q2"], xa, qw)
        if not np.array_equal(replay_a, expected_a):
            sys.exit(
                "assisted replay diverges from golden-assisted.npz — "
                "wrong llama.cpp build?"
            )


def main() -> None:
    """Generate ``tests/data/kquant/golden-assisted.npz``."""
    bin_dir = Path(sys.argv[1]).expanduser()
    lib = _load(bin_dir)
    _check_library(lib)
    arrays: dict[str, np.ndarray] = {}
    for case, (x, qw) in _cases(np.random.default_rng(20260803)).items():
        arrays[f"x_{case}"] = x
        arrays[f"qw_{case}"] = qw
        for name, type_id in TYPES.items():
            arrays[f"{name}_{case}"] = _round_trip(lib, type_id, x, qw)
    # A NULL-pointer marshaling bug would record unassisted outputs
    # under the assisted keys — prove the weights reached the C.
    unassisted = _round_trip(lib, TYPES["q2"], arrays["x_gauss_act"], None)
    if np.array_equal(arrays["q2_gauss_act"], unassisted):
        sys.exit("assisted output equals unassisted — imatrix pointer not marshaled?")
    np.savez_compressed(OUT / "golden-assisted.npz", **arrays)
    print(f"wrote {OUT / 'golden-assisted.npz'} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
