"""Generate `q0-ref` golden fixtures from libggml (ADR-0018).

Drives ``ggml_quantize_chunk`` for ``Q2_0``, ``Q4_0``, and ``Q8_0``
and records the dequantized outputs the torch port must reproduce
bit-exactly. Needs a llama.cpp CPU build carrying ``Q2_0``. The
pinned instrument is 3653e6d6d (b10326), and e9fa078 serves too —
``ggml-quants.c`` and ``ggml-common.h`` are byte-identical across
the two. The script first replays one committed ADR-0018 fixture,
so a wrong library version fails loudly before it writes anything.

Usage:
    uv run python scripts/gen_q0_ref_goldens.py ~/llama.cpp/build/bin
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

ROWS = 4
# A multiple of 64, 32, and 256: one row feeds every type here and
# the ADR-0018 replay below.
N_PER_ROW = 512
# ggml.h type ids.
TYPES = {"q2_0": 42, "q4_0": 2, "q8_0": 8}
DEQUANTIZE = {
    42: "dequantize_row_q2_0",
    2: "dequantize_row_q4_0",
    8: "dequantize_row_q8_0",
}
KQUANT = Path(__file__).parent.parent / "tests" / "data" / "kquant"
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
    for name in (*DEQUANTIZE.values(), "dequantize_row_q3_K"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    return lib


def _round_trip(lib: ctypes.CDLL, type_id: int, x: np.ndarray) -> np.ndarray:
    """Quantize rows through libggml and dequantize them back.

    Args:
        lib: The loaded library.
        type_id: The ggml type enum value.
        x: Input rows, shape ``(rows, n_per_row)``, float32.

    Returns:
        Dequantized values, shape of ``x``.

    Raises:
        RuntimeError: If the quantizer wrote an unexpected size —
            a wrong type id or a row length the type refuses.
    """
    rows, n_per_row = x.shape
    src = np.ascontiguousarray(x, dtype=np.float32)
    row_size = lib.ggml_row_size(type_id, n_per_row)
    buf = ctypes.create_string_buffer(row_size * rows)
    written = lib.ggml_quantize_chunk(
        type_id,
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        buf,
        0,
        rows,
        n_per_row,
        None,
    )
    if written != row_size * rows:
        raise RuntimeError(
            f"quantize wrote {written} bytes, expected {row_size * rows}"
        )
    out = np.empty(rows * n_per_row, dtype=np.float32)
    getattr(lib, DEQUANTIZE[type_id])(
        buf, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), rows * n_per_row
    )
    return out.reshape(rows, n_per_row)


def _cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Build the fixture cases.

    ADR-0018 decision 3 names random, outlier-heavy, constant, zero,
    and subnormal-scale blocks, and adds exact ties. ``ties`` holds
    ``+a`` and ``-a`` at equal magnitude, which is where ``Q4_0``'s
    first-maximum rule decides the whole block's sign. ``halfway``
    lands values exactly on ``.5`` boundaries, where ``Q2_0``'s
    ``roundf`` and ``Q4_0``'s truncating cast disagree.
    ``near_half`` lands one ulp *below* those boundaries, where
    ``roundf`` rounds down but a shift-and-truncate rounds up — the
    float32 add ``0.49999997 + 0.5`` reaches the exact midpoint
    under 1.0 and snaps to it.

    Args:
        rng: Seeded generator, so fixtures regenerate identically.

    Returns:
        Input rows keyed by case name.
    """
    shape = (ROWS, N_PER_ROW)
    gauss = (rng.standard_normal(shape) * 0.05).astype(np.float32)
    outliers = gauss.copy()
    outliers[:, rng.integers(0, N_PER_ROW, 8)] *= 40.0
    ties = np.tile(np.array([0.5, -0.5], dtype=np.float32), (ROWS, N_PER_ROW // 2))
    ties[1] = -ties[1]
    ties[2, ::3] *= 0.5
    # amax is 1.0, so Q2_0 sees w/d on exact halves and Q4_0 sees
    # x*id + 8.5 on exact integers.
    halfway = (rng.integers(-4, 5, shape) / 4.0).astype(np.float32)
    halfway[:, 0] = 1.0
    # One ulp below each half-integer step, against an absmax of 1.
    # Q2_0 sees w/d just under 0.5, and Q8_0 sees w/d just under
    # each of 0.5, 1.5, ... 126.5.
    below = np.nextafter(np.float32(0.5), np.float32(0))
    near_half = np.tile(
        np.array([below, -below, below / 127.0, -below / 127.0], dtype=np.float32),
        (ROWS, N_PER_ROW // 4),
    )
    near_half[:, 0] = 1.0
    return {
        "gauss": gauss,
        "outliers": outliers,
        "constant": np.full(shape, 0.25, dtype=np.float32),
        "zeros": np.zeros(shape, dtype=np.float32),
        "tiny": gauss * np.float32(1e-20),
        "subnormal": gauss * np.float32(1e-38),
        "ties": ties,
        "halfway": halfway,
        "near_half": near_half,
    }


def _check_library(lib: ctypes.CDLL) -> None:
    """Replay one ADR-0018 fixture to prove the library matches.

    Args:
        lib: The loaded library.

    Raises:
        SystemExit: If the Q3_K replay diverges from the committed
            golden — wrong checkout or wrong build.
    """
    with np.load(KQUANT / "golden.npz") as committed:
        x = committed["x_gauss"]
        expected = committed["q3_gauss"]
    row_size = lib.ggml_row_size(11, x.reshape(ROWS, -1).shape[1])
    buf = ctypes.create_string_buffer(row_size * ROWS)
    src = np.ascontiguousarray(x.reshape(ROWS, -1), dtype=np.float32)
    lib.ggml_quantize_chunk(
        11,
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        buf,
        0,
        ROWS,
        src.shape[1],
        None,
    )
    out = np.empty(x.size, dtype=np.float32)
    lib.dequantize_row_q3_K(
        buf, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x.size
    )
    if not np.array_equal(out.reshape(x.shape), expected):
        sys.exit("library replay diverges from golden.npz — wrong llama.cpp build?")
    # The Q3_K replay proves the checkout, not these three types — a
    # build where Q2_0 moved and Q3_K did not would pass it. Replay
    # the committed q0 fixtures too, once they exist.
    existing = OUT / "golden.npz"
    if not existing.exists():
        return
    with np.load(existing) as committed:
        for name, type_id in TYPES.items():
            key = f"{name}_gauss"
            if key not in committed:
                continue
            replay = _round_trip(lib, type_id, committed["x_gauss"])
            if not np.array_equal(replay, committed[key]):
                sys.exit(f"{name} replay diverges from golden.npz — wrong build?")


def main() -> None:
    """Generate ``tests/data/q0_ref/golden.npz``."""
    bin_dir = Path(sys.argv[1]).expanduser()
    lib = _load(bin_dir)
    _check_library(lib)
    arrays: dict[str, np.ndarray] = {}
    for case, x in _cases(np.random.default_rng(20260817)).items():
        arrays[f"x_{case}"] = x
        for name, type_id in TYPES.items():
            arrays[f"{name}_{case}"] = _round_trip(lib, type_id, x)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "golden.npz", **arrays)
    print(f"wrote {OUT / 'golden.npz'} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
