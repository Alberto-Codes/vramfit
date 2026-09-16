"""Generate the assisted ``Q2_0`` reference fixtures (ADR-0032).

The torch encoder in
``src/vramfit/adapters/outbound/scan/q2_0_assisted.py`` reimplements
the reference specification recorded in issue #461. No stock
llama.cpp build carries that fit, so no library can record goldens
the way ``gen_q0_assisted_goldens.py`` does. This script is the
independent oracle instead: a scalar float32 transliteration of the
specification, one element at a time in the reference's sequential
order, written without reading the torch port. It computes every
fixture in ``tests/data/q2_0_assisted/fixtures.json``.

Four crafted cases carry hand-checked expectations, and the test
asserts those directly as well:

- ``signed_scale``: ``-2`` beside 63 ones fits exactly at ``d = -1``
  from the negative seed, an outcome stock ``Q2_0`` cannot reach.
- ``rounding_tie``: ``x / d`` lands on exactly ``0.5``. Half away
  from zero codes it 1 and wins at ``d = 35.5 / 67``. Half to even
  would leave ``d = 2`` and the codes ``[1, 0, ...]``.
- ``zero_block``: every seed is zero, so the block stores ``d = 0``
  and level 0 everywhere, which packs as ``0x55`` bytes.
- ``weighted_selection``: the ``rounding_tie`` block again, with a
  column weight of 50 on its outlier. The unweighted search stores
  ``d = 35.5 / 67`` and the weighted one stores ``d = 0.94921875``,
  so the matrix changed the candidate the block keeps.

Two random cases cover bulk agreement with imatrix weights and with
the unweighted search.

Usage:
    uv run python scripts/gen_q2_0_assisted_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

QK = 64
SEEDS = (np.float32(1.0), np.float32(0.5), np.float32(-0.5))
OUT = (
    Path(__file__).parent.parent / "tests" / "data" / "q2_0_assisted" / "fixtures.json"
)

f32 = np.float32


def fp16(value: np.float32) -> np.float32:
    """Round through fp16 storage, like ``GGML_FP32_TO_FP16``."""
    return f32(np.float16(value))


def roundf(value: np.float32) -> int:
    """Round half away from zero, like C's ``roundf``."""
    return math.floor(abs(float(value)) + 0.5) * (1 if value >= 0 else -1)


def code(x: np.float32, inverse: np.float32) -> int:
    """Code one element: ``clamp(roundf(x * id), -1, 2)``."""
    return max(-1, min(2, roundf(f32(x * inverse))))


def encode_row(row: list[float], qw: list[float] | None) -> list[dict[str, object]]:
    """Encode one row block by block, in the reference's scalar order."""
    xs = [f32(v) for v in row]
    weights: list[np.float32]
    if qw is None:
        weights = [f32(1.0)] * len(xs)
    else:
        sigma2 = f32(0.0)
        for x in xs:
            sigma2 = f32(sigma2 + f32(x * x))
        sigma2 = f32(sigma2 / f32(len(xs)))
        weights = [
            f32(f32(w) * f32(math.sqrt(f32(sigma2 + f32(x * x)))))
            for w, x in zip(qw, xs, strict=True)
        ]
    blocks = []
    for start in range(0, len(xs), QK):
        x = xs[start : start + QK]
        w = weights[start : start + QK]
        amax = f32(0.0)
        for v in x:
            amax = max(amax, f32(abs(v)))
        best_error = f32(np.inf)
        best_d = f32(0.0)
        best_q = [0] * QK
        for seed in SEEDS:
            d = f32(seed * amax)
            for _ in range(4):
                if d == 0:
                    break
                inverse = f32(f32(1.0) / d)
                sum_xq = f32(0.0)
                sum_q2 = f32(0.0)
                for xi, wi in zip(x, w, strict=True):
                    q = code(xi, inverse)
                    sum_xq = f32(sum_xq + f32(f32(wi * xi) * f32(q)))
                    sum_q2 = f32(sum_q2 + f32(f32(wi * f32(q)) * f32(q)))
                if sum_q2 > 0:
                    d = f32(sum_xq / sum_q2)
            d = fp16(d)
            inverse = f32(f32(1.0) / d) if d != 0 else f32(0.0)
            error = f32(0.0)
            codes = []
            for xi, wi in zip(x, w, strict=True):
                q = code(xi, inverse) if d != 0 else 0
                codes.append(q)
                diff = f32(xi - f32(d * f32(q)))
                error = f32(error + f32(f32(wi * diff) * diff))
            if error < best_error:
                best_error = error
                best_d = d
                best_q = codes
        blocks.append({"scale": float(best_d), "levels": best_q})
    return blocks


def crafted() -> dict[str, dict[str, object]]:
    """Build the four hand-checked cases."""
    signed = [1.0] * QK
    signed[0] = -2.0
    tie = [0.5] * QK
    tie[0] = 2.0
    weighted = [0.5] * QK
    weighted[0] = 2.0
    weighted_qw = [1.0] * QK
    weighted_qw[0] = 50.0
    return {
        "signed_scale": {"rows": [signed], "qw": None},
        "rounding_tie": {"rows": [tie], "qw": None},
        "zero_block": {"rows": [[0.0] * QK], "qw": None},
        "weighted_selection": {"rows": [weighted], "qw": [weighted_qw]},
    }


def random_cases() -> dict[str, dict[str, object]]:
    """Build the two random bulk cases."""
    rng = np.random.default_rng(2032)
    rows = rng.standard_normal((4, 128)).astype(np.float32)
    qw = rng.uniform(0.1, 3.0, size=(4, 128)).astype(np.float32)
    return {
        "random_weighted": {"rows": rows.tolist(), "qw": qw.tolist()},
        "random_unweighted": {"rows": rows.tolist(), "qw": None},
    }


def main() -> None:
    """Write every fixture."""
    fixtures: dict[str, object] = {}
    for name, case in {**crafted(), **random_cases()}.items():
        rows = list(case["rows"])  # type: ignore[arg-type]
        qw = case["qw"]
        expected = [
            encode_row(row, None if qw is None else qw[i]) for i, row in enumerate(rows)
        ]
        fixtures[name] = {"rows": rows, "qw": qw, "blocks": expected}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures, indent=1) + "\n")
    print(f"wrote {len(fixtures)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
