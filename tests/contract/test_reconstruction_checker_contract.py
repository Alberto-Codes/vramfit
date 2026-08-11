"""ReconstructionChecker contract: the gguf-py adapter and the fake agree.

The real side reads true GGUF files written with gguf-py — the same
library the adapter dequantizes with — so the suite stays hermetic
(ADR-0009). The measured values themselves are pinned in the
real-only tests below the shared contract: the fake cannot reach the
dequantize path, and the collapse verdict on top of these numbers is
domain arithmetic with its own unit suite.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports

from __future__ import annotations

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="pack extra not installed")
pytest.importorskip("gguf", reason="pack extra not installed")

from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import dequantize, quantize

from tests.fakes import MemoryReconstructionChecker
from vramfit.adapters.outbound.gguf.reconstruction import GgufReconstructionChecker
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.ports.outbound import ReconstructionChecker

pytestmark = pytest.mark.contract

TENSOR = "blk.0.attn_v.weight"
KEPT_F16 = "blk.0.ffn_up.weight"
SHAPE = (4, 64)


def reference_values() -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    return rng.normal(scale=0.02, size=SHAPE).astype(np.float32)


def write_gguf(
    path: Path,
    tensors: dict[str, tuple[np.ndarray, GGMLQuantizationType | None]],
) -> None:
    writer = GGUFWriter(path, arch="llama")
    for name, (data, raw_dtype) in tensors.items():
        if raw_dtype is None:
            writer.add_tensor(name, data.astype(np.float16))
        else:
            writer.add_tensor(name, quantize(data, raw_dtype), raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _real_checker(tmp_path: Path) -> ReconstructionChecker:
    values = reference_values()
    base = tmp_path / "base.gguf"
    packed = tmp_path / "packed.gguf"
    write_gguf(base, {TENSOR: (values, None), KEPT_F16: (values, None)})
    write_gguf(
        packed,
        {
            TENSOR: (values, GGMLQuantizationType.Q8_0),
            KEPT_F16: (values, None),
        },
    )
    return GgufReconstructionChecker(packed=packed, base=base)


def _fake_checker(tmp_path: Path) -> ReconstructionChecker:
    return MemoryReconstructionChecker(errors={TENSOR: 0.0003, KEPT_F16: 0.0})


@pytest.mark.parametrize(
    "build", [_real_checker, _fake_checker], ids=["real-gguf-py", "fake-memory"]
)
class TestReconstructionCheckerContract:
    def test_every_requested_tensor_is_measured(self, build, tmp_path) -> None:
        checker = build(tmp_path)

        errors = checker.rmse((TENSOR, KEPT_F16))

        assert set(errors) == {TENSOR, KEPT_F16}

    def test_measurements_are_finite_and_non_negative(self, build, tmp_path) -> None:
        checker = build(tmp_path)

        errors = checker.rmse((TENSOR,))

        assert all(math.isfinite(v) and v >= 0 for v in errors.values())

    def test_missing_tensor_raises_pack_error(self, build, tmp_path) -> None:
        checker = build(tmp_path)

        with pytest.raises(PackError, match=r"blk\.9\.attn_q\.weight"):
            checker.rmse(("blk.9.attn_q.weight",))


class TestRealCheckerMeasurements:
    def test_quantized_tensor_measures_its_dequantize_error(self, tmp_path) -> None:
        checker = _real_checker(tmp_path)
        # The base stores f16, so the reference is the f16 cast.
        values = reference_values().astype(np.float16).astype(np.float32)
        expected = dequantize(
            quantize(reference_values(), GGMLQuantizationType.Q8_0),
            GGMLQuantizationType.Q8_0,
        )
        rmse = float(np.sqrt(np.mean((expected - values) ** 2)))

        errors = checker.rmse((TENSOR,))

        assert errors[TENSOR] == pytest.approx(rmse, rel=1e-3)
        assert errors[TENSOR] > 0

    def test_f16_tensor_reconstructs_almost_exactly(self, tmp_path) -> None:
        checker = _real_checker(tmp_path)

        errors = checker.rmse((KEPT_F16,))

        # f16 storage of f32 values: only the cast error remains.
        assert errors[KEPT_F16] < 1e-4

    def test_unreadable_file_raises_pack_error(self, tmp_path) -> None:
        bogus = tmp_path / "bogus.gguf"
        bogus.write_bytes(b"not a gguf")
        checker = GgufReconstructionChecker(packed=bogus, base=bogus)

        with pytest.raises(PackError, match="reconstruction"):
            checker.rmse((TENSOR,))
