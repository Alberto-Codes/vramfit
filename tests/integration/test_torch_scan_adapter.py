"""Torch-tier checks of the scan adapter on a tiny offline checkpoint.

These run wherever the scan extra is installed — no CUDA required, no
network. They skip cleanly when torch is absent (ADR-0009).
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")

from typer.testing import CliRunner

from quantfit.adapters.inbound.cli import app
from quantfit.adapters.outbound.scan.calibration import load_calibration
from quantfit.adapters.outbound.scan.kl import (
    mean_kl,
    reference_log_probs,
)
from quantfit.adapters.outbound.scan.quantize import (
    rtn_quantize_dequantize,
)
from quantfit.adapters.outbound.sensitivity_map_json import (
    load_sensitivity_map,
)
from tests.conftest import CALIBRATION_TEXT

pytestmark = [pytest.mark.integration, pytest.mark.slow]

runner = CliRunner()


class TestRtnQuantize:
    def test_8bit_roundtrip_is_near_lossless(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(64, 64)

        assert torch.allclose(w, rtn_quantize_dequantize(w, 8), atol=0.05)

    def test_reconstruction_mse_grows_as_bits_drop(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(64, 64)

        mse = {
            bits: (w - rtn_quantize_dequantize(w, bits)).pow(2).mean().item()
            for bits in (8, 4, 3, 2)
        }

        assert mse[8] < mse[4] < mse[3] < mse[2]

    def test_shape_and_dtype_survive_including_the_padding_path(self) -> None:
        w = torch.randn(17, 5, dtype=torch.bfloat16)

        result = rtn_quantize_dequantize(w, 4, block_size=32)

        assert result.shape == w.shape
        assert result.dtype == w.dtype

    def test_zero_tensor_passes_through_unchanged(self) -> None:
        w = torch.zeros(8, 8)

        assert torch.equal(rtn_quantize_dequantize(w, 2), w)

    def test_bits_below_two_raise(self) -> None:
        with pytest.raises(ValueError, match="bits"):
            rtn_quantize_dequantize(torch.randn(4, 4), 1)

    def test_non_positive_block_size_raises(self) -> None:
        with pytest.raises(ValueError, match="block_size"):
            rtn_quantize_dequantize(torch.randn(4, 4), 4, block_size=0)

    def test_input_tensor_is_never_modified(self) -> None:
        torch.manual_seed(0)
        for shape in ((64, 64), (17, 5)):
            w = torch.randn(*shape)
            before = w.clone()

            result = rtn_quantize_dequantize(w, 2)

            assert torch.equal(w, before)
            assert result.data_ptr() != w.data_ptr()

    @pytest.mark.gpu
    def test_cuda_round_trip_returns_on_cuda_unmodified(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        torch.manual_seed(0)
        w = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
        before = w.clone()

        result = rtn_quantize_dequantize(w, 4)

        assert result.device == w.device
        assert result.dtype == w.dtype
        assert torch.equal(w, before)


class TestMeanKl:
    def test_identical_logits_diverge_by_zero(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(1, 16, 32)

        assert mean_kl(reference_log_probs(logits), logits) == pytest.approx(
            0.0, abs=1e-3
        )

    def test_different_logits_diverge_positively(self) -> None:
        torch.manual_seed(0)
        reference = reference_log_probs(torch.randn(1, 16, 32))

        assert mean_kl(reference, torch.randn(1, 16, 32)) > 0.01

    def test_nan_logits_are_rejected_not_recorded(self) -> None:
        torch.manual_seed(0)
        reference = reference_log_probs(torch.randn(1, 16, 32))
        unstable = torch.full((1, 16, 32), float("nan"))

        with pytest.raises(ValueError, match="numerically"):
            mean_kl(reference, unstable)

    def test_mispaired_reference_is_rejected(self) -> None:
        # A "reference" that is not a normalized log-probability
        # distribution can drive KL strongly negative, which must
        # raise instead of clamping to zero damage.
        bogus_reference = torch.full((1, 4, 8), -40.0)
        bogus_reference[..., 0] = -5.0
        peaked_logits = torch.zeros(1, 4, 8)
        peaked_logits[..., 0] = 100.0

        with pytest.raises(ValueError, match="negative"):
            mean_kl(bogus_reference, peaked_logits)


class TestLoadCalibration:
    def test_token_budget_is_respected(self, tiny_model_dir, tmp_path) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        path = tmp_path / "calib.txt"
        path.write_text(CALIBRATION_TEXT)

        batches, n_tokens = load_calibration(path, tokenizer, max_tokens=64)

        assert 0 < n_tokens <= 64
        assert sum(b.numel() for b in batches) == n_tokens

    def test_too_few_tokens_raises(self, tiny_model_dir, tmp_path) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        path = tmp_path / "calib.txt"
        path.write_text("")

        with pytest.raises(ValueError, match="at least 2"):
            load_calibration(path, tokenizer, max_tokens=64)


@pytest.fixture
def tiny_meter(tiny_model_dir, tmp_path):
    from quantfit.adapters.outbound.scan.meter import TorchDamageMeter

    calibration = tmp_path / "calib.txt"
    calibration.write_text(CALIBRATION_TEXT)
    return TorchDamageMeter(
        str(tiny_model_dir), calibration, max_tokens=128, device="cpu"
    )


class TestTorchDamageMeter:
    def test_layer_grouping_finds_decoder_layers_and_edges(self, tiny_meter) -> None:
        names = {spec.name for spec in tiny_meter.groups()}

        assert "model.layers.0" in names
        assert "model.layers.1" in names
        assert "model.embed_tokens" in names

    def test_tensor_grouping_is_finer_than_layer_grouping(
        self, tiny_meter, tiny_model_dir, tmp_path
    ) -> None:
        from quantfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        by_tensor = TorchDamageMeter(
            str(tiny_model_dir),
            calibration,
            max_tokens=64,
            group_by="tensor",
            device="cpu",
        )

        assert len(by_tensor.groups()) > len(tiny_meter.groups())

    def test_group_bytes_match_parameter_sizes(self, tiny_meter) -> None:
        embed = next(
            spec for spec in tiny_meter.groups() if spec.name == "model.embed_tokens"
        )

        assert embed.bytes_fp16 == 512 * 32 * 2

    def test_gpt2_style_names_group_by_layer(self) -> None:
        from quantfit.adapters.outbound.scan.meter import _discover_groups

        class Gpt2Like(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.h = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(2)]
                )

        groups = _discover_groups(Gpt2Like(), "layer")

        assert set(groups) == {"transformer.h.0", "transformer.h.1"}

    def test_max_memory_maps_cap_and_cpu_for_auto_only(self) -> None:
        from quantfit.adapters.outbound.scan.meter import _max_memory

        assert _max_memory("auto", 17 * 2**30) == {
            0: 17 * 2**30,
            "cpu": 999 * 2**30,
        }
        assert _max_memory("cpu", 17 * 2**30) is None
        assert _max_memory("auto", None) is None

    def test_poisoned_meter_refuses_measure_recipe(self, tiny_meter) -> None:
        tiny_meter._poisoned = True
        recipe = {spec.name: 8 for spec in tiny_meter.groups()}

        with pytest.raises(RuntimeError, match="rebuild the meter"):
            tiny_meter.measure_recipe(recipe)

    def test_poisoned_meter_refuses_measure(self, tiny_meter) -> None:
        tiny_meter._poisoned = True
        group = tiny_meter.groups()[0].name

        with pytest.raises(RuntimeError, match="rebuild the meter"):
            tiny_meter.measure(group, 8)

    @pytest.mark.gpu
    def test_measure_recipe_on_cuda_restores_across_devices(
        self, tiny_model_dir, tmp_path
    ) -> None:
        # The whole-recipe pass stages originals on the CPU and
        # restores host-to-device — the path every GPU validation
        # takes, untestable on the CPU-only contract leg.
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from quantfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        meter = TorchDamageMeter(
            str(tiny_model_dir), calibration, max_tokens=128, device="cuda"
        )
        group = meter.groups()[0].name
        before = meter.measure(group, 2)

        damage = meter.measure_recipe({spec.name: 4 for spec in meter.groups()})

        assert damage >= 0.0
        assert meter.measure(group, 2) == before

    def test_layer_grouping_without_layer_structure_raises(self) -> None:
        from quantfit.adapters.outbound.scan.meter import _discover_groups

        class Flat(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

        with pytest.raises(ValueError, match="--group-by tensor"):
            _discover_groups(Flat(), "layer")


def test_scan_cli_produces_a_valid_map_on_the_tiny_model(
    tiny_model_dir, tmp_path
) -> None:
    calibration = tmp_path / "calib.txt"
    calibration.write_text(CALIBRATION_TEXT)
    out = tmp_path / "sensitivity.json"

    result = runner.invoke(
        app,
        [
            "scan",
            str(tiny_model_dir),
            "--calibration",
            str(calibration),
            "--out",
            str(out),
            "--precisions",
            "8,4,3,2",
            "--max-tokens",
            "64",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    map_ = load_sensitivity_map(out)
    assert map_.scan.precisions == (8, 4, 3, 2)
    assert map_.scan.metric == "kl_divergence"
    assert all(group.sensitivity[2] >= group.sensitivity[8] for group in map_.groups)

    rerun = runner.invoke(
        app,
        [
            "scan",
            str(tiny_model_dir),
            "--calibration",
            str(calibration),
            "--out",
            str(out),
            "--precisions",
            "8,4,3,2",
            "--max-tokens",
            "64",
            "--device",
            "cpu",
        ],
    )

    assert rerun.exit_code == 0, rerun.output
    assert "[1/" not in rerun.output
