"""Torch-tier checks of the scan adapter on a tiny offline checkpoint.

These run wherever the scan extra is installed — no CUDA required, no
network. They skip cleanly when torch is absent (ADR-0009).
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

from typing import Literal, cast

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")

from typer.testing import CliRunner

from tests.conftest import CALIBRATION_TEXT
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.scan.calibration import load_calibration
from vramfit.adapters.outbound.scan.kl import (
    mean_kl,
    reference_log_probs,
)
from vramfit.adapters.outbound.scan.quantize import (
    rtn_quantize_dequantize,
)
from vramfit.adapters.outbound.sensitivity_map_json import (
    load_sensitivity_map,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

runner = CliRunner()


class _MambaLike(torch.nn.Module):
    """One Nemotron-H Mamba layer: conv1d, router, and input projection."""

    def __init__(self) -> None:
        super().__init__()
        layer = torch.nn.Module()
        layer.mixer = torch.nn.Module()
        layer.mixer.conv1d = torch.nn.Conv1d(8, 8, 4, groups=8)
        layer.mixer.gate = torch.nn.Linear(4, 2)
        layer.mixer.in_proj = torch.nn.Linear(4, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])


class _MoeLike(torch.nn.Module):
    """One mixture-of-experts layer, named like the Nemotron family.

    Routed experts carry an index that `stack` grouping collapses. The
    shared expert and the attention projection carry none, so they stay
    separate. Sizes are irrelevant — only the parameter names matter.
    """

    def __init__(self, experts: int) -> None:
        super().__init__()

        def expert() -> torch.nn.Module:
            part = torch.nn.Module()
            part.up_proj = torch.nn.Linear(4, 4)
            part.down_proj = torch.nn.Linear(4, 4)
            return part

        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.ModuleList([expert() for _ in range(experts)])
        layer.mlp.shared_expert = torch.nn.Module()
        layer.mlp.shared_expert.up_proj = torch.nn.Linear(4, 4)
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])


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
    from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

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
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

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

    def test_discovered_groups_match_the_pack_flag_literals(self, tiny_meter) -> None:
        # The GGUF backend keys its embedding and output flags on these
        # exact names (ADR-0012). If discovery ever renames them, the
        # flags disengage and the renamed group surfaces only later,
        # as a PackError for an unmapped group. This pins the drift.
        from vramfit.adapters.outbound.gguf.types import (
            EMBEDDING_GROUPS,
            OUTPUT_GROUP,
        )

        names = {spec.name for spec in tiny_meter.groups()}

        assert names & EMBEDDING_GROUPS
        assert OUTPUT_GROUP in names

    def test_gpt2_style_names_group_by_layer(self) -> None:
        from vramfit.adapters.outbound.scan.discovery import discover_groups

        class Gpt2Like(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.h = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(2)]
                )

        groups = discover_groups(Gpt2Like(), "layer")

        assert set(groups) == {"transformer.h.0", "transformer.h.1"}

    def test_moe_names_group_by_stack_fuses_one_projection(self) -> None:
        # llama.cpp gives one quantization type per fused expert stack
        # (#159), so the map keys on that unit (#161). Discovery must
        # collapse the expert index and nothing else.
        from vramfit.adapters.outbound.scan.discovery import discover_groups

        groups = discover_groups(_MoeLike(experts=4), "stack")

        assert set(groups) == {
            "model.layers.0.mlp.experts.up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.shared_expert.up_proj",
            "model.layers.0.self_attn.q_proj",
        }
        assert len(groups["model.layers.0.mlp.experts.up_proj"]) == 4

    def test_stack_grouping_is_coarser_than_tensor_grouping_on_moe(self) -> None:
        from vramfit.adapters.outbound.scan.discovery import discover_groups

        model = _MoeLike(experts=4)

        by_stack = discover_groups(model, "stack")
        by_tensor = discover_groups(model, "tensor")

        assert len(by_stack) == 4
        assert len(by_tensor) == 10

    def test_nemotron_expert_names_group_by_stack(self) -> None:
        # The real target spells experts "backbone.layers.N.mixer.
        # experts.M.down_proj" (#160), not the ".mlp.experts." of the
        # Qwen family. The index sits in the same place, so one rule
        # covers both. This pins that.
        from vramfit.adapters.outbound.scan.discovery import discover_groups

        class NemotronLike(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                def expert() -> torch.nn.Module:
                    part = torch.nn.Module()
                    part.down_proj = torch.nn.Linear(4, 4)
                    return part

                layer = torch.nn.Module()
                layer.mixer = torch.nn.Module()
                layer.mixer.experts = torch.nn.ModuleList([expert() for _ in range(3)])
                self.backbone = torch.nn.Module()
                self.backbone.layers = torch.nn.ModuleList([layer])

        groups = discover_groups(NemotronLike(), "stack")

        assert set(groups) == {"backbone.layers.0.mixer.experts.down_proj"}
        assert len(groups["backbone.layers.0.mixer.experts.down_proj"]) == 3

    def test_unquantizable_classes_stay_out_of_discovery(self) -> None:
        # The 30B target's `mixer.conv1d.weight` is 3-D, so the rank
        # gate admits it, and `mixer.gate.weight` is a plain 2-D
        # linear. llama-quantize refuses both, so a cell priced for
        # either is a cell no recipe can act on (#204).
        from vramfit.adapters.outbound.scan.discovery import discover_groups

        groups = discover_groups(_MambaLike(), "tensor")

        assert set(groups) == {"model.layers.0.mixer.in_proj"}

    def test_layer_discovery_leaves_unquantizable_classes_out_of_the_layer(
        self,
    ) -> None:
        # Under the default granularity the layer group's members and
        # `bytes_fp16` exclude the skipped classes. The size source
        # keys those by tensor name, so their bytes stay priced (#409).
        from vramfit.adapters.outbound.scan.discovery import discover_groups

        groups = discover_groups(_MambaLike(), "layer")

        assert groups == {"model.layers.0": ["model.layers.0.mixer.in_proj.weight"]}

    def test_stack_grouping_matches_tensor_grouping_on_a_dense_model(self) -> None:
        # A dense model has no expert index to collapse. A pack
        # addresses each of its weights alone, so the two agree.
        from vramfit.adapters.outbound.scan.discovery import discover_groups

        class DenseLike(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(2)]
                )

        model = DenseLike()

        assert discover_groups(model, "stack") == discover_groups(model, "tensor")

    def test_max_memory_maps_cap_and_cpu_for_auto_only(self) -> None:
        from vramfit.adapters.outbound.scan.discovery import max_memory_map

        assert max_memory_map("auto", 17 * 2**30) == {
            0: 17 * 2**30,
            "cpu": 999 * 2**30,
        }
        assert max_memory_map("cpu", 17 * 2**30) is None
        assert max_memory_map("auto", None) is None

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

    def test_kquant_meter_measures_different_damage_than_rtn(
        self, aligned_model_dir, tmp_path
    ) -> None:
        # The flag -> meter -> quantizer chain, end to end. If the
        # dispatch ever falls back to RTN silently, a kquant map
        # records rtn damages under the kquant-ref token — corrupted
        # provenance the golden fixtures cannot catch. 256 must divide
        # the row length, or kquant refuses the cell (#330) and
        # measures nothing.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        def build(method: Literal["rtn", "kquant"]) -> TorchDamageMeter:
            return TorchDamageMeter(
                str(aligned_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group=method,
            )

        rtn_meter = build("rtn")
        kquant_meter = build("kquant")
        group = next(spec.name for spec in rtn_meter.groups() if "layers" in spec.name)

        rtn_damage = rtn_meter.measure(group, 2)
        kquant_damage = kquant_meter.measure(group, 2)

        assert kquant_damage != rtn_damage
        assert kquant_damage >= 0.0

    def test_unknown_within_group_is_refused_before_the_model_loads(
        self, tiny_model_dir, tmp_path
    ) -> None:
        # The method token is not the selector — accepting it would
        # silently measure RTN damages under the kquant-ref label.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        bad_method = cast('Literal["rtn", "kquant"]', "kquant-ref")
        with pytest.raises(ValueError, match="within_group"):
            TorchDamageMeter(
                str(tiny_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group=bad_method,
            )

    def test_assisted_meter_measures_different_damage_than_unassisted(
        self, aligned_model_dir, tmp_path
    ) -> None:
        # The imatrix_weights -> name -> assisted quantizer chain,
        # end to end. A meter that drops the lookup would price
        # unassisted under the assisted label — corrupted provenance
        # the golden fixtures cannot catch.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        def build(weights) -> TorchDamageMeter:
            return TorchDamageMeter(
                str(aligned_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_weights=weights,
            )

        plain = build(None)
        group = next(spec.name for spec in plain.groups() if "layers" in spec.name)
        name = plain._groups[group][0]
        rows = int(plain._param(name).shape[-1])
        spiked = torch.ones(rows)
        spiked[::3] = 100.0
        assisted = build({name: spiked})

        unassisted_damage = plain.measure(group, 2)
        assisted_damage = assisted.measure(group, 2)

        assert assisted_damage != unassisted_damage
        assert assisted_damage >= 0.0

    def test_q0_assisted_meter_differs_at_4_bits_and_matches_at_2(
        self, aligned_model_dir, tmp_path
    ) -> None:
        # q0-imx (ADR-0018, 2026-08-21 amendment): nominal 4 fits
        # with the weights, and nominal 2 keeps the reference
        # arithmetic because quantize_q2_0 discards the matrix.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        def build(weights) -> TorchDamageMeter:
            return TorchDamageMeter(
                str(aligned_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="q0",
                imatrix_weights=weights,
            )

        plain = build(None)
        group = next(spec.name for spec in plain.groups() if "layers" in spec.name)
        name = plain._groups[group][0]
        rows = int(plain._param(name).shape[-1])
        spiked = torch.ones(rows)
        spiked[::3] = 100.0
        assisted = build({name: spiked})

        assert assisted.measure(group, 4) != plain.measure(group, 4)
        assert assisted.measure(group, 2) == plain.measure(group, 2)

    def test_imatrix_weights_with_rtn_are_refused_before_the_model_loads(
        self, tmp_path
    ) -> None:
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        with pytest.raises(ValueError, match="kquant or q0 within-group method"):
            TorchDamageMeter(
                "/nonexistent-model",
                calibration,
                max_tokens=128,
                device="cpu",
                imatrix_weights={"any.weight": torch.ones(8)},
            )

    def test_empty_imatrix_weights_are_refused_before_the_model_loads(
        self, tmp_path
    ) -> None:
        # An empty mapping prices every cell unassisted under the
        # assisted label — the caller must pass None deliberately.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        with pytest.raises(ValueError, match="empty"):
            TorchDamageMeter(
                "/nonexistent-model",
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_weights={},
            )

    def test_non_finite_imatrix_weights_are_refused_at_construction(
        self, aligned_model_dir, tmp_path
    ) -> None:
        # A NaN weight would abort the scan at its first assisted
        # cell, hours in — construction must refuse it up front.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        weights = torch.ones(256)
        weights[3] = float("nan")

        with pytest.raises(ValueError, match="finite"):
            TorchDamageMeter(
                str(aligned_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_weights={"model.layers.0.self_attn.q_proj.weight": weights},
            )

    def test_non_1d_imatrix_weights_are_refused_at_construction(
        self, aligned_model_dir, tmp_path
    ) -> None:
        # A (1, rows) tensor has the right numel — only a dim check
        # stops it passing construction and dying at the first
        # assisted cell.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        with pytest.raises(ValueError, match="1-D"):
            TorchDamageMeter(
                str(aligned_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_weights={
                    "model.layers.0.self_attn.q_proj.weight": torch.ones(1, 256)
                },
            )

    def test_misaligned_covered_parameter_is_refused_at_construction(
        self, tiny_model_dir, tmp_path
    ) -> None:
        # tiny_model_dir rows are 32-wide — a covered parameter that
        # cannot split into 256-element super-blocks can never price
        # assisted, and the first assisted cell would abort mid-scan.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        with pytest.raises(ValueError, match="super-block"):
            TorchDamageMeter(
                str(tiny_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_weights={
                    "model.layers.0.self_attn.q_proj.weight": torch.ones(32)
                },
            )

    def test_imatrix_weights_for_an_unknown_parameter_are_refused(
        self, tiny_model_dir, tmp_path
    ) -> None:
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        with pytest.raises(ValueError, match="unknown parameter"):
            TorchDamageMeter(
                str(tiny_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_weights={"model.layers.0.typo.weight": torch.ones(8)},
            )

    def test_imatrix_weights_with_wrong_length_are_refused(
        self, tiny_model_dir, tmp_path
    ) -> None:
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        with pytest.raises(ValueError, match="rows have"):
            TorchDamageMeter(
                str(tiny_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_weights={
                    "model.layers.0.self_attn.q_proj.weight": torch.ones(8)
                },
            )

    def test_imatrix_path_resolves_weights_and_reports_coverage(
        self, aligned_model_dir, tmp_path
    ) -> None:
        # The full file-to-meter chain (ADR-0020): a real imatrix GGUF
        # covering one parameter must land as that parameter's column
        # weights, with the split reported for the run log.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        gguf = pytest.importorskip("gguf", reason="scan extra not installed")
        np = pytest.importorskip("numpy", reason="scan extra not installed")
        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        imatrix = tmp_path / "im.gguf"
        writer = gguf.GGUFWriter(str(imatrix), "imatrix")
        writer.add_type("imatrix")
        writer.add_tensor(
            "blk.0.attn_q.weight.in_sum2", np.full(256, 8.0, dtype=np.float32)
        )
        writer.add_tensor(
            "blk.0.attn_q.weight.counts", np.array([4.0], dtype=np.float32)
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        meter = TorchDamageMeter(
            str(aligned_model_dir),
            calibration,
            max_tokens=128,
            device="cpu",
            within_group="kquant",
            imatrix_path=imatrix,
        )

        assert meter.imatrix_covered_count == 1
        covered = meter._imatrix_weights["model.layers.0.self_attn.q_proj.weight"]
        assert torch.equal(covered, torch.full((256,), 2.0))
        assert meter.imatrix_uncovered is not None
        assert "model.layers.0.self_attn.q_proj.weight" not in meter.imatrix_uncovered
        assert "model.embed_tokens.weight" in meter.imatrix_uncovered

    def test_q0_imatrix_path_resolves_rows_a_super_block_refuses(
        self, tiny_model_dir, tmp_path
    ) -> None:
        # The reader-family dispatch, file to meter. tiny_model_dir
        # rows are 32-wide, which the kquant reader reports
        # misaligned — it would refuse at zero coverage. A covered
        # resolution therefore proves the meter routed the file
        # through the q0 reader (ADR-0018, 2026-08-21 amendment,
        # decision 2).
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        gguf = pytest.importorskip("gguf", reason="scan extra not installed")
        np = pytest.importorskip("numpy", reason="scan extra not installed")
        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        imatrix = tmp_path / "im.gguf"
        writer = gguf.GGUFWriter(str(imatrix), "imatrix")
        writer.add_type("imatrix")
        writer.add_tensor(
            "blk.0.attn_q.weight.in_sum2", np.full(32, 8.0, dtype=np.float32)
        )
        writer.add_tensor(
            "blk.0.attn_q.weight.counts", np.array([4.0], dtype=np.float32)
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        meter = TorchDamageMeter(
            str(tiny_model_dir),
            calibration,
            max_tokens=128,
            device="cpu",
            within_group="q0",
            imatrix_path=imatrix,
        )

        assert meter.imatrix_covered_count == 1
        covered = meter._imatrix_weights["model.layers.0.self_attn.q_proj.weight"]
        assert torch.equal(covered, torch.full((32,), 2.0))

    def test_wrong_model_imatrix_is_refused_at_construction(
        self, aligned_model_dir, tmp_path
    ) -> None:
        # A well-formed imatrix from another model maps to none of
        # the discovered parameters — the constructor's resolve call
        # must refuse, not price every cell unassisted under the
        # assisted label.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        gguf = pytest.importorskip("gguf", reason="scan extra not installed")
        np = pytest.importorskip("numpy", reason="scan extra not installed")
        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        imatrix = tmp_path / "other-model.gguf"
        writer = gguf.GGUFWriter(str(imatrix), "imatrix")
        writer.add_type("imatrix")
        writer.add_tensor(
            "blk.99.attn_q.weight.in_sum2", np.full(256, 8.0, dtype=np.float32)
        )
        writer.add_tensor(
            "blk.99.attn_q.weight.counts", np.array([4.0], dtype=np.float32)
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        with pytest.raises(ValueError, match="covers none"):
            TorchDamageMeter(
                str(aligned_model_dir),
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_path=imatrix,
            )

    def test_imatrix_path_with_weights_is_refused_before_the_model_loads(
        self, tmp_path
    ) -> None:
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        with pytest.raises(ValueError, match="not both"):
            TorchDamageMeter(
                "/nonexistent-model",
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_weights={"any.weight": torch.ones(256)},
                imatrix_path=tmp_path / "im.gguf",
            )

    def test_imatrix_path_with_rtn_is_refused_before_the_model_loads(
        self, tmp_path
    ) -> None:
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)

        with pytest.raises(ValueError, match="kquant or q0 within-group method"):
            TorchDamageMeter(
                "/nonexistent-model",
                calibration,
                max_tokens=128,
                device="cpu",
                imatrix_path=tmp_path / "im.gguf",
            )

    def test_malformed_imatrix_file_is_refused_before_the_model_loads(
        self, tmp_path
    ) -> None:
        # The imatrix loads first — a bad file must refuse in
        # milliseconds, not after minutes of shard loading. The
        # nonexistent model proves the ordering.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        gguf = pytest.importorskip("gguf", reason="scan extra not installed")
        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        imatrix = tmp_path / "not-an-imatrix.gguf"
        writer = gguf.GGUFWriter(str(imatrix), "llama")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        with pytest.raises(ValueError, match="not an imatrix"):
            TorchDamageMeter(
                "/nonexistent-model",
                calibration,
                max_tokens=128,
                device="cpu",
                within_group="kquant",
                imatrix_path=imatrix,
            )

    def test_kquant_meter_refuses_uncovered_bits(
        self, aligned_model_dir, tmp_path
    ) -> None:
        # The follow-up measure needs rows that 256 divides (#330), so
        # the aligned checkpoint carries this test.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        meter = TorchDamageMeter(
            str(aligned_model_dir),
            calibration,
            max_tokens=128,
            device="cpu",
            within_group="kquant",
        )
        group = meter.groups()[0].name

        with pytest.raises(ValueError, match="supports bits in"):
            meter.measure(group, 6)

        # The refusal happened mid-perturbation path — the meter must
        # still be usable, not silently poisoned.
        assert meter.measure(group, 2) >= 0.0

    def test_kquant_meter_on_straddling_rows_refuses_and_leaves_the_meter_usable(
        self, tiny_model_dir, tmp_path
    ) -> None:
        # tiny_model_dir rows are 32 or 64 wide and Q2_K blocks 256.
        # The refusal (ADR-0018, 2026-08-17 amendment, decision 4)
        # must name the row length. Q8_0 blocks 32, so the 8-bit cell
        # on the same rows still measures, and that proves the meter
        # usable and the refusal keyed to the mapped type's block.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib.txt"
        calibration.write_text(CALIBRATION_TEXT)
        meter = TorchDamageMeter(
            str(tiny_model_dir),
            calibration,
            max_tokens=128,
            device="cpu",
            within_group="kquant",
        )
        group = meter.groups()[0].name
        rows = int(meter._param(meter._groups[group][0]).shape[-1])

        with pytest.raises(ValueError, match=f"does not divide the row length {rows}"):
            meter.measure(group, 2)

        assert meter.measure(group, 8) >= 0.0

    @pytest.mark.gpu
    def test_measure_recipe_on_cuda_restores_across_devices(
        self, tiny_model_dir, tmp_path
    ) -> None:
        # The whole-recipe pass stages originals on the CPU and
        # restores host-to-device — the path every GPU validation
        # takes, untestable on the CPU-only contract leg.
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

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
        from vramfit.adapters.outbound.scan.discovery import discover_groups

        class Flat(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

        with pytest.raises(ValueError, match="--group-by tensor"):
            discover_groups(Flat(), "layer")


@pytest.fixture
def tiny_moe_meter(tiny_moe_model_dir, tmp_path):
    from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

    calibration = tmp_path / "calib.txt"
    calibration.write_text(CALIBRATION_TEXT)
    return TorchDamageMeter(
        str(tiny_moe_model_dir), calibration, max_tokens=128, device="cpu"
    )


class TestMeasureSlices:
    """The slice perturbation path (ADR-0026, the #200 amendment)."""

    UP = "model.layers.0.mlp.experts.gate_up_proj"
    DOWN = "model.layers.0.mlp.experts.down_proj"

    def test_fixture_loads_fused_expert_stacks(self, tiny_moe_meter) -> None:
        # The path exists for the fused layout, so the fixture must
        # produce it — a dense or indexed layout would test nothing.
        param = tiny_moe_meter._param(self.UP)

        assert param.ndim == 3
        assert param.shape[0] == 8

    def test_single_expert_cell_measures_and_restores_bit_exactly(
        self, tiny_moe_meter
    ) -> None:
        before = {
            name: tiny_moe_meter._param(name).clone() for name in (self.UP, self.DOWN)
        }

        damage = tiny_moe_meter.measure_slices(
            {self.UP: (0, 1), self.DOWN: (0, 1)}, bits=2
        )

        assert damage >= 0.0
        for name, original in before.items():
            assert torch.equal(tiny_moe_meter._param(name), original)

    def test_full_range_slice_equals_the_whole_tensor_measure(
        self, tiny_moe_meter, tiny_moe_model_dir, tmp_path
    ) -> None:
        # A full range names the same tensor the whole-tensor cell
        # perturbs, so the two entry points must agree exactly — a
        # difference means the slice path measures something else.
        from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

        calibration = tmp_path / "calib-tensor.txt"
        calibration.write_text(CALIBRATION_TEXT)
        by_tensor = TorchDamageMeter(
            str(tiny_moe_model_dir),
            calibration,
            max_tokens=128,
            group_by="tensor",
            device="cpu",
        )
        group = next(
            spec.name for spec in by_tensor.groups() if spec.tensors == (self.UP,)
        )

        whole = by_tensor.measure(group, 2)
        sliced = by_tensor.measure_slices({self.UP: (0, 8)}, bits=2)

        assert sliced == whole

    def test_slice_perturbs_only_the_named_range_and_restores(
        self, tiny_moe_meter, monkeypatch
    ) -> None:
        # Snapshot the weights mid-measurement: the named range must
        # hold the round-tripped values, every other expert must hold
        # the originals, and the restore must put the originals back.
        from vramfit.adapters.outbound.scan import meter as meter_module
        from vramfit.adapters.outbound.scan.quantize import rtn_quantize_dequantize

        before = tiny_moe_meter._param(self.UP).clone()
        captured = {}

        def snapshot(model, batches, reference) -> float:
            captured["perturbed"] = tiny_moe_meter._param(self.UP).clone()
            return 0.0

        monkeypatch.setattr(meter_module, "mean_damage", snapshot)
        tiny_moe_meter.measure_slices({self.UP: (2, 4)}, bits=2)

        perturbed = captured["perturbed"]
        assert torch.equal(perturbed[2:4], rtn_quantize_dequantize(before[2:4], 2))
        assert torch.equal(perturbed[0:2], before[0:2])
        assert torch.equal(perturbed[4:], before[4:])
        assert torch.equal(tiny_moe_meter._param(self.UP), before)

    def test_smaller_slice_damages_no_more_than_the_full_stack(
        self, tiny_moe_meter
    ) -> None:
        # Not an axiom — a sanity bound for this tiny fixture: one
        # expert of eight should not out-damage all eight.
        one = tiny_moe_meter.measure_slices({self.UP: (0, 1)}, bits=2)
        all_eight = tiny_moe_meter.measure_slices({self.UP: (0, 8)}, bits=2)

        assert one <= all_eight

    def test_expert_band_kquant_round_trip_matches_the_whole_tensor(
        self, tiny_moe_meter
    ) -> None:
        # The fixture sizes each expert's slice at a multiple of 256
        # elements, so a band's K-quant blocks are the whole tensor's
        # blocks and every fit is block-local — a band slice cell
        # quantizes the same values the whole-stack cell quantizes.
        from vramfit.adapters.outbound.scan.kquant import kquant_quantize_dequantize

        param = tiny_moe_meter._param(self.UP).detach()

        whole = kquant_quantize_dequantize(param, 2)
        band = kquant_quantize_dequantize(param[2:6], 2)

        assert torch.equal(whole[2:6], band)

    def test_measurement_failure_restores_and_does_not_poison(
        self, tiny_moe_meter, monkeypatch
    ) -> None:
        from vramfit.adapters.outbound.scan import meter as meter_module

        before = tiny_moe_meter._param(self.UP).clone()

        def unstable(model, batches, reference) -> float:
            raise ValueError("numerically unstable")

        monkeypatch.setattr(meter_module, "mean_damage", unstable)
        with pytest.raises(ValueError, match="numerically unstable"):
            tiny_moe_meter.measure_slices({self.UP: (2, 4)}, bits=2)

        assert not tiny_moe_meter._poisoned
        assert torch.equal(tiny_moe_meter._param(self.UP), before)
        monkeypatch.undo()
        assert tiny_moe_meter.measure_slices({self.UP: (2, 4)}, bits=8) >= 0.0

    def test_restore_failure_through_the_slice_path_poisons(
        self, tiny_moe_meter, monkeypatch
    ) -> None:
        # Fail the sliced restore itself: swap the parameter lookup
        # to a mismatched tensor mid-measurement, so the range copy
        # raises inside the finally clause. The in-flight
        # measurement error must keep the stage and the meter must
        # poison with a recorded reason.
        from vramfit.adapters.outbound.scan import meter as meter_module

        def unstable(model, batches, reference) -> float:
            tiny_moe_meter._offloaded[self.UP] = torch.zeros(1, 2, 2)
            raise ValueError("numerically unstable")

        monkeypatch.setattr(meter_module, "mean_damage", unstable)
        with pytest.raises(ValueError, match="numerically unstable"):
            tiny_moe_meter.measure_slices({self.UP: (2, 4)}, bits=2)

        assert tiny_moe_meter._poisoned
        assert tiny_moe_meter._poisoned_reason
        with pytest.raises(RuntimeError, match="rebuild the meter"):
            tiny_moe_meter.measure_slices({self.UP: (2, 4)}, bits=2)

    def test_unknown_parameter_refuses_through_the_meter(self, tiny_moe_meter) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            tiny_moe_meter.measure_slices({"model.layers.9.nope": (0, 1)}, bits=2)

    def test_dense_parameter_refuses_through_the_meter(self, tiny_moe_meter) -> None:
        with pytest.raises(ValueError, match="fused expert stack"):
            tiny_moe_meter.measure_slices(
                {"model.layers.0.self_attn.q_proj.weight": (0, 1)}, bits=2
            )

    def test_bits_below_two_refuse(self, tiny_moe_meter) -> None:
        with pytest.raises(ValueError, match="bits"):
            tiny_moe_meter.measure_slices({self.UP: (0, 1)}, bits=1)

    def test_poisoned_meter_refuses_measure_slices(self, tiny_moe_meter) -> None:
        tiny_moe_meter._poisoned = True

        with pytest.raises(RuntimeError, match="rebuild the meter"):
            tiny_moe_meter.measure_slices({self.UP: (0, 1)}, bits=2)


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
