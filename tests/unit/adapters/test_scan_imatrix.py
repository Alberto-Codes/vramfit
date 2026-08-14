"""Checks of the imatrix adapter: name mapping and GGUF loading.

The loader tests write miniature imatrix GGUF files with
``gguf.GGUFWriter`` — the same library the adapter reads with — so
they stay hermetic. They skip cleanly where the scan extra is
absent (ADR-0009).
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
gguf = pytest.importorskip("gguf", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.imatrix import (
    assisted_weights_for_params,
    expert_stack_count_vectors,
    gguf_tensor_name,
    load_imatrix,
    resolve_assisted_weights,
    resolve_imatrix_counts,
)

pytestmark = pytest.mark.unit

# Nemotron 3.5 Lightning's routed-expert dimensions (#160): 128
# experts per expert stack, hidden 2688, expert intermediate 1856.
EXPERTS = 128
HIDDEN = 2688


def _stack_params(layer: int, projection: str, experts: int = EXPERTS) -> list[str]:
    """Name one expert stack's routed-expert parameters, in order."""
    return [
        f"backbone.layers.{layer}.mixer.experts.{i}.{projection}.weight"
        for i in range(experts)
    ]


def _write_imatrix(
    path: Path,
    tensors: dict[str, np.ndarray],
    general_type: str | None = "imatrix",
) -> Path:
    writer = gguf.GGUFWriter(str(path), "imatrix")
    if general_type is not None:
        writer.add_type(general_type)
    for name, data in tensors.items():
        writer.add_tensor(name, data.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class TestGgufTensorName:
    @pytest.mark.parametrize(
        ("param", "expected"),
        [
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.41.self_attn.k_proj.weight", "blk.41.attn_k.weight"),
            ("model.layers.7.self_attn.v_proj.weight", "blk.7.attn_v.weight"),
            ("model.layers.7.self_attn.o_proj.weight", "blk.7.attn_output.weight"),
            ("model.layers.79.mlp.gate_proj.weight", "blk.79.ffn_gate.weight"),
            ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            ("model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight"),
            ("lm_head.weight", "output.weight"),
            ("model.embed_tokens.weight", "token_embd.weight"),
        ],
    )
    def test_llama_family_names_map(self, param: str, expected: str) -> None:
        assert gguf_tensor_name(param) == expected

    @pytest.mark.parametrize(
        ("param", "expected"),
        [
            (
                "backbone.layers.3.mixer.experts.57.down_proj.weight",
                "blk.3.ffn_down_exps.weight",
            ),
            (
                "backbone.layers.20.mixer.experts.0.up_proj.weight",
                "blk.20.ffn_up_exps.weight",
            ),
            (
                "model.layers.9.mlp.experts.127.gate_proj.weight",
                "blk.9.ffn_gate_exps.weight",
            ),
        ],
    )
    def test_routed_experts_map_to_their_expert_stack(
        self, param: str, expected: str
    ) -> None:
        # llama.cpp fuses one projection's experts into one tensor
        # (#159), so every expert shares one GGUF name.
        assert gguf_tensor_name(param) == expected

    def test_every_expert_shares_one_gguf_name(self) -> None:
        names = {gguf_tensor_name(p) for p in _stack_params(3, "up_proj")}

        assert names == {"blk.3.ffn_up_exps.weight"}

    @pytest.mark.parametrize(
        ("param", "expected"),
        [
            (
                "model.layers.3.mixer.experts.up_proj",
                "blk.3.ffn_up_exps.weight",
            ),
            (
                "backbone.layers.20.mixer.experts.down_proj",
                "blk.20.ffn_down_exps.weight",
            ),
        ],
    )
    def test_a_fused_expert_stack_maps_to_the_same_gguf_name(
        self, param: str, expected: str
    ) -> None:
        # transformers 5 fuses the routed experts at load (#202).
        # The loaded module reports one 3D parameter per projection,
        # with no ".weight" suffix and no expert index. #202 measured
        # the model root. The backbone root joins because #186 pairs
        # both roots for this family's dense names.
        assert gguf_tensor_name(param) == expected

    @pytest.mark.parametrize(
        "param",
        [
            # Nemotron 3.5 Lightning's experts are ungated (#160). A
            # fused gate stays out until a loaded model reports one.
            "model.layers.3.mixer.experts.gate_proj",
            # No loaded model reports a fused llama-family spelling.
            "model.layers.3.mlp.experts.up_proj",
            # A fused parameter carries no ".weight" suffix (#202).
            "model.layers.3.mixer.experts.up_proj.weight",
        ],
    )
    def test_an_unproved_fused_spelling_returns_none(self, param: str) -> None:
        assert gguf_tensor_name(param) is None

    @pytest.mark.parametrize(
        "param",
        [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.unknown_proj.weight",
            "transformer.h.0.attn.c_attn.weight",
            # Mixtral spells its expert projections w1/w2/w3. The
            # table omits them rather than guess the mapping.
            "model.layers.0.block_sparse_moe.experts.4.w1.weight",
            "backbone.layers.3.mixer.unknown_proj.weight",
        ],
    )
    def test_unmapped_names_return_none(self, param: str) -> None:
        assert gguf_tensor_name(param) is None

    @pytest.mark.parametrize(
        ("param", "expected"),
        [
            ("backbone.layers.0.mixer.in_proj.weight", "blk.0.ssm_in.weight"),
            ("backbone.layers.51.mixer.out_proj.weight", "blk.51.ssm_out.weight"),
            ("backbone.layers.5.mixer.q_proj.weight", "blk.5.attn_q.weight"),
            ("backbone.layers.5.mixer.k_proj.weight", "blk.5.attn_k.weight"),
            ("backbone.layers.5.mixer.v_proj.weight", "blk.5.attn_v.weight"),
            ("backbone.layers.5.mixer.o_proj.weight", "blk.5.attn_output.weight"),
            ("backbone.layers.3.mixer.gate.weight", "blk.3.ffn_gate_inp.weight"),
            (
                "backbone.layers.3.mixer.shared_experts.up_proj.weight",
                "blk.3.ffn_up_shexp.weight",
            ),
            (
                "backbone.layers.3.mixer.shared_experts.down_proj.weight",
                "blk.3.ffn_down_shexp.weight",
            ),
        ],
    )
    def test_nemotron_h_dense_names_map(self, param: str, expected: str) -> None:
        # Nemotron-H spells attention, Mamba-2, the router, and the
        # shared expert under one "mixer." module (#186). No
        # llama-family name reaches any of them.
        assert gguf_tensor_name(param) == expected

    def test_the_router_bias_is_not_a_weight(self) -> None:
        # "mixer.gate" maps, and the router carries a second tensor
        # beside its weight. Only the weight is a matrix.
        bias = "backbone.layers.3.mixer.gate.e_score_correction_bias"

        assert gguf_tensor_name(bias) is None

    @pytest.mark.parametrize(
        "param",
        [
            "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "model.layers.0.cross_attn.layers.2.self_attn.q_proj.weight",
        ],
    )
    def test_a_trunk_outside_the_decoder_stays_uncovered(self, param: str) -> None:
        # A vision tower carries its own layers.N. Mapping one onto
        # blk.N would price it against the decoder's columns, which
        # is the failure this module exists to refuse. An unsupported
        # root reports uncovered instead.
        assert gguf_tensor_name(param) is None


class TestLoadImatrix:
    def test_weights_are_sums_over_counts(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.arange(8.0),
                "blk.0.attn_q.weight.counts": np.array([4.0]),
            },
        )

        entries = load_imatrix(path)

        expected = (torch.arange(8.0) / 4.0).reshape(1, 8)
        assert torch.equal(entries["blk.0.attn_q.weight"].column_weights, expected)
        assert torch.equal(entries["blk.0.attn_q.weight"].counts, torch.tensor([4.0]))

    def test_zero_count_columns_weigh_one(self, tmp_path) -> None:
        # The llama-quantize load formula: a column with no chunks
        # falls back to weight 1, not 0 — zero would erase it from
        # the fit.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([3.0, 5.0]),
                "t.weight.counts": np.array([0.0]),
            },
        )

        entries = load_imatrix(path)

        assert torch.equal(entries["t.weight"].column_weights, torch.ones(1, 2))

    def test_per_expert_counts_normalize_row_wise(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([[2.0, 4.0], [30.0, 90.0]]),
                "t.weight.counts": np.array([[2.0, 3.0]]),
            },
        )

        entries = load_imatrix(path)

        assert torch.equal(
            entries["t.weight"].column_weights, torch.tensor([[1.0, 2.0], [10.0, 30.0]])
        )

    def test_an_expert_stack_keeps_one_row_per_expert(self, tmp_path) -> None:
        # llama.cpp declares in_sum2 with ne of [columns, matrices]
        # and counts as one float per matrix (imatrix.cpp:595-607).
        # ne runs fastest axis first, so the buffer reads as
        # (matrices, columns) in NumPy order. That is the order
        # llama-quantize slices per expert (llama-quant.cpp:1256-1262).
        sums = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.ffn_up_exps.weight.in_sum2": sums,
                "blk.0.ffn_up_exps.weight.counts": np.array([[1.0, 2.0, 4.0, 8.0]]),
            },
        )

        entry = load_imatrix(path)["blk.0.ffn_up_exps.weight"]

        assert entry.column_weights.shape == (4, 3)
        assert torch.equal(entry.counts, torch.tensor([1.0, 2.0, 4.0, 8.0]))
        assert torch.equal(
            entry.column_weights,
            torch.from_numpy(sums) / torch.tensor([[1.0], [2.0], [4.0], [8.0]]),
        )

    def test_a_zero_count_expert_weighs_one_and_keeps_its_count(self, tmp_path) -> None:
        # ADR-0026 decision 1: a zero count uses the unassisted fit.
        # Decision 5 reports the expert, so the count survives the
        # load as 0 and does not fold into the weight.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.ffn_up_exps.weight.in_sum2": np.array([[2.0, 4.0], [9.0, 9.0]]),
                "blk.0.ffn_up_exps.weight.counts": np.array([[2.0, 0.0]]),
            },
        )

        entry = load_imatrix(path)["blk.0.ffn_up_exps.weight"]

        assert torch.equal(entry.column_weights, torch.tensor([[1.0, 2.0], [1.0, 1.0]]))
        assert torch.equal(entry.counts, torch.tensor([2.0, 0.0]))

    def test_sum_without_counts_twin_raises(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {"t.weight.in_sum2": np.array([1.0])},
        )

        with pytest.raises(ValueError, match="counts twin"):
            load_imatrix(path)

    def test_counts_without_sums_twin_raises(self, tmp_path) -> None:
        # An orphan counts tensor means a sums tensor was misnamed or
        # lost — its columns would vanish from coverage silently.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([1.0]),
                "t.weight.counts": np.array([1.0]),
                "u.weight.counts": np.array([1.0]),
            },
        )

        with pytest.raises(ValueError, match="in_sum2 twin"):
            load_imatrix(path)

    def test_unrecognized_tensor_name_raises(self, tmp_path) -> None:
        # Suffix drift in a future imatrix format must fail loudly,
        # not shrink coverage silently.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([1.0]),
                "t.weight.counts": np.array([1.0]),
                "t.weight.sums": np.array([1.0]),
            },
        )

        with pytest.raises(ValueError, match="unexpected tensor"):
            load_imatrix(path)

    def test_imatrix_without_data_raises(self, tmp_path) -> None:
        path = _write_imatrix(tmp_path / "im.gguf", {"dummy.counts": np.array([1.0])})

        with pytest.raises(ValueError, match=r"no \.in_sum2 tensors|in_sum2 twin"):
            load_imatrix(path)

    def test_sums_not_divisible_by_counts_raises(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([1.0, 2.0, 3.0]),
                "t.weight.counts": np.array([1.0, 2.0]),
            },
        )

        with pytest.raises(ValueError, match="not divisible"):
            load_imatrix(path)

    @pytest.mark.parametrize(
        ("counts", "match"),
        [
            (np.array([float("inf")]), "not finite"),
            (np.array([float("nan")]), "not finite"),
            (np.array([-5.0]), "negative"),
        ],
        ids=["inf", "nan", "negative"],
    )
    def test_a_count_that_is_not_a_tally_is_refused(
        self, tmp_path, counts, match: str
    ) -> None:
        # An infinite count drives every column weight to zero. A
        # negative or nan count lands in the zero-count branch and
        # weighs every column 1. Both pass check_imatrix_weights, so
        # a 30-hour scan would price against a dead signal.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([3.0, 5.0]),
                "t.weight.counts": counts,
            },
        )

        with pytest.raises(ValueError, match=match):
            load_imatrix(path)

    def test_a_transposed_sums_tensor_is_refused(self, tmp_path) -> None:
        # A writer that emits (columns, matrices) in NumPy order
        # passes the divisibility check, because numel matches either
        # way. The reshape would then stripe every expert's row
        # across all the others, and no later gate would notice.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.ffn_up_exps.weight.in_sum2": np.ones((3, 4), dtype=np.float32),
                "blk.0.ffn_up_exps.weight.counts": np.array([[1.0, 2.0, 4.0, 8.0]]),
            },
        )

        with pytest.raises(ValueError, match="reads as"):
            load_imatrix(path)

    def test_one_dimensional_sums_with_many_counts_is_refused(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "t.weight.in_sum2": np.array([2.0, 4.0, 30.0, 90.0]),
                "t.weight.counts": np.array([[2.0, 3.0]]),
            },
        )

        with pytest.raises(ValueError, match="one dimension"):
            load_imatrix(path)

    def test_non_imatrix_file_is_refused(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "not-im.gguf",
            {"t.weight.in_sum2": np.array([1.0]), "t.weight.counts": np.array([1.0])},
            general_type="model",
        )

        with pytest.raises(ValueError, match="not an imatrix"):
            load_imatrix(path)


class TestAssistedWeightsForParams:
    def test_covered_and_uncovered_split(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.ones(256),
                "blk.0.attn_q.weight.counts": np.array([1.0]),
            },
        )
        shapes = {
            "model.layers.0.self_attn.q_proj.weight": (8, 256),
            "model.layers.1.self_attn.q_proj.weight": (8, 256),
            "model.embed_tokens.weight": (16, 256),
        }

        covered, uncovered = assisted_weights_for_params(path, shapes)

        assert set(covered) == {"model.layers.0.self_attn.q_proj.weight"}
        assert uncovered == (
            "model.layers.1.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
        )

    def test_zero_coverage_raises(self, tmp_path) -> None:
        # An imatrix that covers nothing is the wrong file — a scan
        # run on it would price every cell unassisted under the
        # assisted label.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.ones(8),
                "blk.0.attn_q.weight.counts": np.array([1.0]),
            },
        )

        with pytest.raises(ValueError, match="covers none"):
            assisted_weights_for_params(
                path, {"model.layers.9.mlp.up_proj.weight": (8, 8)}
            )

    def test_row_length_mismatch_raises(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.ones(256),
                "blk.0.attn_q.weight.counts": np.array([1.0]),
            },
        )

        with pytest.raises(ValueError, match="rows have 512"):
            assisted_weights_for_params(
                path, {"model.layers.0.self_attn.q_proj.weight": (8, 512)}
            )

    def test_misaligned_covered_rows_fall_back_to_uncovered(self, tmp_path) -> None:
        # A covered parameter whose rows do not divide into
        # super-blocks cannot price assisted (ADR-0020) — it joins
        # the uncovered set instead of refusing the whole scan.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.ones(96),
                "blk.0.attn_q.weight.counts": np.array([1.0]),
                "blk.0.attn_k.weight.in_sum2": np.ones(256),
                "blk.0.attn_k.weight.counts": np.array([1.0]),
            },
        )
        shapes = {
            "model.layers.0.self_attn.q_proj.weight": (8, 96),
            "model.layers.0.self_attn.k_proj.weight": (8, 256),
        }

        covered, uncovered = assisted_weights_for_params(path, shapes)

        assert set(covered) == {"model.layers.0.self_attn.k_proj.weight"}
        assert uncovered == ("model.layers.0.self_attn.q_proj.weight",)


# Nemotron 3.5 Lightning's dense column widths, read from the
# checkpoint's own config.json. A column width is the parameter's
# input dimension, which is shape[-1] of an HF linear weight.
MAMBA_INNER = 4096  # mamba_num_heads 64 x mamba_head_dim 64
ATTN_INNER = 4096  # num_attention_heads 32 x head_dim 128
SHARED_EXPERT = 3712  # moe_shared_expert_intermediate_size
# Every dense stem the model carries, with its HF module suffix and
# its column width. 23 layers are Mamba-2, 23 are MoE, and 6 are
# attention (#160).
DENSE_STEMS = (
    ("ssm_in", "mixer.in_proj", HIDDEN, 23),
    ("ssm_out", "mixer.out_proj", MAMBA_INNER, 23),
    ("ffn_gate_inp", "mixer.gate", HIDDEN, 23),
    ("ffn_up_shexp", "mixer.shared_experts.up_proj", HIDDEN, 23),
    ("ffn_down_shexp", "mixer.shared_experts.down_proj", SHARED_EXPERT, 23),
    ("attn_q", "mixer.q_proj", HIDDEN, 6),
    ("attn_k", "mixer.k_proj", HIDDEN, 6),
    ("attn_v", "mixer.v_proj", HIDDEN, 6),
    ("attn_output", "mixer.o_proj", ATTN_INNER, 6),
)


def _write_dense_imatrix(path: Path, layers: int) -> tuple[Path, dict[str, tuple]]:
    """Write this model's dense entries, and the shapes they price.

    Layer indices are arbitrary here. Name resolution reads the index
    and never the layer's type, so the counts are what matter.
    """
    tensors: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, int]] = {}
    for stem, suffix, columns, present in DENSE_STEMS:
        for layer in range(min(layers, present)):
            tensors[f"blk.{layer}.{stem}.weight.in_sum2"] = np.ones(columns)
            tensors[f"blk.{layer}.{stem}.weight.counts"] = np.array([2.0])
            shapes[f"backbone.layers.{layer}.{suffix}.weight"] = (8, columns)
    return _write_imatrix(path, tensors), shapes


class TestNemotronHDenseCoverage:
    def test_only_the_four_thousand_ninety_six_wide_stems_price_assisted(
        self, tmp_path
    ) -> None:
        # Every dense stem now maps, so the super-block gate is the
        # only remaining bar (ADR-0020). 2688 and 3712 both leave 128
        # over 256, so only the two 4096-wide stems clear it.
        path, shapes = _write_dense_imatrix(tmp_path / "im.gguf", layers=1)

        covered, uncovered = assisted_weights_for_params(path, shapes)

        assert set(covered) == {
            "backbone.layers.0.mixer.out_proj.weight",
            "backbone.layers.0.mixer.o_proj.weight",
        }
        assert len(uncovered) == 7

    def test_twenty_nine_of_the_hundred_thirty_nine_dense_entries_price_assisted(
        self, tmp_path
    ) -> None:
        # The whole-model count: 23 ssm_out plus 6 attn_output (#186).
        path, shapes = _write_dense_imatrix(tmp_path / "im.gguf", layers=52)

        covered, uncovered = assisted_weights_for_params(path, shapes)

        assert len(shapes) == 139
        assert len(covered) == 29
        assert len(uncovered) == 110

    def test_a_misaligned_stem_reports_the_gate_not_a_missing_name(
        self, tmp_path
    ) -> None:
        # The diagnosis this ticket changes. Every dense name now
        # maps, so a stem that covers nothing blames the super-block
        # gate rather than the name table.
        path, shapes = _write_dense_imatrix(tmp_path / "im.gguf", layers=52)
        in_proj = {n: s for n, s in shapes.items() if n.endswith("in_proj.weight")}

        with pytest.raises(ValueError) as refusal:
            assisted_weights_for_params(path, in_proj)

        assert "0 names have no GGUF mapping" in str(refusal.value)
        assert "23 have rows that do not divide" in str(refusal.value)

    def test_the_router_reports_a_chunk_tally_not_a_routing_frequency(
        self, tmp_path
    ) -> None:
        # The router is one of the 139 dense entries, so it maps. Its
        # count is the calibration chunk tally for the router matmul.
        # It is not a routing frequency: the glossary defines that for
        # a routed expert, and ADR-0026 decisions 4 and 5 read only
        # expert-stack rows. The return shape carries the difference —
        # a dense name reads a scalar, never a vector (#193).
        path, _ = _write_dense_imatrix(tmp_path / "im.gguf", layers=1)
        router = "backbone.layers.0.mixer.gate.weight"
        expert = "backbone.layers.0.mixer.experts.0.up_proj.weight"

        counts, uncovered = resolve_imatrix_counts(
            load_imatrix(path), {router: (128, HIDDEN), expert: (1856, HIDDEN)}
        )

        assert counts == {router: 2}
        assert isinstance(counts[router], int)
        assert uncovered == (expert,)


def _write_stack(path: Path, columns: int, experts: int = EXPERTS) -> Path:
    """Write a one-entry imatrix whose rows differ per expert."""
    sums = np.arange(experts * columns, dtype=np.float32).reshape(experts, columns)
    return _write_imatrix(
        path,
        {
            "blk.3.ffn_up_exps.weight.in_sum2": sums,
            "blk.3.ffn_up_exps.weight.counts": np.arange(
                1.0, experts + 1.0, dtype=np.float32
            ).reshape(1, experts),
        },
    )


class TestExpertStackResolution:
    def test_each_expert_parameter_reads_its_own_row(self, tmp_path) -> None:
        # The expert stack holds 128 matrices and the checkpoint
        # holds 128 parameters. Each must land on its own row (#177).
        path = _write_stack(tmp_path / "im.gguf", columns=256)
        params = _stack_params(3, "up_proj")

        covered, uncovered = resolve_assisted_weights(
            load_imatrix(path), dict.fromkeys(params, 256)
        )

        assert uncovered == ()
        assert len(covered) == EXPERTS
        expected = torch.arange(EXPERTS * 256.0).reshape(EXPERTS, 256) / torch.arange(
            1.0, EXPERTS + 1.0
        ).reshape(EXPERTS, 1)
        for index, name in enumerate(params):
            assert torch.equal(covered[name], expected[index])

    def test_expert_stack_rows_are_distinct_per_expert(self, tmp_path) -> None:
        # A flattened expert stack would hand every expert the same
        # vector, or refuse them all on row length. Both are the
        # defect #177 names.
        path = _write_stack(tmp_path / "im.gguf", columns=256)
        params = _stack_params(3, "up_proj")

        covered, _ = resolve_assisted_weights(
            load_imatrix(path), dict.fromkeys(params, 256)
        )

        first, second = covered[params[0]], covered[params[1]]
        assert not torch.equal(first, second)
        assert first.numel() == 256

    def test_expert_index_past_the_expert_stack_raises(self, tmp_path) -> None:
        # A checkpoint with more experts than the imatrix means the
        # two describe different models.
        path = _write_stack(tmp_path / "im.gguf", columns=256, experts=8)

        with pytest.raises(ValueError, match="is expert 8"):
            resolve_assisted_weights(
                load_imatrix(path),
                {"backbone.layers.3.mixer.experts.8.up_proj.weight": 256},
            )

    def test_dense_parameter_on_a_many_matrix_entry_raises(self, tmp_path) -> None:
        # Reading row 0 anyway would price a dense tensor against one
        # expert's columns, silently.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.ones((4, 256), dtype=np.float32),
                "blk.0.attn_q.weight.counts": np.ones((1, 4), dtype=np.float32),
            },
        )

        with pytest.raises(ValueError, match="carries no expert index"):
            resolve_assisted_weights(
                load_imatrix(path), {"model.layers.0.self_attn.q_proj.weight": 256}
            )

    def test_expert_zero_on_a_one_matrix_entry_raises(self, tmp_path) -> None:
        # Expert 0 is the one index that fits a one-matrix entry. It
        # must refuse with experts 1 and up, not price against the
        # dense row while the other experts raise.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.3.ffn_up_exps.weight.in_sum2": np.ones(256, dtype=np.float32),
                "blk.3.ffn_up_exps.weight.counts": np.array([7.0]),
            },
        )

        with pytest.raises(ValueError, match="holds one matrix"):
            resolve_assisted_weights(
                load_imatrix(path),
                {"backbone.layers.3.mixer.experts.0.up_proj.weight": 256},
            )

    def test_two_parameters_claiming_one_row_are_refused(self, tmp_path) -> None:
        # _SUFFIX_TO_GGUF carries two suffixes per attention stem,
        # and no checkpoint spells both (#186). If one ever does,
        # both parameters would read row 0 and one would price
        # against the wrong columns. The resolver now refuses the
        # second claimant (#193).
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.5.attn_q.weight.in_sum2": np.ones(256, dtype=np.float32),
                "blk.5.attn_q.weight.counts": np.array([1.0]),
            },
        )

        with pytest.raises(ValueError, match="both claim"):
            resolve_assisted_weights(
                load_imatrix(path),
                {
                    "model.layers.5.self_attn.q_proj.weight": 256,
                    "backbone.layers.5.mixer.q_proj.weight": 256,
                },
            )

    def test_zero_coverage_names_each_cause(self, tmp_path) -> None:
        # Naming the file alone sends an operator to regenerate a
        # correct matrix, which costs GPU hours and fails the same
        # way. The counts point at the cause instead.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.0.attn_q.weight.in_sum2": np.ones(96, dtype=np.float32),
                "blk.0.attn_q.weight.counts": np.array([1.0]),
            },
        )
        rows = {
            "model.norm.weight": 256,
            "model.layers.9.mlp.up_proj.weight": 256,
            "model.layers.0.self_attn.q_proj.weight": 96,
            "model.layers.0.mixer.experts.up_proj": 2688,
        }

        with pytest.raises(ValueError) as caught:
            resolve_assisted_weights(load_imatrix(path), rows)

        message = str(caught.value)
        assert "1 names have no GGUF mapping" in message
        assert "1 are fused expert stacks" in message
        assert "1 mapped to a tensor the file does not hold" in message
        assert "1 have rows that do not divide" in message


class TestFusedExpertStackResolution:
    def test_a_fused_expert_stack_reports_uncovered_by_rule(self, tmp_path) -> None:
        # ADR-0026 (2026-08-13): the stacks stay unassisted until a
        # non-k-quant assisted fit exists. Rows of 256 would pass the
        # super-block gate and reach _matrix_row, which would refuse
        # with a wrong diagnosis. The rule refuses first.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.3.ffn_up_exps.weight.in_sum2": np.ones(
                    (EXPERTS, 256), dtype=np.float32
                ),
                "blk.3.ffn_up_exps.weight.counts": np.ones(
                    (1, EXPERTS), dtype=np.float32
                ),
                "output.weight.in_sum2": np.ones(256, dtype=np.float32),
                "output.weight.counts": np.array([1.0]),
            },
        )
        fused = "model.layers.3.mixer.experts.up_proj"

        covered, uncovered = resolve_assisted_weights(
            load_imatrix(path), {fused: 256, "lm_head.weight": 256}
        )

        assert uncovered == (fused,)
        assert set(covered) == {"lm_head.weight"}


class TestResolveImatrixCounts:
    def test_each_expert_parameter_reads_its_own_count(self, tmp_path) -> None:
        path = _write_stack(tmp_path / "im.gguf", columns=256)
        params = _stack_params(3, "up_proj")

        counts, _ = resolve_imatrix_counts(
            load_imatrix(path), dict.fromkeys(params, (8, 256))
        )

        assert [counts[name] for name in params] == list(range(1, EXPERTS + 1))

    def test_misaligned_expert_rows_still_report_counts(self, tmp_path) -> None:
        # Nemotron 3.5 Lightning's expert rows are 2688 and 1856.
        # Neither divides the 256-element super-block, so the
        # assisted weights report uncovered (ADR-0020). The counts
        # are routing frequency, not a fit, so they still resolve
        # (ADR-0026 decisions 4 and 5).
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.3.ffn_up_exps.weight.in_sum2": np.ones(
                    (4, HIDDEN), dtype=np.float32
                ),
                "blk.3.ffn_up_exps.weight.counts": np.array([[1.0, 2.0, 3.0, 4.0]]),
                "output.weight.in_sum2": np.ones(256, dtype=np.float32),
                "output.weight.counts": np.array([9.0]),
            },
        )
        params = _stack_params(3, "up_proj", experts=4)
        entries = load_imatrix(path)

        counts, _ = resolve_imatrix_counts(
            entries,
            dict.fromkeys(params, (8, HIDDEN)) | {"lm_head.weight": (8, 256)},
        )
        covered, uncovered = resolve_assisted_weights(
            entries, dict.fromkeys(params, HIDDEN) | {"lm_head.weight": 256}
        )

        assert [counts[name] for name in params] == [1, 2, 3, 4]
        assert counts["lm_head.weight"] == 9
        assert uncovered == tuple(params)
        assert set(covered) == {"lm_head.weight"}

    def test_uncovered_names_are_reported(self, tmp_path) -> None:
        # Uncovered, never zero. ADR-0026 decision 5 reports a zero
        # count as a real measurement, so the two must not merge.
        path = _write_stack(tmp_path / "im.gguf", columns=256, experts=4)

        names = ["model.layers.0.self_attn.q_proj.weight", "model.norm.weight"]

        counts, uncovered = resolve_imatrix_counts(
            load_imatrix(path), dict.fromkeys(names, (8, 256))
        )

        assert counts == {}
        assert uncovered == tuple(names)

    def test_zero_count_expert_reports_zero(self, tmp_path) -> None:
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.3.ffn_up_exps.weight.in_sum2": np.ones((2, 256), dtype=np.float32),
                "blk.3.ffn_up_exps.weight.counts": np.array([[5.0, 0.0]]),
            },
        )
        params = _stack_params(3, "up_proj", experts=2)

        counts, _ = resolve_imatrix_counts(
            load_imatrix(path), dict.fromkeys(params, (8, 256))
        )

        assert [counts[name] for name in params] == [5, 0]

    def test_a_count_rounds_half_away_from_zero(self, tmp_path) -> None:
        # The file stores counts as float32 and the C loader applies
        # std::lround (imatrix-loader.cpp:158). torch.round breaks a
        # tie to even, so 4.5 would read 4 and 2.5 would read 2.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.3.ffn_up_exps.weight.in_sum2": np.ones((4, 256), dtype=np.float32),
                "blk.3.ffn_up_exps.weight.counts": np.array([[4.7, 4.2, 4.5, 2.5]]),
            },
        )
        params = _stack_params(3, "up_proj", experts=4)

        counts, _ = resolve_imatrix_counts(
            load_imatrix(path), dict.fromkeys(params, (8, 256))
        )

        assert [counts[name] for name in params] == [5, 4, 5, 3]

    def test_a_fused_expert_stack_reads_one_count_vector(self, tmp_path) -> None:
        # ADR-0026 (2026-08-13): a fused expert stack's counts return
        # as one vector, keyed by the loaded parameter name. Element
        # i is expert i's routing frequency. The read constructs no
        # indexed names — constructed names are how #177 missed the
        # fusion.
        path = _write_stack(tmp_path / "im.gguf", columns=256)
        stack = "model.layers.3.mixer.experts.up_proj"

        counts, uncovered = resolve_imatrix_counts(
            load_imatrix(path), {stack: (EXPERTS, 1856, 256)}
        )

        assert uncovered == ()
        assert counts[stack] == tuple(range(1, EXPERTS + 1))

    def test_a_fused_vector_keeps_zeros_and_rounds_half_away_from_zero(
        self, tmp_path
    ) -> None:
        # Zero is a real count that ADR-0026 decision 5 reports, and
        # the vector rounds the C loader's way, the same as a scalar.
        # Rows of 2688 refuse a k-quant fit, and the counts still
        # resolve — routing frequency takes no super-block gate.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.3.ffn_up_exps.weight.in_sum2": np.ones(
                    (4, HIDDEN), dtype=np.float32
                ),
                "blk.3.ffn_up_exps.weight.counts": np.array([[5.0, 0.0, 4.5, 2.5]]),
            },
        )
        stack = "backbone.layers.3.mixer.experts.up_proj"

        counts, _ = resolve_imatrix_counts(
            load_imatrix(path), {stack: (4, 1856, HIDDEN)}
        )

        assert counts[stack] == (5, 0, 5, 3)

    def test_a_fused_stack_that_is_not_three_dimensional_raises(self, tmp_path) -> None:
        # The name decides the fused fork and the shape vouches for
        # it (ADR-0026, 2026-08-13). A 2D shape under a fused name
        # means the loaded model and the name table disagree.
        path = _write_stack(tmp_path / "im.gguf", columns=256)

        with pytest.raises(ValueError, match="dimensions, not 3"):
            resolve_imatrix_counts(
                load_imatrix(path),
                {"model.layers.3.mixer.experts.up_proj": (EXPERTS, 256)},
            )

    def test_a_fused_count_length_against_first_dimension_mismatch_raises(
        self, tmp_path
    ) -> None:
        # The shape assertion is the vouching mechanism, not a
        # version floor (ADR-0026, 2026-08-13). 64 experts against
        # 128 counts means the imatrix describes another model.
        path = _write_stack(tmp_path / "im.gguf", columns=256)

        with pytest.raises(ValueError, match="holds 64 experts"):
            resolve_imatrix_counts(
                load_imatrix(path),
                {"model.layers.3.mixer.experts.up_proj": (64, 1856, 256)},
            )

    def test_a_fused_stack_on_a_dense_entry_raises(self, tmp_path) -> None:
        # A one-matrix entry cannot be an expert stack. Reading it as
        # a length-1 vector would record a chunk tally as a routing
        # frequency, which is the defect #193 separates.
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.3.ffn_up_exps.weight.in_sum2": np.ones(256, dtype=np.float32),
                "blk.3.ffn_up_exps.weight.counts": np.array([7.0]),
            },
        )

        with pytest.raises(ValueError, match="counts 1 matrices"):
            resolve_imatrix_counts(
                load_imatrix(path),
                {"model.layers.3.mixer.experts.up_proj": (EXPERTS, 1856, 256)},
            )

    def test_a_fused_stack_the_file_does_not_hold_reports_uncovered(
        self, tmp_path
    ) -> None:
        path = _write_stack(tmp_path / "im.gguf", columns=256)
        down = "model.layers.3.mixer.experts.down_proj"

        counts, uncovered = resolve_imatrix_counts(
            load_imatrix(path), {down: (EXPERTS, 256, 1856)}
        )

        assert counts == {}
        assert uncovered == (down,)


class TestExpertStackCountVectors:
    """The #201 amendment's selection rule: vectors only, all or nothing."""

    UP = "model.layers.3.mixer.experts.up_proj"
    DOWN = "model.layers.3.mixer.experts.down_proj"
    DENSE = "model.layers.3.mixer.out_proj.weight"

    def test_covered_stacks_pool_and_dense_scalars_stay_out(self) -> None:
        counts = {self.UP: (1, 2), self.DOWN: (3, 4), self.DENSE: 421_370}

        vectors = expert_stack_count_vectors(counts, [self.UP, self.DOWN, self.DENSE])

        assert vectors == ((1, 2), (3, 4))

    def test_group_without_an_expert_stack_selects_none(self) -> None:
        assert expert_stack_count_vectors({self.DENSE: 421_370}, [self.DENSE]) is None

    def test_one_unresolved_stack_makes_the_group_all_or_nothing(self) -> None:
        counts: dict[str, int | tuple[int, ...]] = {self.UP: (1, 2)}

        assert expert_stack_count_vectors(counts, [self.UP, self.DOWN]) is None

    def test_empty_resolution_selects_none_and_never_raises(self) -> None:
        # The #202 regression shape: an empty resolution leaves the
        # summary absent and never refuses (the #198 amendment).
        assert expert_stack_count_vectors({}, [self.UP, self.DENSE]) is None

    def test_indexed_expert_scalars_never_enter(self) -> None:
        # An unfused layout resolves one scalar per indexed expert.
        # The summary reduces count vectors only, so the group
        # selects nothing.
        indexed = "model.layers.3.mixer.experts.0.up_proj.weight"

        assert expert_stack_count_vectors({indexed: 7}, [indexed]) is None
