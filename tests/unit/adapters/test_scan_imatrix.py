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
    gguf_tensor_name,
    load_imatrix,
    resolve_assisted_weights,
    resolve_imatrix_counts,
)
from vramfit.domain.model import ImatrixCountSummary

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
        "param",
        [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.unknown_proj.weight",
            "transformer.h.0.attn.c_attn.weight",
            # Mixtral spells its expert projections w1/w2/w3. The
            # table omits them rather than guess the mapping.
            "model.layers.0.block_sparse_moe.experts.4.w1.weight",
            # A shared expert carries no index and is not one row of
            # an expert stack.
            "backbone.layers.3.mixer.shared_experts.up_proj.weight",
        ],
    )
    def test_unmapped_names_return_none(self, param: str) -> None:
        assert gguf_tensor_name(param) is None

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
        }

        with pytest.raises(ValueError) as caught:
            resolve_assisted_weights(load_imatrix(path), rows)

        message = str(caught.value)
        assert "1 names have no GGUF mapping" in message
        assert "1 mapped to a tensor the file does not hold" in message
        assert "1 have rows that do not divide" in message


class TestResolveImatrixCounts:
    def test_each_expert_parameter_reads_its_own_count(self, tmp_path) -> None:
        path = _write_stack(tmp_path / "im.gguf", columns=256)
        params = _stack_params(3, "up_proj")

        counts, _ = resolve_imatrix_counts(load_imatrix(path), params)

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

        counts, _ = resolve_imatrix_counts(entries, [*params, "lm_head.weight"])
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

        counts, uncovered = resolve_imatrix_counts(load_imatrix(path), names)

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

        counts, _ = resolve_imatrix_counts(load_imatrix(path), params)

        assert [counts[name] for name in params] == [5, 0]

    def test_a_fused_stacks_counts_reduce_to_the_maps_summary(self, tmp_path) -> None:
        # ADR-0026 decision 4, the meter's chain without a model: an
        # expert stack's counts resolve per expert, then reduce to
        # the three numbers the map carries. The rows here are 2688
        # wide, the real target's width, which no k-quant fit can
        # price — the counts read anyway (#177).
        path = _write_imatrix(
            tmp_path / "im.gguf",
            {
                "blk.20.ffn_up_exps.weight.in_sum2": np.ones(
                    (4, HIDDEN), dtype=np.float32
                ),
                "blk.20.ffn_up_exps.weight.counts": np.array(
                    [[192_191.0, 426.0, 18_114.0, 823.0]]
                ),
            },
        )
        params = _stack_params(20, "up_proj", experts=4)

        counts, uncovered = resolve_imatrix_counts(load_imatrix(path), params)
        summary = ImatrixCountSummary.from_counts(counts[name] for name in params)

        assert uncovered == ()
        assert summary.minimum == 426
        assert summary.median == pytest.approx(9_468.5)
        assert summary.maximum == 192_191

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

        counts, _ = resolve_imatrix_counts(load_imatrix(path), params)

        assert [counts[name] for name in params] == [5, 4, 5, 3]
