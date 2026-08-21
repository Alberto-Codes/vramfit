"""Checks of the q0 imatrix reader (ADR-0018, 2026-08-21 amendment).

The reader accepts fused expert stacks — one weight row per expert —
where the kquant resolver refuses them by rule (ADR-0026). Entries
are built in memory: `resolve_q0_assisted_weights` consumes what
`load_imatrix` produces, and the loader has its own suite.
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.imatrix import (
    ImatrixEntry,
    resolve_assisted_weights,
)
from vramfit.adapters.outbound.scan.imatrix_q0 import (
    check_q0_imatrix_weights,
    resolve_q0_assisted_weights,
)

pytestmark = pytest.mark.unit

STACK = "backbone.layers.3.mixer.experts.up_proj"
DENSE = "backbone.layers.3.mixer.q_proj.weight"


def _entry(matrices: int, columns: int, seed: int = 0) -> ImatrixEntry:
    generator = torch.Generator().manual_seed(seed)
    return ImatrixEntry(
        column_weights=torch.rand((matrices, columns), generator=generator) + 0.01,
        counts=torch.full((matrices,), 5.0),
    )


class TestFusedExpertStacks:
    def test_a_fused_stack_resolves_one_weight_row_per_expert(self) -> None:
        entry = _entry(matrices=4, columns=64)

        covered, uncovered = resolve_q0_assisted_weights(
            {"blk.3.ffn_up_exps.weight": entry}, {STACK: (4, 8, 64)}
        )

        assert uncovered == ()
        assert torch.equal(covered[STACK], entry.column_weights)

    def test_the_kquant_resolver_still_refuses_the_same_stack(self) -> None:
        # One reader per method family (decision 2): the q0 build
        # must not relax the kquant resolver's fused-stack rule.
        entry = _entry(matrices=4, columns=256)
        with pytest.raises(ValueError, match="fused expert stacks"):
            resolve_assisted_weights({"blk.3.ffn_up_exps.weight": entry}, {STACK: 256})

    def test_a_non_3d_fused_shape_refuses(self) -> None:
        entry = _entry(matrices=4, columns=64)
        with pytest.raises(ValueError, match="dimensions, not 3"):
            resolve_q0_assisted_weights(
                {"blk.3.ffn_up_exps.weight": entry}, {STACK: (8, 64)}
            )

    def test_an_expert_count_mismatch_refuses(self) -> None:
        entry = _entry(matrices=4, columns=64)
        with pytest.raises(ValueError, match="not describe this checkpoint"):
            resolve_q0_assisted_weights(
                {"blk.3.ffn_up_exps.weight": entry}, {STACK: (5, 8, 64)}
            )

    def test_a_column_mismatch_refuses(self) -> None:
        entry = _entry(matrices=4, columns=64)
        with pytest.raises(ValueError, match="parameter rows"):
            resolve_q0_assisted_weights(
                {"blk.3.ffn_up_exps.weight": entry}, {STACK: (4, 8, 96)}
            )


class TestDenseAndIndexedParameters:
    def test_a_dense_parameter_reads_row_zero(self) -> None:
        entry = _entry(matrices=1, columns=64)

        covered, uncovered = resolve_q0_assisted_weights(
            {"blk.3.attn_q.weight": entry}, {DENSE: (32, 64)}
        )

        assert uncovered == ()
        assert torch.equal(covered[DENSE], entry.column_weights[0])

    def test_an_indexed_expert_reads_its_own_row(self) -> None:
        entry = _entry(matrices=4, columns=64)
        name = "backbone.layers.3.mixer.experts.2.up_proj.weight"

        covered, _ = resolve_q0_assisted_weights(
            {"blk.3.ffn_up_exps.weight": entry}, {name: (8, 64)}
        )

        assert torch.equal(covered[name], entry.column_weights[2])

    def test_two_parameters_claiming_one_row_refuse(self) -> None:
        # #193's rule carries over: a duplicate claim prices one of
        # the two against the wrong columns.
        entry = _entry(matrices=1, columns=64)
        other = "model.layers.3.self_attn.q_proj.weight"
        with pytest.raises(ValueError, match="both claim"):
            resolve_q0_assisted_weights(
                {"blk.3.attn_q.weight": entry},
                {DENSE: (32, 64), other: (32, 64)},
            )


class TestUncoveredFallbacks:
    def test_an_unmapped_name_reports_uncovered(self) -> None:
        entry = _entry(matrices=1, columns=64)

        covered, uncovered = resolve_q0_assisted_weights(
            {"blk.3.attn_q.weight": entry},
            {DENSE: (32, 64), "vision_tower.layers.0.w.weight": (32, 64)},
        )

        assert DENSE in covered
        assert uncovered == ("vision_tower.layers.0.w.weight",)

    def test_an_absent_entry_reports_uncovered(self) -> None:
        entry = _entry(matrices=1, columns=64)

        _, uncovered = resolve_q0_assisted_weights(
            {"blk.3.attn_q.weight": entry},
            {DENSE: (32, 64), "backbone.layers.3.mixer.k_proj.weight": (32, 64)},
        )

        assert uncovered == ("backbone.layers.3.mixer.k_proj.weight",)

    def test_misaligned_rows_report_uncovered_not_refused(self) -> None:
        # 48 % 32 != 0: the assisted fit cannot align its weights, so
        # the parameter prices unassisted — the fallback beats
        # refusing a multi-day scan over one tensor.
        entry = _entry(matrices=1, columns=64)

        covered, uncovered = resolve_q0_assisted_weights(
            {"blk.3.attn_q.weight": entry},
            {DENSE: (32, 64), "backbone.layers.3.mixer.o_proj.weight": (32, 48)},
        )

        assert DENSE in covered
        assert uncovered == ("backbone.layers.3.mixer.o_proj.weight",)

    def test_zero_coverage_names_each_cause(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            resolve_q0_assisted_weights(
                {"blk.3.attn_output.weight": _entry(matrices=1, columns=48)},
                {
                    "vision_tower.layers.0.w.weight": (32, 64),
                    "backbone.layers.3.mixer.k_proj.weight": (32, 64),
                    "backbone.layers.3.mixer.o_proj.weight": (32, 48),
                },
            )
        message = str(excinfo.value)
        assert "covers none of the 3 parameters" in message
        assert "1 names have no GGUF mapping" in message
        assert "1 mapped to a tensor the file does not hold" in message
        assert "1 have rows that do not divide" in message


class TestCheckQ0ImatrixWeights:
    def test_valid_1d_and_2d_weights_pass(self) -> None:
        check_q0_imatrix_weights(
            {DENSE: torch.rand(64), STACK: torch.rand(4, 64)},
            {DENSE: (32, 64), STACK: (4, 8, 64)},
        )

    def test_an_unknown_name_refuses(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter"):
            check_q0_imatrix_weights({"nope": torch.rand(64)}, {DENSE: (32, 64)})

    def test_2d_weights_against_a_2d_parameter_refuse(self) -> None:
        with pytest.raises(ValueError, match="3-D"):
            check_q0_imatrix_weights({DENSE: torch.rand(4, 64)}, {DENSE: (32, 64)})

    def test_2d_weights_with_a_wrong_expert_count_refuse(self) -> None:
        with pytest.raises(ValueError, match="3-D"):
            check_q0_imatrix_weights({STACK: torch.rand(3, 64)}, {STACK: (4, 8, 64)})

    def test_a_wrong_column_count_refuses(self) -> None:
        with pytest.raises(ValueError, match="64"):
            check_q0_imatrix_weights({DENSE: torch.rand(32)}, {DENSE: (32, 64)})

    def test_misaligned_rows_refuse(self) -> None:
        with pytest.raises(ValueError, match="32-element Q4_0"):
            check_q0_imatrix_weights({DENSE: torch.rand(48)}, {DENSE: (32, 48)})

    def test_garbage_weights_refuse(self) -> None:
        bad = torch.rand(64)
        bad[0] = float("nan")
        with pytest.raises(ValueError, match="finite and non-negative"):
            check_q0_imatrix_weights({DENSE: bad}, {DENSE: (32, 64)})
