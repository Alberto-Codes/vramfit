"""The 256 super-block decision reads the measured row width (#515).

Issue 515 found that `rows_refuse_super_block` matched a class-name
list, and a dogfood scan of Qwen3-Coder-30B-A3B found the same defect
from the other side: that target's routed-expert rows are 2048 and
768, both of which divide 256, and the name-based routing sent them
to the ADR-0028 table anyway. The suite holds both directions, and it
holds the plan's price against the pack's emitted type.
"""

from __future__ import annotations

import pytest

from tests.unit.conftest import make_map
from vramfit.adapters.outbound.gguf.types import (
    EXPERT_STACK_TYPE_BY_BITS,
    GGML_TYPE_BY_BITS,
    PackError,
    tensor_overrides,
)
from vramfit.adapters.outbound.sensitivity_map_json import map_from_dict
from vramfit.domain.model import Recipe
from vramfit.domain.runtime import (
    EFFECTIVE_BITS,
    EXPERT_STACK_EFFECTIVE_BITS,
    LLAMA_CPP,
)
from vramfit.domain.sizes import SizeSourceError
from vramfit.domain.solver import group_bytes, solve

pytestmark = pytest.mark.unit

# The 30B Nemotron target's dense rows (#159, #183). No k-quant
# super-block divides 2688.
NEMOTRON_CLASS = "model.layers.0.mixer.in_proj"
NEMOTRON_ROWS = 2688

# Qwen3-Coder-30B-A3B's routed-expert stacks. Both widths divide 256,
# so both take the ADR-0012 k-quant table.
QWEN_UP = "model.layers.0.mlp.experts.up_proj"
QWEN_DOWN = "model.layers.0.mlp.experts.down_proj"
QWEN_ROWS = {QWEN_UP: 2048, QWEN_DOWN: 768}

CURVE = {8: 0.001, 6: 0.002, 4: 0.01, 3: 0.02, 2: 0.1}
PRECISIONS = (8, 6, 4, 3, 2)


def plan(group: str, rows: int, bits: int, bytes_fp16: int = 160_000) -> Recipe:
    """Solve one group at one pinned precision under a stated row width.

    Args:
        group: The group to price.
        rows: Its measured row width.
        bits: The precision to pin it at.
        bytes_fp16: Its size at reference precision.

    Returns:
        The one-assignment recipe.
    """
    map_ = map_from_dict(make_map([(group, bytes_fp16, CURVE)], precisions=PRECISIONS))
    return solve(
        map_,
        weight_budget_bytes=10**9,
        vram_budget_bytes=10**9 + 1000,
        kv_headroom_bytes=1000,
        runtime=LLAMA_CPP,
        pins={group: bits},
        # The Nemotron-H family needs a size source whatever this
        # suite measures: the scan skips its refused classes, so only
        # the checkpoint prices them (ADR-0029 decision 3, #409).
        discovered_bytes={group: bytes_fp16},
        row_widths={group: rows},
        format_overhead=0.0,
    )


class TestRowsThatRefuseTheSuperBlock:
    """Acceptance 1: every group refused by name is still refused."""

    @pytest.mark.parametrize("bits", [8, 6, 4, 2])
    def test_a_2688_wide_class_prices_through_the_adr_0028_table(
        self, bits: int
    ) -> None:
        recipe = plan(NEMOTRON_CLASS, NEMOTRON_ROWS, bits)

        assert recipe.assignments[0].bytes == group_bytes(
            160_000, EXPERT_STACK_EFFECTIVE_BITS[LLAMA_CPP][bits], 0.0
        )

    @pytest.mark.parametrize("bits", [8, 6, 4, 2])
    def test_a_2688_wide_class_packs_the_adr_0028_type(self, bits: int) -> None:
        recipe = plan(NEMOTRON_CLASS, NEMOTRON_ROWS, bits)

        (override,) = tensor_overrides(recipe, {NEMOTRON_CLASS: NEMOTRON_ROWS})
        assert override.quant_type == EXPERT_STACK_TYPE_BY_BITS[bits]

    def test_a_2688_wide_class_still_refuses_nominal_3(self) -> None:
        # ADR-0028 decision 2: no GGUF type lands between 2.25 and
        # 4.25 bits per weight on rows the super-block refuses.
        recipe = plan(NEMOTRON_CLASS, NEMOTRON_ROWS, 3)

        with pytest.raises(PackError, match="cannot pack at nominal 3"):
            tensor_overrides(recipe, {NEMOTRON_CLASS: NEMOTRON_ROWS})


class TestRowsThatDivideTheSuperBlock:
    """Acceptance 2: a stack of 256-dividing rows takes the k-quant table."""

    @pytest.mark.parametrize("group", [QWEN_UP, QWEN_DOWN])
    @pytest.mark.parametrize("bits", [8, 6, 4, 3, 2])
    def test_a_qwen_stack_prices_through_the_kquant_table(
        self, group: str, bits: int
    ) -> None:
        recipe = plan(group, QWEN_ROWS[group], bits)

        assert recipe.assignments[0].bytes == group_bytes(
            160_000, EFFECTIVE_BITS[LLAMA_CPP][bits], 0.0
        )

    @pytest.mark.parametrize("group", [QWEN_UP, QWEN_DOWN])
    @pytest.mark.parametrize("bits", [8, 6, 4, 3, 2])
    def test_a_qwen_stack_packs_the_kquant_type(self, group: str, bits: int) -> None:
        recipe = plan(group, QWEN_ROWS[group], bits)

        (override,) = tensor_overrides(recipe, {group: QWEN_ROWS[group]})
        assert override.quant_type == GGML_TYPE_BY_BITS[bits]

    @pytest.mark.parametrize("group", [QWEN_UP, QWEN_DOWN])
    def test_a_qwen_stack_accepts_nominal_3(self, group: str) -> None:
        # The name-based routing banned nominal 3 across 94.95 % of
        # this target's parameters and would have lost to stock
        # Q4_K_M at equal bytes (#515).
        recipe = plan(group, QWEN_ROWS[group], 3)

        (override,) = tensor_overrides(recipe, {group: QWEN_ROWS[group]})
        assert override.quant_type == "q3_k"


class TestPredictionMatchesEmission:
    """Acceptance 3: one width drives the price and the emitted type."""

    @pytest.mark.parametrize(
        ("group", "rows", "effective", "types"),
        [
            (
                NEMOTRON_CLASS,
                NEMOTRON_ROWS,
                EXPERT_STACK_EFFECTIVE_BITS[LLAMA_CPP],
                EXPERT_STACK_TYPE_BY_BITS,
            ),
            (QWEN_UP, 2048, EFFECTIVE_BITS[LLAMA_CPP], GGML_TYPE_BY_BITS),
            (QWEN_DOWN, 768, EFFECTIVE_BITS[LLAMA_CPP], GGML_TYPE_BY_BITS),
        ],
        ids=["nemotron-2688", "qwen-2048", "qwen-768"],
    )
    @pytest.mark.parametrize("bits", [8, 6, 4, 2])
    def test_the_domain_prices_from_the_table_the_pack_emits_from(
        self,
        group: str,
        rows: int,
        effective: dict[int, float],
        types: dict[int, str],
        bits: int,
    ) -> None:
        # Nominal 6 is where the two tables disagree most: Q5_1 at
        # 6.00 bits per weight against Q6_K's 6.5625. A domain that
        # routed by name while the pack read the width would drift
        # 0.5625 bits per weight and never say so.
        recipe = plan(group, rows, bits)

        (override,) = tensor_overrides(recipe, {group: rows})
        assert recipe.assignments[0].bytes == group_bytes(160_000, effective[bits], 0.0)
        assert override.quant_type == types[bits]


class TestUnmeasuredRows:
    """A width the plan cannot measure refuses rather than defaults."""

    def test_a_class_group_without_a_measured_width_refuses_and_names_the_flag(
        self,
    ) -> None:
        map_ = map_from_dict(
            make_map([(QWEN_UP, 160_000, CURVE)], precisions=PRECISIONS)
        )

        with pytest.raises(SizeSourceError, match="no measured row width"):
            solve(
                map_,
                weight_budget_bytes=10**9,
                vram_budget_bytes=10**9 + 1000,
                kv_headroom_bytes=1000,
                runtime=LLAMA_CPP,
                discovered_bytes={QWEN_UP: 160_000},
            )

    def test_the_pack_refuses_a_group_it_measured_no_width_for(self) -> None:
        recipe = plan(QWEN_UP, 2048, 4)

        with pytest.raises(PackError, match="no measured row width"):
            tensor_overrides(recipe, {})

    def test_a_whole_layer_group_needs_no_width(self) -> None:
        # A layer group holds classes of several widths and keeps the
        # ADR-0012 k-quant table, as it did before the width reached
        # the decision.
        map_ = map_from_dict(
            make_map([("model.layers.0", 160_000, CURVE)], precisions=PRECISIONS)
        )

        recipe = solve(
            map_,
            weight_budget_bytes=10**9,
            vram_budget_bytes=10**9 + 1000,
            kv_headroom_bytes=1000,
            runtime=LLAMA_CPP,
            format_overhead=0.0,
        )

        (override,) = tensor_overrides(recipe, {})
        assert override.quant_type == GGML_TYPE_BY_BITS[8]
