"""The 256 super-block decision reads the measured row width (#515).

Issue 515 found that `rows_refuse_super_block` matched a class-name
list, and a dogfood scan of Qwen3-Coder-30B-A3B found the same defect
from the other side: that target's routed-expert rows are 2048 and
768, both of which divide 256, and the name-based routing sent them
to the ADR-0028 table anyway. The suite holds both directions, and it
holds the plan's price against the pack's emitted type.
"""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fakes import MemoryRecipePacker
from tests.unit.conftest import make_map
from vramfit.adapters.inbound import cli_pack
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.gguf.pack import checkpoint_row_widths
from vramfit.adapters.outbound.gguf.types import (
    EXPERT_STACK_TYPE_BY_BITS,
    GGML_TYPE_BY_BITS,
    PackError,
    tensor_overrides,
)
from vramfit.adapters.outbound.recipe_json import load_recipe, save_recipe
from vramfit.adapters.outbound.sensitivity_map_json import (
    MAP_SCHEMA_VERSION,
    map_from_dict,
)
from vramfit.domain.model import Assignment, PlanMeta, Recipe, SensitivityMap
from vramfit.domain.runtime import (
    EFFECTIVE_BITS,
    EXPERT_STACK_EFFECTIVE_BITS,
    LLAMA_CPP,
    VLLM,
)
from vramfit.domain.sizes import SizeSourceError, map_row_widths
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

# A layer-class group under a root the scan name table supports and
# the ADR-0029 reconcile table does not.
FOREIGN_ROOT_CLASS = "transformer.h.3.mlp.up_proj"

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

    def test_the_pack_surfaces_the_root_refusal_rather_than_the_width_one(
        self,
    ) -> None:
        # `transformer.` sits in the scan name table and outside the
        # ADR-0029 reconcile table. The width lookup asks that table
        # for a second spelling, and the table's own refusal names
        # the root. The missing-width message would name nothing.
        recipe = plan(FOREIGN_ROOT_CLASS, 2048, 4)

        with pytest.raises(SizeSourceError, match="root the table does not carry"):
            tensor_overrides(recipe, {})

    def test_a_single_unmeasured_group_reads_as_one_group(self) -> None:
        map_ = map_from_dict(
            make_map([(QWEN_UP, 160_000, CURVE)], precisions=PRECISIONS)
        )

        with pytest.raises(SizeSourceError) as caught:
            solve(
                map_,
                weight_budget_bytes=10**9,
                vram_budget_bytes=10**9 + 1000,
                kv_headroom_bytes=1000,
                runtime=LLAMA_CPP,
                discovered_bytes={QWEN_UP: 160_000},
            )

        assert f'group "{QWEN_UP}" has no measured row width' in str(caught.value)
        assert "1 groups" not in str(caught.value)

    def test_a_checkpoint_that_misses_one_group_does_not_repeat_the_flag(
        self,
    ) -> None:
        # The plan already read a checkpoint, so telling the operator
        # to pass --checkpoint repeats what they did. The checkpoint
        # carries QWEN_UP and not QWEN_DOWN.
        map_ = map_from_dict(
            make_map(
                [(QWEN_UP, 160_000, CURVE), (QWEN_DOWN, 160_000, CURVE)],
                precisions=PRECISIONS,
            )
        )

        with pytest.raises(SizeSourceError) as caught:
            solve(
                map_,
                weight_budget_bytes=10**9,
                vram_budget_bytes=10**9 + 1000,
                kv_headroom_bytes=1000,
                runtime=LLAMA_CPP,
                discovered_bytes={QWEN_UP: 160_000, QWEN_DOWN: 160_000},
                row_widths={QWEN_UP: 2048},
            )

        assert QWEN_DOWN in str(caught.value)
        assert "--checkpoint" not in str(caught.value)
        assert "Name the checkpoint the scan measured" in str(caught.value)

    def test_no_size_source_at_all_names_the_flag(self) -> None:
        map_ = map_from_dict(
            make_map([(QWEN_UP, 160_000, CURVE)], precisions=PRECISIONS)
        )

        with pytest.raises(SizeSourceError, match="Plan with --checkpoint"):
            solve(
                map_,
                weight_budget_bytes=10**9,
                vram_budget_bytes=10**9 + 1000,
                kv_headroom_bytes=1000,
                runtime=LLAMA_CPP,
            )

    def test_a_group_no_checkpoint_can_reach_states_the_root_limit(self) -> None:
        # `transformer.` sits in the scan name table and outside the
        # ADR-0029 root table, so --checkpoint would refuse again
        # (#551). The refusal must not send the operator there.
        map_ = map_from_dict(
            make_map([(FOREIGN_ROOT_CLASS, 160_000, CURVE)], precisions=PRECISIONS)
        )

        with pytest.raises(SizeSourceError) as caught:
            solve(
                map_,
                weight_budget_bytes=10**9,
                vram_budget_bytes=10**9 + 1000,
                kv_headroom_bytes=1000,
                runtime=LLAMA_CPP,
            )

        assert "No checkpoint can supply it" in str(caught.value)
        assert "Plan with --checkpoint" not in str(caught.value)

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


class TestGroupSpelledUnderTheCheckpointRoot:
    """One lookup serves the plan and the pack, so both take both roots."""

    # The scan names a Nemotron-H group under the checkpoint's own
    # root, and `discovered_group_rows` keys every width under the
    # map root (ADR-0029 decision 7). The plan must reconcile before
    # the lookup, as the pack does.
    BACKBONE_CLASS = "backbone.layers.0.mlp.up_proj"
    MAP_ROOTED = "model.layers.0.mlp.up_proj"

    def _solve(self) -> Recipe:
        map_ = map_from_dict(
            make_map([(self.BACKBONE_CLASS, 160_000, CURVE)], precisions=PRECISIONS)
        )
        return solve(
            map_,
            weight_budget_bytes=10**9,
            vram_budget_bytes=10**9 + 1000,
            kv_headroom_bytes=1000,
            runtime=LLAMA_CPP,
            pins={self.BACKBONE_CLASS: 6},
            row_widths={self.MAP_ROOTED: NEMOTRON_ROWS},
            format_overhead=0.0,
        )

    def test_the_plan_prices_it_from_the_reconciled_width(self) -> None:
        recipe = self._solve()

        assert recipe.assignments[0].bytes == group_bytes(
            160_000, EXPERT_STACK_EFFECTIVE_BITS[LLAMA_CPP][6], 0.0
        )

    def test_the_pack_emits_the_type_that_price_assumed(self) -> None:
        recipe = self._solve()

        (override,) = tensor_overrides(recipe, {self.MAP_ROOTED: NEMOTRON_ROWS})
        assert override.quant_type == EXPERT_STACK_TYPE_BY_BITS[6]


class TestRuntimesWithNoTypeTable:
    """The width routes between two tables, so a runtime without them is exempt."""

    @pytest.mark.parametrize("runtime", [VLLM, None], ids=["vllm", "no-runtime"])
    def test_a_stack_map_plans_without_a_size_source(self, runtime: str | None) -> None:
        # vLLM carries neither effective-bits table, so every group
        # prices at nominal bits and no width moves a byte. Refusing
        # would block a solve the width cannot improve.
        map_ = map_from_dict(
            make_map([(QWEN_UP, 160_000, CURVE)], precisions=PRECISIONS)
        )

        recipe = solve(
            map_,
            weight_budget_bytes=10**9,
            vram_budget_bytes=10**9 + 1000,
            kv_headroom_bytes=1000,
            runtime=runtime,
            format_overhead=0.0,
        )

        assert recipe.assignments[0].bytes == group_bytes(160_000, 8, 0.0)

    @pytest.mark.parametrize("runtime", [VLLM, None], ids=["vllm", "no-runtime"])
    def test_a_foreign_rooted_stack_map_plans_too(self, runtime: str | None) -> None:
        map_ = map_from_dict(
            make_map([(FOREIGN_ROOT_CLASS, 160_000, CURVE)], precisions=PRECISIONS)
        )

        recipe = solve(
            map_,
            weight_budget_bytes=10**9,
            vram_budget_bytes=10**9 + 1000,
            kv_headroom_bytes=1000,
            runtime=runtime,
            format_overhead=0.0,
        )

        assert recipe.assignments[0].group == FOREIGN_ROOT_CLASS


def write_checkpoint(model_dir: Path, shapes: dict[str, list[int]]) -> Path:
    """Write one bf16 safetensors shard stating each tensor's shape.

    The header is the safetensors byte contract: an 8-byte
    little-endian length, then the JSON header the reader parses.

    Args:
        model_dir: Directory to write the shard into.
        shapes: Shape per checkpoint tensor name. The last dimension
            is the row width the super-block decision reads.

    Returns:
        The checkpoint directory.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for name, shape in shapes.items():
        span = 2
        for dim in shape:
            span *= dim
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + span],
        }
        offset += span
    blob = json.dumps(header).encode("utf-8")
    (model_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return model_dir


class TestCheckpointRowWidths:
    """The pack measures the widths from the same headers the plan read."""

    def test_it_keys_stack_widths_under_the_map_root(self, tmp_path) -> None:
        model_dir = write_checkpoint(
            tmp_path / "ckpt",
            {
                "backbone.layers.0.mixer.experts.0.up_proj.weight": [4, 2688],
                "backbone.layers.0.mixer.experts.1.up_proj.weight": [4, 2688],
                "backbone.layers.0.mlp.down_proj.weight": [4, 768],
            },
        )

        assert checkpoint_row_widths(model_dir) == {
            "model.layers.0.mixer.experts.up_proj": 2688,
            "model.layers.0.mlp.down_proj": 768,
        }

    def test_a_refusal_from_the_source_names_the_checkpoint(self, tmp_path) -> None:
        # An empty directory refuses inside the size source. The pack
        # re-words it so the operator learns which directory failed.
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(PackError, match="cannot measure the checkpoint"):
            checkpoint_row_widths(empty)

    def test_two_widths_in_one_group_refuse_through_the_pack(self, tmp_path) -> None:
        model_dir = write_checkpoint(
            tmp_path / "ckpt",
            {
                "backbone.layers.0.mixer.experts.0.up_proj.weight": [4, 2688],
                "backbone.layers.0.mixer.experts.1.up_proj.weight": [4, 2048],
            },
        )

        with pytest.raises(PackError, match="rows of 2688 and 2048"):
            checkpoint_row_widths(model_dir)

    def test_an_unreadable_shard_names_the_checkpoint(self, tmp_path) -> None:
        # A directory under the shard glob raises IsADirectoryError,
        # which is an OSError and no refusal the source words itself.
        model_dir = tmp_path / "ckpt"
        model_dir.mkdir()
        (model_dir / "model.safetensors").mkdir()

        with pytest.raises(PackError, match="cannot read the checkpoint"):
            checkpoint_row_widths(model_dir)


runner = CliRunner()


def make_toolchain(root: Path) -> Path:
    """Lay out the llama.cpp paths the pack command checks for."""
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").touch()
    (root / "build" / "bin" / "llama-quantize").touch()
    (root / "build" / "bin" / "llama-perplexity").touch()
    return root


def make_routed_recipe(model_id: str, groups: dict[str, int]) -> Recipe:
    """Build an unprotected recipe assigning each group its precision."""
    return Recipe(
        model_id=model_id,
        plan=PlanMeta(
            vram_budget_bytes=4_000,
            kv_headroom_bytes=1_000,
            weight_budget_bytes=3_000,
            predicted_total_bytes=2_500,
            predicted_damage=0.05,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=tuple(
            Assignment(group=group, bits=bits, bytes=500, damage=0.01)
            for group, bits in groups.items()
        ),
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


class TestPlanCommandWithoutACheckpoint:
    """The CLI half of the exemption, which `plan --help` states."""

    def test_a_vllm_plan_of_a_stack_map_needs_no_checkpoint(self, tmp_path) -> None:
        # vLLM carries neither effective-bits table, so the width
        # routes nothing and the refusal never fires. `--runtime` is
        # the only CLI path to that exemption, because the option
        # defaults to llama.cpp.
        map_path = tmp_path / "sensitivity.json"
        map_path.write_text(
            json.dumps(make_map([(QWEN_UP, 160_000, CURVE)], precisions=PRECISIONS))
        )
        out = tmp_path / "recipe.json"

        result = runner.invoke(
            app,
            [
                "plan",
                str(map_path),
                "--vram",
                "200000",
                "--kv-headroom",
                "50000",
                "--runtime",
                VLLM,
                "--out",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        assert load_recipe(out).assignments[0].group == QWEN_UP


class TestPackPreflight:
    """The width refusal lands before the convert stage (ADR-0022, #367)."""

    def test_an_unprotected_recipe_refuses_before_convert(
        self, tmp_path, monkeypatch
    ) -> None:
        # `--model` names a sibling snapshot that carries layer 9 and
        # not layer 0. Convert would write a full-size base GGUF
        # first, and the refusal would land at the quantize stage.
        packer = MemoryRecipePacker(packed_bytes=100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: packer)
        model_dir = write_checkpoint(
            tmp_path / "ckpt", {"backbone.layers.9.mixer.in_proj.weight": [4, 2688]}
        )
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_routed_recipe(str(model_dir), {"model.layers.0.mixer.in_proj": 4}),
            recipe_path,
        )

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(make_toolchain(tmp_path / "llama.cpp")),
                "--out",
                str(tmp_path / "packed.gguf"),
            ],
        )

        assert result.exit_code == 1
        assert "no measured row width" in result.output
        assert "model.layers.0.mixer.in_proj" in result.output
        assert packer.has_base is False

    def test_an_unmappable_recipe_keeps_the_mapping_refusal(
        self, tmp_path, monkeypatch
    ) -> None:
        # A `--group-by tensor` map of an MoE model names each expert
        # by index. No class-table stem maps that suffix, so the
        # recipe is unmappable whatever the checkpoint states, and
        # `checkpoint_row_widths` measures at stack granularity and
        # would never hold that spelling. Blaming --model would send
        # the operator after a checkpoint that is already correct.
        packer = MemoryRecipePacker(packed_bytes=100)
        monkeypatch.setattr(cli_pack, "_build_packer", lambda *args: packer)
        model_dir = tmp_path / "no-shards"
        model_dir.mkdir()
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_routed_recipe(
                str(model_dir), {"model.layers.0.mlp.experts.57.up_proj": 4}
            ),
            recipe_path,
        )

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(make_toolchain(tmp_path / "llama.cpp")),
                "--out",
                str(tmp_path / "packed.gguf"),
            ],
        )

        assert result.exit_code == 1
        assert "no GGUF tensor mapping" in result.output
        assert "no measured row width" not in result.output

    def test_a_recipe_of_unquantizable_classes_reads_no_checkpoint(
        self, tmp_path, monkeypatch
    ) -> None:
        # The Nemotron-H layer-granularity shape: the only
        # width-shaped groups are the classes ADR-0029 holds at the
        # convert dtype, and nothing consults a width for them. An
        # absent shard is not this command's problem (#409).
        seen: dict[str, object] = {}

        def recorder(*args):
            seen["row_widths"] = args[-1]
            return MemoryRecipePacker(packed_bytes=100)

        monkeypatch.setattr(cli_pack, "_build_packer", recorder)
        model_dir = tmp_path / "no-shards"
        model_dir.mkdir()
        recipe_path = tmp_path / "recipe.json"
        save_recipe(
            make_routed_recipe(
                str(model_dir),
                {"model.layers.0": 4, "model.layers.0.mixer.conv1d": 16},
            ),
            recipe_path,
        )

        result = runner.invoke(
            app,
            [
                "pack",
                str(recipe_path),
                "--llama-cpp",
                str(make_toolchain(tmp_path / "llama.cpp")),
                "--out",
                str(tmp_path / "packed.gguf"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert seen["row_widths"] == {}


def map_with_widths(
    groups: dict[str, int], bytes_fp16: int = 160_000, group_by: str = "stack"
) -> dict:
    """Build a schema-5 map that records its own row widths (#558).

    Args:
        groups: Row width per group name.
        bytes_fp16: Each group's size at reference precision.
        group_by: The granularity the map recorded.

    Returns:
        A raw map dict accepted by `map_from_dict`. It declares the
        version whose producer records the field, because no earlier
        producer emits this shape and the reader refuses it.
    """
    raw = make_map(
        [(name, bytes_fp16, CURVE) for name in groups], precisions=PRECISIONS
    )
    raw["vramfit_schema"] = MAP_SCHEMA_VERSION
    raw["scan"]["group_by"] = group_by
    for entry in raw["groups"]:
        entry["row_width"] = groups[entry["name"]]
    return raw


@pytest.mark.unit
class TestAMapPlansOnItsOwnWidths:
    """Issue #558: a published map routes without the checkpoint.

    Before schema 5 the map discarded the width the scan measured,
    so `vramfit plan MAP --vram 24GiB --runtime llama.cpp` refused a
    stack map with no checkpoint beside it. A Space that holds the
    map and no weights is the case that ended.
    """

    def solve_map_only(self, raw: dict, group: str = QWEN_UP) -> Recipe:
        return solve(
            map_from_dict(raw),
            weight_budget_bytes=10**9,
            vram_budget_bytes=10**9 + 1000,
            kv_headroom_bytes=1000,
            runtime=LLAMA_CPP,
            pins={group: 4},
            format_overhead=0.0,
        )

    def test_a_recorded_width_solves_with_no_checkpoint(self) -> None:
        recipe = self.solve_map_only(map_with_widths({QWEN_UP: NEMOTRON_ROWS}))

        assert recipe.assignments[0].bytes == group_bytes(
            160_000, EXPERT_STACK_EFFECTIVE_BITS[LLAMA_CPP][4], 0.0
        )

    def test_the_recorded_width_routes_the_table_the_checkpoint_would(self) -> None:
        # 2688 refuses the 256 super-block, so the map's own width
        # must reach the ADR-0028 table, not the k-quant one.
        from_map = self.solve_map_only(map_with_widths({QWEN_UP: NEMOTRON_ROWS}))
        from_checkpoint = plan(QWEN_UP, NEMOTRON_ROWS, 4)

        assert from_map.assignments[0].bytes == from_checkpoint.assignments[0].bytes

    def test_a_width_that_divides_256_keeps_the_k_quant_table(self) -> None:
        recipe = self.solve_map_only(map_with_widths({QWEN_UP: 2048}))

        assert recipe.assignments[0].bytes == group_bytes(
            160_000, EFFECTIVE_BITS[LLAMA_CPP][4], 0.0
        )

    def test_a_map_without_the_width_still_refuses_loudly(self) -> None:
        # The silent misprice is taking the k-quant table by
        # omission. A map below schema 5 records no width, so the
        # solve refuses rather than routing off a number it lacks.
        raw = make_map([(QWEN_UP, 160_000, CURVE)], precisions=PRECISIONS)
        raw["scan"]["group_by"] = "stack"

        with pytest.raises(SizeSourceError) as caught:
            self.solve_map_only(raw)

        assert "no measured row width" in str(caught.value)
        assert "records no row width" in str(caught.value)
        assert "--checkpoint" in str(caught.value)

    def test_one_missing_width_refuses_even_beside_recorded_ones(self) -> None:
        raw = map_with_widths({QWEN_UP: 2048, QWEN_DOWN: 768})
        del raw["groups"][1]["row_width"]

        with pytest.raises(SizeSourceError, match=QWEN_DOWN):
            solve(
                map_from_dict(raw),
                weight_budget_bytes=10**9,
                vram_budget_bytes=10**9 + 1000,
                kv_headroom_bytes=1000,
                runtime=LLAMA_CPP,
                format_overhead=0.0,
            )

    def test_a_partly_recorded_map_is_not_blamed_on_its_schema(self) -> None:
        # This map is schema 5 and states a width for its other group,
        # so blaming "every map below schema 5" would be false and
        # would send a partial back-fill after the wrong remedy.
        raw = map_with_widths({QWEN_UP: 2048, QWEN_DOWN: 768})
        del raw["groups"][1]["row_width"]

        with pytest.raises(SizeSourceError) as caught:
            solve(
                map_from_dict(raw),
                weight_budget_bytes=10**9,
                vram_budget_bytes=10**9 + 1000,
                kv_headroom_bytes=1000,
                runtime=LLAMA_CPP,
                format_overhead=0.0,
            )

        assert "states a row width for other groups" in str(caught.value)
        assert "below schema 5" not in str(caught.value)

    def test_a_map_recording_no_width_at_all_names_the_schema(self) -> None:
        raw = make_map([(QWEN_UP, 160_000, CURVE)], precisions=PRECISIONS)
        raw["scan"]["group_by"] = "stack"

        with pytest.raises(SizeSourceError) as caught:
            self.solve_map_only(raw)

        assert "records no row width at all" in str(caught.value)
        assert "below schema 5" in str(caught.value)

    def test_the_checkpoint_wins_where_both_state_a_width(self) -> None:
        # `pack` quantizes the checkpoint, so the plan must predict
        # against the width that file states. The map says 2048,
        # which would take the k-quant table.
        recipe = solve(
            map_from_dict(map_with_widths({QWEN_UP: 2048})),
            weight_budget_bytes=10**9,
            vram_budget_bytes=10**9 + 1000,
            kv_headroom_bytes=1000,
            runtime=LLAMA_CPP,
            pins={QWEN_UP: 4},
            row_widths={QWEN_UP: NEMOTRON_ROWS},
            format_overhead=0.0,
        )

        assert recipe.assignments[0].bytes == group_bytes(
            160_000, EXPERT_STACK_EFFECTIVE_BITS[LLAMA_CPP][4], 0.0
        )

    def test_a_vllm_plan_needs_no_width_at_all(self) -> None:
        # vLLM prices at nominal bits, so no table routes by width
        # and the refusal would block a solve it cannot improve.
        raw = make_map([(QWEN_UP, 160_000, CURVE)], precisions=PRECISIONS)
        raw["scan"]["group_by"] = "stack"

        recipe = solve(
            map_from_dict(raw),
            weight_budget_bytes=10**9,
            vram_budget_bytes=10**9 + 1000,
            kv_headroom_bytes=1000,
            runtime=VLLM,
            format_overhead=0.0,
        )

        assert recipe.assignments


@pytest.mark.unit
class TestTheMapAndTheCheckpointDisagree:
    """One of the two describes another checkpoint (#558)."""

    def plan_widths(
        self,
        tmp_path,
        widths: dict[str, tuple[int, int]],
        runtime: str | None = None,
    ):
        raw = map_with_widths({name: pair[0] for name, pair in widths.items()})
        map_path = tmp_path / "map.json"
        map_path.write_text(json.dumps(raw))
        model_dir = write_checkpoint(
            tmp_path / "ckpt",
            {f"{name}.weight": [4, pair[1]] for name, pair in widths.items()},
        )
        argv = [
            "plan",
            str(map_path),
            "--vram",
            "24GiB",
            "--checkpoint",
            str(model_dir),
            "--out",
            str(tmp_path / "recipe.json"),
        ]
        if runtime is not None:
            argv += ["--runtime", runtime]
        return runner.invoke(app, argv)

    def plan_against(
        self,
        tmp_path,
        map_width: int,
        checkpoint_width: int,
        runtime: str | None = None,
    ):
        return self.plan_widths(
            tmp_path, {QWEN_UP: (map_width, checkpoint_width)}, runtime
        )

    def contest_lines(self, result) -> list[str]:
        return [
            line
            for line in result.stderr.splitlines()
            if "different row widths" in line
        ]

    def test_a_contested_width_draws_a_warning_naming_both(self, tmp_path) -> None:
        result = self.plan_against(tmp_path, map_width=2048, checkpoint_width=2688)

        assert result.exit_code == 0
        # Pinned to stderr: the warning is emitted with `err=True`, and
        # `result.output` mixes both streams, so it would pass even if
        # the warning moved off the human channel.
        assert f'for group "{QWEN_UP}"' in result.stderr
        assert "the map records 2048 elements" in result.stderr
        assert "this checkpoint states 2688" in result.stderr

    def test_agreeing_widths_draw_no_warning(self, tmp_path) -> None:
        result = self.plan_against(tmp_path, map_width=2688, checkpoint_width=2688)

        assert result.exit_code == 0
        assert self.contest_lines(result) == []

    def test_many_contested_groups_draw_one_line(self, tmp_path) -> None:
        # A stack map against a same-family checkpoint of another size
        # contests every routed group. A line each would bury the
        # merged-projection, held-class, and missing-group warnings
        # this module prints after it.
        result = self.plan_widths(
            tmp_path, {QWEN_UP: (2048, 2688), QWEN_DOWN: (768, 2688)}
        )

        assert result.exit_code == 0
        lines = self.contest_lines(result)
        assert len(lines) == 1
        assert f'2 groups, the first being "{QWEN_DOWN}"' in lines[0]
        assert "the map records 768 elements" in lines[0]
        assert "this checkpoint states 2688" in lines[0]

    def test_the_warning_states_precedence_and_not_a_price(self, tmp_path) -> None:
        # vLLM has no effective-bits table, so the solver prices every
        # group at nominal bits and neither width moves a byte. The
        # warning must still print and still ask its question, without
        # claiming the plan priced the checkpoint's width.
        result = self.plan_against(
            tmp_path, map_width=2048, checkpoint_width=2688, runtime=VLLM
        )

        assert result.exit_code == 0
        assert "the map records 2048 elements" in result.stderr
        assert "this checkpoint states 2688" in result.stderr
        assert "takes precedence" in result.stderr
        assert "is this the checkpoint the scan measured?" in result.stderr
        assert "prices the checkpoint's width" not in result.stderr


@pytest.mark.unit
class TestTheDomainGatesAnUnreachedWidth:
    """A width reaches routing only for a group routing reaches.

    The JSON reader refuses a width on a group the 256 super-block
    decision does not reach, so this covers the map a caller builds
    in memory. Without the domain gate, a `group_by: layer` map
    carrying a row width on a whole-layer group prices that layer
    through the ADR-0028 expert-stack table, while `pack` gates on
    `consults_row_width` and uses the k-quant table — a silent
    misprice with no refusal.
    """

    LAYER = "model.layers.0"

    def layer_map_with_width(self) -> SensitivityMap:
        map_ = map_from_dict(
            make_map([(self.LAYER, 160_000, CURVE)], precisions=PRECISIONS)
        )
        group = replace(map_.groups[0], row_width=NEMOTRON_ROWS)
        return replace(map_, groups=(group,))

    def test_a_width_on_a_whole_layer_group_is_dropped(self) -> None:
        assert map_row_widths(self.layer_map_with_width()) == {}

    def test_a_whole_layer_group_prices_through_the_k_quant_table(self) -> None:
        recipe = solve(
            self.layer_map_with_width(),
            weight_budget_bytes=10**9,
            vram_budget_bytes=10**9 + 1000,
            kv_headroom_bytes=1000,
            runtime=LLAMA_CPP,
            pins={self.LAYER: 4},
            format_overhead=0.0,
        )

        assert recipe.assignments[0].bytes == group_bytes(
            160_000, EFFECTIVE_BITS[LLAMA_CPP][4], 0.0
        )
