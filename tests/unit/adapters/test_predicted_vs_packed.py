"""Predicted bytes match packed bytes on a synthetic layout (#409).

Publication #2 fit its budget on two cancelling errors: the passthrough
under-priced the F32 classes by 16.9 MB, and the residual overhead
over-priced the quantized classes by 33.8 MB. This suite writes a GGUF
with the tensor types a recipe drives, measures its bytes per type
through the pack step's own reader, and holds the solver's prediction
at zero overhead against them. A per-type equality cannot cancel.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

np = pytest.importorskip("numpy", reason="pack extra not installed")
pytest.importorskip("gguf", reason="pack extra not installed")

from gguf import GGMLQuantizationType, GGUFWriter

from tests.unit.conftest import make_map
from vramfit.adapters.outbound.gguf.file_type import read_layout
from vramfit.adapters.outbound.gguf.types import (
    expert_stack_type_for,
    tensor_overrides,
)
from vramfit.adapters.outbound.sensitivity_map_json import map_from_dict
from vramfit.domain.solver import solve

# gguf-py 0.19.0 cannot name Q2_0, so the id is written raw.
Q2_0_TYPE_ID = 42
TYPE_IDS = {
    "Q8_0": int(GGMLQuantizationType.Q8_0),
    "Q4_0": int(GGMLQuantizationType.Q4_0),
    "Q2_0": Q2_0_TYPE_ID,
    "F32": int(GGMLQuantizationType.F32),
}
# Bytes per 32-weight block (ggml block layouts), and F32's four
# bytes per weight.
BLOCK_BYTES = {"Q8_0": 34, "Q4_0": 18, "Q2_0": 9}

# The synthetic layout: recipe group, GGUF tensor, weights, and the
# type the pack drives. Weight counts are multiples of 32 and every
# tensor lands on the 32-byte alignment, so no padding blurs a type.
LAYOUT = (
    ("model.layers.1.mixer.experts.up_proj", "blk.1.ffn_up_exps.weight", 512, "Q4_0"),
    (
        "model.layers.1.mixer.experts.down_proj",
        "blk.1.ffn_down_exps.weight",
        1024,
        "Q2_0",
    ),
    ("model.layers.0.mixer.in_proj", "blk.0.ssm_in.weight", 512, "Q8_0"),
    ("model.layers.0.mixer.conv1d", "blk.0.ssm_conv1d.weight", 96, "F32"),
    ("model.layers.1.mixer.gate", "blk.1.ffn_gate_inp.weight", 128, "F32"),
)


def packed_bytes(weights: int, gguf_type: str) -> int:
    if gguf_type == "F32":
        return weights * 4
    return weights // 32 * BLOCK_BYTES[gguf_type]


def write_layout(path: Path) -> None:
    writer = GGUFWriter(path, arch="llama")
    writer.add_file_type(2)
    for _, tensor, weights, gguf_type in LAYOUT:
        raw = cast("GGMLQuantizationType", TYPE_IDS[gguf_type])
        data = np.zeros(packed_bytes(weights, gguf_type), dtype=np.int8)
        writer.add_tensor(tensor, data, raw_dtype=raw)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.unit
class TestPredictedMatchesPacked:
    def test_every_type_predicts_its_packed_bytes_exactly(self, tmp_path: Path) -> None:
        map_ = map_from_dict(
            make_map(
                [
                    (LAYOUT[0][0], 512 * 2, {4: 0.01, 2: 0.1}),
                    (LAYOUT[1][0], 1024 * 2, {4: 0.01, 2: 0.1}),
                ],
                precisions=(4, 2),
            )
        )
        discovered = {group: weights * 2 for group, _, weights, _ in LAYOUT}
        recipe = solve(
            map_,
            weight_budget_bytes=10**6,
            vram_budget_bytes=10**6 + 1000,
            kv_headroom_bytes=1000,
            runtime="llama.cpp",
            discovered_bytes=discovered,
            pins={LAYOUT[1][0]: 2, LAYOUT[2][0]: 8},
            format_overhead=0.0,
        )
        types_by_group = {group: gguf_type for group, _, _, gguf_type in LAYOUT}
        predicted: dict[str, int] = {}
        for assignment in recipe.assignments:
            gguf_type = types_by_group[assignment.group]
            predicted[gguf_type] = predicted.get(gguf_type, 0) + assignment.bytes

        packed = tmp_path / "packed.gguf"
        write_layout(packed)
        layout = read_layout(packed)

        assert layout.bytes_by_type == predicted
        assert sum(layout.bytes_by_type.values()) == recipe.plan.predicted_total_bytes

    def test_the_layout_drives_the_types_the_pack_would(self) -> None:
        # The synthetic layout's types are the ones the pack step
        # maps: the stack table for the expert stacks and the
        # super-block-refused class, no override for the F32 classes.
        assert expert_stack_type_for(4, LAYOUT[0][0]) == "q4_0"
        assert expert_stack_type_for(2, LAYOUT[1][0]) == "q2_0"
        map_ = map_from_dict(
            make_map([(LAYOUT[2][0], 1024, {8: 0.0})], precisions=(8,))
        )
        recipe = solve(
            map_,
            weight_budget_bytes=10**6,
            vram_budget_bytes=10**6 + 1000,
            kv_headroom_bytes=1000,
            runtime="llama.cpp",
            discovered_bytes={LAYOUT[3][0]: 192, LAYOUT[4][0]: 256},
        )
        overrides = tensor_overrides(recipe)
        assert [o.quant_type for o in overrides] == ["q8_0"]
