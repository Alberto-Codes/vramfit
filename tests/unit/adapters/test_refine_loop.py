"""Unit tests for the refinement pass's measure loop, on verified fakes."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import (
    MemoryRecipePacker,
    MemoryRuntimeDivergenceMeter,
    stack_row_widths,
)
from vramfit.adapters.inbound.refine_loop import run_pass, stride
from vramfit.domain.evals import CorpusReference
from vramfit.domain.model import (
    Assignment,
    LayerGroup,
    PlanMeta,
    Recipe,
    ScanMeta,
    SensitivityMap,
)
from vramfit.domain.refinement import neighbours
from vramfit.domain.refinement_record import CONTROL_ARM, MeasurementFrame

pytestmark = pytest.mark.unit

# Routed-expert stacks, the unit the C4 protocol swapped between. The
# GGUF backend maps these names, so the packer fake prices them the
# way the real adapter would.
G0 = "model.layers.0.mixer.experts.up_proj"
G1 = "model.layers.1.mixer.experts.up_proj"
G2 = "model.layers.2.mixer.experts.up_proj"
G3 = "model.layers.3.mixer.experts.up_proj"

PRECISIONS = (8, 4, 2)
SIZE_AT = {8: 800, 4: 400, 2: 200}
CONTROL_CHUNKS = (0.30, 0.30, 0.30, 0.30, 0.30, 0.30)


def _frame() -> MeasurementFrame:
    return MeasurementFrame(
        runtime_build="b10362",
        hardware="H100 SXM",
        corpus=CorpusReference(file="wiki.test.raw"),
        reference="f16 base logits",
        imatrix=None,
    )


def _map(names, curves=None):
    curves = curves or {
        name: {8: 0.0, 4: 0.1 * (i + 1), 2: 0.4 * (i + 1)}
        for i, name in enumerate(names)
    }
    return SensitivityMap(
        model_id="test/model",
        scan=ScanMeta(
            metric="kl_divergence",
            calibration="/work/calibration.txt",
            calibration_tokens=131072,
            precisions=PRECISIONS,
            group_by="stack",
            started_at="2026-07-27T00:00:00Z",
        ),
        groups=tuple(
            LayerGroup(
                name=name,
                tensors=(f"{name}.weight",),
                bytes_fp16=1600,
                sensitivity=curves[name],
            )
            for name in names
        ),
    )


def _recipe(bits, pins=None):
    assignments = tuple(
        Assignment(group=n, bits=b, bytes=SIZE_AT[b], damage=0.1)
        for n, b in bits.items()
    )
    plan = PlanMeta(
        vram_budget_bytes=10**9,
        kv_headroom_bytes=0,
        weight_budget_bytes=10**9,
        predicted_total_bytes=sum(a.bytes for a in assignments),
        predicted_damage=1.0,
        solver="greedy-damage-per-byte",
        pins=pins or {},
        protections={},
        format_overhead=0.0,
        trace=(),
    )
    return Recipe(
        model_id="test/model",
        plan=plan,
        assignments=assignments,
        runtime=None,
        within_group=None,
        imatrix=None,
        protected_tensors=(),
    )


def _packer_for(recorder):
    def build(path: str):
        recorder.append(path)
        return MemoryRecipePacker(
            packed_bytes=500,
            has_base=True,
            row_widths=stack_row_widths([G0, G1, G2, G3]),
            out_path=Path(path),
        )

    return build


def _run(tmp_path, meter, bits=None, limit=10, bar=1.0):
    bits = bits or {G0: 2, G1: 2, G2: 4, G3: 4}
    return run_pass(
        _recipe(bits),
        _map(list(bits)),
        _packer_for([]),
        meter,
        _frame(),
        bar=bar,
        limit=limit,
        out_dir=tmp_path,
        row_widths={},
    )


def test_stride_returns_every_candidate_when_the_limit_covers_them() -> None:
    recipe = _recipe({G0: 2, G1: 2, G2: 4, G3: 4})
    found = neighbours(recipe, _map([G0, G1, G2, G3]), {})
    assert stride(found, 10) == found


def test_stride_spreads_across_the_enumeration() -> None:
    recipe = _recipe({G0: 2, G1: 2, G2: 4, G3: 4})
    found = neighbours(recipe, _map([G0, G1, G2, G3]), {})
    taken = stride(found, 2)
    assert len(taken) == 2
    assert taken[0] == found[0]
    assert taken[1] == found[len(found) // 2]


def test_stride_refuses_a_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        stride((), 0)


def test_run_pass_measures_the_control_first(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    _run(tmp_path, meter)

    assert meter.measured[0].endswith(f"{CONTROL_ARM}.gguf")


def test_run_pass_measures_every_arm_it_selected(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    sidecar = _run(tmp_path, meter, limit=3)

    assert len(sidecar.arms) == 3
    # One control plus three arms.
    assert len(meter.measured) == 4


def test_run_pass_keeps_the_arm_that_measured_best(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(
        default=CONTROL_CHUNKS,
        series={
            str(tmp_path / "arm01.gguf"): tuple(c - 0.01 for c in CONTROL_CHUNKS),
            str(tmp_path / "arm02.gguf"): tuple(c - 0.05 for c in CONTROL_CHUNKS),
        },
    )

    sidecar = _run(tmp_path, meter, limit=3)

    assert sidecar.winner == "arm02"
    assert sidecar.declined is None


def test_run_pass_records_every_arm_including_the_losers(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(
        default=CONTROL_CHUNKS,
        series={str(tmp_path / "arm01.gguf"): tuple(c + 0.05 for c in CONTROL_CHUNKS)},
    )

    sidecar = _run(tmp_path, meter, limit=3)

    assert {a.arm for a in sidecar.arms} == {"arm01", "arm02", "arm03"}
    assert sidecar.arms[0].delta > 0


def test_run_pass_keeps_nothing_when_no_arm_clears_the_bar(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    sidecar = _run(tmp_path, meter, limit=2, bar=7.8)

    assert sidecar.winner is None
    assert sidecar.declined is None


def test_run_pass_records_the_control_measurement(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    sidecar = _run(tmp_path, meter, limit=2)

    assert sidecar.control is not None
    assert sidecar.control.arm == CONTROL_ARM
    assert sidecar.control.delta == pytest.approx(0.0)
    assert sidecar.control.mean == pytest.approx(0.30)


def test_run_pass_declines_a_recipe_with_no_legal_swap(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    sidecar = _run(tmp_path, meter, bits={G0: 4, G1: 4, G2: 4})

    assert sidecar.declined is not None
    assert sidecar.arms == ()
    assert sidecar.winner is None
    assert sidecar.control is None


def test_run_pass_declining_measures_nothing_at_all(tmp_path) -> None:
    """The 49B case cost 0.00 dollars, and the loop must keep it that way."""
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    _run(tmp_path, meter, bits={G0: 4, G1: 4, G2: 4})

    assert meter.measured == []


def test_run_pass_deletes_each_pack_by_default(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)
    (tmp_path / f"{CONTROL_ARM}.gguf").write_bytes(b"stale")

    _run(tmp_path, meter, limit=2)

    assert not (tmp_path / f"{CONTROL_ARM}.gguf").exists()


def test_run_pass_reports_its_progress(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)
    seen: list[str] = []

    run_pass(
        _recipe({G0: 2, G1: 2, G2: 4, G3: 4}),
        _map([G0, G1, G2, G3]),
        _packer_for([]),
        meter,
        _frame(),
        bar=1.0,
        limit=1,
        out_dir=tmp_path,
        row_widths={},
        report=lambda event, fields: seen.append(event),
    )

    assert seen[0] == "refine_started"
    assert "arm_packed" in seen
    assert seen[-1] == "refine_finished"


def test_run_pass_arm_recipes_drop_the_solver_trace(tmp_path) -> None:
    """A refined recipe's trace no longer explains it (ADR-0031 decision 4)."""
    packed: list = []

    def build(path: str):
        packer = MemoryRecipePacker(
            packed_bytes=500,
            has_base=True,
            row_widths=stack_row_widths([G0, G1, G2, G3]),
        )
        packed.append(packer)
        return packer

    run_pass(
        _recipe({G0: 2, G1: 2, G2: 4, G3: 4}),
        _map([G0, G1, G2, G3]),
        build,
        MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS),
        _frame(),
        bar=1.0,
        limit=1,
        out_dir=tmp_path,
        row_widths={},
    )

    arm = packed[-1].packed[-1]
    assert arm.plan.trace == ()
    assert arm.plan.solver.endswith("+refine")


def test_run_pass_arm_recipes_keep_the_control_byte_total(tmp_path) -> None:
    packed: list = []

    def build(path: str):
        packer = MemoryRecipePacker(
            packed_bytes=500,
            has_base=True,
            row_widths=stack_row_widths([G0, G1, G2, G3]),
        )
        packed.append(packer)
        return packer

    recipe = _recipe({G0: 2, G1: 2, G2: 4, G3: 4})
    run_pass(
        recipe,
        _map([G0, G1, G2, G3]),
        build,
        MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS),
        _frame(),
        bar=1.0,
        limit=4,
        out_dir=tmp_path,
        row_widths={},
    )

    total = sum(a.bytes for a in recipe.assignments)
    for packer in packed[1:]:
        arm = packer.packed[-1]
        assert sum(a.bytes for a in arm.assignments) == total
        assert arm.plan.predicted_total_bytes == total


def test_a_strided_pass_records_the_whole_neighbourhood(tmp_path) -> None:
    """15 arms of 15 and 15 arms of 385 must not read alike."""
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    sidecar = _run(tmp_path, meter, limit=1)

    assert len(sidecar.arms) == 1
    assert sidecar.neighbourhood_moves == 4


def test_an_unstrided_pass_counts_its_arms_and_its_neighbourhood_alike(
    tmp_path,
) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    sidecar = _run(tmp_path, meter, limit=10)

    assert len(sidecar.arms) == sidecar.neighbourhood_moves == 4


def test_an_empty_neighbourhood_declines_at_zero(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    sidecar = _run(tmp_path, meter, bits={G0: 4, G1: 4, G2: 4})

    assert sidecar.declined is not None
    assert sidecar.neighbourhood_moves == 0


def test_a_pin_miss_decline_records_the_moves_it_enumerated(tmp_path) -> None:
    """A pin the map cannot resolve is not an empty neighbourhood."""
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)
    bits = {G0: 2, G1: 2, G2: 4, G3: 4}
    recipe = _recipe(bits, pins={"model.layers.9.mixer.in_proj": 8})

    sidecar = run_pass(
        recipe,
        _map(list(bits)),
        _packer_for([]),
        meter,
        _frame(),
        bar=1.0,
        limit=10,
        out_dir=tmp_path,
        row_widths={},
    )

    assert sidecar.declined is not None
    assert "cannot resolve pin" in sidecar.declined
    assert sidecar.neighbourhood_moves == 4
    assert sidecar.arms == ()


def test_refine_declined_carries_the_enumerated_count(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)
    bits = {G0: 2, G1: 2, G2: 4, G3: 4}
    seen: list[tuple[str, dict]] = []

    run_pass(
        _recipe(bits, pins={"model.layers.9.mixer.in_proj": 8}),
        _map(list(bits)),
        _packer_for([]),
        meter,
        _frame(),
        bar=1.0,
        limit=10,
        out_dir=tmp_path,
        row_widths={},
        report=lambda event, fields: seen.append((event, dict(fields))),
    )

    declined = next(fields for event, fields in seen if event == "refine_declined")
    assert declined["neighbourhood_moves"] == 4


def test_refine_started_carries_both_counts(tmp_path) -> None:
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)
    seen: list[tuple[str, dict]] = []

    run_pass(
        _recipe({G0: 2, G1: 2, G2: 4, G3: 4}),
        _map([G0, G1, G2, G3]),
        _packer_for([]),
        meter,
        _frame(),
        bar=1.0,
        limit=1,
        out_dir=tmp_path,
        row_widths={},
        report=lambda event, fields: seen.append((event, dict(fields))),
    )

    started = next(fields for event, fields in seen if event == "refine_started")
    assert started["arms"] == 1
    assert started["neighbourhood_moves"] == 4


def test_the_loop_drops_every_arm_pack_it_measured(tmp_path) -> None:
    """A sixteen-arm 30B pass must not fill a rented pod's disk."""
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    _run(tmp_path, meter, limit=3)

    assert meter.measured
    assert list(tmp_path.glob("*.gguf")) == []


def test_the_loop_drops_an_arm_pack_only_after_measuring_it(tmp_path) -> None:
    """Deleting before the meter reads it would measure nothing."""
    meter = MemoryRuntimeDivergenceMeter(default=CONTROL_CHUNKS)

    _run(tmp_path, meter, limit=3)

    assert meter.present
    assert all(meter.present)
