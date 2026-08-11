"""DamageMeter contract: the torch meter and the memory fake agree.

The fake side is hermetic and runs in the default suite. The real side
loads torch and a tiny random checkpoint, so it carries the
``integration`` and ``slow`` markers and runs where the scan extra is
installed (ADR-0009: expensive ports prove their fakes out of band).
"""

from __future__ import annotations

import math

import pytest

from tests.conftest import CALIBRATION_TEXT
from tests.fakes import MemoryDamageMeter
from vramfit.domain.scan import GroupSpec
from vramfit.ports.outbound import DamageMeter

pytestmark = pytest.mark.contract

FAKE_SPECS = (
    GroupSpec(name="model.layers.0", tensors=("model.layers.0.w",), bytes_fp16=1000),
    GroupSpec(name="model.layers.1", tensors=("model.layers.1.w",), bytes_fp16=2000),
)
FAKE_DAMAGES = {
    (spec.name, bits): damage
    for spec in FAKE_SPECS
    for bits, damage in {8: 0.0001, 4: 0.01, 3: 0.1, 2: 1.0}.items()
}


@pytest.fixture(
    params=[
        pytest.param("fake", id="fake-memory"),
        pytest.param(
            "real",
            id="real-torch",
            marks=[pytest.mark.integration, pytest.mark.slow],
        ),
        pytest.param(
            "offloaded",
            id="real-torch-offloaded",
            marks=[pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu],
        ),
    ]
)
def meter(request, tmp_path) -> DamageMeter:
    if request.param == "fake":
        return MemoryDamageMeter(
            specs=FAKE_SPECS, damages=dict(FAKE_DAMAGES), tokens=64
        )
    if request.param == "offloaded":
        # The port contract must hold when groups measure through the
        # weights map (ADR-0015) — same behavior, different devices.
        return request.getfixturevalue("offloaded_contract_meter")
    tiny = request.getfixturevalue("tiny_model_dir")
    from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

    calibration = tmp_path / "calib.txt"
    calibration.write_text(CALIBRATION_TEXT)
    return TorchDamageMeter(str(tiny), calibration, max_tokens=128, device="cpu")


@pytest.fixture(scope="session")
def offloaded_contract_meter(offload_model_dir, tmp_path_factory) -> DamageMeter:
    """One capped meter for the whole contract run — loads are expensive.

    Sharing is safe: every port operation restores the model, and the
    suite asserts exactly that.
    """
    torch = pytest.importorskip("torch", reason="scan extra not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from tests.conftest import OFFLOAD_GPU_CAP
    from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

    calibration = tmp_path_factory.mktemp("contract-calib") / "calib.txt"
    calibration.write_text(CALIBRATION_TEXT)
    return TorchDamageMeter(
        str(offload_model_dir),
        calibration,
        max_tokens=128,
        device="auto",
        max_gpu_memory=OFFLOAD_GPU_CAP,
    )


def test_groups_are_unique_and_positively_sized(meter: DamageMeter) -> None:
    specs = meter.groups()

    names = [spec.name for spec in specs]
    assert len(specs) > 0
    assert len(set(names)) == len(names)
    assert all(spec.bytes_fp16 > 0 for spec in specs)
    assert all(spec.tensors for spec in specs)


def test_calibration_tokens_is_positive(meter: DamageMeter) -> None:
    assert meter.calibration_tokens() > 0


def test_measure_returns_finite_non_negative_damage(meter: DamageMeter) -> None:
    group = meter.groups()[0].name

    damage = meter.measure(group, 8)

    assert math.isfinite(damage)
    assert damage >= 0.0


def test_measure_is_deterministic_and_restores_the_model(meter: DamageMeter) -> None:
    group = meter.groups()[0].name

    coarse = meter.measure(group, 2)
    fine = meter.measure(group, 8)
    again = meter.measure(group, 2)

    assert coarse == again
    assert fine <= coarse


def test_two_bit_damage_is_at_least_eight_bit_damage(meter: DamageMeter) -> None:
    group = meter.groups()[-1].name

    curve = {bits: meter.measure(group, bits) for bits in (8, 2)}

    assert 0.0 <= curve[8] <= curve[2]


def test_measure_on_unknown_group_raises_value_error(meter: DamageMeter) -> None:
    with pytest.raises(ValueError, match="unknown group"):
        meter.measure("no.such.group", 8)


def test_measure_below_two_bits_raises_value_error(meter: DamageMeter) -> None:
    group = meter.groups()[0].name

    with pytest.raises(ValueError, match="bits"):
        meter.measure(group, 1)


def test_measure_recipe_returns_finite_non_negative_damage(
    meter: DamageMeter,
) -> None:
    recipe = {spec.name: 8 for spec in meter.groups()}

    damage = meter.measure_recipe(recipe)

    assert math.isfinite(damage)
    assert damage >= 0.0


def test_measure_recipe_on_one_group_equals_measure(meter: DamageMeter) -> None:
    group = meter.groups()[0].name

    assert meter.measure_recipe({group: 4}) == meter.measure(group, 4)


def test_measure_recipe_restores_the_model(meter: DamageMeter) -> None:
    group = meter.groups()[0].name
    before = meter.measure(group, 2)

    meter.measure_recipe({spec.name: 4 for spec in meter.groups()})

    assert meter.measure(group, 2) == before


def test_measure_recipe_coarse_damages_at_least_fine(meter: DamageMeter) -> None:
    specs = meter.groups()

    fine = meter.measure_recipe({spec.name: 8 for spec in specs})
    coarse = meter.measure_recipe({spec.name: 2 for spec in specs})

    assert 0.0 <= fine <= coarse


def test_measure_recipe_on_unknown_group_raises_value_error(
    meter: DamageMeter,
) -> None:
    with pytest.raises(ValueError, match="unknown group"):
        meter.measure_recipe({"no.such.group": 8})


def test_measure_recipe_below_two_bits_raises_value_error(
    meter: DamageMeter,
) -> None:
    group = meter.groups()[0].name

    with pytest.raises(ValueError, match="bits"):
        meter.measure_recipe({group: 1})


def test_measure_recipe_checks_group_names_before_bits(meter: DamageMeter) -> None:
    group = meter.groups()[0].name

    with pytest.raises(ValueError, match="unknown group"):
        meter.measure_recipe({group: 1, "no.such.group": 8})


def test_measure_recipe_with_no_assignments_raises_value_error(
    meter: DamageMeter,
) -> None:
    with pytest.raises(ValueError, match="empty"):
        meter.measure_recipe({})
