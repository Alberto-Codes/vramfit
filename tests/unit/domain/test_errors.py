from __future__ import annotations

import pytest

from vramfit.adapters.inbound.cli_scan import ScanExtraMissingError
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.domain.errors import VramfitError
from vramfit.domain.solver import InfeasibleBudgetError, PinError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("exc", "legacy_base"),
    [
        (PinError, ValueError),
        (InfeasibleBudgetError, Exception),
        (ArtifactError, ValueError),
        (ScanExtraMissingError, RuntimeError),
    ],
    ids=["pin", "infeasible", "artifact", "extra-missing"],
)
def test_every_vramfit_exception_inherits_the_root_and_its_legacy_base(
    exc: type[BaseException], legacy_base: type[BaseException]
) -> None:
    assert issubclass(exc, VramfitError)
    assert issubclass(exc, legacy_base)
