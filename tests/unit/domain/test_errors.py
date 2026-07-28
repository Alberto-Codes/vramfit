from __future__ import annotations

import pytest

from quantfit.adapters.inbound.cli_scan import ScanExtraMissingError
from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.domain.errors import QuantfitError
from quantfit.domain.solver import InfeasibleBudgetError, PinError

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
def test_every_quantfit_exception_inherits_the_root_and_its_legacy_base(
    exc: type[BaseException], legacy_base: type[BaseException]
) -> None:
    assert issubclass(exc, QuantfitError)
    assert issubclass(exc, legacy_base)
