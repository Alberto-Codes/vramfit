from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

import pytest

from quantfit.domain.errors import QuantfitError
from quantfit.domain.runtime import (
    RUNTIME_CAPABILITIES,
    RuntimeCapabilityError,
    serveable_precisions,
)

pytestmark = pytest.mark.unit


class TestServeablePrecisions:
    def test_llama_cpp_serves_the_full_scan_set(self) -> None:
        assert serveable_precisions((8, 6, 5, 4, 3, 2), "llama.cpp") == (
            8,
            6,
            5,
            4,
            3,
            2,
        )

    def test_vllm_filters_to_its_kernel_set(self) -> None:
        assert serveable_precisions((8, 4, 3, 2), "vllm") == (8, 4)

    def test_filter_preserves_descending_order(self) -> None:
        assert serveable_precisions((8, 5, 2), "llama.cpp") == (8, 5, 2)

    def test_unknown_runtime_raises_capability_error(self) -> None:
        with pytest.raises(RuntimeCapabilityError, match='unknown runtime "tgi"'):
            serveable_precisions((8, 4), "tgi")

    def test_empty_intersection_raises_capability_error(self) -> None:
        with pytest.raises(RuntimeCapabilityError, match="serves none"):
            serveable_precisions((3, 2), "vllm")

    def test_capability_error_inherits_the_quantfit_root(self) -> None:
        assert issubclass(RuntimeCapabilityError, QuantfitError)
        assert issubclass(RuntimeCapabilityError, ValueError)

    def test_capability_table_is_read_only(self) -> None:
        table = cast("MutableMapping[str, frozenset[int]]", RUNTIME_CAPABILITIES)

        with pytest.raises(TypeError):
            table["new"] = frozenset()
