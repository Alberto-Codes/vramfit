"""ImatrixCountSource contract: the gguf-py adapter and the fake agree.

The real side reads true GGUF imatrix files written with gguf-py —
the same library the adapter reads them back with — so the suite
stays hermetic (ADR-0009). No zero-count expert exists on the
north-star MoE target and its corpus (issue #162, 0 uncovered
across 2,944 cells), so every starved case here is written by hand.
That is the whole reason ADR-0026 decision 5 needs a fake.

The real-only tests below the shared contract pin what the fake
structurally cannot reach: which tensors of a file the reader
touches, and how it rounds a stored float count.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="pack extra not installed")
pytest.importorskip("gguf", reason="pack extra not installed")

from gguf import GGUFWriter

from tests.fakes import MemoryImatrixCounts
from vramfit.adapters.outbound.gguf.imatrix_counts import GgufImatrixCounts
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import ZeroCountExpert
from vramfit.ports.outbound import ImatrixCountSource

pytestmark = pytest.mark.contract

STACK = "blk.1.ffn_up_exps.weight"
OTHER_STACK = "blk.0.ffn_down_exps.weight"
DENSE = "blk.0.attn_v.weight"
COLUMNS = 8
EXPERTS = 4

# The starved cells this suite writes: expert 2 of one stack and
# expert 0 of another. Deliberately out of sorted order, so the
# contract pins the reader's ordering rather than the input's.
STARVED = (
    ZeroCountExpert(stack=STACK, expert=2),
    ZeroCountExpert(stack=OTHER_STACK, expert=0),
)


def write_imatrix(path: Path, counts: dict[str, list[float]]) -> None:
    """Write a GGUF imatrix holding the given per-matrix counts.

    Each entry gets the ``in_sum2``/``counts`` pair ``llama-imatrix``
    writes. The sums are ones — this reader never looks at them.
    """
    writer = GGUFWriter(path, arch="llama")
    writer.add_type("imatrix")
    for name, tally in counts.items():
        writer.add_tensor(
            f"{name}.in_sum2",
            np.ones((len(tally), COLUMNS), dtype=np.float32),
        )
        writer.add_tensor(f"{name}.counts", np.array(tally, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def healthy_counts() -> dict[str, list[float]]:
    """Every expert fired, which is what issue #162 measured."""
    return {
        DENSE: [421_370.0],
        STACK: [19_752.0] * EXPERTS,
        OTHER_STACK: [426.0] * EXPERTS,
    }


def starved_counts() -> dict[str, list[float]]:
    """Two experts the router never fired, in two different stacks."""
    counts = healthy_counts()
    counts[STACK][2] = 0.0
    counts[OTHER_STACK][0] = 0.0
    return counts


def _real_source(tmp_path: Path, starved: bool = False) -> ImatrixCountSource:
    path = tmp_path / "model.imatrix.gguf"
    write_imatrix(path, starved_counts() if starved else healthy_counts())
    return GgufImatrixCounts(imatrix=path)


def _fake_source(tmp_path: Path, starved: bool = False) -> ImatrixCountSource:
    return MemoryImatrixCounts(experts=STARVED if starved else ())


@pytest.mark.parametrize(
    "build", [_real_source, _fake_source], ids=["real-gguf-py", "fake-memory"]
)
class TestImatrixCountSourceContract:
    def test_matrix_covering_every_expert_reports_none(self, build, tmp_path) -> None:
        source = build(tmp_path)

        assert source.zero_count_experts() == ()

    def test_zero_count_expert_is_named_by_stack_and_index(
        self, build, tmp_path
    ) -> None:
        source = build(tmp_path, starved=True)

        assert source.zero_count_experts() == (
            ZeroCountExpert(stack=OTHER_STACK, expert=0),
            ZeroCountExpert(stack=STACK, expert=2),
        )

    def test_report_is_sorted_by_stack_then_expert(self, build, tmp_path) -> None:
        source = build(tmp_path, starved=True)

        experts = source.zero_count_experts()

        assert list(experts) == sorted(experts, key=lambda e: (e.stack, e.expert))

    def test_reading_twice_returns_the_same_report(self, build, tmp_path) -> None:
        source = build(tmp_path, starved=True)

        assert source.zero_count_experts() == source.zero_count_experts()


class TestGgufImatrixCountsReads:
    """Real-adapter reads the memory fake structurally cannot cover."""

    def test_dense_tensor_at_zero_is_not_an_expert_report(self, tmp_path) -> None:
        # One matrix means a dense tensor. A zero count there
        # flattens the whole tensor, which ADR-0016's uncovered
        # record and ADR-0023's exclusions already fence. Decision 5
        # names an expert inside a covered stack, so this stays out.
        path = tmp_path / "model.imatrix.gguf"
        counts = healthy_counts()
        counts[DENSE] = [0.0]
        write_imatrix(path, counts)

        assert GgufImatrixCounts(imatrix=path).zero_count_experts() == ()

    def test_every_expert_of_a_stack_at_zero_is_reported(self, tmp_path) -> None:
        path = tmp_path / "model.imatrix.gguf"
        counts = healthy_counts()
        counts[STACK] = [0.0] * EXPERTS
        write_imatrix(path, counts)

        experts = GgufImatrixCounts(imatrix=path).zero_count_experts()

        assert experts == tuple(
            ZeroCountExpert(stack=STACK, expert=i) for i in range(EXPERTS)
        )

    def test_count_below_half_rounds_to_zero_and_is_reported(self, tmp_path) -> None:
        # The file stores a float and the C loader tests the rounded
        # value (llama.cpp tools/imatrix/imatrix-loader.cpp:158).
        path = tmp_path / "model.imatrix.gguf"
        counts = healthy_counts()
        counts[STACK][1] = 0.4
        write_imatrix(path, counts)

        assert GgufImatrixCounts(imatrix=path).zero_count_experts() == (
            ZeroCountExpert(stack=STACK, expert=1),
        )

    def test_count_of_half_rounds_away_from_zero_and_is_not_reported(
        self, tmp_path
    ) -> None:
        # std::lround breaks the tie away from zero, so 0.5 is a
        # count of 1 and the expert prices assisted.
        path = tmp_path / "model.imatrix.gguf"
        counts = healthy_counts()
        counts[STACK][1] = 0.5
        write_imatrix(path, counts)

        assert GgufImatrixCounts(imatrix=path).zero_count_experts() == ()

    def test_unreadable_file_raises_pack_error_naming_it(self, tmp_path) -> None:
        # An empty result must mean a healthy matrix. A failed read
        # returning one would report the same thing.
        path = tmp_path / "not-a-gguf.imatrix.gguf"
        path.write_bytes(b"not a gguf file at all")

        with pytest.raises(PackError, match="not-a-gguf"):
            GgufImatrixCounts(imatrix=path).zero_count_experts()

    def test_missing_file_raises_pack_error_naming_it(self, tmp_path) -> None:
        path = tmp_path / "absent.imatrix.gguf"

        with pytest.raises(PackError, match="absent"):
            GgufImatrixCounts(imatrix=path).zero_count_experts()
