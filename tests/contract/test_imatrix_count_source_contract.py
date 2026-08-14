"""ImatrixCountSource contract: the gguf-py adapter and the fake agree.

The real side reads true GGUF files written with gguf-py — the same
library the adapter reads with — so the suite stays hermetic
(ADR-0009). The #198 amendment's closed refusal list, the silent
skip, and the half-up rounding are real-only pins below the shared
contract: the fake holds no file to malform. No zero-count expert
exists on the chart's model and corpus (#162), so every starved case
here is written by hand.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="pack extra not installed")
pytest.importorskip("gguf", reason="pack extra not installed")

from gguf import GGUFWriter

from tests.fakes import MemoryImatrixCountSource
from vramfit.adapters.outbound.gguf.imatrix_counts import GgufImatrixCounts
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.ports.outbound import ImatrixCountSource

pytestmark = pytest.mark.contract

STACK = "blk.0.ffn_up_exps.weight"
DENSE = "blk.0.attn_v.weight"
# Numpy shape (experts, rows, columns) writes GGUF ne (columns,
# rows, experts) — the adapter reads the expert count from ne[2].
EXPERTS = 2
STACK_SHAPE = (EXPERTS, 4, 8)
DENSE_SHAPE = (4, 8)
# Expert 1 is the hand-written starved case.
STACK_COUNTS = (3.0, 0.0)
DENSE_COUNT = (5.0,)


def write_base(path: Path, extra: dict[str, tuple[int, ...]] | None = None) -> None:
    writer = GGUFWriter(path, arch="llama")
    shapes = {STACK: STACK_SHAPE, DENSE: DENSE_SHAPE, **(extra or {})}
    for name, shape in shapes.items():
        writer.add_tensor(name, np.zeros(shape, dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_imatrix(
    path: Path,
    counts: dict[str, tuple[float, ...]],
    sums: dict[str, np.ndarray] | None = None,
    general_type: str | None = "imatrix",
    raw: dict[str, np.ndarray] | None = None,
) -> None:
    writer = GGUFWriter(path, arch="llama")
    if general_type is not None:
        writer.add_type(general_type)
    if sums is None:
        sums = {
            name: np.ones((len(values), 8), dtype=np.float32)
            for name, values in counts.items()
        }
    for name, data in sums.items():
        writer.add_tensor(f"{name}.in_sum2", data)
    for name, values in counts.items():
        writer.add_tensor(f"{name}.counts", np.array(values, dtype=np.float32))
    for name, data in (raw or {}).items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _real_source(tmp_path: Path) -> ImatrixCountSource:
    base = tmp_path / "base.gguf"
    matrix = tmp_path / "model.imatrix.gguf"
    write_base(base)
    write_imatrix(matrix, {STACK: STACK_COUNTS, DENSE: DENSE_COUNT})
    return GgufImatrixCounts(imatrix=matrix, base_gguf=base)


def _fake_source(tmp_path: Path) -> ImatrixCountSource:
    return MemoryImatrixCountSource(stack_counts={STACK: (3, 0)})


def _failing_real(tmp_path: Path) -> ImatrixCountSource:
    base = tmp_path / "base.gguf"
    matrix = tmp_path / "model.imatrix.gguf"
    write_base(base)
    write_imatrix(matrix, {STACK: STACK_COUNTS}, general_type=None)
    return GgufImatrixCounts(imatrix=matrix, base_gguf=base)


def _failing_fake(tmp_path: Path) -> ImatrixCountSource:
    return MemoryImatrixCountSource(fail=True)


@pytest.mark.parametrize(
    "build", [_real_source, _fake_source], ids=["real-gguf-py", "fake-memory"]
)
class TestImatrixCountSourceContract:
    def test_expert_stack_reads_one_count_per_expert(self, build, tmp_path) -> None:
        source = build(tmp_path)

        counts = source.expert_stack_counts()

        assert counts[STACK] == (3, 0)

    def test_dense_entries_stay_out_of_the_result(self, build, tmp_path) -> None:
        source = build(tmp_path)

        counts = source.expert_stack_counts()

        assert set(counts) == {STACK}

    def test_zero_count_survives_the_read_verbatim(self, build, tmp_path) -> None:
        source = build(tmp_path)

        counts = source.expert_stack_counts()

        assert counts[STACK][1] == 0


@pytest.mark.parametrize(
    "build", [_failing_real, _failing_fake], ids=["real-gguf-py", "fake-memory"]
)
class TestImatrixCountSourceRefusal:
    def test_unvouchable_source_raises_pack_error(self, build, tmp_path) -> None:
        source = build(tmp_path)

        with pytest.raises(PackError, match="imatrix"):
            source.expert_stack_counts()


class TestRealReaderRefusals:
    """The #198 amendment's closed refusal list, pinned case by case."""

    def test_missing_general_type_refuses(self, tmp_path) -> None:
        source = _failing_real(tmp_path)

        with pytest.raises(PackError, match=r"general\.type"):
            source.expert_stack_counts()

    def test_non_gguf_imatrix_refuses(self, tmp_path) -> None:
        base = tmp_path / "base.gguf"
        write_base(base)
        bogus = tmp_path / "bogus.gguf"
        bogus.write_bytes(b"not a gguf")

        with pytest.raises(PackError, match="not a GGUF"):
            GgufImatrixCounts(imatrix=bogus, base_gguf=base).expert_stack_counts()

    def test_no_counts_tensor_refuses(self, tmp_path) -> None:
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(matrix, {}, sums={STACK: np.ones((EXPERTS, 8), dtype=np.float32)})

        with pytest.raises(PackError, match=r"no \.counts"):
            GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

    def test_unknown_suffix_refuses(self, tmp_path) -> None:
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(
            matrix,
            {STACK: STACK_COUNTS},
            raw={f"{DENSE}.in_sum3": np.ones(8, dtype=np.float32)},
        )

        with pytest.raises(PackError, match="in_sum3"):
            GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

    def test_sums_without_counts_twin_refuses(self, tmp_path) -> None:
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(
            matrix,
            {STACK: STACK_COUNTS},
            sums={
                STACK: np.ones((EXPERTS, 8), dtype=np.float32),
                DENSE: np.ones((1, 8), dtype=np.float32),
            },
        )

        with pytest.raises(PackError, match="no counts twin"):
            GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

    def test_negative_count_refuses(self, tmp_path) -> None:
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(matrix, {STACK: (3.0, -1.0)})

        with pytest.raises(PackError, match="negative or not finite"):
            GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

    def test_non_finite_count_refuses(self, tmp_path) -> None:
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(matrix, {STACK: (3.0, float("nan"))})

        with pytest.raises(PackError, match="negative or not finite"):
            GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

    def test_stack_count_length_mismatch_refuses(self, tmp_path) -> None:
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(matrix, {STACK: (3.0, 4.0, 5.0)})

        with pytest.raises(PackError, match="expects 2"):
            GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

    def test_dense_entry_with_two_counts_refuses(self, tmp_path) -> None:
        # PR #195's replaced rule read any 2+-count entry as a stack.
        # The base shape says DENSE is 2D, so two counts refuse.
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(matrix, {DENSE: (3.0, 4.0)})

        with pytest.raises(PackError, match="expects 1"):
            GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()


class TestRealReaderReads:
    def test_one_expert_stack_reads_as_a_stack(self, tmp_path) -> None:
        # The replaced PR #195 rule inferred a stack from 2+ counts
        # and missed this case. The base shape rule catches it.
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        one = "blk.1.ffn_down_exps.weight"
        write_base(base, extra={one: (1, 4, 8)})
        write_imatrix(matrix, {one: (7.0,)})

        counts = GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

        assert counts[one] == (7,)

    def test_entry_naming_no_base_tensor_skips_silently(self, tmp_path) -> None:
        # A matrix over an MTP-bearing export packs a backbone-only
        # base this way (the #198 amendment).
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(
            matrix,
            {STACK: STACK_COUNTS, "blk.9.ffn_up_exps.weight": (1.0, 2.0, 3.0)},
        )

        counts = GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

        assert set(counts) == {STACK}

    def test_counts_round_half_up_before_the_zero_test(self, tmp_path) -> None:
        # std::lround's tie break (imatrix-loader.cpp:158): 0.4 is a
        # zero count and 0.5 is not. Pack and scan pin the same tie.
        base = tmp_path / "base.gguf"
        matrix = tmp_path / "model.imatrix.gguf"
        write_base(base)
        write_imatrix(matrix, {STACK: (0.4, 0.5)})

        counts = GgufImatrixCounts(imatrix=matrix, base_gguf=base).expert_stack_counts()

        assert counts[STACK] == (0, 1)

    def test_non_gguf_base_refuses(self, tmp_path) -> None:
        matrix = tmp_path / "model.imatrix.gguf"
        write_imatrix(matrix, {STACK: STACK_COUNTS})
        bogus = tmp_path / "base.gguf"
        bogus.write_bytes(b"not a gguf")

        with pytest.raises(PackError, match="base GGUF"):
            GgufImatrixCounts(imatrix=matrix, base_gguf=bogus).expert_stack_counts()

    def test_missing_gguf_names_the_pack_extra(self, tmp_path, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def no_gguf(name, *args, **kwargs):
            if name == "gguf":
                raise ImportError("No module named 'gguf'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_gguf)
        source = GgufImatrixCounts(
            imatrix=tmp_path / "m.gguf", base_gguf=tmp_path / "b.gguf"
        )

        with pytest.raises(PackError, match="pack"):
            source.expert_stack_counts()
