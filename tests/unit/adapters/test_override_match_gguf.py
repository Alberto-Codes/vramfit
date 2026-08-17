"""`base_tensor_names` against a written GGUF (#303).

The rest of the override matching runs without gguf-py in
`test_override_match.py`. Only the header read needs a real file, so
only this module carries the guard.

CI installs no extras on the default test jobs, so these skip there.
The `test-gguf` job installs gguf-py and selects `-m contract`, which
this module does not carry — ADR-0009 reserves `contract` for
verified-fake port suites and #207 owns whether that marker widens.
"""

# ruff: noqa: E402 - the importorskip guard must run before gguf imports

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="pack extra not installed")
pytest.importorskip("gguf", reason="pack extra not installed")

from gguf import GGUFWriter

from vramfit.adapters.outbound.gguf.override_match import (
    base_tensor_names,
    check_overrides_match,
)
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import TypeOverride

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def base_gguf_names() -> None:
    """Override the suite-wide stub — this module reads real files."""


def write_gguf(path: Path, names: tuple[str, ...]) -> None:
    writer = GGUFWriter(path, arch="llama")
    for name in names:
        writer.add_tensor(name, np.zeros((2, 2), dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestBaseTensorNames:
    def test_written_names_read_back_in_file_order(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_gguf(base, ("blk.0.attn_v.weight", "blk.1.attn_v.weight"))
        assert base_tensor_names(base) == (
            "blk.0.attn_v.weight",
            "blk.1.attn_v.weight",
        )

    def test_non_gguf_file_refuses(self, tmp_path: Path) -> None:
        bogus = tmp_path / "base.gguf"
        bogus.write_bytes(b"not a gguf")
        with pytest.raises(PackError, match="is not a GGUF"):
            base_tensor_names(bogus)


class TestCheckAgainstARealFile:
    def test_override_reaching_a_written_tensor_passes(self, tmp_path: Path) -> None:
        base = tmp_path / "base.gguf"
        write_gguf(base, ("blk.0.attn_v.weight",))
        check_overrides_match((TypeOverride(r"blk\.0\.", "q4_k"),), base)

    def test_override_naming_an_absent_layer_refuses(self, tmp_path: Path) -> None:
        # A recipe packed against the wrong checkpoint, or one naming
        # more layers than it carries.
        base = tmp_path / "base.gguf"
        write_gguf(base, ("blk.0.attn_v.weight", "blk.1.attn_v.weight"))
        overrides = (
            TypeOverride(r"blk\.7\.", "q4_k"),
            TypeOverride(r"blk\.8\.", "q2_k"),
        )
        with pytest.raises(PackError, match="no tensor for 2 of 2"):
            check_overrides_match(overrides, base)

    def test_prefixed_tree_matches_and_passes(self, tmp_path: Path) -> None:
        # `blk\.0\.` is a substring of `v.blk.0.attn_v.weight`, so the
        # quantizer's own search matches it. The check passes, and
        # #236 owns the resulting mis-application.
        base = tmp_path / "base.gguf"
        write_gguf(base, ("v.blk.0.attn_v.weight",))
        check_overrides_match((TypeOverride(r"blk\.0\.", "q4_k"),), base)
