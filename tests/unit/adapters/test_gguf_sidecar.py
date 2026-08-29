"""The projector sidecar ships byte-identical (ADR-0030 decision 2).

Drives `ship_sidecar` over real files, so the copy, the hash proof,
and the refusals are the unit under test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vramfit.adapters.outbound.gguf import sidecar
from vramfit.adapters.outbound.gguf.sidecar import ship_sidecar

pytestmark = pytest.mark.unit

MMPROJ_BYTES = b"GGUF-mmproj-payload"


def write_mmproj(tmp_path: Path) -> Path:
    source_dir = tmp_path / "vendor"
    source_dir.mkdir()
    mmproj = source_dir / "mmproj-model-f16.gguf"
    mmproj.write_bytes(MMPROJ_BYTES)
    return mmproj


def test_ship_sidecar_copies_the_file_beside_the_artifact(tmp_path: Path) -> None:
    mmproj = write_mmproj(tmp_path)
    out = tmp_path / "packed.gguf"

    result = ship_sidecar(mmproj, beside=out)

    assert result.path == tmp_path / "mmproj-model-f16.gguf"
    assert result.path.read_bytes() == MMPROJ_BYTES
    assert result.n_bytes == len(MMPROJ_BYTES)


def test_ship_sidecar_digest_matches_the_source_bytes(tmp_path: Path) -> None:
    mmproj = write_mmproj(tmp_path)
    out = tmp_path / "packed.gguf"

    result = ship_sidecar(mmproj, beside=out)

    assert result.sha256 == hashlib.sha256(MMPROJ_BYTES).hexdigest()


def test_ship_sidecar_source_already_beside_out_skips_the_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(MMPROJ_BYTES)
    out = tmp_path / "packed.gguf"

    def refuse_copy(src: str, dst: str) -> None:
        raise AssertionError("the in-place source must not be copied")

    monkeypatch.setattr(sidecar.shutil, "copyfile", refuse_copy)

    result = ship_sidecar(mmproj, beside=out)

    assert result.path == mmproj
    assert result.path.read_bytes() == MMPROJ_BYTES
    assert result.sha256 == hashlib.sha256(MMPROJ_BYTES).hexdigest()


def test_ship_sidecar_replaces_a_stale_copy(tmp_path: Path) -> None:
    mmproj = write_mmproj(tmp_path)
    out = tmp_path / "packed.gguf"
    stale = tmp_path / "mmproj-model-f16.gguf"
    stale.write_bytes(b"stale bytes from an earlier run")

    result = ship_sidecar(mmproj, beside=out)

    assert result.path.read_bytes() == MMPROJ_BYTES


def test_ship_sidecar_with_the_out_file_name_refuses(tmp_path: Path) -> None:
    source_dir = tmp_path / "vendor"
    source_dir.mkdir()
    mmproj = source_dir / "packed.gguf"
    mmproj.write_bytes(MMPROJ_BYTES)
    out = tmp_path / "packed.gguf"

    with pytest.raises(ValueError, match="overwrite the decoder"):
        ship_sidecar(mmproj, beside=out)


def test_ship_sidecar_dangling_symlink_destination_refuses(tmp_path: Path) -> None:
    # `copyfile` follows the link and would land the payload at its
    # target, outside the artifact directory.
    mmproj = write_mmproj(tmp_path)
    out = tmp_path / "packed.gguf"
    (tmp_path / "mmproj-model-f16.gguf").symlink_to(tmp_path / "elsewhere.bin")

    with pytest.raises(RuntimeError, match="is a symlink"):
        ship_sidecar(mmproj, beside=out)
    assert not (tmp_path / "elsewhere.bin").exists()


def test_ship_sidecar_symlink_to_the_source_refuses(tmp_path: Path) -> None:
    # A link would ship where the record promises a copy.
    mmproj = write_mmproj(tmp_path)
    out = tmp_path / "packed.gguf"
    (tmp_path / "mmproj-model-f16.gguf").symlink_to(mmproj)

    with pytest.raises(RuntimeError, match="is a symlink"):
        ship_sidecar(mmproj, beside=out)


def test_ship_sidecar_hash_mismatch_refuses_and_removes_the_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A copy that lands different bytes must refuse and leave no
    # wrong-byte file under the vendor name (ADR-0030 decision 2).
    mmproj = write_mmproj(tmp_path)
    out = tmp_path / "packed.gguf"

    def corrupt_copy(src: str, dst: str) -> None:
        Path(dst).write_bytes(b"corrupted")

    monkeypatch.setattr(sidecar.shutil, "copyfile", corrupt_copy)

    with pytest.raises(RuntimeError, match="did not match the source"):
        ship_sidecar(mmproj, beside=out)
    assert not (tmp_path / "mmproj-model-f16.gguf").exists()
    assert mmproj.read_bytes() == MMPROJ_BYTES
