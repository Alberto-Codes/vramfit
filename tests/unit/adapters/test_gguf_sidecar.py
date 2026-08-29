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


def test_ship_sidecar_copies_the_file_and_proves_the_hash(tmp_path: Path) -> None:
    source_dir = tmp_path / "vendor"
    source_dir.mkdir()
    mmproj = source_dir / "mmproj-model-f16.gguf"
    mmproj.write_bytes(MMPROJ_BYTES)
    out = tmp_path / "packed.gguf"

    result = ship_sidecar(mmproj, beside=out)

    assert result.path == tmp_path / "mmproj-model-f16.gguf"
    assert result.path.read_bytes() == MMPROJ_BYTES
    assert result.n_bytes == len(MMPROJ_BYTES)
    assert result.sha256 == hashlib.sha256(MMPROJ_BYTES).hexdigest()


def test_ship_sidecar_source_already_beside_out_skips_the_copy(
    tmp_path: Path,
) -> None:
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(MMPROJ_BYTES)
    out = tmp_path / "packed.gguf"

    result = ship_sidecar(mmproj, beside=out)

    assert result.path == mmproj
    assert result.path.read_bytes() == MMPROJ_BYTES
    assert result.sha256 == hashlib.sha256(MMPROJ_BYTES).hexdigest()


def test_ship_sidecar_replaces_a_stale_copy(tmp_path: Path) -> None:
    source_dir = tmp_path / "vendor"
    source_dir.mkdir()
    mmproj = source_dir / "mmproj-model-f16.gguf"
    mmproj.write_bytes(MMPROJ_BYTES)
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


def test_ship_sidecar_hash_mismatch_refuses(tmp_path: Path, monkeypatch) -> None:
    # A copy that lands different bytes must refuse, never ship — the
    # clause demands byte identity (ADR-0030 decision 2).
    source_dir = tmp_path / "vendor"
    source_dir.mkdir()
    mmproj = source_dir / "mmproj-model-f16.gguf"
    mmproj.write_bytes(MMPROJ_BYTES)
    out = tmp_path / "packed.gguf"

    def corrupt_copy(src: str, dst: str) -> None:
        Path(dst).write_bytes(b"corrupted")

    monkeypatch.setattr(sidecar.shutil, "copyfile", corrupt_copy)

    with pytest.raises(RuntimeError, match="does not match the source"):
        ship_sidecar(mmproj, beside=out)
