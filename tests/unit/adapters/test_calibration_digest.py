from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vramfit.adapters.outbound.calibration_digest import calibration_identity

pytestmark = pytest.mark.unit


def test_identity_matches_the_files_sha256_and_size(tmp_path: Path) -> None:
    path = tmp_path / "calibration.txt"
    path.write_bytes(b"It is a truth universally acknowledged")

    digest, n_bytes = calibration_identity(path)

    assert (
        digest == hashlib.sha256(b"It is a truth universally acknowledged").hexdigest()
    )
    assert n_bytes == 38


def test_a_file_larger_than_one_slab_hashes_whole(tmp_path: Path) -> None:
    # The reader hashes in 1 MiB slabs, so a file spanning several
    # must still hash to one digest over every byte.
    payload = b"x" * ((1 << 20) * 2 + 17)
    path = tmp_path / "calibration.txt"
    path.write_bytes(payload)

    digest, n_bytes = calibration_identity(path)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert n_bytes == len(payload)


def test_reissued_bytes_behind_one_path_change_the_digest(tmp_path: Path) -> None:
    path = tmp_path / "calibration.txt"
    path.write_bytes(b"first issue")
    first, _ = calibration_identity(path)
    path.write_bytes(b"second issue")

    second, _ = calibration_identity(path)

    assert first != second


def test_a_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        calibration_identity(tmp_path / "absent.txt")
