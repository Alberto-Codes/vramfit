"""Ship the projector sidecar beside the packed decoder GGUF.

The artifact ships the vendor mmproj beside the decoder GGUF,
byte-identical (ADR-0030 decision 2). This module copies the file
and proves the copy: it hashes the source and the copy with SHA-256
and refuses a mismatch. The sidecar stays unquantized until #419
prices the quantized alternative — the copy is the whole mechanism.

The hash also serves publication: the sidecar reaches hashing and
upload beside the decoder GGUF (ADR-0030 consequences), and the
run log records the digest this module computes.

Examples:
    Ship an mmproj beside a packed artifact:

    ```python
    from pathlib import Path

    from vramfit.adapters.outbound.gguf.sidecar import ship_sidecar

    result = ship_sidecar(Path("mmproj.gguf"), beside=Path("packed.gguf"))
    print(result.sha256)
    ```

See Also:
    - [vramfit.adapters.inbound.cli_pack][]: Wires this into the
      ``pack`` command's ``--mmproj`` option.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

# Hash in 1 MiB slabs: the Gemma 4 31B mmproj is 1.118 GiB, and a
# whole-file read would hold it in memory twice.
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class SidecarResult:
    """One shipped sidecar: where it landed and what it hashes to.

    Attributes:
        path (Path): The shipped copy beside the decoder GGUF.
        n_bytes (int): The copy's size in bytes.
        sha256 (str): SHA-256 hex digest of the copy, proven equal
            to the source's.

    Examples:
        Read the digest for a publication record:

        ```python
        result = ship_sidecar(Path("mmproj.gguf"), beside=Path("packed.gguf"))
        digest = result.sha256
        ```
    """

    path: Path
    n_bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    """Hash a file's bytes with SHA-256.

    Args:
        path: The file to hash.

    Returns:
        The hex digest.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def ship_sidecar(mmproj: Path, beside: Path) -> SidecarResult:
    """Copy the vendor mmproj beside a packed artifact, byte-identical.

    The copy keeps the vendor file name and lands in the packed
    artifact's directory (ADR-0030 decision 2). The function hashes
    the source, copies, hashes the copy, and refuses a mismatch. A
    source already at the destination path is hashed in place and
    not copied.

    Args:
        mmproj: The vendor mmproj file.
        beside: The packed decoder GGUF the sidecar ships beside.

    Returns:
        The shipped copy's path, size, and SHA-256 digest.

    Raises:
        ValueError: If the mmproj carries the packed artifact's own
            file name — the copy would overwrite the decoder.
        RuntimeError: If the copy's hash differs from the source's.
        OSError: If a read, write, or stat fails.
    """
    destination = beside.with_name(mmproj.name)
    if destination == beside:
        raise ValueError(
            f'mmproj "{mmproj}" carries the packed artifact\'s file name '
            "— the sidecar copy would overwrite the decoder GGUF"
        )
    source_digest = _sha256(mmproj)
    if not (destination.exists() and destination.samefile(mmproj)):
        shutil.copyfile(mmproj, destination)
    copy_digest = _sha256(destination)
    if copy_digest != source_digest:
        raise RuntimeError(
            f'sidecar copy "{destination}" does not match the source '
            f'"{mmproj}": {copy_digest} != {source_digest}'
        )
    return SidecarResult(
        path=destination,
        n_bytes=destination.stat().st_size,
        sha256=copy_digest,
    )
