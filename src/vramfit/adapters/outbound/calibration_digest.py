"""Read a corpus file's content identity.

The scan records the calibration text by content, not by name, and
the refinement pass records the evaluation text the same way. This
module reads the file once and returns the SHA-256 hex digest with
the byte count. `vramfit.domain.scan.scan_fingerprint` folds both, so
a re-issued file behind an unchanged path refuses the old checkpoint,
and the sensitivity map carries both so a reader can prove which
file produced a damage value. The scan measures a prefix of that
file, bounded by ``--max-tokens``, so the pair names the file and
never the measured part.

The digest records the bytes the scan read, at the moment it read
them. It never certifies an earlier run: a file that has since
changed under the same name hashes to today's bytes. Where a past
run's calibration file does not survive, the honest record is NOT
RECORDED — `ScanMeta` takes None for both fields, and nothing in the
pipeline back-fills a digest from a fresh download.

The digest pins the corpus, not the chunking. The tokenizer stays
unpinned. A digest match does not promise the same
`calibration_tokens` count.

Examples:
    Read the identity the scan records:

    ```python
    from pathlib import Path

    from vramfit.adapters.outbound.calibration_digest import (
        content_identity,
    )

    sha256, n_bytes = content_identity(Path("calibration.txt"))
    ```

See Also:
    - [vramfit.adapters.inbound.cli_scan][]: Records the identity in
      the map's `ScanMeta`.
    - [vramfit.adapters.inbound.cli_refine][]: Records the evaluation
      corpus's identity in the refinement sidecar's frame.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Hash in 1 MiB slabs, as the sidecar does: a corpus file has no
# fixed size, and a whole-file read would hold all of it in memory.
_HASH_CHUNK_BYTES = 1 << 20


def content_identity(path: Path) -> tuple[str, int]:
    """Hash a corpus file and count its bytes.

    Args:
        path: The text file a stage measures against — the scan's
            calibration corpus, or the refinement pass's evaluation
            corpus.

    Returns:
        The SHA-256 hex digest and the byte count, in that order.
        Both come from one read of the file this call opens.

    Raises:
        OSError: If the file cannot be read.

    Examples:
        Record the identity the published calibration text carries:

        ```python
        sha256, n_bytes = content_identity(path)
        assert n_bytes == 772386
        ```
    """
    digest = hashlib.sha256()
    n_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            n_bytes += len(chunk)
    return digest.hexdigest(), n_bytes
