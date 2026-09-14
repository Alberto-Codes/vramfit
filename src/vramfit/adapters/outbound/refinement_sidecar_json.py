"""JSON writer for the refinement sidecar (ADR-0031 decision 3).

One JSON document per refinement pass, published beside the recipe it
refined, carrying the `vramfit_schema` envelope
(`REFINEMENT_SIDECAR_SCHEMA_VERSION`). Breaking changes bump it.

The document records every arm the pass measured, not only the one it
kept. The losing arms are the evidence that the sensitivity map did
not order the neighbourhood, which is the finding the whole stage
rests on.

It also records `neighbourhood_moves`, how many byte-neutral moves
the neighbourhood held. The arms are a sample of that whenever the
caller's budget was smaller, so a reader needs both numbers before
reading any outcome.

`predicted_delta` serializes as provenance. Nothing in this package or
the domain orders, filters, or selects on it (ADR-0031 decision 6).

Examples:
    Write a pass's record beside its recipe:

    ```python
    from pathlib import Path

    from vramfit.adapters.outbound.refinement_sidecar_json import (
        JsonRefinementSidecarFile,
    )

    JsonRefinementSidecarFile(Path("recipe.refinement.json")).save(sidecar)
    ```

See Also:
    - [vramfit.domain.refinement_record][]: The types this writes.
    - [vramfit.ports.outbound][]: `RefinementSidecarSink`, which this
      satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.json_common import _save_json
from vramfit.domain.evals import CorpusReference
from vramfit.domain.refinement_record import ArmRecord, RefinementSidecar

# Schema version 1: the first refinement sidecar. It carries the
# frame, the stated bar, the control, every arm, the neighbourhood
# the arms were drawn from, and the outcome. `neighbourhood_moves`
# joined version 1 rather than bumping it: no sidecar had been
# published when the field landed, so nothing reads a document
# without it.
REFINEMENT_SIDECAR_SCHEMA_VERSION: Final[int] = 1


def _corpus_to_dict(corpus: CorpusReference) -> dict[str, Any]:
    """Serialize the evaluation corpus reference.

    Every field is written, null where the pass recorded nothing. A
    null reads as unrecorded, never as a claim about the bytes behind
    that name today.

    Args:
        corpus: The corpus reference.

    Returns:
        The corpus entry's JSON object.
    """
    return {
        "source": corpus.source,
        "revision": corpus.revision,
        "file": corpus.file,
        "sha256": corpus.sha256,
        "size_bytes": corpus.size_bytes,
        "provenance": corpus.provenance,
    }


def _arm_to_dict(arm: ArmRecord) -> dict[str, Any]:
    """Serialize one measured arm.

    Args:
        arm: The arm record.

    Returns:
        One ``arms`` entry's JSON object.
    """
    return {
        "arm": arm.arm,
        "promoted": arm.promoted,
        "demoted": arm.demoted,
        "from_bits": arm.from_bits,
        "to_bits": arm.to_bits,
        "mean": arm.mean,
        "delta": arm.delta,
        "sigma": arm.sigma,
        "better_chunks": arm.better_chunks,
        "chunks": arm.chunks,
        # Provenance only. Nothing orders, filters, or selects on this
        # (ADR-0031 decision 6).
        "predicted_delta": arm.predicted_delta,
        "packed_bytes": arm.packed_bytes,
    }


def sidecar_to_dict(sidecar: RefinementSidecar) -> dict[str, Any]:
    """Serialize a refinement sidecar with the schema envelope.

    Writes ``neighbourhood_moves`` beside ``arms``, so a reader can
    always tell a whole search from a sample of one.

    Args:
        sidecar: The record to serialize.

    Returns:
        The artifact's top-level JSON object.
    """
    return {
        "vramfit_schema": REFINEMENT_SIDECAR_SCHEMA_VERSION,
        "model_id": sidecar.model_id,
        "frame": {
            "runtime_build": sidecar.frame.runtime_build,
            "hardware": sidecar.frame.hardware,
            "reference": sidecar.frame.reference,
            "corpus": _corpus_to_dict(sidecar.frame.corpus),
        },
        "bar": sidecar.bar,
        "control": (None if sidecar.control is None else _arm_to_dict(sidecar.control)),
        "arms": [_arm_to_dict(a) for a in sidecar.arms],
        # Read beside len(arms): the two differ whenever the arm
        # budget was smaller than the neighbourhood.
        "neighbourhood_moves": sidecar.neighbourhood_moves,
        "winner": sidecar.winner,
        "declined": sidecar.declined,
    }


def save_refinement_sidecar(sidecar: RefinementSidecar, path: Path) -> None:
    """Write a refinement sidecar as pretty-printed JSON, atomically.

    Args:
        sidecar: The record to write.
        path: Destination file.
    """
    _save_json(sidecar_to_dict(sidecar), path)


@dataclass(frozen=True, slots=True)
class JsonRefinementSidecarFile:
    """`RefinementSidecarSink` adapter for one file.

    Attributes:
        path (Path): The file to write.

    Examples:
        Use as a port implementation:

        ```python
        sink = JsonRefinementSidecarFile(Path("recipe.refinement.json"))
        sink.save(sidecar)
        ```
    """

    path: Path

    def save(self, sidecar: RefinementSidecar) -> None:
        """Persist the refinement sidecar to `path` as JSON.

        Args:
            sidecar: The record to persist.
        """
        save_refinement_sidecar(sidecar, self.path)
