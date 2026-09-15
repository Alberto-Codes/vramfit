"""JSON writer for the refinement sidecar (ADR-0031 decision 3).

One JSON document per refinement pass, published beside the recipe it
refined, carrying the `vramfit_schema` envelope
(`REFINEMENT_SIDECAR_SCHEMA_VERSION`). Breaking changes bump it.

The document records every arm the pass measured, not only the one it
kept, and that includes an arm the weight budget excluded from
selection — a measured arm is never dropped from the record. The
losing arms are the evidence that the sensitivity map did not order
the neighbourhood, which is the finding the whole stage rests on.

The pass writes this document as it runs, so a document may describe
a pass still under way. `_save_json` replaces the file in one step,
which is what lets it: an interrupted write leaves the record before
it rather than a truncated one. `finished` says which state the
document describes, and a reader reads it before `winner`.

It also records `neighbourhood_moves`, how many byte-neutral moves
the neighbourhood held. The arms are a sample of that whenever the
caller's budget was smaller, so a reader needs both numbers before
reading any outcome.

The frame carries every input whose substitution would change the
number (ADR-0031 decision 5), so it names the reference logits and
the importance matrix by content beside the evaluation corpus. An
unassisted pass records a null matrix rather than omitting the
field.

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
from vramfit.domain.refinement_record import (
    ArmRecord,
    FileIdentity,
    RefinementSidecar,
)

# Schema version 2: version 1 carried the frame, the stated bar, the
# control, every arm, the neighbourhood the arms were drawn from, and
# the outcome. `neighbourhood_moves`, the frame's `imatrix`, the
# frame's `reference` as a content identity, and each arm's
# `budget_margin` all joined version 1 rather than bumping it: no
# sidecar had been published when any of those landed, and each added
# a fact without changing one already written.
#
# `finished` is the first change of the second kind, so it bumps. A
# version 1 document's null `winner` means the pass judged every arm
# and kept none. In version 2 it means that only when `finished` is
# true, because the pass now writes this document as it runs and a
# document it wrote at arm 12 of 16 carries a null winner too. A
# reader that cannot tell the versions apart reads a stopped pass as
# a completed one.
REFINEMENT_SIDECAR_SCHEMA_VERSION: Final[int] = 2


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


def _identity_to_dict(identity: FileIdentity | None) -> dict[str, Any] | None:
    """Serialize one file the pass consumed, named by its bytes.

    Args:
        identity: The file identity, or None where the field is
            optional and the pass used no such file.

    Returns:
        The entry's JSON object, or null. A null on the matrix reads
        as "this pass ran unassisted", which is a recorded state.
    """
    if identity is None:
        return None
    return {
        "file": identity.file,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }


def _arm_to_dict(arm: ArmRecord) -> dict[str, Any]:
    """Serialize one measured arm.

    Writes ``budget_margin`` beside ``packed_bytes``: a negative
    margin marks an arm the weight budget excluded from selection,
    which an arm that was never evaluated is not — that one has no
    entry here at all.

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
        # Negative means the arm packed over the weight budget, so
        # selection could not keep it. The arm is still here with its
        # measurement: it was evaluated and excluded, which an arm
        # that was never evaluated is not — that one has no entry.
        "budget_margin": arm.budget_margin,
    }


def sidecar_to_dict(sidecar: RefinementSidecar) -> dict[str, Any]:
    """Serialize a refinement sidecar with the schema envelope.

    Writes ``neighbourhood_moves`` beside ``arms``, so a reader can
    always tell a whole search from a sample of one, and the frame's
    ``reference`` and ``imatrix`` by content, so two passes that
    measured against different bytes never read alike.

    Writes ``finished`` beside ``winner``. The pass writes this
    document as it runs, so a null winner means the pass kept nothing
    only when ``finished`` is true.

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
            # Every divergence is measured against these bytes, so
            # two passes against different reference logits must not
            # serialize alike.
            "reference": _identity_to_dict(sidecar.frame.reference),
            "corpus": _corpus_to_dict(sidecar.frame.corpus),
            # Null records an unassisted pass. Without this an
            # assisted and an unassisted pass serialize alike.
            "imatrix": _identity_to_dict(sidecar.frame.imatrix),
        },
        "bar": sidecar.bar,
        "control": (None if sidecar.control is None else _arm_to_dict(sidecar.control)),
        "arms": [_arm_to_dict(a) for a in sidecar.arms],
        # Read beside len(arms): the two differ whenever the arm
        # budget was smaller than the neighbourhood.
        "neighbourhood_moves": sidecar.neighbourhood_moves,
        "winner": sidecar.winner,
        # Read before `winner`. False marks a pass that stopped before
        # selection ran, whose arms are real measurements the pass
        # banked as it took them.
        "finished": sidecar.finished,
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
