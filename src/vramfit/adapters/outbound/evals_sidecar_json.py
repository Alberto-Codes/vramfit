"""JSON file adapter for the evals sidecar artifact (ADR-0025).

Owns serialization of the sidecar schema, including the
``vramfit_schema`` envelope (version 2 since the envelope key
renamed, #118). Writer only — the reader
path for card tooling is deferred (issue #65). Absent tiers
serialize as JSON null, as do the toolchain's tier-3 fields, so
every schema-2 sidecar carries the same key set. The domain types
(`vramfit.domain.evals`) enforce the value invariants before
anything reaches this adapter.

Examples:
    Write a sidecar beside its artifact:

    ```python
    from vramfit.adapters.outbound.evals_sidecar_json import (
        save_evals_sidecar,
    )

    save_evals_sidecar(sidecar, path)
    ```

See Also:
    - [vramfit.ports.outbound][]: `EvalsSidecarSink`, which
      `JsonEvalsSidecarFile` satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.json_common import _save_json
from vramfit.domain.evals import (
    EvalsSidecar,
    Tier1Result,
    Tier2Result,
    Tier3Result,
)

# The evals-sidecar schema version (ADR-0025). Versions advance per
# artifact (ADR-0013) — a breaking change here bumps this constant
# and nothing else's.
EVALS_SIDECAR_SCHEMA_VERSION: Final[int] = 2


def _tier1_to_dict(tier1: Tier1Result) -> dict[str, Any]:
    """Serialize the tier-1 block.

    Args:
        tier1: The tier-1 result.

    Returns:
        The ``tier1`` JSON object.
    """
    return {
        "date": tier1.date,
        "dataset": tier1.dataset,
        "chunks": tier1.chunks,
        "ppl": tier1.ppl,
        "ppl_stderr": tier1.ppl_stderr,
    }


def _tier2_to_dict(tier2: Tier2Result) -> dict[str, Any]:
    """Serialize the tier-2 block.

    Args:
        tier2: The tier-2 result.

    Returns:
        The ``tier2`` JSON object.
    """
    return {
        "reference": tier2.reference,
        "dataset": tier2.dataset,
        "windows": [
            {
                "date": w.date,
                "chunks": w.chunks,
                "mean_kld": w.mean_kld,
                "kld_stderr": w.kld_stderr,
                "same_top_pct": w.same_top_pct,
                "same_top_stderr_pct": w.same_top_stderr_pct,
            }
            for w in tier2.windows
        ],
    }


def _tier3_to_dict(tier3: Tier3Result) -> dict[str, Any]:
    """Serialize the tier-3 block.

    Args:
        tier3: The tier-3 result.

    Returns:
        The ``tier3`` JSON object.
    """
    return {
        "tasks": [
            {
                "date": t.date,
                "name": t.name,
                "version": t.version,
                "few_shot": t.few_shot,
                "n": t.n,
                "metric": t.metric,
                "score": t.score,
                "stderr": t.stderr,
                "wall_clock_seconds": t.wall_clock_seconds,
            }
            for t in tier3.tasks
        ],
    }


def sidecar_to_dict(sidecar: EvalsSidecar) -> dict[str, Any]:
    """Serialize a sidecar to a JSON dict with the schema envelope.

    Absent tiers serialize as JSON null, as do the toolchain's
    tier-3 fields — the key set never varies within schema 2.

    Args:
        sidecar: The sidecar to serialize.

    Returns:
        The artifact's top-level JSON object.
    """
    return {
        "vramfit_schema": EVALS_SIDECAR_SCHEMA_VERSION,
        "artifact": {
            "file": sidecar.artifact.file,
            "sha256": sidecar.artifact.sha256,
            "size_bytes": sidecar.artifact.size_bytes,
        },
        "toolchain": {
            "llama_cpp_build": sidecar.toolchain.llama_cpp_build,
            "lm_eval": sidecar.toolchain.lm_eval,
            "llama_cpp_python": sidecar.toolchain.llama_cpp_python,
            "lane": sidecar.toolchain.lane,
        },
        "tier1": None if sidecar.tier1 is None else _tier1_to_dict(sidecar.tier1),
        "tier2": None if sidecar.tier2 is None else _tier2_to_dict(sidecar.tier2),
        "tier3": None if sidecar.tier3 is None else _tier3_to_dict(sidecar.tier3),
    }


def save_evals_sidecar(sidecar: EvalsSidecar, path: Path) -> None:
    """Write a sidecar as pretty-printed JSON, atomically.

    Args:
        sidecar: The sidecar to write.
        path: Destination file.
    """
    _save_json(sidecar_to_dict(sidecar), path)


@dataclass(frozen=True, slots=True)
class JsonEvalsSidecarFile:
    """`EvalsSidecarSink` adapter backed by a JSON file.

    Attributes:
        path (Path): The file to write.

    Examples:
        Use as a port implementation:

        ```python
        sink = JsonEvalsSidecarFile(Path("model.gguf.evals.json"))
        sink.save(sidecar)
        ```
    """

    path: Path

    def save(self, sidecar: EvalsSidecar) -> None:
        """Persist the sidecar to `path` as JSON.

        Args:
            sidecar: The sidecar to persist.
        """
        save_evals_sidecar(sidecar, self.path)
