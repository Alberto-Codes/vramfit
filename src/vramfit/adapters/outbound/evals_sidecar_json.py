"""JSON file adapter for the evals sidecar artifact (ADR-0025).

Owns serialization of the sidecar schema, including the
``vramfit_schema`` envelope (version 2 since the envelope key
renamed, #118). Absent tiers serialize as JSON null, as do the
toolchain's tier-3 fields, so every schema-2 sidecar carries the same
key set. The domain types (`vramfit.domain.evals`) enforce the value
invariants, and the reader restates a domain refusal as an
`ArtifactError` naming the JSON path.

The reader landed with #137. The sidecar was the last published
artifact with no reader, so nobody could verify the five shipped
files by loading them (ADR-0025).

A field the reader does not know warns and loads (#261, ADR-0013's
2026-08-16 amendment). A save then drops it, because no domain type
carries it, and the warning says so. The sidecar is the sharpest
case: ADR-0025 publishes it beside the weights.

Examples:
    Read a published sidecar:

    ```python
    from vramfit.adapters.outbound.evals_sidecar_json import (
        load_evals_sidecar,
    )

    sidecar = load_evals_sidecar(path)
    ```

See Also:
    - [vramfit.ports.outbound][]: `EvalsSidecarSource` and
      `EvalsSidecarSink`, which `JsonEvalsSidecarFile` satisfies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.json_common import (
    ArtifactError,
    _check_schema_version,
    _get_dict,
    _get_float,
    _get_int,
    _get_list,
    _get_str,
    _load_json,
    _require,
    _save_json,
    _warn_unknown_fields,
)
from vramfit.domain.evals import (
    EvalsSidecar,
    EvalToolchain,
    EvaluatedArtifact,
    Tier1Result,
    Tier2Result,
    Tier2Window,
    Tier3Result,
    Tier3Task,
)

# The evals-sidecar schema version (ADR-0025). Versions advance per
# artifact (ADR-0013) — a breaking change here bumps this constant
# and nothing else's.
EVALS_SIDECAR_SCHEMA_VERSION: Final[int] = 2

# Every key the reader carries, per object (#261). A key outside these
# sets warns and loads (ADR-0013, the 2026-08-16 amendment). The writer
# emits one key set in every schema-2 sidecar, so a hand edit is the
# only way a document reaches a report.
SIDECAR_ROOT_FIELDS: Final[frozenset[str]] = frozenset(
    {"vramfit_schema", "artifact", "toolchain", "tier1", "tier2", "tier3"}
)
ARTIFACT_FIELDS: Final[frozenset[str]] = frozenset({"file", "sha256", "size_bytes"})
TOOLCHAIN_FIELDS: Final[frozenset[str]] = frozenset(
    {"llama_cpp_build", "lm_eval", "llama_cpp_python", "lane"}
)
TIER1_FIELDS: Final[frozenset[str]] = frozenset(
    {"date", "dataset", "chunks", "ppl", "ppl_stderr"}
)
TIER2_FIELDS: Final[frozenset[str]] = frozenset({"reference", "dataset", "windows"})
TIER2_WINDOW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "date",
        "chunks",
        "mean_kld",
        "kld_stderr",
        "same_top_pct",
        "same_top_stderr_pct",
    }
)
TIER3_FIELDS: Final[frozenset[str]] = frozenset({"tasks"})
TIER3_TASK_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "date",
        "name",
        "version",
        "few_shot",
        "n",
        "metric",
        "score",
        "stderr",
        "wall_clock_seconds",
    }
)


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


def _built[T](path: str, build: Callable[[], T]) -> T:
    """Construct a domain value, reporting its invariants by JSON path.

    The domain types enforce the value rules in ``__post_init__``
    (ADR-0008). A reader must not leak a bare `ValueError` naming no
    field. This restates the failure as an `ArtifactError`.

    Every caller extracts its fields first and passes ``build`` a
    constructor call and nothing else. That keeps this ``except``
    narrow. A ``build`` that also parsed would relabel any unrelated
    `ValueError` as the reader's fault, at the enclosing block's path
    rather than the failing field's — the error-labeling bug class
    ADR-0011 exists to prevent.

    Args:
        path: JSON path of the object being built.
        build: Zero-argument constructor call. It must not parse.

    Returns:
        The constructed domain value.

    Raises:
        ArtifactError: If the domain type rejects the values.
    """
    try:
        return build()
    except ArtifactError:
        raise
    except ValueError as exc:
        raise ArtifactError(path, str(exc)) from exc


def _get_opt_str(obj: dict[str, Any], key: str, path: str) -> str | None:
    """Return the non-empty string at ``key``, or None for JSON null.

    Args:
        obj: Parent JSON object.
        key: Key to read.
        path: JSON path of the parent for error reporting.

    Returns:
        The string value, or None when the field is null.

    Raises:
        ArtifactError: If the key is missing, or holds neither null
            nor a non-empty string.
    """
    _require(key in obj, path, f'missing required field "{key}"')
    value = obj[key]
    if value is None:
        return None
    _require(isinstance(value, str), f"{path}.{key}", "expected a string or null")
    _require(value != "", f"{path}.{key}", "must not be empty — use null")
    return value


def _artifact_from_dict(obj: dict[str, Any], path: str) -> EvaluatedArtifact:
    """Parse the ``artifact`` block.

    Extracts every field first, then constructs. `_built` wraps the
    constructor alone, so it never relabels a parse failure.

    The reader reports a field the block does not carry, then loads
    it (#261).

    Args:
        obj: The block's JSON object.
        path: Its JSON path.

    Returns:
        The evaluated artifact.

    Raises:
        ArtifactError: If a field is missing or invalid.
    """
    _warn_unknown_fields(obj, path, ARTIFACT_FIELDS)
    file = _get_str(obj, "file", path)
    sha256 = _get_str(obj, "sha256", path)
    size_bytes = _get_int(obj, "size_bytes", path)
    return _built(path, lambda: EvaluatedArtifact(file, sha256, size_bytes))


def _toolchain_from_dict(obj: dict[str, Any], path: str) -> EvalToolchain:
    """Parse the ``toolchain`` block.

    Extracts every field first, then constructs. `_built` wraps the
    constructor alone, so it never relabels a parse failure.

    The three tier-3 fields serialize as null on a tiers-1-2 sidecar,
    so each reads as optional here. `EvalsSidecar` enforces their
    pairing with tier 3.

    The reader reports a field the block does not carry, then loads
    it (#261).

    Args:
        obj: The block's JSON object.
        path: Its JSON path.

    Returns:
        The toolchain record.

    Raises:
        ArtifactError: If a field is missing or invalid.
    """
    _warn_unknown_fields(obj, path, TOOLCHAIN_FIELDS)
    build = _get_str(obj, "llama_cpp_build", path)
    lm_eval = _get_opt_str(obj, "lm_eval", path)
    binding = _get_opt_str(obj, "llama_cpp_python", path)
    lane = _get_opt_str(obj, "lane", path)
    return _built(path, lambda: EvalToolchain(build, lm_eval, binding, lane))


def _tier1_from_dict(obj: dict[str, Any], path: str) -> Tier1Result:
    """Parse the ``tier1`` block.

    Extracts every field first, then constructs. `_built` wraps the
    constructor alone, so it never relabels a parse failure.

    The reader reports a field the block does not carry, then loads
    it (#261).

    Args:
        obj: The block's JSON object.
        path: Its JSON path.

    Returns:
        The tier-1 result.

    Raises:
        ArtifactError: If a field is missing or invalid.
    """
    _warn_unknown_fields(obj, path, TIER1_FIELDS)
    date = _get_str(obj, "date", path)
    dataset = _get_str(obj, "dataset", path)
    chunks = _get_int(obj, "chunks", path)
    ppl = _get_float(obj, "ppl", path)
    ppl_stderr = _get_float(obj, "ppl_stderr", path)
    return _built(path, lambda: Tier1Result(date, dataset, chunks, ppl, ppl_stderr))


def _tier2_window_from_dict(obj: dict[str, Any], path: str) -> Tier2Window:
    """Parse one ``tier2.windows`` entry.

    Extracts every field first, then constructs. `_built` wraps the
    constructor alone, so it never relabels a parse failure.

    The reader reports a field the block does not carry, then loads
    it (#261).

    Args:
        obj: The window's JSON object.
        path: Its JSON path.

    Returns:
        The window.

    Raises:
        ArtifactError: If a field is missing or invalid.
    """
    _warn_unknown_fields(obj, path, TIER2_WINDOW_FIELDS)
    date = _get_str(obj, "date", path)
    chunks = _get_int(obj, "chunks", path)
    mean_kld = _get_float(obj, "mean_kld", path)
    kld_stderr = _get_float(obj, "kld_stderr", path)
    same_top = _get_float(obj, "same_top_pct", path)
    same_top_stderr = _get_float(obj, "same_top_stderr_pct", path)
    return _built(
        path,
        lambda: Tier2Window(
            date, chunks, mean_kld, kld_stderr, same_top, same_top_stderr
        ),
    )


def _tier2_from_dict(obj: dict[str, Any], path: str) -> Tier2Result:
    """Parse the ``tier2`` block.

    Extracts every field first, then constructs. `_built` wraps the
    constructor alone, so it never relabels a parse failure.

    The reader reports a field the block does not carry, then loads
    it (#261).

    Args:
        obj: The block's JSON object.
        path: Its JSON path.

    Returns:
        The tier-2 result.

    Raises:
        ArtifactError: If a field is missing or invalid, or two
            windows share a chunk count.
    """
    _warn_unknown_fields(obj, path, TIER2_FIELDS)
    raw = _get_list(obj, "windows", path)
    windows = tuple(
        _tier2_window_from_dict(
            _get_dict(entry, f"{path}.windows[{index}]"), f"{path}.windows[{index}]"
        )
        for index, entry in enumerate(raw)
    )
    reference = _get_str(obj, "reference", path)
    dataset = _get_str(obj, "dataset", path)
    return _built(path, lambda: Tier2Result(reference, dataset, windows))


def _tier3_task_from_dict(obj: dict[str, Any], path: str) -> Tier3Task:
    """Parse one ``tier3.tasks`` entry.

    Extracts every field first, then constructs. `_built` wraps the
    constructor alone, so it never relabels a parse failure.

    The reader reports a field the block does not carry, then loads
    it (#261).

    Args:
        obj: The task's JSON object.
        path: Its JSON path.

    Returns:
        The task row.

    Raises:
        ArtifactError: If a field is missing or invalid.
    """
    _warn_unknown_fields(obj, path, TIER3_TASK_FIELDS)
    date = _get_str(obj, "date", path)
    name = _get_str(obj, "name", path)
    version = _get_str(obj, "version", path)
    few_shot = _get_int(obj, "few_shot", path)
    n = _get_int(obj, "n", path)
    metric = _get_str(obj, "metric", path)
    score = _get_float(obj, "score", path)
    stderr = _get_float(obj, "stderr", path)
    seconds = _get_float(obj, "wall_clock_seconds", path)
    return _built(
        path,
        lambda: Tier3Task(
            date, name, version, few_shot, n, metric, score, stderr, seconds
        ),
    )


def _tier3_from_dict(obj: dict[str, Any], path: str) -> Tier3Result:
    """Parse the ``tier3`` block.

    The reader reports a field the block does not carry, then loads
    it (#261).

    Args:
        obj: The block's JSON object.
        path: Its JSON path.

    Returns:
        The tier-3 result.

    Raises:
        ArtifactError: If a field is missing or invalid, or two tasks
            share a name.
    """
    _warn_unknown_fields(obj, path, TIER3_FIELDS)
    raw = _get_list(obj, "tasks", path)
    tasks = tuple(
        _tier3_task_from_dict(
            _get_dict(entry, f"{path}.tasks[{index}]"), f"{path}.tasks[{index}]"
        )
        for index, entry in enumerate(raw)
    )
    return _built(path, lambda: Tier3Result(tasks=tasks))


def _required_block[T](
    obj: dict[str, Any], key: str, root: str, parse: Callable[[dict[str, Any], str], T]
) -> T:
    """Parse a block that every sidecar carries.

    The missing case reports itself as missing. `dict.get` would turn
    an absent key into None and report the wrong reason, which sends
    the reader hunting for a type error in a block that is not there.

    Args:
        obj: Top-level artifact object.
        key: The block's key, e.g. ``artifact``.
        root: JSON path of the artifact root.
        parse: The block's parser.

    Returns:
        The parsed block.

    Raises:
        ArtifactError: If the key is missing or is not a JSON object.
    """
    _require(key in obj, root, f'missing required field "{key}"')
    path = f"{root}.{key}"
    return parse(_get_dict(obj[key], path), path)


def _optional_block[T](
    obj: dict[str, Any], key: str, root: str, parse: Callable[[dict[str, Any], str], T]
) -> T | None:
    """Parse a tier block that serializes as null when absent.

    Args:
        obj: Top-level artifact object.
        key: The tier's key, e.g. ``tier1``.
        root: JSON path of the artifact root.
        parse: The block's parser.

    Returns:
        The parsed block, or None when the field is null.

    Raises:
        ArtifactError: If the key is missing, or holds neither null
            nor a JSON object.
    """
    _require(key in obj, root, f'missing required field "{key}"')
    value = obj[key]
    if value is None:
        return None
    path = f"{root}.{key}"
    return parse(_get_dict(value, path), path)


def sidecar_from_dict(data: dict[str, Any]) -> EvalsSidecar:
    """Validate a JSON dict and build the sidecar it describes.

    The inverse of `sidecar_to_dict`. Every tier key must be present.
    The writer emits one key set in every schema-2 sidecar, so a null
    value means the tier did not run. A field the reader does not know
    reports and loads (#261).

    Args:
        data: The artifact's top-level JSON object.

    Returns:
        The validated sidecar.

    Raises:
        ArtifactError: If the envelope is not
            `EVALS_SIDECAR_SCHEMA_VERSION`, if the document carries
            the pre-rename envelope key (#118, #154), or if any field
            is missing or invalid.

    Examples:
        A version mismatch refuses by path:

        ```python
        sidecar_from_dict({"vramfit_schema": 99})  # raises ArtifactError
        ```
    """
    root = "$"
    obj = _get_dict(data, root)
    _check_schema_version(obj, root, EVALS_SIDECAR_SCHEMA_VERSION)
    _warn_unknown_fields(obj, root, SIDECAR_ROOT_FIELDS)
    artifact = _required_block(obj, "artifact", root, _artifact_from_dict)
    toolchain = _required_block(obj, "toolchain", root, _toolchain_from_dict)
    tier1 = _optional_block(obj, "tier1", root, _tier1_from_dict)
    tier2 = _optional_block(obj, "tier2", root, _tier2_from_dict)
    tier3 = _optional_block(obj, "tier3", root, _tier3_from_dict)
    return _built(root, lambda: EvalsSidecar(artifact, toolchain, tier1, tier2, tier3))


def load_evals_sidecar(path: Path) -> EvalsSidecar:
    """Read and validate an evals sidecar from a JSON file.

    Args:
        path: The sidecar file to read.

    Returns:
        The validated sidecar.

    Raises:
        ArtifactError: If the file cannot be read, is not valid JSON,
            or fails validation.

    Examples:
        Load a published sidecar:

        ```python
        sidecar = load_evals_sidecar(Path("model.gguf.evals.json"))
        ```
    """
    return sidecar_from_dict(_load_json(path, "$"))


def save_evals_sidecar(sidecar: EvalsSidecar, path: Path) -> None:
    """Write a sidecar as pretty-printed JSON, atomically.

    Args:
        sidecar: The sidecar to write.
        path: Destination file.
    """
    _save_json(sidecar_to_dict(sidecar), path)


@dataclass(frozen=True, slots=True)
class JsonEvalsSidecarFile:
    """`EvalsSidecarSource` and `EvalsSidecarSink` adapter for one file.

    Attributes:
        path (Path): The file to read or write.

    Examples:
        Use as a port implementation:

        ```python
        source = JsonEvalsSidecarFile(Path("model.gguf.evals.json"))
        sidecar = source.load()
        ```
    """

    path: Path

    def load(self) -> EvalsSidecar:
        """Load and validate the sidecar from `path`.

        Returns:
            The validated sidecar.

        Raises:
            ArtifactError: If the file is not valid JSON or fails
                validation.
        """
        return load_evals_sidecar(self.path)

    def save(self, sidecar: EvalsSidecar) -> None:
        """Persist the sidecar to `path` as JSON.

        Args:
            sidecar: The sidecar to persist.
        """
        save_evals_sidecar(sidecar, self.path)
