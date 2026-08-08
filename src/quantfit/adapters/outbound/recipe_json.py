"""JSON file adapter for the recipe artifact.

Owns (de)serialization and validation of the recipe schema, including
the ``quantfit_schema`` envelope (version 3 since recipes record
their protections, ADR-0022). A known runtime must serve every
assigned and protected precision — an unknown runtime name loads
untouched.
Mirrors the strict reject-don't-normalize stance of the
sensitivity-map adapter. Two fields are additive rather than
strict: ``within_group`` (ADR-0019) and ``imatrix`` (ADR-0020)
default to None when absent, because recipes written before the
fields existed do not record which map priced them. A present
``imatrix`` must pair with the assisted method token — the loader
rejects a recipe whose provenance contradicts itself.

Examples:
    Round-trip a recipe through a file:

    ```python
    from quantfit.adapters.outbound.recipe_json import (
        load_recipe,
        save_recipe,
    )

    save_recipe(recipe, path)
    assert load_recipe(path) == recipe
    ```

See Also:
    - [quantfit.ports.outbound][]: `RecipeSink`, which `JsonRecipeFile`
      satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from quantfit.adapters.outbound.json_common import (
    _as_int,
    _check_schema_version,
    _get_dict,
    _get_float,
    _get_int,
    _get_list,
    _get_str,
    _load_json,
    _require,
    _save_json,
)
from quantfit.domain.model import (
    Assignment,
    PlanMeta,
    ProtectedTensor,
    Recipe,
    TraceStep,
)
from quantfit.domain.runtime import RUNTIME_CAPABILITIES
from quantfit.domain.scan import KQUANT_IMX_METHOD

# The recipe schema version. Bumped to 2 when recipes gained the
# required (nullable) ``runtime`` field (ADR-0013), and to 3 when
# they gained protections (ADR-0022) — a version-2 reader that
# dropped the protections record would silently pack a different
# artifact than the recipe intends.
RECIPE_SCHEMA_VERSION: Final[int] = 3


def recipe_from_dict(data: object) -> Recipe:
    """Validate parsed JSON and build a `Recipe`.

    Args:
        data: Parsed JSON value, expected to be the artifact's top-level
            object.

    Returns:
        The validated recipe.

    When ``runtime`` names a runtime this version's capability table
    knows, every assignment's precision must be servable by it. An
    unknown runtime name loads untouched — a newer quantfit's recipe
    stays readable, and pack backends judge it at use (ADR-0013).
    ``within_group`` and ``imatrix`` load as None when absent or
    null — recipes written before the fields existed do not record
    their map's method (ADR-0019) or imatrix (ADR-0020). A present
    ``imatrix`` must pair with the assisted method token.
    ``protected_tensors`` and ``plan.protections`` are required and
    must agree about whether the recipe is protected — a known
    runtime must also serve every protected precision (ADR-0022).

    Raises:
        ArtifactError: If any field is missing, mistyped, or violates a
            schema rule — including non-positive ``bits`` or ``bytes``
            in any assignment, and a known runtime that cannot serve
            an assigned precision.

    Examples:
        Reject an unsupported schema version:

        ```python
        recipe_from_dict({"quantfit_schema": 1})  # raises ArtifactError
        ```
    """
    root = _get_dict(data, "$")
    _check_schema_version(root, "$", expected=RECIPE_SCHEMA_VERSION)
    model_id = _get_str(root, "model_id", "$")
    _require("runtime" in root, "$", 'missing required field "runtime"')
    runtime = None if root["runtime"] is None else _get_str(root, "runtime", "$")
    # Optional and additive (ADR-0019, ADR-0020): recipes written
    # before the fields existed do not record their map's method or
    # imatrix. The pairing rules mirror Recipe's own invariant,
    # re-stated here for JSON-path errors.
    within_group = (
        _get_str(root, "within_group", "$")
        if root.get("within_group") is not None
        else None
    )
    imatrix = (
        _get_str(root, "imatrix", "$") if root.get("imatrix") is not None else None
    )
    _require(
        not (within_group == KQUANT_IMX_METHOD and imatrix is None),
        "$.imatrix",
        f'within_group "{KQUANT_IMX_METHOD}" requires the imatrix field (ADR-0020)',
    )
    _require(
        not (imatrix is not None and within_group != KQUANT_IMX_METHOD),
        "$.imatrix",
        f'imatrix provenance requires within_group "{KQUANT_IMX_METHOD}", '
        f'got "{within_group}" (ADR-0020)',
    )
    _require("plan" in root, "$", 'missing required field "plan"')
    plan = _parse_plan_meta(_get_dict(root["plan"], "$.plan"))
    raw_assignments = _get_list(root, "assignments", "$")
    _require(len(raw_assignments) > 0, "$.assignments", "must not be empty")
    assignments: list[Assignment] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_assignments):
        path = f"$.assignments[{i}]"
        obj = _get_dict(raw, path)
        bits = _get_int(obj, "bits", path)
        _require(bits > 0, f"{path}.bits", "must be positive")
        size = _get_int(obj, "bytes", path)
        _require(size > 0, f"{path}.bytes", "must be positive")
        assignment = Assignment(
            group=_get_str(obj, "group", path),
            bits=bits,
            bytes=size,
            damage=_get_float(obj, "damage", path),
        )
        _require(
            assignment.group not in seen,
            f"{path}.group",
            f'duplicate group "{assignment.group}"',
        )
        seen.add(assignment.group)
        assignments.append(assignment)
    if runtime is not None and runtime in RUNTIME_CAPABILITIES:
        capability = RUNTIME_CAPABILITIES[runtime]
        for i, assignment in enumerate(assignments):
            _require(
                assignment.bits in capability,
                f"$.assignments[{i}].bits",
                f'precision {assignment.bits} is not servable by runtime "{runtime}"',
            )
    raw_protected = _get_list(root, "protected_tensors", "$")
    protected: list[ProtectedTensor] = []
    seen_tensors: set[str] = set()
    for i, raw in enumerate(raw_protected):
        path = f"$.protected_tensors[{i}]"
        obj = _get_dict(raw, path)
        bits = _get_int(obj, "bits", path)
        _require(bits > 0, f"{path}.bits", "must be positive")
        pair = ProtectedTensor(tensor=_get_str(obj, "tensor", path), bits=bits)
        _require(
            pair.tensor not in seen_tensors,
            f"{path}.tensor",
            f'duplicate protected tensor "{pair.tensor}"',
        )
        if runtime is not None and runtime in RUNTIME_CAPABILITIES:
            _require(
                pair.bits in RUNTIME_CAPABILITIES[runtime],
                f"{path}.bits",
                f'precision {pair.bits} is not servable by runtime "{runtime}"',
            )
        seen_tensors.add(pair.tensor)
        protected.append(pair)
    _require(
        bool(plan.protections) == bool(protected),
        "$.protected_tensors",
        "plan.protections and protected_tensors must both be empty or both "
        "be present (ADR-0022)",
    )
    return Recipe(
        model_id=model_id,
        plan=plan,
        assignments=tuple(assignments),
        runtime=runtime,
        within_group=within_group,
        imatrix=imatrix,
        protected_tensors=tuple(protected),
    )


def recipe_to_dict(recipe: Recipe) -> dict[str, Any]:
    """Serialize a recipe to a JSON dict with the schema envelope.

    An unconstrained recipe serializes its runtime as JSON null, and
    unknown provenance serializes ``within_group`` and ``imatrix``
    as null. An unprotected recipe serializes ``plan.protections``
    as an empty object and ``protected_tensors`` as an empty list —
    the fields are always present (ADR-0022).

    Args:
        recipe: The recipe to serialize.

    Returns:
        A dict that `recipe_from_dict` accepts and round-trips to an
        equal recipe.
    """
    plan = recipe.plan
    return {
        "quantfit_schema": RECIPE_SCHEMA_VERSION,
        "model_id": recipe.model_id,
        "runtime": recipe.runtime,
        "within_group": recipe.within_group,
        "imatrix": recipe.imatrix,
        "plan": {
            "vram_budget_bytes": plan.vram_budget_bytes,
            "kv_headroom_bytes": plan.kv_headroom_bytes,
            "weight_budget_bytes": plan.weight_budget_bytes,
            "predicted_total_bytes": plan.predicted_total_bytes,
            "predicted_damage": plan.predicted_damage,
            "solver": plan.solver,
            "pins": dict(plan.pins),
            "protections": dict(plan.protections),
            "format_overhead": plan.format_overhead,
            "trace": [
                {
                    "step": t.step,
                    "group": t.group,
                    "from_bits": t.from_bits,
                    "to_bits": t.to_bits,
                    "damage_delta": t.damage_delta,
                    "bytes_freed": t.bytes_freed,
                    "ratio": t.ratio,
                }
                for t in plan.trace
            ],
        },
        "assignments": [
            {"group": a.group, "bits": a.bits, "bytes": a.bytes, "damage": a.damage}
            for a in recipe.assignments
        ],
        "protected_tensors": [
            {"tensor": p.tensor, "bits": p.bits} for p in recipe.protected_tensors
        ],
    }


def load_recipe(path: Path) -> Recipe:
    """Read and validate a recipe file.

    Args:
        path: JSON file written by ``quantfit plan`` or `save_recipe`.

    Returns:
        The validated recipe.

    Raises:
        ArtifactError: If the file is not valid JSON or fails validation.
    """
    return recipe_from_dict(_load_json(path, "$"))


def save_recipe(recipe: Recipe, path: Path) -> None:
    """Write a recipe as pretty-printed JSON.

    Args:
        recipe: The recipe to write.
        path: Destination file.
    """
    _save_json(recipe_to_dict(recipe), path)


@dataclass(frozen=True, slots=True)
class JsonRecipeFile:
    """`RecipeSink` adapter backed by a JSON file.

    Attributes:
        path (Path): The file to write.

    Examples:
        Use as a port implementation:

        ```python
        sink = JsonRecipeFile(Path("recipe.json"))
        sink.save(recipe)
        ```
    """

    path: Path

    def save(self, recipe: Recipe) -> None:
        """Persist the recipe to `path` as JSON.

        Args:
            recipe: The recipe to persist.
        """
        save_recipe(recipe, self.path)


def _parse_plan_meta(obj: dict[str, Any]) -> PlanMeta:
    """Validate the ``plan`` section of a recipe.

    Args:
        obj: The ``plan`` JSON object.

    Returns:
        The validated plan metadata.

    Raises:
        ArtifactError: If a field is missing or invalid. All plan fields
            are required — including ``protections`` (ADR-0022) — the
            writer always emits them, so a missing
            field means a truncated or hand-edited artifact.
    """
    path = "$.plan"
    _require("pins" in obj, path, 'missing required field "pins"')
    pins_obj = _get_dict(obj["pins"], f"{path}.pins")
    pins = {
        pattern: _as_int(bits, f"{path}.pins.{pattern}")
        for pattern, bits in pins_obj.items()
    }
    _require("protections" in obj, path, 'missing required field "protections"')
    protections_obj = _get_dict(obj["protections"], f"{path}.protections")
    protections = {
        pattern: _as_int(floor, f"{path}.protections.{pattern}")
        for pattern, floor in protections_obj.items()
    }
    trace_raw = _get_list(obj, "trace", path)
    trace: list[TraceStep] = []
    for i, raw in enumerate(trace_raw):
        step_path = f"{path}.trace[{i}]"
        step_obj = _get_dict(raw, step_path)
        trace.append(
            TraceStep(
                step=_get_int(step_obj, "step", step_path),
                group=_get_str(step_obj, "group", step_path),
                from_bits=_get_int(step_obj, "from_bits", step_path),
                to_bits=_get_int(step_obj, "to_bits", step_path),
                damage_delta=_get_float(step_obj, "damage_delta", step_path),
                bytes_freed=_get_int(step_obj, "bytes_freed", step_path),
                ratio=_get_float(step_obj, "ratio", step_path),
            )
        )
    return PlanMeta(
        vram_budget_bytes=_get_int(obj, "vram_budget_bytes", path),
        kv_headroom_bytes=_get_int(obj, "kv_headroom_bytes", path),
        weight_budget_bytes=_get_int(obj, "weight_budget_bytes", path),
        predicted_total_bytes=_get_int(obj, "predicted_total_bytes", path),
        predicted_damage=_get_float(obj, "predicted_damage", path),
        solver=_get_str(obj, "solver", path),
        pins=pins,
        protections=protections,
        format_overhead=_get_float(obj, "format_overhead", path),
        trace=tuple(trace),
    )
