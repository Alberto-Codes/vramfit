"""JSON file adapter for the recipe artifact.

Owns (de)serialization and validation of the recipe schema, including
the ``vramfit_schema`` envelope (version 6 since the envelope key
renamed, #118). A known runtime must serve every
assigned and protected precision — an unknown runtime name loads
untouched.
Mirrors the strict reject-don't-normalize stance of the
sensitivity-map adapter. Two fields are additive rather than
strict: ``within_group`` (ADR-0019) and ``imatrix`` (ADR-0020)
default to None when absent, because recipes written before the
fields existed do not record which map priced them. A present
``imatrix`` must pair with an assisted method token, ``kquant-imx``
or ``q0-imx`` — the loader
rejects a recipe whose provenance contradicts itself. A field the
reader does not know warns and loads (#261, ADR-0013's 2026-08-16
amendment). A save then drops it, and the warning says so.

Examples:
    Round-trip a recipe through a file:

    ```python
    from vramfit.adapters.outbound.recipe_json import (
        load_recipe,
        save_recipe,
    )

    save_recipe(recipe, path)
    assert load_recipe(path) == recipe
    ```

See Also:
    - [vramfit.ports.outbound][]: `RecipeSink`, which `JsonRecipeFile`
      satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vramfit.adapters.outbound.json_common import (
    _as_int,
    _as_str,
    _check_schema_version,
    _get_bool,
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
from vramfit.domain.model import (
    Assignment,
    PlanMeta,
    ProtectedTensor,
    Recipe,
    TraceStep,
)
from vramfit.domain.runtime import RUNTIME_CAPABILITIES
from vramfit.domain.scan import ASSISTED_METHODS

# The recipe schema version. Bumped to 2 when recipes gained the
# required (nullable) ``runtime`` field (ADR-0013), to 3 when
# they gained protections (ADR-0022), to 4 when they gained
# imatrix exclusions (ADR-0023) — a reader that dropped either
# record would silently pack a different artifact than the recipe
# intends — and to 5 when no-op protection pairs stopped resolving
# (issue #59). A schema-4 reader rejects a protection record with
# zero pairs, and a schema-4 recipe can carry no-op pairs that
# falsely fail the reconstruction gate — re-plan it.
RECIPE_SCHEMA_VERSION: Final[int] = 6

# Every key the reader carries, per object the schema fixes (#261).
# A key outside these sets warns and loads (ADR-0013, the 2026-08-16
# amendment). ``plan.pins`` and ``plan.protections`` are absent here
# on purpose: their keys are tensor-name patterns, and the solver
# validates them.
RECIPE_ROOT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "vramfit_schema",
        "model_id",
        "runtime",
        "within_group",
        "imatrix",
        "plan",
        "assignments",
        "protected_tensors",
    }
)
PLAN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "vram_budget_bytes",
        "kv_headroom_bytes",
        "weight_budget_bytes",
        "predicted_total_bytes",
        "predicted_damage",
        "solver",
        "pins",
        "protections",
        "imatrix_exclusions",
        "format_overhead",
        "trace",
    }
)
TRACE_STEP_FIELDS: Final[frozenset[str]] = frozenset(
    {"step", "group", "from_bits", "to_bits", "damage_delta", "bytes_freed", "ratio"}
)
ASSIGNMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"group", "bits", "bytes", "damage"}
)
PROTECTED_TENSOR_FIELDS: Final[frozenset[str]] = frozenset(
    {"tensor", "bits", "exclude_imatrix"}
)


def recipe_from_dict(data: object) -> Recipe:
    """Validate parsed JSON and build a `Recipe`.

    Args:
        data: Parsed JSON value, expected to be the artifact's top-level
            object.

    Returns:
        The validated recipe.

    When ``runtime`` names a runtime this version's capability table
    knows, every assignment's precision must be servable by it. An
    unknown runtime name loads untouched — a newer vramfit's recipe
    stays readable, and pack backends judge it at use (ADR-0013).
    ``within_group`` and ``imatrix`` load as None when absent or
    null — recipes written before the fields existed do not record
    their map's method (ADR-0019) or imatrix (ADR-0020). A present
    ``imatrix`` must pair with an assisted method token,
    ``kquant-imx`` or ``q0-imx``.
    ``protected_tensors`` and ``plan.protections`` are required.
    Resolved pairs need a protection record, and a known runtime
    must serve every protected precision (ADR-0022). A record with
    zero pairs is legal — every floor can be a per-tensor no-op
    (issue #59). Each protected pair carries a required
    ``exclude_imatrix`` mark, and ``plan.imatrix_exclusions`` must
    agree with the marks about whether the recipe excludes imatrix
    rows — the solver refuses an exclusion whose every pair dropped
    (ADR-0023).

    Raises:
        ArtifactError: If any field is missing, mistyped, or violates a
            schema rule — including non-positive ``bits`` or ``bytes``
            in any assignment, and a known runtime that cannot serve
            an assigned precision. A field the reader does not know
            reports and loads instead (#261).

    Examples:
        Reject an unsupported schema version:

        ```python
        recipe_from_dict({"vramfit_schema": 1})  # raises ArtifactError
        ```
    """
    root = _get_dict(data, "$")
    _check_schema_version(root, "$", expected=RECIPE_SCHEMA_VERSION)
    _warn_unknown_fields(root, "$", RECIPE_ROOT_FIELDS)
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
        not (within_group in ASSISTED_METHODS and imatrix is None),
        "$.imatrix",
        f'within_group "{within_group}" requires the imatrix field (ADR-0020)',
    )
    _require(
        not (imatrix is not None and within_group not in ASSISTED_METHODS),
        "$.imatrix",
        "imatrix provenance requires an assisted within_group "
        f'({", ".join(ASSISTED_METHODS)}), got "{within_group}" (ADR-0020)',
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
        _warn_unknown_fields(obj, path, ASSIGNMENT_FIELDS)
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
        _warn_unknown_fields(obj, path, PROTECTED_TENSOR_FIELDS)
        bits = _get_int(obj, "bits", path)
        _require(bits > 0, f"{path}.bits", "must be positive")
        pair = ProtectedTensor(
            tensor=_get_str(obj, "tensor", path),
            bits=bits,
            exclude_imatrix=_get_bool(obj, "exclude_imatrix", path),
        )
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
        not protected or bool(plan.protections),
        "$.protected_tensors",
        "protected_tensors requires plan.protections — resolved pairs "
        "cannot exist without the rules that made them (ADR-0022)",
    )
    _require(
        bool(plan.imatrix_exclusions) == any(p.exclude_imatrix for p in protected),
        "$.plan.imatrix_exclusions",
        "plan.imatrix_exclusions and the exclude_imatrix marks must both be "
        "empty or both be present (ADR-0023)",
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
    the fields are always present (ADR-0022), as is
    ``plan.imatrix_exclusions`` (ADR-0023).

    Args:
        recipe: The recipe to serialize.

    Returns:
        A dict that `recipe_from_dict` accepts and round-trips to an
        equal recipe.
    """
    plan = recipe.plan
    return {
        "vramfit_schema": RECIPE_SCHEMA_VERSION,
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
            "imatrix_exclusions": list(plan.imatrix_exclusions),
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
            {"tensor": p.tensor, "bits": p.bits, "exclude_imatrix": p.exclude_imatrix}
            for p in recipe.protected_tensors
        ],
    }


def load_recipe(path: Path) -> Recipe:
    """Read and validate a recipe file.

    Args:
        path: JSON file written by ``vramfit plan`` or `save_recipe`.

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
            are required — including ``protections`` (ADR-0022) and
            ``imatrix_exclusions`` (ADR-0023) — the
            writer always emits them, so a missing
            field means a truncated or hand-edited artifact. A field
            the section does not carry reports and loads (#261).
    """
    path = "$.plan"
    _warn_unknown_fields(obj, path, PLAN_FIELDS)
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
    exclusions_raw = _get_list(obj, "imatrix_exclusions", path)
    exclusions = tuple(
        _as_str(pattern, f"{path}.imatrix_exclusions[{i}]")
        for i, pattern in enumerate(exclusions_raw)
    )
    trace_raw = _get_list(obj, "trace", path)
    trace: list[TraceStep] = []
    for i, raw in enumerate(trace_raw):
        step_path = f"{path}.trace[{i}]"
        step_obj = _get_dict(raw, step_path)
        _warn_unknown_fields(step_obj, step_path, TRACE_STEP_FIELDS)
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
        imatrix_exclusions=exclusions,
    )
