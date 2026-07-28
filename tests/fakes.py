from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from quantfit.adapters.outbound.gguf.types import (
    PackError,
    base_type,
    check_runtime,
    tensor_overrides,
    token_embedding_type,
)
from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.domain.budget import ModelShape
from quantfit.domain.model import Recipe, SensitivityMap
from quantfit.domain.pack import PackResult
from quantfit.domain.scan import GroupSpec, Measurement


@dataclass
class MemorySensitivityMapSource:
    """In-memory `SensitivityMapSource`. Raises like the JSON adapter when empty."""

    map_: SensitivityMap | None = None

    def load(self) -> SensitivityMap:
        if self.map_ is None:
            raise ArtifactError("$", "no sensitivity map configured")
        return self.map_


@dataclass
class MemoryRecipeSink:
    """In-memory `RecipeSink` capturing every save. Last one wins."""

    saved: list[Recipe] = field(default_factory=list)

    def save(self, recipe: Recipe) -> None:
        self.saved.append(recipe)

    @property
    def last(self) -> Recipe:
        return self.saved[-1]


@dataclass
class MemoryModelShapeSource:
    """In-memory `ModelShapeSource`. Raises like the config adapter when empty."""

    shape_: ModelShape | None = None

    def load(self) -> ModelShape:
        if self.shape_ is None:
            raise ValueError("no model shape configured")
        return self.shape_


@dataclass
class MemorySensitivityMapSink:
    """In-memory `SensitivityMapSink` capturing every save. Last one wins."""

    saved: list[SensitivityMap] = field(default_factory=list)

    def save(self, map_: SensitivityMap) -> None:
        self.saved.append(map_)

    @property
    def last(self) -> SensitivityMap:
        return self.saved[-1]


@dataclass
class MemoryDamageMeter:
    """In-memory `DamageMeter`. Damages are configured per (group, bits) cell.

    `measure_recipe` sums the configured cells and adds
    ``interaction_damage`` when two or more groups are perturbed —
    the configurable additivity leak. A single-group recipe equals
    `measure` for that cell, like the real meter.
    """

    specs: tuple[GroupSpec, ...] = ()
    damages: dict[tuple[str, int], float] = field(default_factory=dict)
    tokens: int = 1024
    interaction_damage: float = 0.0
    calls: list[tuple[str, int]] = field(default_factory=list)
    recipe_calls: list[dict[str, int]] = field(default_factory=list)

    def groups(self) -> tuple[GroupSpec, ...]:
        return self.specs

    def calibration_tokens(self) -> int:
        return self.tokens

    def _cell(self, group: str, bits: int) -> float:
        if group not in {spec.name for spec in self.specs}:
            raise ValueError(f'unknown group "{group}"')
        if bits < 2:
            raise ValueError("bits must be at least 2")
        if (group, bits) not in self.damages:
            raise ValueError(f"no damage configured for ({group}, {bits})")
        return self.damages[(group, bits)]

    def measure(self, group: str, bits: int) -> float:
        damage = self._cell(group, bits)
        self.calls.append((group, bits))
        return damage

    def measure_recipe(self, assignments: Mapping[str, int]) -> float:
        if not assignments:
            raise ValueError("assignments must not be empty")
        total = sum(self._cell(group, bits) for group, bits in assignments.items())
        if len(assignments) > 1:
            total += self.interaction_damage
        self.recipe_calls.append(dict(assignments))
        return total


@dataclass
class MemoryScanCheckpointStore:
    """In-memory `ScanCheckpointStore`. Raises like the JSON adapter on mismatch."""

    fingerprint: str | None = None
    measurements: list[Measurement] = field(default_factory=list)

    def _check(self, fingerprint: str) -> None:
        if self.fingerprint is not None and fingerprint != self.fingerprint:
            raise ArtifactError(
                "$.fingerprint",
                f'checkpoint belongs to a different scan ("{self.fingerprint}" '
                f'!= "{fingerprint}")',
            )

    def load(self, fingerprint: str) -> tuple[Measurement, ...]:
        if self.fingerprint is None:
            return ()
        self._check(fingerprint)
        return tuple(self.measurements)

    def append(self, fingerprint: str, measurement: Measurement) -> None:
        self._check(fingerprint)
        self.fingerprint = fingerprint
        self.measurements.append(measurement)


@dataclass
class MemoryRecipePacker:
    """In-memory `RecipePacker`. Sizes are configured, the type mapping is shared.

    The mapping comes from the same pure functions the real adapter
    uses, so the fake cannot drift from the ADR-0012 tables. Like the
    real adapter, an existing base wins before a broken convert tool,
    and a mapping failure packs nothing.
    """

    base_bytes: int = 1_000
    packed_bytes: int = 500
    fail_stage: Literal["convert", "quantize"] | None = None
    has_base: bool = False
    packed: list[Recipe] = field(default_factory=list)

    def convert(self) -> int:
        if self.has_base:
            return self.base_bytes
        if self.fail_stage == "convert":
            raise PackError("convert failed with exit code 3:\nconfigured failure")
        self.has_base = True
        return self.base_bytes

    def pack(self, recipe: Recipe) -> PackResult:
        check_runtime(recipe)
        if not self.has_base:
            raise PackError("base GGUF does not exist — run convert first")
        if self.fail_stage == "quantize":
            raise PackError("quantize failed with exit code 3:\nconfigured failure")
        result = PackResult(
            packed_bytes=self.packed_bytes,
            base_type=base_type(recipe),
            token_embedding_type=token_embedding_type(recipe),
            overrides=tensor_overrides(recipe),
        )
        self.packed.append(recipe)
        return result


@dataclass
class MemoryRunLog:
    """In-memory `RunLogSink` recording events in order."""

    events: list[tuple[str, dict]] = field(default_factory=list)

    def emit(self, event: str, fields) -> None:
        self.events.append((event, dict(fields)))
