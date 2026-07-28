from __future__ import annotations

from dataclasses import dataclass, field

from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.domain.budget import ModelShape
from quantfit.domain.model import Recipe, SensitivityMap
from quantfit.domain.scan import GroupSpec, Measurement


@dataclass
class MemorySensitivityMapSource:
    """In-memory `SensitivityMapSource`; raises like the JSON adapter when empty."""

    map_: SensitivityMap | None = None

    def load(self) -> SensitivityMap:
        if self.map_ is None:
            raise ArtifactError("$", "no sensitivity map configured")
        return self.map_


@dataclass
class MemoryRecipeSink:
    """In-memory `RecipeSink` capturing every save; last one wins."""

    saved: list[Recipe] = field(default_factory=list)

    def save(self, recipe: Recipe) -> None:
        self.saved.append(recipe)

    @property
    def last(self) -> Recipe:
        return self.saved[-1]


@dataclass
class MemoryModelShapeSource:
    """In-memory `ModelShapeSource`; raises like the config adapter when empty."""

    shape_: ModelShape | None = None

    def load(self) -> ModelShape:
        if self.shape_ is None:
            raise ValueError("no model shape configured")
        return self.shape_


@dataclass
class MemorySensitivityMapSink:
    """In-memory `SensitivityMapSink` capturing every save; last one wins."""

    saved: list[SensitivityMap] = field(default_factory=list)

    def save(self, map_: SensitivityMap) -> None:
        self.saved.append(map_)

    @property
    def last(self) -> SensitivityMap:
        return self.saved[-1]


@dataclass
class MemoryDamageMeter:
    """In-memory `DamageMeter`; damages configured per (group, bits) cell."""

    specs: tuple[GroupSpec, ...] = ()
    damages: dict[tuple[str, int], float] = field(default_factory=dict)
    tokens: int = 1024
    calls: list[tuple[str, int]] = field(default_factory=list)

    def groups(self) -> tuple[GroupSpec, ...]:
        return self.specs

    def calibration_tokens(self) -> int:
        return self.tokens

    def measure(self, group: str, bits: int) -> float:
        if group not in {spec.name for spec in self.specs}:
            raise ValueError(f'unknown group "{group}"')
        self.calls.append((group, bits))
        if (group, bits) not in self.damages:
            raise ValueError(f"no damage configured for ({group}, {bits})")
        return self.damages[(group, bits)]


@dataclass
class MemoryScanCheckpointStore:
    """In-memory `ScanCheckpointStore`; raises like the JSON adapter on mismatch."""

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
