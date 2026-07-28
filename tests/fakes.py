from __future__ import annotations

from dataclasses import dataclass, field

from quantfit.adapters.outbound.json_common import ArtifactError
from quantfit.domain.budget import ModelShape
from quantfit.domain.model import Recipe, SensitivityMap


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

    def shape(self) -> ModelShape:
        if self.shape_ is None:
            raise ValueError("no model shape configured")
        return self.shape_
