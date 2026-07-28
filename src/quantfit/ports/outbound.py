"""Outbound (driven) ports: what the application needs from the world.

Each protocol names one capability the inbound side orchestrates against.
Concrete implementations live in [quantfit.adapters.outbound][].

Examples:
    A test double satisfying `RecipeSink`:

    ```python
    class MemorySink:
        def __init__(self) -> None:
            self.saved = None

        def save(self, recipe) -> None:
            self.saved = recipe
    ```

See Also:
    - [quantfit.domain.model][]: The types these ports carry.
"""

from __future__ import annotations

from typing import Protocol

from quantfit.domain.budget import ModelShape
from quantfit.domain.model import Recipe, SensitivityMap


class SensitivityMapSource(Protocol):
    """Provides a validated sensitivity map from somewhere.

    Examples:
        The JSON file adapter satisfies this port:

        ```python
        from quantfit.adapters.outbound.sensitivity_map_json import (
            JsonSensitivityMapFile,
        )

        source: SensitivityMapSource = JsonSensitivityMapFile(path)
        ```
    """

    def load(self) -> SensitivityMap:
        """Load and validate the sensitivity map.

        Returns:
            The validated map.
        """
        ...


class RecipeSink(Protocol):
    """Accepts a finished recipe for persistence.

    Examples:
        The JSON file adapter satisfies this port:

        ```python
        from quantfit.adapters.outbound.recipe_json import JsonRecipeFile

        sink: RecipeSink = JsonRecipeFile(path)
        ```
    """

    def save(self, recipe: Recipe) -> None:
        """Persist the recipe.

        Args:
            recipe: The recipe to persist.
        """
        ...


class ModelShapeSource(Protocol):
    """Provides a model's attention geometry from somewhere.

    Examples:
        The Hugging Face config adapter satisfies this port:

        ```python
        from quantfit.adapters.outbound.hf_config import HfConfigFile

        source: ModelShapeSource = HfConfigFile(path)
        ```
    """

    def shape(self) -> ModelShape:
        """Load the model's attention geometry.

        Returns:
            The parsed shape.
        """
        ...
