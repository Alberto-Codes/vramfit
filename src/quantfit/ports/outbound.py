"""Outbound (driven) ports: what the application needs from the world.

Each protocol names one capability the inbound side orchestrates
against. Artifact IO ports carry whole domain values. The scan ports
(`DamageMeter`, `ScanCheckpointStore`) carry the scan grid cell by
cell so a crashed scan can resume. Concrete implementations live in
[quantfit.adapters.outbound][].

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
from quantfit.domain.scan import GroupSpec, Measurement


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

    def load(self) -> ModelShape:
        """Load and parse the model's attention geometry.

        Returns:
            The parsed shape.

        Raises:
            ValueError: If the backing source is missing or invalid.
        """
        ...


class SensitivityMapSink(Protocol):
    """Accepts a finished sensitivity map for persistence.

    Examples:
        The JSON file adapter satisfies this port:

        ```python
        from quantfit.adapters.outbound.sensitivity_map_json import (
            JsonSensitivityMapFile,
        )

        sink: SensitivityMapSink = JsonSensitivityMapFile(path)
        ```
    """

    def save(self, map_: SensitivityMap) -> None:
        """Persist the sensitivity map.

        Args:
            map_: The map to persist.
        """
        ...


class DamageMeter(Protocol):
    """Measures damage per (group x precision) on a loaded model.

    The expensive port: one `measure` call is a calibration pass over a
    perturbed model. The torch adapter implements it behind the scan
    extra. Orchestration is tested against the verified fake
    (ADR-0009) — no GPU in the hermetic suites.

    Examples:
        The scan loop drives the port like this:

        ```python
        for group, bits in todo:
            damage = meter.measure(group, bits)
        ```
    """

    def groups(self) -> tuple[GroupSpec, ...]:
        """Discover the model's layer groups.

        Returns:
            All groups, in scan order.
        """
        ...

    def calibration_tokens(self) -> int:
        """Count the calibration tokens each measurement runs over.

        Returns:
            The token count.
        """
        ...

    def measure(self, group: str, bits: int) -> float:
        """Measure one group's damage at one candidate precision.

        Args:
            group: Name of the group to perturb.
            bits: Candidate precision to quantize the group to.

        Returns:
            The measured damage. Always finite and non-negative —
            implementations reject unstable measurements instead of
            recording them.

        Raises:
            ValueError: If the group name is unknown, ``bits`` is below
                2, or the measurement is numerically unstable.
        """
        ...


class ScanCheckpointStore(Protocol):
    """Persists scan measurements incrementally for resume.

    A checkpoint belongs to exactly one scan, identified by its
    fingerprint (`quantfit.domain.scan.scan_fingerprint`). Loading with
    a different fingerprint fails rather than mixing two scans' numbers.

    Examples:
        Resume, then record one new cell:

        ```python
        done = store.load(fingerprint)
        store.append(fingerprint, measurement)
        ```
    """

    def load(self, fingerprint: str) -> tuple[Measurement, ...]:
        """Load prior measurements for this scan.

        Args:
            fingerprint: The scan's identity string.

        Returns:
            All checkpointed measurements, empty when none exist.

        Raises:
            ValueError: If the stored checkpoint carries a different
                fingerprint or is corrupt.
            OSError: If the backing store exists but cannot be read.
        """
        ...

    def append(self, fingerprint: str, measurement: Measurement) -> None:
        """Record one finished measurement.

        Callers append each grid cell at most once — the store does not
        deduplicate, and `quantfit.domain.scan.plan_measurements`
        rejects a checkpoint that repeats a cell.

        Args:
            fingerprint: The scan's identity string.
            measurement: The finished cell.

        Raises:
            ValueError: If the stored checkpoint carries a different
                fingerprint.
            OSError: If the write fails.
        """
        ...
