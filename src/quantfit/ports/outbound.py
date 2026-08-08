"""Outbound (driven) ports: what the application needs from the world.

Each protocol names one capability the inbound side orchestrates
against. Artifact IO ports carry whole domain values, and the run-log
port carries one machine event at a time (ADR-0011). The scan ports
(`DamageMeter`, `ScanCheckpointStore`) carry the scan grid cell by
cell so a crashed scan can resume — the meter also measures a whole
recipe at once for the validation pass (ADR-0006). The pack port (`RecipePacker`)
carries its two toolchain stages separately so the composition root
can log each (ADR-0012), the smoke port (`SmokeTester`) carries
the post-pack proof that the artifact emits language (ADR-0017),
and the reconstruction port (`ReconstructionChecker`) carries the
per-tensor measurement that guards protected packs against fit
collapse (ADR-0022).
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

from collections.abc import Mapping
from typing import Protocol

from quantfit.domain.budget import ModelShape
from quantfit.domain.model import Recipe, SensitivityMap
from quantfit.domain.pack import PackResult
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

    def measure_recipe(self, assignments: Mapping[str, int]) -> float:
        """Measure whole-recipe damage: all groups perturbed at once.

        The validation-pass measurement (ADR-0006): every listed group
        is quantized to its assigned precision in one pass, so the
        result includes the interactions marginal scanning misses.
        Perturbing one group this way equals `measure` for that cell.

        Args:
            assignments: Assigned precision per group name.

        Returns:
            The measured damage. Always finite and non-negative.

        Raises:
            ValueError: If ``assignments`` is empty, names an unknown
                group, assigns bits below 2, or the measurement is
                numerically unstable.
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


class RecipePacker(Protocol):
    """Packs a recipe into a checkpoint a target runtime can serve.

    The second expensive port after `DamageMeter`: both stages shell
    out to a runtime's toolchain and run for minutes. The port splits
    them so the composition root can log each stage (ADR-0012):
    `convert` materializes the full-precision base GGUF once, and
    `pack` drives the runtime's quantizer with the recipe's type
    mapping. The llama.cpp adapter implements it by subprocess.

    Examples:
        The pack command drives the port like this:

        ```python
        base_bytes = packer.convert()
        result = packer.pack(recipe)
        ```
    """

    def convert(self) -> int:
        """Materialize the full-precision base GGUF, reusing any existing one.

        Returns:
            Size of the base GGUF in bytes.

        Raises:
            RuntimeError: If the conversion tool fails or writes no
                file.
        """
        ...

    def pack(self, recipe: Recipe) -> PackResult:
        """Quantize the base GGUF into the recipe's packed model.

        Args:
            recipe: The recipe to apply.

        Returns:
            The accounting record, with the real packed size.

        Raises:
            RuntimeError: If the base GGUF is missing, the recipe has
                a group or precision the backend cannot map, or the
                quantizer fails.
        """
        ...


class ReconstructionChecker(Protocol):
    """Measures per-tensor reconstruction error of a packed model.

    The reconstruction check's measurement half (ADR-0022): dequantize
    named tensors from a packed file and compare each against the f16
    base. The verdict against the unprotected reference is domain
    arithmetic (`quantfit.domain.pack.collapsed_tensors`) — fit
    collapse is invisible to the smoke test, so protected imatrix
    packs gate on this instead. The gguf-py adapter implements it in
    seconds of CPU.

    Examples:
        The pack command drives the port like this:

        ```python
        protected = checker.rmse(tensors)
        reference = reference_checker.rmse(tensors)
        collapsed = collapsed_tensors(protected, reference)
        ```
    """

    def rmse(self, tensors: tuple[str, ...]) -> Mapping[str, float]:
        """Measure reconstruction error for the named tensors.

        Args:
            tensors: Tensor names in the packed file's own naming,
                e.g. ``blk.4.attn_v.weight`` for GGUF.

        Returns:
            Root-mean-square error against the f16 base, per tensor.
            Every requested tensor is present in the result.

        Raises:
            RuntimeError: If a file cannot be read, or a requested
                tensor is missing from either file.
        """
        ...


class SmokeTester(Protocol):
    """Measures a packed model's perplexity over a few chunks.

    The post-pack smoke test (ADR-0017): a cheap proof that the
    packed artifact emits language before anything downstream trusts
    it. The llama.cpp adapter drives ``llama-perplexity`` by
    subprocess. The measurement is the port's whole job — the verdict
    against the ceiling is domain arithmetic
    (`quantfit.domain.pack.smoke_passed`).

    Examples:
        The pack command drives the port like this:

        ```python
        perplexity = tester.smoke()
        passed = smoke_passed(perplexity, threshold)
        ```
    """

    def smoke(self) -> float:
        """Run the smoke chunks and report the final perplexity.

        Returns:
            The tool's final perplexity estimate. May be non-finite —
            a destroyed artifact's NaN is a valid measurement and the
            caller's ceiling rejects it.

        Raises:
            RuntimeError: If the tool cannot start, exits nonzero, or
                reports no final estimate.
        """
        ...


class RunLogSink(Protocol):
    """Accepts run-log events for durable, machine-readable recording.

    The run log is the machine channel (ADR-0011): one event per call,
    appended in order. Human CLI output never routes through it.

    Examples:
        Record one measured cell:

        ```python
        sink.emit("cell_measured", {"group": "g0", "bits": 4})
        ```
    """

    def emit(self, event: str, fields: Mapping[str, object]) -> None:
        """Record one event with its fields.

        The keys ``event``, ``ts``, and ``quantfit_runlog`` belong to
        the sink's envelope, and the inbound ``SafeRunLog`` wrapper
        stamps ``run_id`` — callers must not pass any of them in
        ``fields``.

        Args:
            event: Past-tense event name, e.g. ``cell_measured``.
            fields: JSON-representable payload for the event.

        Raises:
            OSError: If the backing store cannot be written.
            TypeError: If a field is not JSON-representable.
            ValueError: If a field is NaN or infinite.
        """
        ...
