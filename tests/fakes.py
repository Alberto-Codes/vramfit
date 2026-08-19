from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vramfit.adapters.outbound.gguf.exclusion_match import unmatched_exclusions
from vramfit.adapters.outbound.gguf.override_match import (
    floored_layers,
    unmatched_flags,
    unmatched_patterns,
)
from vramfit.adapters.outbound.gguf.pack import TypeFallbackError
from vramfit.adapters.outbound.gguf.types import (
    PackError,
    all_overrides,
    base_type,
    check_runtime,
    imatrix_exclusion_names,
    output_group_type,
    output_tensor_type,
    token_embedding_type,
)
from vramfit.adapters.outbound.json_common import ArtifactError
from vramfit.domain.budget import ModelShape
from vramfit.domain.evals import EvalsSidecar
from vramfit.domain.model import Recipe, SensitivityMap
from vramfit.domain.pack import PackResult
from vramfit.domain.scan import GroupSpec, Measurement
from vramfit.domain.sizes import SizeSourceError, TensorSize


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
        # Match the real meter's validation order: all group names
        # first, then bits, before any cell is read.
        unknown = sorted(set(assignments) - {spec.name for spec in self.specs})
        if unknown:
            raise ValueError(f'unknown group "{unknown[0]}"')
        if any(bits < 2 for bits in assignments.values()):
            raise ValueError("bits must be at least 2")
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

    ``base_tensor_names`` gives the fake the real adapter's pre-run
    override refusal (#303), its dedicated-flag refusal (#306), and
    its uncovered-layer report (#307). It holds the names a base GGUF
    would declare, so a suite that sets it must name the embedding
    tensor a real file carries. None skips all three, because most
    suites configure no model shape and only care about the sizes.
    ``imatrix_entry_names`` does the same for the exclusion refusal
    (#309), and None skips it the same way.
    """

    base_bytes: int = 1_000
    packed_bytes: int = 500
    fail_stage: Literal["convert", "quantize"] | None = None
    has_base: bool = False
    imatrix: str | None = None
    imatrix_uncovered: tuple[str, ...] = ()
    type_fallbacks: tuple[tuple[str, str, str], ...] = ()
    base_tensor_names: tuple[str, ...] | None = None
    imatrix_entry_names: tuple[str, ...] | None = None
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
        # The mapping evaluates in the real adapter's order, so the
        # same malformed recipe raises the same refusal first. A
        # configured type-fallback pair then halts after the mapping
        # proved drivable (ADR-0028 decision 3).
        base = base_type(recipe)
        embedding = token_embedding_type(recipe)
        output = output_tensor_type(recipe)
        overrides = all_overrides(recipe)
        layer_gaps: tuple[str, ...] = ()
        if self.base_tensor_names is not None:
            # Parity with the real adapter's pre-run checks (#303,
            # #306, #307). A fake left at None keeps the old behavior,
            # so a suite that does not care about them stays
            # unchanged. An empty override set reports no floored
            # layer, as `check_base_coverage` does.
            unmatched = unmatched_patterns(overrides, self.base_tensor_names)
            if unmatched:
                raise PackError(
                    f"the base GGUF carries no tensor for {len(unmatched)} "
                    f"of {len(overrides)} override patterns: "
                    + ", ".join(f'"{pattern}"' for pattern in unmatched)
                )
            # The tied fallback emits a flag the record already rules
            # a no-op, so only a scanned lm_head group is held here.
            unreached = unmatched_flags(
                self.base_tensor_names,
                embedding=embedding is not None,
                output=output_group_type(recipe) is not None,
            )
            if unreached:
                raise PackError(
                    f"the base GGUF carries no target tensor for "
                    f"{len(unreached)} dedicated flag"
                    f"{'' if len(unreached) == 1 else 's'}: "
                    + ", ".join(
                        f"{flag} (needs "
                        + " or ".join(f'"{name}"' for name in targets)
                        + ")"
                        for flag, targets in unreached
                    )
                    + ". The quantizer binds each flag by exact tensor name "
                    "and exits 0 when the file carries none, so that tensor "
                    "would take the --pure floor while the record states the "
                    "recipe's type (#306). Check the recipe's embedding and "
                    "lm_head groups against the base GGUF"
                )
            if overrides:
                layer_gaps = floored_layers(overrides, self.base_tensor_names)
        # The real adapter resolves the exclusion names after the
        # base-GGUF checks, so a recipe failing both reports the
        # mapping error (`pack.py`). This block sits here to match.
        excluded = imatrix_exclusion_names(recipe) if self.imatrix is not None else ()
        if excluded and self.imatrix_entry_names is not None:
            # Parity with the real adapter's exclusion refusal (#309).
            # A fake left at None keeps the old behavior. The message
            # carries the real one's remedy, so a suite asserting the
            # operator's next step cannot pass against a bare summary.
            unreached = unmatched_exclusions(excluded, self.imatrix_entry_names)
            if unreached:
                raise PackError(
                    f"the imatrix carries no row for {len(unreached)} of "
                    f"{len(dict.fromkeys(excluded))} recipe exclusions: "
                    + ", ".join(f'"{name}"' for name in unreached)
                    + ". The quantizer erases no row for such a name and "
                    "exits 0. The tensor then keeps the assisted fit the "
                    "recipe asked to drop, and the record states an "
                    "exclusion that never applied (ADR-0023). Check the "
                    "recipe's protected tensors against the imatrix's "
                    "entry names"
                )
        if self.type_fallbacks:
            raise TypeFallbackError(self.type_fallbacks, Path("packed.gguf"))
        result = PackResult(
            packed_bytes=self.packed_bytes,
            base_type=base,
            token_embedding_type=embedding,
            output_tensor_type=output,
            overrides=overrides,
            imatrix_path=self.imatrix,
            imatrix_uncovered=(
                tuple(n for n in self.imatrix_uncovered if n not in excluded)
                if self.imatrix is not None
                else ()
            ),
            imatrix_excluded=excluded,
            floored_layers=layer_gaps,
        )
        self.packed.append(recipe)
        return result


@dataclass
class MemoryImatrixCountSource:
    """In-memory `ImatrixCountSource`. Counts are configured per stack.

    Like the real adapter, an unvouchable source raises `PackError`
    rather than returning an empty mapping — an empty report is what
    a healthy matrix returns, so a silent failure would read as a
    clean bill of health (ADR-0026, #198 amendment).
    """

    stack_counts: dict[str, tuple[int, ...]] = field(default_factory=dict)
    fail: bool = False
    reads: int = 0

    def expert_stack_counts(self) -> dict[str, tuple[int, ...]]:
        if self.fail:
            raise PackError("imatrix.gguf is not an imatrix GGUF: configured failure")
        self.reads += 1
        return dict(self.stack_counts)


@dataclass
class MemoryTensorSizeSource:
    """In-memory `TensorSizeSource`. Sizes are configured per tensor.

    Like the safetensors adapter, it drops the MTP block (ADR-0029
    decision 2) and refuses a checkpoint it cannot read rather than
    returning an empty mapping — an empty result would price the
    whole model at zero bytes.
    """

    sizes: dict[str, TensorSize] = field(default_factory=dict)
    fail: bool = False
    reads: int = 0

    def tensor_sizes(self) -> dict[str, TensorSize]:
        if self.fail:
            raise SizeSourceError(
                "checkpoint: no *.safetensors shards found: configured failure"
            )
        self.reads += 1
        return {
            name: size
            for name, size in self.sizes.items()
            if not name.startswith("mtp.")
        }


@dataclass
class MemoryReconstructionChecker:
    """In-memory `ReconstructionChecker`. Errors are configured per tensor.

    Like the real adapter, a tensor without a configured measurement
    raises `PackError` — a missing tensor must never read as a
    passing one.
    """

    errors: dict[str, float] = field(default_factory=dict)
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def rmse(self, tensors: tuple[str, ...]) -> Mapping[str, float]:
        self.calls.append(tensors)
        missing = [name for name in tensors if name not in self.errors]
        if missing:
            raise PackError(f'tensor "{missing[0]}" is not in the packed file')
        return {name: self.errors[name] for name in tensors}


@dataclass
class MemorySmokeTester:
    """In-memory `SmokeTester`. The measurement is configured, NaN included.

    Like the real adapter, a tool failure raises `PackError` and a
    successful run returns the estimate verbatim — the verdict
    belongs to the caller.
    """

    perplexity: float = 9.5
    fail: bool = False
    runs: int = 0

    def smoke(self) -> float:
        if self.fail:
            raise PackError("smoke failed with exit code 3:\nconfigured failure")
        self.runs += 1
        return self.perplexity


@dataclass
class MemoryRunLog:
    """In-memory `RunLogSink` recording events in order."""

    events: list[tuple[str, dict]] = field(default_factory=list)

    def emit(self, event: str, fields) -> None:
        self.events.append((event, dict(fields)))


@dataclass
class MemoryEvalsSidecarStore:
    """In-memory `EvalsSidecarSource` and `EvalsSidecarSink`.

    Named `Store` because it serves both ports, matching
    `MemoryScanCheckpointStore`. Every save is captured and the last
    one wins, so `load` returns what `save` last accepted. An empty
    store refuses `load` with `ArtifactError`, which is the error type
    the JSON adapter raises for an unreadable source.
    """

    saved: list[EvalsSidecar] = field(default_factory=list)

    def save(self, sidecar: EvalsSidecar) -> None:
        self.saved.append(sidecar)

    def load(self) -> EvalsSidecar:
        if not self.saved:
            raise ArtifactError("$", "no evals sidecar configured")
        return self.saved[-1]

    @property
    def last(self) -> EvalsSidecar:
        return self.saved[-1]


def decoder_tensor_names(layers: int = 64) -> tuple[str, ...]:
    """Name every GGUF tensor a llama-family decoder of ``layers`` carries.

    A stand-in for `base_tensor_names` in suites that stub the base
    GGUF as opaque bytes. It covers every pattern this backend emits
    — the seven quantized projections, the three fused expert
    stacks, the two Mamba projections, the embedding, and the output
    head — so a recipe under the decoder root matches and the
    override check passes (#303).

    Args:
        layers: How many `blk.<n>.` layers to name.

    Returns:
        The tensor names, embedding and head first.

    Examples:
        Stub the reader in a CLI suite:

        ```python
        monkeypatch.setattr(
            override_match, "base_tensor_names", lambda _: decoder_tensor_names()
        )
        ```
    """
    suffixes = (
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ssm_in",
        "ssm_out",
    )
    names = ["token_embd.weight", "output.weight"]
    names += [
        f"blk.{index}.{suffix}.weight" for index in range(layers) for suffix in suffixes
    ]
    return tuple(names)


def decoder_imatrix_entry_names(layers: int = 64) -> tuple[str, ...]:
    """Name the entries an imatrix over that decoder would carry.

    A stand-in for `imatrix_entry_names` in suites that stub the
    matrix as a path to no file. It is a strict subset of
    `decoder_tensor_names`, because a matrix prices fewer tensors
    than the base GGUF carries. ADR-0016's acceptance evidence
    measured `token_embd` as the expected uncovered tensor, so the
    embedding and the output head stay out.

    Keeping the two lists distinguishable matters: a check handed the
    base GGUF's names in place of the matrix's would pass every
    stub that served one list for both.

    Args:
        layers: How many `blk.<n>.` layers to price.

    Returns:
        The entry names, in tensor order.

    Examples:
        Stub the matrix read in a contract suite:

        ```python
        monkeypatch.setattr(
            exclusion_match,
            "imatrix_entry_names",
            lambda _: decoder_imatrix_entry_names(),
        )
        ```
    """
    head = {"token_embd.weight", "output.weight"}
    return tuple(n for n in decoder_tensor_names(layers) if n not in head)
