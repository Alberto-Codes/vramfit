"""`DamageMeter` adapter: perturb one group, measure KL on real weights.

The core scan loop per ADR-0006: quantize-dequantize one layer group in
place, run the calibration set, compare final logits against the cached
reference distribution, restore the group. The validation pass reuses
the same machinery with every group perturbed at once
(`TorchDamageMeter.measure_recipe`). Reference log-probabilities
are cached on the CPU in float16 (~0.25 GiB per 1024 tokens at a 128k
vocabulary) so the reference model runs exactly once.

`TorchDamageMeter.measure_slices` reuses the same machinery on a
dim-0 expert range of a fused expert stack — the slice perturbation
path (ADR-0026, the 2026-08-13 #200 amendment). Slice cells rank and
weight in the scan frame and never set a recipe's price, so the path
sits on the adapter only. The `DamageMeter` port does not carry it.

Only floating-point tensors with 2+ dimensions join layer groups —
norms and biases stay at reference precision and are not scanned.
Tied names that alias one storage collapse to one group.
``group_by`` decides the rest. ``layer`` collapses each decoder layer
into one group. ``tensor`` keeps every weight apart. ``stack`` also
keeps every weight apart, except a layer's routed experts. Those fuse
into one group per projection, which is the unit a pack addresses
(#161). [vramfit.domain.scan][] holds the naming rule.
A measurement that fails to restore the model poisons the meter: it
refuses further cells rather than measure against corrupt weights.
Groups that ``auto`` sharding offloaded to host RAM perturb through
accelerate's weights map (ADR-0015). Offloaded weights the map cannot
reach — disk spill, or an unrecognized accelerate layout — are
refused at construction.

Examples:
    Measure one cell on a local checkpoint:

    ```python
    from vramfit.adapters.outbound.scan.meter import TorchDamageMeter

    meter = TorchDamageMeter("./model", Path("calibration.txt"), max_tokens=4096)
    damage = meter.measure(meter.groups()[0].name, bits=4)
    ```

The within-group method is selected at construction (ADR-0018):
round-to-nearest by default, or the K-quant-faithful port. The
kquant method can price covered tensors with imatrix column weights
(assisted pricing, ADR-0020) — passed directly, or resolved from a
GGUF imatrix file that loads before the model. Construction refuses
weights the
model cannot consume — wrong names, wrong lengths, misaligned rows,
non-finite values — before the scan spends an hour, and reports the
coverage split for the run log. A file-resolved meter also reads
the matrix's counts and pools each group's expert-stack vectors
into the map's count summary through
[vramfit.adapters.outbound.scan.discovery][] (ADR-0026 decision 4,
the #201 amendment).

See Also:
    - [vramfit.ports.outbound][]: `DamageMeter`, the port this
      satisfies.
    - [vramfit.adapters.outbound.scan.quantize][]: The v1 within-group
      method.
    - [vramfit.adapters.outbound.scan.kquant][]: The K-quant-faithful
      method.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vramfit.adapters.outbound.scan.calibration import load_calibration
from vramfit.adapters.outbound.scan.discovery import (
    discover_groups,
    group_count_summaries,
    max_memory_map,
)
from vramfit.adapters.outbound.scan.imatrix import (
    check_imatrix_weights,
    load_imatrix,
    resolve_assisted_weights,
    resolve_imatrix_counts,
)
from vramfit.adapters.outbound.scan.kl import mean_damage, reference_pass
from vramfit.adapters.outbound.scan.kquant import kquant_quantize_dequantize
from vramfit.adapters.outbound.scan.kquant_assisted import (
    kquant_assisted_quantize_dequantize,
)
from vramfit.adapters.outbound.scan.offload import (
    ShardReader,
    dedupe_aliased_groups,
    open_shard_reader,
    resolve_offloaded_params,
)
from vramfit.adapters.outbound.scan.quantize import (
    DEFAULT_BLOCK_SIZE,
    MIN_BITS,
    rtn_quantize_dequantize,
)
from vramfit.adapters.outbound.scan.slices import check_slice_cell
from vramfit.domain.model import ImatrixCountSummary
from vramfit.domain.scan import GroupSpec

# One weight perturbation: parameter name, candidate precision, and
# the dim-0 expert range for a slice cell (None perturbs the whole
# tensor).
_Perturbation = tuple[str, int, tuple[int, int] | None]


class TorchDamageMeter:
    """`DamageMeter` backed by torch and transformers.

    Holds the loaded model, the tokenized calibration batches, the
    cached reference distributions, and a poisoned flag set when a
    failed measurement leaves the weights unrestored. `measure`,
    `measure_recipe`, and `measure_slices` share one
    perturb-measure-restore path — a single-group recipe measures
    exactly like the scan's cell, and a slice cell perturbs a dim-0
    expert range instead of whole tensors (ADR-0026, the #200
    amendment). All three quantize through the configured
    within-group method (ADR-0018) — with per-parameter imatrix
    column weights when assisted pricing is on (ADR-0020).
    A single group's originals stay on their device. A multi-group
    perturbation stages GPU-resident originals on the CPU and
    restores offloaded originals from the safetensors shards
    (ADR-0015).

    Attributes:
        model_id (str): The Hugging Face id or local path the model
            loaded from.
        offloaded_group_count (int): How many groups hold at least one
            offloaded parameter (ADR-0015). Zero when the model fits
            the card.
        imatrix_covered_count (int | None): How many discovered
            parameters price assisted (ADR-0020). None when the
            meter is unassisted — distinct from a real zero, which
            construction refuses.
        imatrix_uncovered (tuple[str, ...] | None): Discovered
            parameters the imatrix does not cover, in discovery
            order — they price unassisted, the ``llama-quantize``
            fallback. None when the meter is unassisted.

    Examples:
        Discover groups on a tiny local model:

        ```python
        meter = TorchDamageMeter("./tiny", Path("calib.txt"), max_tokens=256)
        print([spec.name for spec in meter.groups()])
        ```
    """

    def __init__(
        self,
        model_id: str,
        calibration_path: Path,
        max_tokens: int,
        group_by: Literal["layer", "tensor", "stack"] = "layer",
        device: str = "auto",
        trust_remote_code: bool = False,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_gpu_memory: int | None = None,
        within_group: Literal["rtn", "kquant"] = "rtn",
        imatrix_weights: Mapping[str, torch.Tensor] | None = None,
        imatrix_path: Path | None = None,
    ) -> None:
        """Load the model, tokenize the calibration set, discover groups.

        Discovery and the count pooling live in
        [vramfit.adapters.outbound.scan.discovery][]. Offloaded
        parameters resolve to their weights-map backing tensors, and
        tied names that alias one storage collapse to one group
        (ADR-0015).

        Args:
            model_id: Hugging Face model id or local checkpoint path.
            calibration_path: UTF-8 calibration text file.
            max_tokens: Upper bound on calibration tokens.
            group_by: Grouping granularity — one group per decoder
                layer, one per tensor, or one per pack-addressable
                stack.
            device: transformers ``device_map`` value (``auto``,
                ``cpu``, or ``cuda``).
            trust_remote_code: Allow model repos with custom modeling
                code (the north-star target needs this).
            block_size: Elements per quantization scale block.
            max_gpu_memory: Byte cap on GPU 0 model shards for ``auto``
                sharding. Without a cap, sharding packs the card full
                and leaves no workspace for activations and logits.
            within_group: Within-group method (ADR-0018). ``rtn`` is
                the v1 round-to-nearest. ``kquant`` round-trips cells
                through the ported K-quant reference quantizers and
                refuses precisions outside their coverage.
            imatrix_weights: Imatrix column weights per parameter
                name for assisted pricing (ADR-0020). Requires the
                ``kquant`` method and must not be empty — an empty
                mapping would price every cell unassisted under the
                assisted label. A parameter absent from a non-empty
                mapping prices unassisted — the ``llama-quantize``
                fallback for a NULL imatrix row.
            imatrix_path: GGUF imatrix file to resolve column
                weights from instead of ``imatrix_weights`` — the
                two exclude each other. The file loads before the
                model, so a malformed imatrix refuses in
                milliseconds. Name resolution runs after group
                discovery, when the parameter shapes exist. A
                file-resolved meter also reads the counts and pools
                each group's expert-stack vectors into its map
                summary (ADR-0026 decision 4) — all or nothing per
                group, absent on an empty resolution. A fused entry
                whose shape or count length contradicts the model
                still refuses, the #202 vouching rule.

        Raises:
            ValueError: If ``within_group`` is not a known method —
                an unknown value must not fall back to RTN and record
                damages under the wrong token — imatrix input
                arrives with the ``rtn`` method (RTN has no weighted
                C counterpart), ``imatrix_weights`` and
                ``imatrix_path`` arrive together, the imatrix file
                is malformed or covers no parameter, a weighted name
                is unknown or its length does not match the
                parameter's rows, the calibration file yields too
                few tokens, or a quantizable group was offloaded
                beyond host RAM (an unperturbable weight would
                record zero damage).
            OSError: If the model, calibration, or imatrix file
                cannot be read.
        """
        # Checked before the model load: a silent RTN fallback under
        # a mistyped method corrupts every damage the meter measures.
        if within_group not in ("rtn", "kquant"):
            raise ValueError(
                f'within_group must be "rtn" or "kquant", got "{within_group}"'
            )
        if imatrix_weights is not None and imatrix_path is not None:
            raise ValueError(
                "give imatrix_weights or imatrix_path, not both — two "
                "weight sources cannot both be the provenance"
            )
        if imatrix_weights is not None or imatrix_path is not None:
            if within_group != "kquant":
                raise ValueError(
                    "imatrix weights require the kquant within-group method "
                    "(ADR-0020) — RTN has no weighted C counterpart"
                )
            if imatrix_weights is not None and not imatrix_weights:
                raise ValueError(
                    "imatrix_weights is empty — every cell would price "
                    "unassisted under the assisted label (ADR-0020); pass "
                    "None to price unassisted deliberately"
                )
        # The imatrix file loads before the model: a malformed file
        # refuses in milliseconds, not after minutes of shard loading.
        # Name resolution waits for group discovery below.
        pending_imatrix = (
            load_imatrix(imatrix_path) if imatrix_path is not None else None
        )
        self.model_id = model_id
        self._within_group = within_group
        self._imatrix_weights = dict(imatrix_weights or {})
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
            max_memory=max_memory_map(device, max_gpu_memory),
            trust_remote_code=trust_remote_code,
        )
        self._model.eval()
        self._model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self._batches, self._n_tokens = load_calibration(
            calibration_path, tokenizer, max_tokens
        )
        self._block_size = block_size
        groups = discover_groups(self._model, group_by)
        backing = resolve_offloaded_params(self._model, groups)
        self._groups, self._offloaded = dedupe_aliased_groups(groups, backing)
        self.offloaded_group_count = sum(
            1
            for members in self._groups.values()
            if any(name in self._offloaded for name in members)
        )
        rows_by_param = {
            name: int(self._param(name).shape[-1])
            for members in self._groups.values()
            for name in members
        }
        # Decision 4's count summary per group (ADR-0026, the #201
        # amendment): expert-stack vectors only, all or nothing per
        # group. An empty resolution leaves every summary absent —
        # no zero-coverage refusal (the #198 amendment). A fused
        # entry whose shape or count length contradicts the model
        # still refuses, the #202 vouching rule.
        self._imatrix_count_summaries: dict[str, ImatrixCountSummary] = {}
        if pending_imatrix is not None:
            shapes_by_param = {
                name: tuple(self._param(name).shape) for name in rows_by_param
            }
            resolved_counts, _ = resolve_imatrix_counts(
                pending_imatrix, shapes_by_param
            )
            self._imatrix_count_summaries = group_count_summaries(
                resolved_counts, self._groups
            )
            self._imatrix_weights, _ = resolve_assisted_weights(
                pending_imatrix, rows_by_param
            )
        # None when unassisted — distinct from zero coverage, which
        # the resolvers refuse (ADR-0020).
        self.imatrix_covered_count: int | None = None
        self.imatrix_uncovered: tuple[str, ...] | None = None
        if self._imatrix_weights:
            self.imatrix_covered_count = len(self._imatrix_weights)
            self.imatrix_uncovered = tuple(
                name for name in rows_by_param if name not in self._imatrix_weights
            )
        self._shards: ShardReader | None = None
        self._reference: list[torch.Tensor] | None = None
        self._poisoned = False
        self._poisoned_reason = ""
        check_imatrix_weights(self._imatrix_weights, rows_by_param)

    def groups(self) -> tuple[GroupSpec, ...]:
        """Discover the model's layer groups.

        Returns:
            All groups in module order, sized at 2 bytes per parameter
            (the bf16 reference), with per-tensor sizes for the
            protection pricing (ADR-0022) and, on a file-resolved
            assisted meter, each group's pooled expert-stack count
            summary (ADR-0026 decision 4). A meter built from
            in-memory weights records no summary — the counts live
            in the file.
        """
        return tuple(
            GroupSpec(
                name=name,
                tensors=tuple(tensors),
                bytes_fp16=sum(t.numel() * 2 for t in map(self._param, tensors)),
                # Per-tensor sizes ride into the map so protections
                # can price against them (ADR-0022).
                tensor_bytes={t: self._param(t).numel() * 2 for t in tensors},
                imatrix_counts=self._imatrix_count_summaries.get(name),
            )
            for name, tensors in self._groups.items()
        )

    def calibration_tokens(self) -> int:
        """Count the calibration tokens each measurement runs over.

        Returns:
            The token count.
        """
        return self._n_tokens

    def measure(self, group: str, bits: int) -> float:
        """Measure one group's damage at one candidate precision.

        A one-assignment case of the shared perturb-measure-restore
        path, expanded to whole-tensor targets by `_measure_groups`.
        The group's originals stay on their own device — no host
        round-trip on the scan's hot path. An offloaded group's own
        device is host RAM, so its originals clone there.

        Args:
            group: Name of the group to perturb.
            bits: Candidate precision to quantize the group to.

        Returns:
            Token-weighted mean KL divergence against the reference.
            Always finite and non-negative.

        Raises:
            ValueError: If the group name is unknown, ``bits`` is below
                2, or the perturbed forward pass is numerically
                unstable.
            RuntimeError: If a prior measurement failed to restore the
                model — the meter refuses to produce corrupt numbers.
        """
        self._refuse_poisoned()
        if group not in self._groups:
            raise ValueError(f'unknown group "{group}"')
        return self._measure_groups({group: bits})

    def measure_recipe(self, assignments: Mapping[str, int]) -> float:
        """Measure whole-recipe damage: all groups perturbed at once.

        The validation-pass measurement (ADR-0006), expanded to
        whole-tensor targets by `_measure_groups`. GPU-resident
        originals stage on the CPU during this pass — the card must
        keep its workspace. Offloaded originals are not staged at
        all: they restore from the model's safetensors shards
        (ADR-0015), verified restorable before any weight changes.

        Args:
            assignments: Assigned precision per group name.

        Returns:
            Token-weighted mean KL divergence against the reference.
            Always finite and non-negative.

        Raises:
            ValueError: If ``assignments`` is empty, names an unknown
                group, assigns bits below 2, perturbs offloaded groups
                without a local safetensors directory to restore from,
                or the perturbed forward pass is numerically unstable.
            RuntimeError: If a prior measurement failed to restore the
                model — the meter refuses to produce corrupt numbers.
        """
        self._refuse_poisoned()
        if not assignments:
            raise ValueError("assignments must not be empty")
        unknown = sorted(set(assignments) - set(self._groups))
        if unknown:
            raise ValueError(f'unknown group "{unknown[0]}"')
        return self._measure_groups(dict(assignments))

    def measure_slices(self, slices: Mapping[str, tuple[int, int]], bits: int) -> float:
        """Measure one slice cell: expert ranges quantized at ``bits``.

        The slice perturbation path (ADR-0026, the 2026-08-13 #200
        amendment). On a fused expert layout a per-expert cell is a
        dim-0 slice of the parameter, not a name. The meter quantizes
        each named range in place through the configured within-group
        method, keeps every other weight at reference precision, and
        measures damage as usual. A slice cell ranks or weights in
        the scan frame — its damage never sets a recipe's price.
        Validation resolves only the named parameters, so a probe
        pays no per-cell walk over the whole model.

        Args:
            slices: Half-open expert index range per fused expert
                stack, keyed by the loaded parameter name. A
                single-expert cell names one expert's range in both
                projections. A band cell names one contiguous range
                in one projection.
            bits: Candidate precision for every named range.

        Returns:
            Token-weighted mean KL divergence against the reference.
            Always finite and non-negative.

        Raises:
            ValueError: If ``slices`` is empty, names an unknown or
                non-3D parameter, a range is empty or out of bounds,
                ``bits`` is below 2, or the perturbed forward pass
                is numerically unstable.
            RuntimeError: If a prior measurement failed to restore
                the model — the meter refuses to produce corrupt
                numbers.
        """
        self._refuse_poisoned()
        # Resolve only the named parameters — a probe calls this per
        # cell, and a name outside the discovered set stays absent,
        # which is the validator's unknown-parameter refusal.
        known = {name for members in self._groups.values() for name in members}
        check_slice_cell(
            slices,
            {name: self._param(name) for name in slices if name in known},
        )
        targets: list[_Perturbation] = [
            (name, bits, expert_range) for name, expert_range in slices.items()
        ]
        return self._measure_perturbed(targets, stage_on_cpu=False, shard_plan=None)

    def _refuse_poisoned(self) -> None:
        """Refuse to measure on a meter with unrestored weights.

        Raises:
            RuntimeError: If a prior measurement failed to restore the
                model — the meter refuses to produce corrupt numbers.
                The message carries the recorded restore failure.
        """
        if self._poisoned:
            raise RuntimeError(
                "a prior measurement failed to restore the model "
                f"({self._poisoned_reason}) — rebuild the meter before "
                "measuring again"
            )

    def _measure_groups(self, assignments: dict[str, int]) -> float:
        """Measure whole groups: expand each group to its member tensors.

        Bits validate here, before shard planning spends any I/O.
        A single group's originals stay on their own device — the card
        already keeps workspace for one group, and a host round-trip
        per scan cell would cost hours over a full scan. A multi-group
        perturbation stages originals on the CPU instead: cloning
        every quantizable tensor on the card would double the model's
        footprint. Offloaded originals in a multi-group pass are not
        staged at all — staging them would double the model's host
        footprint — and restore from the safetensors shards instead
        (ADR-0015).

        Args:
            assignments: Assigned precision per validated group name.

        Returns:
            Token-weighted mean KL divergence against the reference.

        Raises:
            ValueError: If any assigned bits are below 2, offloaded
                originals cannot restore from local shards, or the
                perturbed forward pass is numerically unstable.
            RuntimeError: If the restore fails after a completed
                measurement.
        """
        # Bits validate before shard planning: rejecting a bad input
        # must not first spend shard I/O on `reader.verify` (the
        # contract pins names before bits, and bits precede the plan).
        if any(bits < MIN_BITS for bits in assignments.values()):
            raise ValueError(f"bits must be at least {MIN_BITS}")
        targets: list[_Perturbation] = [
            (name, bits, None)
            for group, bits in assignments.items()
            for name in self._groups[group]
        ]
        stage_on_cpu = len(assignments) > 1
        shard_plan = self._plan_shard_restore(assignments) if stage_on_cpu else None
        return self._measure_perturbed(targets, stage_on_cpu, shard_plan)

    def _measure_perturbed(
        self,
        targets: list[_Perturbation],
        stage_on_cpu: bool,
        shard_plan: tuple[ShardReader, set[str]] | None,
    ) -> float:
        """Quantize the targeted weights in place, measure, restore.

        The perturbation runs through the configured within-group
        method (ADR-0018), keyed by parameter name so assisted
        pricing can select each tensor's column weights (ADR-0020).
        A target carrying an expert range perturbs only that dim-0
        slice, and only the slice is saved and restored (ADR-0026,
        the #200 amendment). The restore runs in a finally clause, so
        every exit either restores the weights or poisons the meter.

        Args:
            targets: The weight perturbations to apply.
            stage_on_cpu: Stage saved originals on the CPU instead of
                their own device.
            shard_plan: The verified reader and the parameter names
                that restore from the safetensors shards, or None.

        Returns:
            Token-weighted mean KL divergence against the reference.

        Raises:
            ValueError: If any assigned bits are below 2, or the
                perturbed forward pass is numerically unstable.
            RuntimeError: If the restore fails after a completed
                measurement.
        """
        if any(bits < MIN_BITS for _, bits, _ in targets):
            raise ValueError(f"bits must be at least {MIN_BITS}")
        reference = self._ensure_reference()
        from_shards = shard_plan[1] if shard_plan else set()
        ranges = {name: rng for name, _, rng in targets if rng is not None}
        originals: dict[str, torch.Tensor] = {}
        # The restore runs in a finally so no exception — including an
        # interrupt between measurement and restore — can leave the
        # weights perturbed without either restoring or poisoning.
        try:
            with torch.no_grad():
                for name, bits, expert_range in targets:
                    param = self._param(name)
                    view = (
                        param
                        if expert_range is None
                        else param[expert_range[0] : expert_range[1]]
                    )
                    if name not in from_shards:
                        saved = view.detach()
                        originals[name] = (
                            saved.to("cpu", copy=True)
                            if stage_on_cpu
                            else saved.clone()
                        )
                    view.copy_(self._quantize_dequantize(view, bits, name))
            damage = mean_damage(self._model, self._batches, reference)
        finally:
            self._restore(
                originals, ranges, shard_plan, in_flight=sys.exception() is not None
            )
        return damage

    def _quantize_dequantize(
        self, param: torch.Tensor, bits: int, name: str
    ) -> torch.Tensor:
        """Round-trip one tensor through the configured within-group method.

        Args:
            param: The tensor to perturb.
            bits: Candidate precision.
            name: The parameter's dotted name — selects its imatrix
                column weights when assisted pricing is on
                (ADR-0020).

        Returns:
            The dequantized tensor, same shape, dtype, and device.

        Raises:
            ValueError: If the ``kquant`` method has no port for
                ``bits`` (ADR-0018 covers 8, 4, 3, and 2).
        """
        if self._within_group == "kquant":
            column_weights = self._imatrix_weights.get(name)
            if column_weights is not None:
                return kquant_assisted_quantize_dequantize(param, bits, column_weights)
            return kquant_quantize_dequantize(param, bits)
        return rtn_quantize_dequantize(param, bits, self._block_size)

    def _param(self, name: str) -> torch.Tensor:
        """Look up a parameter tensor by its dotted name.

        Args:
            name: The parameter's dotted name.

        Returns:
            The parameter tensor, or its backing CPU tensor when
            ``auto`` sharding offloaded the parameter (ADR-0015).
        """
        backing = self._offloaded.get(name)
        return backing if backing is not None else self._model.get_parameter(name)

    def _plan_shard_restore(
        self, assignments: dict[str, int]
    ) -> tuple[ShardReader, set[str]] | None:
        """Choose the offloaded tensors a multi-group pass restores from disk.

        Runs before any weight changes, so an unrestorable pass
        refuses cleanly instead of poisoning the meter.

        Args:
            assignments: Assigned precision per validated group name.

        Returns:
            The verified reader and the parameter names that restore
            from the shards. None when no assigned group is offloaded.

        Raises:
            ValueError: If the model is not a local safetensors
                directory, or a tensor has no shard entry matching the
                live tensor's shape and sample values.
            OSError: If a shard file cannot be read.
        """
        names = {
            name
            for group in assignments
            for name in self._groups[group]
            if name in self._offloaded
        }
        if not names:
            return None
        reader = self._shards or open_shard_reader(self.model_id)
        if reader is None:
            raise ValueError(
                "a whole-recipe pass over offloaded groups restores originals "
                f'from the model\'s safetensors shards, and "{self.model_id}" '
                "is not a local safetensors directory (ADR-0015)"
            )
        problem = reader.verify({name: self._offloaded[name] for name in names})
        if problem:
            raise ValueError(
                f"cannot restore offloaded originals from {self.model_id}: {problem}"
            )
        self._shards = reader
        return reader, names

    def _restore(
        self,
        originals: dict[str, torch.Tensor],
        ranges: dict[str, tuple[int, int]],
        shard_plan: tuple[ShardReader, set[str]] | None,
        in_flight: bool,
    ) -> None:
        """Copy saved weights back, poisoning the meter if that fails.

        Any failure — including an interrupt mid-copy — poisons the
        meter: a partial restore leaves corrupt weights, and the flag
        is what stops them being measured. Interrupts and exits always
        propagate. A restore failure while another exception is in
        flight is recorded on the meter and noted on the in-flight
        exception, so the root cause keeps the stage without losing
        the restore detail.

        Args:
            originals: Saved tensors keyed by parameter name.
            ranges: Perturbed dim-0 expert range per sliced parameter
                name — its saved original covers only that range.
            shard_plan: The verified reader and the names that restore
                from the safetensors shards, or None.
            in_flight: True when the caller is unwinding another
                exception.

        Raises:
            RuntimeError: If the restore itself fails and no other
                exception is already propagating.
        """
        try:
            with torch.no_grad():
                for name, original in originals.items():
                    target = self._param(name)
                    expert_range = ranges.get(name)
                    if expert_range is not None:
                        target = target[expert_range[0] : expert_range[1]]
                    target.copy_(original)
            if shard_plan is not None:
                reader, from_shards = shard_plan
                reader.read_into({name: self._param(name) for name in from_shards})
        except BaseException as exc:
            self._poisoned = True
            self._poisoned_reason = repr(exc)
            if not isinstance(exc, Exception):
                raise
            if not in_flight:
                raise RuntimeError(
                    f"failed to restore original weights: {exc}"
                ) from exc
            # Raising inside the unwind chains the in-flight exception
            # as __context__ — note the restore failure on it so the
            # detail survives past this suppression.
            if exc.__context__ is not None:
                exc.__context__.add_note(
                    f"the weight restore also failed: {exc!r} — the meter is poisoned"
                )

    def _ensure_reference(self) -> list[torch.Tensor]:
        """Run `reference_pass` on the first call and cache its result.

        Returns:
            The cached per-batch reference log-probabilities.
        """
        if self._reference is None:
            self._reference = reference_pass(self._model, self._batches)
        return self._reference
