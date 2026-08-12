"""`DamageMeter` adapter: perturb one group, measure KL on real weights.

The core scan loop per ADR-0006: quantize-dequantize one layer group in
place, run the calibration set, compare final logits against the cached
reference distribution, restore the group. The validation pass reuses
the same machinery with every group perturbed at once
(`TorchDamageMeter.measure_recipe`). Reference log-probabilities
are cached on the CPU in float16 (~0.25 GiB per 1024 tokens at a 128k
vocabulary) so the reference model runs exactly once.

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
coverage split for the run log. An assisted run resolved from a
file also summarizes each group's imatrix counts into the map, as
provenance the solver never reads (ADR-0026 decision 4).

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
from vramfit.adapters.outbound.scan.imatrix import (
    check_imatrix_weights,
    load_imatrix,
    resolve_assisted_weights,
    resolve_imatrix_counts,
)
from vramfit.adapters.outbound.scan.kl import mean_kl, reference_log_probs
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
from vramfit.domain.model import ImatrixCountSummary
from vramfit.domain.scan import GroupSpec, group_key, matches_a_layer


class TorchDamageMeter:
    """`DamageMeter` backed by torch and transformers.

    Holds the loaded model, the tokenized calibration batches, the
    cached reference distributions, and a poisoned flag set when a
    failed measurement leaves the weights unrestored. `measure` and
    `measure_recipe` share one perturb-measure-restore path — a
    single-group recipe measures exactly like the scan's cell. Both
    quantize through the configured within-group method (ADR-0018) —
    with per-parameter imatrix column weights when assisted pricing
    is on (ADR-0020).
    A single group's originals stay on their device. A multi-group
    perturbation stages GPU-resident originals on the CPU and
    restores offloaded originals from the safetensors shards
    (ADR-0015). An assisted meter resolved from a file also holds
    each parameter's imatrix count, which `groups` reduces to the
    per-group summary the map records (ADR-0026 decision 4).

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

        Offloaded parameters resolve to their weights-map backing
        tensors, and tied names that alias one storage collapse to
        one group (ADR-0015).

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
                discovery, when the parameter shapes exist. This
                path also reads each parameter's imatrix count for
                the map's provenance (ADR-0026 decision 4).
                ``imatrix_weights`` carries no counts, so a meter
                built that way records no summary.

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
            max_memory=_max_memory(device, max_gpu_memory),
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
        groups = _discover_groups(self._model, group_by)
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
        # Counts read through their own resolver, never the assisted
        # one: they are routing frequency, not a fit, so no
        # super-block gate applies and an expert stack whose rows
        # refuse a k-quant fit still reports its distribution (#177,
        # ADR-0026 decision 4).
        self._imatrix_counts: dict[str, int] = {}
        if pending_imatrix is not None:
            self._imatrix_weights, _ = resolve_assisted_weights(
                pending_imatrix, rows_by_param
            )
            self._imatrix_counts, _ = resolve_imatrix_counts(
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
            protection pricing (ADR-0022) and, on an assisted run
            resolved from a file, the group's imatrix count summary
            (ADR-0026 decision 4).
        """
        return tuple(
            GroupSpec(
                name=name,
                tensors=tuple(tensors),
                bytes_fp16=sum(t.numel() * 2 for t in map(self._param, tensors)),
                # Per-tensor sizes ride into the map so protections
                # can price against them (ADR-0022).
                tensor_bytes={t: self._param(t).numel() * 2 for t in tensors},
                imatrix_counts=self._count_summary(tensors),
            )
            for name, tensors in self._groups.items()
        )

    def _count_summary(self, tensors: list[str]) -> ImatrixCountSummary | None:
        """Reduce one group's imatrix counts to its map provenance.

        Args:
            tensors: The group's member parameter names.

        Returns:
            The summary over the members the imatrix covers, or None
            when it covers none of them (ADR-0026 decision 4). A
            partly covered group summarizes the members it does
            cover — the alternative drops a 128-expert stack's whole
            distribution over one missing member. The summary's
            ``covered`` field carries how many members that was, so a
            reader can size the distribution rather than assume it
            spans the group.
        """
        counts = [
            self._imatrix_counts[name]
            for name in tensors
            if name in self._imatrix_counts
        ]
        return ImatrixCountSummary.from_counts(counts) if counts else None

    def calibration_tokens(self) -> int:
        """Count the calibration tokens each measurement runs over.

        Returns:
            The token count.
        """
        return self._n_tokens

    def measure(self, group: str, bits: int) -> float:
        """Measure one group's damage at one candidate precision.

        A one-assignment case of the shared perturb-measure-restore
        path. The group's originals stay on their own device — no
        host round-trip on the scan's hot path. An offloaded group's
        own device is host RAM, so its originals clone there.

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
        return self._measure_perturbed({group: bits})

    def measure_recipe(self, assignments: Mapping[str, int]) -> float:
        """Measure whole-recipe damage: all groups perturbed at once.

        The validation-pass measurement (ADR-0006). GPU-resident
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
        return self._measure_perturbed(dict(assignments))

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

    def _measure_perturbed(self, assignments: dict[str, int]) -> float:
        """Quantize the assigned groups in place, measure, restore.

        The perturbation runs through the configured within-group
        method (ADR-0018), keyed by parameter name so assisted
        pricing can select each tensor's column weights (ADR-0020).
        A single group's originals stay on their own device — the card
        already keeps workspace for one group, and a host round-trip
        per scan cell would cost hours over a full scan. A multi-group
        perturbation stages originals on the CPU instead: cloning
        every quantizable tensor on the card would double the model's
        footprint. Offloaded originals in a multi-group pass are not
        staged at all — staging them would double the model's host
        footprint — and restore from the safetensors shards instead
        (ADR-0015). The restore runs in a finally clause, so every
        exit either restores the weights or poisons the meter.

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
        if any(bits < MIN_BITS for bits in assignments.values()):
            raise ValueError(f"bits must be at least {MIN_BITS}")
        reference = self._ensure_reference()
        stage_on_cpu = len(assignments) > 1
        shard_plan = self._plan_shard_restore(assignments) if stage_on_cpu else None
        from_shards = shard_plan[1] if shard_plan else set()
        originals: dict[str, torch.Tensor] = {}
        # The restore runs in a finally so no exception — including an
        # interrupt between measurement and restore — can leave the
        # weights perturbed without either restoring or poisoning.
        try:
            with torch.no_grad():
                for group, bits in assignments.items():
                    for name in self._groups[group]:
                        param = self._param(name)
                        if name not in from_shards:
                            saved = param.detach()
                            originals[name] = (
                                saved.to("cpu", copy=True)
                                if stage_on_cpu
                                else saved.clone()
                            )
                        param.copy_(self._quantize_dequantize(param, bits, name))
            damage = self._mean_damage(reference)
        finally:
            self._restore(originals, shard_plan, in_flight=sys.exception() is not None)
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
                    self._param(name).copy_(original)
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
        """Run the unperturbed model once and cache its distributions.

        Returns:
            The cached per-batch reference log-probabilities.
        """
        if self._reference is None:
            reference: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch in self._batches:
                    logits = self._model(batch.to(self._model.device)).logits
                    reference.append(reference_log_probs(logits))
            self._reference = reference
        return self._reference

    def _mean_damage(self, reference: list[torch.Tensor]) -> float:
        """Run the perturbed model over all batches and average KL.

        Args:
            reference: Cached per-batch reference log-probabilities.

        Returns:
            Token-weighted mean KL divergence.
        """
        total = 0.0
        tokens = 0
        with torch.inference_mode():
            for batch, ref in zip(self._batches, reference, strict=True):
                logits = self._model(batch.to(self._model.device)).logits
                total += mean_kl(ref, logits) * batch.numel()
                tokens += batch.numel()
        return total / tokens


def _max_memory(device: str, max_gpu_memory: int | None) -> dict[int | str, int] | None:
    """Build the accelerate ``max_memory`` map for a GPU shard cap.

    The cap applies to GPU 0 only — the reference box has one card.
    The integer device key is required: accelerate rejects ``"0"``.

    Args:
        device: The ``device_map`` value.
        max_gpu_memory: Byte cap on GPU 0 shards, or None for no cap.

    Returns:
        The map for ``auto`` sharding with a cap, otherwise None.
    """
    if max_gpu_memory is None or device != "auto":
        return None
    return {0: max_gpu_memory, "cpu": 999 * 2**30}


def _discover_groups(
    model: torch.nn.Module, group_by: Literal["layer", "tensor", "stack"]
) -> dict[str, list[str]]:
    """Group the model's quantizable parameters.

    Discovery walks the parameters and filters them. The naming rule
    itself lives in [vramfit.domain.scan][] (`group_key`), so the
    fast suite covers every granularity without torch.

    Args:
        model: The loaded model.
        group_by: Grouping granularity, passed through to `group_key`.

    Returns:
        Ordered mapping of group name to member parameter names. Only
        floating-point tensors with 2+ dimensions are included.

    Raises:
        ValueError: If no quantizable parameters are found, or
            ``layer`` grouping finds no per-layer structure — silently
            degrading to per-tensor groups would misrepresent the map.
    """
    groups: dict[str, list[str]] = {}
    layer_matches = 0
    for name, param in model.named_parameters():
        if param.ndim < 2 or not param.is_floating_point():  # noqa: PLR2004
            continue
        layer_matches += group_by == "layer" and matches_a_layer(name)
        groups.setdefault(group_key(name, group_by), []).append(name)
    if not groups:
        raise ValueError(f"no quantizable parameters found in {model.__class__}")
    if group_by == "layer" and layer_matches == 0:
        raise ValueError(
            "no per-layer structure found in this model's parameter names — "
            "pass --group-by tensor"
        )
    return groups
