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
A measurement that fails to restore the model poisons the meter: it
refuses further cells rather than measure against corrupt weights.
Groups that ``auto`` sharding offloaded to host RAM perturb through
accelerate's weights map (ADR-0015). Offloaded weights the map cannot
reach — disk spill, or an unrecognized accelerate layout — are
refused at construction.

Examples:
    Measure one cell on a local checkpoint:

    ```python
    from quantfit.adapters.outbound.scan.meter import TorchDamageMeter

    meter = TorchDamageMeter("./model", Path("calibration.txt"), max_tokens=4096)
    damage = meter.measure(meter.groups()[0].name, bits=4)
    ```

See Also:
    - [quantfit.ports.outbound][]: `DamageMeter`, the port this
      satisfies.
    - [quantfit.adapters.outbound.scan.quantize][]: The v1 within-group
      method.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantfit.adapters.outbound.scan.calibration import load_calibration
from quantfit.adapters.outbound.scan.kl import mean_kl, reference_log_probs
from quantfit.adapters.outbound.scan.offload import (
    ShardReader,
    dedupe_aliased_groups,
    open_shard_reader,
    resolve_offloaded_params,
)
from quantfit.adapters.outbound.scan.quantize import (
    DEFAULT_BLOCK_SIZE,
    MIN_BITS,
    rtn_quantize_dequantize,
)
from quantfit.domain.scan import GroupSpec

# Decoder-layer prefixes across common naming families: llama-style
# ".layers.N.", GPT-2-style ".h.N.", and ".blocks.N.".
_LAYER_PREFIX = re.compile(r"^(.*\.(?:layers|h|blocks)\.\d+)\.")


class TorchDamageMeter:
    """`DamageMeter` backed by torch and transformers.

    Holds the loaded model, the tokenized calibration batches, the
    cached reference distributions, and a poisoned flag set when a
    failed measurement leaves the weights unrestored. `measure` and
    `measure_recipe` share one perturb-measure-restore path — a
    single-group recipe measures exactly like the scan's cell. A
    single group's originals stay on their device. A multi-group
    perturbation stages GPU-resident originals on the CPU and
    restores offloaded originals from the safetensors shards
    (ADR-0015).

    Attributes:
        model_id (str): The Hugging Face id or local path the model
            loaded from.
        offloaded_group_count (int): How many groups hold at least one
            offloaded parameter (ADR-0015). Zero when the model fits
            the card.

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
        group_by: Literal["layer", "tensor"] = "layer",
        device: str = "auto",
        trust_remote_code: bool = False,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_gpu_memory: int | None = None,
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
                layer, or one per tensor.
            device: transformers ``device_map`` value (``auto``,
                ``cpu``, or ``cuda``).
            trust_remote_code: Allow model repos with custom modeling
                code (the north-star target needs this).
            block_size: Elements per quantization scale block.
            max_gpu_memory: Byte cap on GPU 0 model shards for ``auto``
                sharding. Without a cap, sharding packs the card full
                and leaves no workspace for activations and logits.

        Raises:
            ValueError: If the calibration file yields too few tokens,
                or a quantizable group was offloaded beyond host RAM —
                an unperturbable weight would record zero damage.
            OSError: If the model or calibration file cannot be read.
        """
        self.model_id = model_id
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
        self._shards: ShardReader | None = None
        self._reference: list[torch.Tensor] | None = None
        self._poisoned = False
        self._poisoned_reason = ""

    def groups(self) -> tuple[GroupSpec, ...]:
        """Discover the model's layer groups.

        Returns:
            All groups in module order, sized at 2 bytes per parameter
            (the bf16 reference).
        """
        return tuple(
            GroupSpec(
                name=name,
                tensors=tuple(tensors),
                bytes_fp16=sum(t.numel() * 2 for t in map(self._param, tensors)),
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
                        param.copy_(
                            rtn_quantize_dequantize(param, bits, self._block_size)
                        )
            damage = self._mean_damage(reference)
        finally:
            self._restore(originals, shard_plan, in_flight=sys.exception() is not None)
        return damage

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
    model: torch.nn.Module, group_by: Literal["layer", "tensor"]
) -> dict[str, list[str]]:
    """Group the model's quantizable parameters.

    Args:
        model: The loaded model.
        group_by: ``layer`` collapses each decoder layer into one
            group. ``tensor`` keeps every weight separate.

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
        if group_by == "tensor":
            key = name.removesuffix(".weight")
        else:
            match = _LAYER_PREFIX.match(name)
            layer_matches += bool(match)
            key = match.group(1) if match else name.removesuffix(".weight")
        groups.setdefault(key, []).append(name)
    if not groups:
        raise ValueError(f"no quantizable parameters found in {model.__class__}")
    if group_by == "layer" and layer_matches == 0:
        raise ValueError(
            "no per-layer structure found in this model's parameter names — "
            "pass --group-by tensor"
        )
    return groups
