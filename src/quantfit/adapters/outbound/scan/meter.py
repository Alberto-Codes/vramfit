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
A measurement that fails to restore the model poisons the meter: it
refuses further cells rather than measure against corrupt weights.
A model whose groups get offloaded off real devices is refused at
construction — offloaded weights cannot be perturbed in place.

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
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantfit.adapters.outbound.scan.calibration import load_calibration
from quantfit.adapters.outbound.scan.kl import mean_kl, reference_log_probs
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
    single group's originals stay on their device; a multi-group
    perturbation stages originals on the CPU.

    Attributes:
        model_id (str): The Hugging Face id or local path the model
            loaded from.

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
                or sharding offloaded any quantizable group off real
                devices — offloaded weights cannot be perturbed in
                place, and measuring them would record zero damage.
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
        self._groups = _discover_groups(self._model, group_by)
        _reject_offloaded_groups(self._model, self._groups)
        self._reference: list[torch.Tensor] | None = None
        self._poisoned = False

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
        host round-trip on the scan's hot path.

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

        The validation-pass measurement (ADR-0006). Original weights
        stage on the CPU during this pass — a whole-recipe
        perturbation clones every quantizable tensor, and the card
        must keep its workspace.

        Args:
            assignments: Assigned precision per group name.

        Returns:
            Token-weighted mean KL divergence against the reference.
            Always finite and non-negative.

        Raises:
            ValueError: If ``assignments`` is empty, names an unknown
                group, assigns bits below 2, or the perturbed forward
                pass is numerically unstable.
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
        """
        if self._poisoned:
            raise RuntimeError(
                "a prior measurement failed to restore the model — "
                "rebuild the meter before measuring again"
            )

    def _measure_perturbed(self, assignments: dict[str, int]) -> float:
        """Quantize the assigned groups in place, measure, restore.

        A single group's originals stay on their own device — the card
        already keeps workspace for one group, and a host round-trip
        per scan cell would cost hours over a full scan. A multi-group
        perturbation stages originals on the CPU instead: cloning
        every quantizable tensor on the card would double the model's
        footprint.

        Args:
            assignments: Assigned precision per validated group name.

        Returns:
            Token-weighted mean KL divergence against the reference.

        Raises:
            ValueError: If any assigned bits are below 2, or the
                perturbed forward pass is numerically unstable.
            RuntimeError: If the restore fails after a completed
                measurement.
        """
        if any(bits < MIN_BITS for bits in assignments.values()):
            raise ValueError(f"bits must be at least {MIN_BITS}")
        reference = self._ensure_reference()
        stage_on_cpu = len(assignments) > 1
        originals: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for group, bits in assignments.items():
                    for name in self._groups[group]:
                        param = self._param(name)
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
        except BaseException:
            self._restore(originals, in_flight=True)
            raise
        self._restore(originals, in_flight=False)
        return damage

    def _param(self, name: str) -> torch.Tensor:
        """Look up a parameter tensor by its dotted name.

        Args:
            name: The parameter's dotted name.

        Returns:
            The parameter tensor.
        """
        return self._model.get_parameter(name)

    def _restore(self, originals: dict[str, torch.Tensor], in_flight: bool) -> None:
        """Copy saved weights back, poisoning the meter if that fails.

        Any failure — including an interrupt mid-copy — poisons the
        meter: a partial restore leaves corrupt weights, and the flag
        is what stops them being measured. Interrupts and exits always
        propagate. A restore failure while another exception is in
        flight is recorded but not raised, so the root cause keeps the
        stage.

        Args:
            originals: Saved tensors keyed by parameter name.
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
        except BaseException as exc:
            self._poisoned = True
            if not isinstance(exc, Exception):
                raise
            if not in_flight:
                raise RuntimeError(
                    f"failed to restore original weights: {exc}"
                ) from exc

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


def _reject_offloaded_groups(
    model: torch.nn.Module, groups: dict[str, list[str]]
) -> None:
    """Refuse a model whose quantizable groups left real devices.

    ``auto`` sharding offloads overflow modules and exposes their
    parameters on the ``meta`` device. An in-place perturbation of a
    meta parameter is either an error or a silent no-op, and a no-op
    measures as zero damage — a poisoned map. Refusing is the only
    honest option until offload-aware perturbation exists.

    Args:
        model: The loaded model.
        groups: Discovered group membership.

    Raises:
        ValueError: If any group contains a meta parameter. The
            message counts affected groups and names the first three.
    """
    params = dict(model.named_parameters())
    offloaded = [
        name
        for name, members in groups.items()
        if any(params[m].is_meta for m in members)
    ]
    if offloaded:
        shown = ", ".join(offloaded[:3])
        raise ValueError(
            f"{len(offloaded)} of {len(groups)} groups were offloaded off the "
            f"GPU and cannot be measured (first: {shown}) — raise --gpu-memory, "
            "use a smaller model, or wait for offload-aware scanning"
        )


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
