"""`DamageMeter` adapter: perturb one group, measure KL on real weights.

The core scan loop per ADR-0006: quantize-dequantize one layer group in
place, run the calibration set, compare final logits against the cached
reference distribution, restore the group. Reference log-probabilities
are cached on the CPU in float16 (~0.25 GiB per 1024 tokens at a 128k
vocabulary) so the reference model runs exactly once.

Only floating-point tensors with 2+ dimensions join layer groups —
norms and biases stay at reference precision and are not scanned.

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
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantfit.adapters.outbound.scan.calibration import load_calibration
from quantfit.adapters.outbound.scan.kl import mean_kl, reference_log_probs
from quantfit.adapters.outbound.scan.quantize import (
    DEFAULT_BLOCK_SIZE,
    rtn_quantize_dequantize,
)
from quantfit.domain.scan import GroupSpec

_LAYER_PREFIX = re.compile(r"^(.*\.layers\.\d+)\.")


class TorchDamageMeter:
    """`DamageMeter` backed by torch and transformers.

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
    ) -> None:
        """Load the model and tokenize the calibration set.

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

        Raises:
            ValueError: If the calibration file yields too few tokens.
            OSError: If the model or calibration file cannot be read.
        """
        self.model_id = model_id
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
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
        self._reference: list[torch.Tensor] | None = None

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

        Args:
            group: Name of the group to perturb.
            bits: Candidate precision to quantize the group to.

        Returns:
            Token-weighted mean KL divergence against the reference.

        Raises:
            ValueError: If the group name is unknown or ``bits`` is
                below 2.
        """
        if group not in self._groups:
            raise ValueError(f'unknown group "{group}"')
        self._ensure_reference()
        originals: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name in self._groups[group]:
                    param = self._param(name)
                    originals[name] = param.detach().clone()
                    param.copy_(rtn_quantize_dequantize(param, bits, self._block_size))
            return self._mean_damage()
        finally:
            with torch.no_grad():
                for name, original in originals.items():
                    self._param(name).copy_(original)

    def _param(self, name: str) -> torch.Tensor:
        """Look up a parameter tensor by its dotted name.

        Args:
            name: The parameter's dotted name.

        Returns:
            The parameter tensor.
        """
        return self._model.get_parameter(name)

    def _ensure_reference(self) -> None:
        """Run the unperturbed model once and cache its distributions."""
        if self._reference is not None:
            return
        reference: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in self._batches:
                logits = self._model(batch.to(self._model.device)).logits
                reference.append(reference_log_probs(logits))
        self._reference = reference

    def _mean_damage(self) -> float:
        """Run the perturbed model over all batches and average KL.

        Returns:
            Token-weighted mean KL divergence.
        """
        assert self._reference is not None  # noqa: S101 - _ensure_reference ran
        total = 0.0
        tokens = 0
        with torch.inference_mode():
            for batch, ref in zip(self._batches, self._reference, strict=True):
                logits = self._model(batch.to(self._model.device)).logits
                total += mean_kl(ref, logits) * batch.numel()
                tokens += batch.numel()
        return total / tokens


def _discover_groups(
    model: torch.nn.Module, group_by: Literal["layer", "tensor"]
) -> dict[str, list[str]]:
    """Group the model's quantizable parameters.

    Args:
        model: The loaded model.
        group_by: ``layer`` collapses each decoder layer into one
            group; ``tensor`` keeps every weight separate.

    Returns:
        Ordered mapping of group name to member parameter names. Only
        floating-point tensors with 2+ dimensions are included.

    Raises:
        ValueError: If no quantizable parameters are found.
    """
    groups: dict[str, list[str]] = {}
    for name, param in model.named_parameters():
        if param.ndim < 2 or not param.is_floating_point():  # noqa: PLR2004
            continue
        if group_by == "tensor":
            key = name.removesuffix(".weight")
        else:
            match = _LAYER_PREFIX.match(name)
            key = match.group(1) if match else name.removesuffix(".weight")
        groups.setdefault(key, []).append(name)
    if not groups:
        raise ValueError(f"no quantizable parameters found in {model.__class__}")
    return groups
