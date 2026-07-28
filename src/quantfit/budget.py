"""VRAM budget math: size parsing, KV-cache cost, and the weight budget.

Implements the arithmetic from ``docs/explanation/vram-budget.md``:
``weight_budget = vram_total − kv_cache − runtime_overhead``. KV cost is a
sum over attention layers, not ``layers × constant``, because NAS-derived
models like the north-star target delete attention from some layers.

Attributes:
    KV_DTYPE_BYTES (dict[str, int]): Bytes per KV-cache element by dtype
        name (``fp16``, ``bf16``, ``fp8``).
    DEFAULT_RUNTIME_OVERHEAD_BYTES (int): Planning figure for CUDA
        context, workspace, and fragmentation (2 GiB).

Examples:
    Compute the weight budget for the north-star target:

    ```python
    from quantfit.budget import Budget, ModelShape, kv_cache_bytes, parse_size

    shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
    budget = Budget(
        vram_total_bytes=parse_size("24GiB"),
        kv_cache_bytes=kv_cache_bytes(shape, context=16384, kv_dtype="fp8"),
        runtime_overhead_bytes=parse_size("2GiB"),
    )
    print(budget.weight_budget_bytes)
    ```

See Also:
    - [quantfit.solver][]: Packs weights against the budget computed here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

KV_DTYPE_BYTES: Final[dict[str, int]] = {"fp16": 2, "bf16": 2, "fp8": 1}
DEFAULT_RUNTIME_OVERHEAD_BYTES: Final[int] = 2 * 2**30

_SIZE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]i?B|B)?\s*$", re.IGNORECASE
)
_UNIT_BYTES: Final[dict[str, int]] = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}


def parse_size(text: str) -> int:
    """Parse a human-readable size into bytes.

    Binary units (``GiB``) are powers of 1024; decimal units (``GB``) are
    powers of 1000. A bare number is bytes.

    Args:
        text: Size string, e.g. ``"24GiB"``, ``"4 GB"``, ``"1073741824"``.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the string is not a recognizable size.

    Examples:
        Binary vs decimal units:

        ```python
        from quantfit.budget import parse_size

        assert parse_size("1GiB") == 2**30
        assert parse_size("1GB") == 10**9
        ```
    """
    match = _SIZE_RE.match(text)
    if match is None:
        raise ValueError(f'not a recognizable size: "{text}"')
    number = float(match.group("number"))
    unit = (match.group("unit") or "B").lower()
    return int(number * _UNIT_BYTES[unit])


def format_size(n_bytes: int) -> str:
    """Render a byte count in binary units with two decimals.

    Args:
        n_bytes: Byte count to render.

    Returns:
        A string like ``"19.42 GiB"``; plain ``"512 B"`` below 1 KiB.

    Examples:
        Render two gibibytes:

        ```python
        from quantfit.budget import format_size

        assert format_size(2 * 2**30) == "2.00 GiB"
        ```
    """
    magnitude = abs(n_bytes)
    for unit, size in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if magnitude >= size:
            return f"{n_bytes / size:.2f} {unit}"
    return f"{n_bytes} B"


@dataclass(frozen=True, slots=True)
class ModelShape:
    """The attention geometry that determines KV-cache cost.

    Attributes:
        kv_heads_per_layer (tuple[int, ...]): KV head count for each
            attention layer. Layers without attention (NAS ``no_op``
            blocks) have no entry.
        head_dim (int): Dimension of each attention head.

    Examples:
        The north-star target's shape, built by hand:

        ```python
        from quantfit.budget import ModelShape

        shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
        assert len(shape.kv_heads_per_layer) == 49
        ```
    """

    kv_heads_per_layer: tuple[int, ...]
    head_dim: int

    @classmethod
    def uniform(cls, attn_layers: int, kv_heads: int, head_dim: int) -> ModelShape:
        """Build a shape where every attention layer is identical.

        Args:
            attn_layers: Number of layers that have attention.
            kv_heads: KV heads per attention layer.
            head_dim: Dimension of each attention head.

        Returns:
            The uniform shape.
        """
        return cls(
            kv_heads_per_layer=(kv_heads,) * attn_layers,
            head_dim=head_dim,
        )

    @classmethod
    def from_config_json(cls, path: Path) -> ModelShape:
        """Build a shape from a Hugging Face ``config.json``.

        Handles two config families: DeciLM-style NAS configs (the
        north-star target) with per-block ``block_configs`` where
        attention can be ``no_op``, and standard llama-style configs with
        uniform layers.

        Args:
            path: Path to the model's ``config.json``.

        Returns:
            The parsed shape.

        Raises:
            ValueError: If required fields are missing or the file is not
                valid JSON.

        Examples:
            Parse a downloaded config:

            ```python
            from pathlib import Path

            from quantfit.budget import ModelShape

            shape = ModelShape.from_config_json(Path("config.json"))
            ```
        """
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"{path}: expected a JSON object")  # noqa: TRY004
        if "block_configs" in config:
            return cls._from_decilm_config(config, path)
        return cls._from_llama_config(config, path)

    @classmethod
    def _from_decilm_config(cls, config: dict[str, Any], path: Path) -> ModelShape:
        """Parse a DeciLM-style NAS config with per-block attention.

        Args:
            config: Parsed ``config.json`` containing ``block_configs``.
            path: Source path, for error messages.

        Returns:
            The parsed shape, with ``no_op`` attention blocks skipped.

        Raises:
            ValueError: If required fields are missing.
        """
        heads = _config_int(config, "num_attention_heads", path)
        head_dim = _head_dim(config, heads, path)
        kv_heads_per_layer: list[int] = []
        for i, block in enumerate(config["block_configs"]):
            attention = block.get("attention") if isinstance(block, dict) else None
            if not isinstance(attention, dict):
                raise ValueError(f"{path}: block_configs[{i}] has no attention object")  # noqa: TRY004
            if attention.get("no_op") or attention.get("replace_with_linear"):
                continue
            group_size = attention.get("n_heads_in_group")
            if not isinstance(group_size, int) or group_size <= 0:
                raise ValueError(
                    f"{path}: block_configs[{i}].attention.n_heads_in_group "
                    "must be a positive integer"
                )
            kv_heads_per_layer.append(heads // group_size)
        return cls(kv_heads_per_layer=tuple(kv_heads_per_layer), head_dim=head_dim)

    @classmethod
    def _from_llama_config(cls, config: dict[str, Any], path: Path) -> ModelShape:
        """Parse a standard llama-style config with uniform layers.

        Args:
            config: Parsed ``config.json``.
            path: Source path, for error messages.

        Returns:
            The parsed uniform shape.

        Raises:
            ValueError: If required fields are missing.
        """
        layers = _config_int(config, "num_hidden_layers", path)
        kv_heads = _config_int(config, "num_key_value_heads", path)
        heads = _config_int(config, "num_attention_heads", path)
        return cls.uniform(
            attn_layers=layers,
            kv_heads=kv_heads,
            head_dim=_head_dim(config, heads, path),
        )


def _config_int(config: dict[str, Any], key: str, path: Path) -> int:
    """Read a required positive integer from a model config.

    Args:
        config: Parsed ``config.json``.
        key: Field to read.
        path: Source path, for error messages.

    Returns:
        The integer value.

    Raises:
        ValueError: If the field is missing or not a positive integer.
    """
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{path}: "{key}" must be a positive integer')
    return value


def _head_dim(config: dict[str, Any], num_heads: int, path: Path) -> int:
    """Derive the attention head dimension from a model config.

    Args:
        config: Parsed ``config.json``.
        num_heads: The model's attention head count.
        path: Source path, for error messages.

    Returns:
        ``head_dim`` if present, otherwise ``hidden_size // num_heads``.

    Raises:
        ValueError: If neither ``head_dim`` nor ``hidden_size`` is usable.
    """
    head_dim = config.get("head_dim")
    if isinstance(head_dim, int) and not isinstance(head_dim, bool) and head_dim > 0:
        return head_dim
    return _config_int(config, "hidden_size", path) // num_heads


def kv_bytes_per_token(shape: ModelShape, kv_dtype: str = "fp16") -> int:
    """Compute KV-cache bytes stored per token across the whole stack.

    The formula is ``2 (keys + values) × head_dim × Σ kv_heads ×
    bytes_per_element``, summing over attention layers only.

    Args:
        shape: The model's attention geometry.
        kv_dtype: KV-cache element dtype (``fp16``, ``bf16``, or ``fp8``).

    Returns:
        Bytes per token.

    Raises:
        KeyError: If ``kv_dtype`` is not a known dtype.

    Examples:
        The north-star target stores ~196 KiB per token at fp16:

        ```python
        from quantfit.budget import ModelShape, kv_bytes_per_token

        shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
        assert kv_bytes_per_token(shape) == 200_704
        ```
    """
    element_bytes = KV_DTYPE_BYTES[kv_dtype]
    return 2 * shape.head_dim * sum(shape.kv_heads_per_layer) * element_bytes


def kv_cache_bytes(
    shape: ModelShape,
    context: int,
    kv_dtype: str = "fp16",
    sequences: int = 1,
) -> int:
    """Compute total KV-cache bytes for a context length and batch.

    Args:
        shape: The model's attention geometry.
        context: Context length in tokens.
        kv_dtype: KV-cache element dtype.
        sequences: Concurrent sequences sharing the card.

    Returns:
        Total KV-cache bytes.

    Examples:
        Cache for 16k context on the north-star target:

        ```python
        from quantfit.budget import ModelShape, kv_cache_bytes

        shape = ModelShape.uniform(attn_layers=49, kv_heads=8, head_dim=128)
        total = kv_cache_bytes(shape, context=16384)
        ```
    """
    return kv_bytes_per_token(shape, kv_dtype) * context * sequences


@dataclass(frozen=True, slots=True)
class Budget:
    """The VRAM ledger that yields the weight budget.

    Attributes:
        vram_total_bytes (int): The card's total VRAM.
        kv_cache_bytes (int): Bytes reserved for the KV cache.
        runtime_overhead_bytes (int): Bytes reserved for CUDA context,
            workspace, and fragmentation.

    Examples:
        A ledger that leaves ~19 GiB for weights:

        ```python
        from quantfit.budget import Budget, parse_size

        budget = Budget(
            vram_total_bytes=parse_size("24GiB"),
            kv_cache_bytes=parse_size("3GiB"),
            runtime_overhead_bytes=parse_size("2GiB"),
        )
        assert budget.weight_budget_bytes == parse_size("19GiB")
        ```
    """

    vram_total_bytes: int
    kv_cache_bytes: int
    runtime_overhead_bytes: int

    @property
    def weight_budget_bytes(self) -> int:
        """Bytes left for weights; negative when overcommitted.

        Returns:
            ``vram_total − kv_cache − runtime_overhead``.
        """
        return self.vram_total_bytes - self.kv_cache_bytes - self.runtime_overhead_bytes
