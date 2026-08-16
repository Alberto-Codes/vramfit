"""Hugging Face ``config.json`` adapter: model file → `ModelShape`.

Handles two config families: DeciLM-style NAS configs (the north-star
target) with per-block ``block_configs`` where attention can be deleted
(``no_op``) or replaced with a linear layer (``replace_with_linear``) —
both are excluded from KV accounting — and standard llama-style configs
with uniform layers. Invalid
geometry (non-divisible GQA group sizes, non-divisible head dimensions)
is rejected rather than silently truncated.

The model publisher owns this file. vramfit reads it and never writes
it, and it still refuses a file that defines one key twice (#283). The
alternative keeps the last value, so a repeated ``num_hidden_layers``
would give a wrong `ModelShape` and a wrong weight budget with no
report.

Examples:
    Parse a downloaded config:

    ```python
    from pathlib import Path

    from vramfit.adapters.outbound.hf_config import shape_from_config_json

    shape = shape_from_config_json(Path("config.json"))
    ```

See Also:
    - [vramfit.ports.outbound][]: `ModelShapeSource`, which
      `HfConfigFile` satisfies.
    - [vramfit.domain.budget][]: Consumes the `ModelShape`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.domain.budget import ModelShape


def shape_from_config_json(path: Path) -> ModelShape:
    """Build a `ModelShape` from a Hugging Face ``config.json``.

    The publisher owns this file, so vramfit reads it and never writes
    it. A repeated key still refuses (#283). `json.loads` would keep the
    last value, and a repeated ``num_hidden_layers`` would then give a
    wrong `ModelShape` and a wrong weight budget, with no report.

    Args:
        path: Path to the model's ``config.json``.

    Returns:
        The parsed shape.

    Raises:
        ValueError: If the file is not UTF-8, is not valid JSON, defines
            the same key twice, required fields are missing, or the
            attention geometry is inconsistent.

    Examples:
        Standard llama-style configs parse to uniform layers:

        ```python
        shape = shape_from_config_json(Path("config.json"))
        print(len(shape.kv_heads_per_layer))
        ```
    """
    try:
        config = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=object_from_pairs
        )
    except DuplicateKeyError as exc:
        raise ValueError(f"{path}: {exc.message}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not valid UTF-8: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if "block_configs" in config:
        return _from_decilm_config(config, path)
    return _from_llama_config(config, path)


@dataclass(frozen=True, slots=True)
class HfConfigFile:
    """`ModelShapeSource` adapter backed by a ``config.json`` file.

    Attributes:
        path (Path): The config file to read.

    Examples:
        Use as a port implementation:

        ```python
        source = HfConfigFile(Path("config.json"))
        shape = source.load()
        ```
    """

    path: Path

    def load(self) -> ModelShape:
        """Read and parse the attention geometry from `path`.

        Named per the `ModelShapeSource` port contract.

        Returns:
            The parsed shape.

        Raises:
            ValueError: If the config is missing or inconsistent.
        """
        return shape_from_config_json(self.path)


def _from_decilm_config(config: dict[str, Any], path: Path) -> ModelShape:
    """Parse a DeciLM-style NAS config with per-block attention.

    Args:
        config: Parsed ``config.json`` containing ``block_configs``.
        path: Source path, for error messages.

    Returns:
        The parsed shape, with ``no_op`` and ``replace_with_linear``
        attention blocks skipped.

    Raises:
        ValueError: If required fields are missing, ``block_configs`` is
            not a list, no block has real attention, a skip flag is not
            a boolean, or a block's ``n_heads_in_group`` is not a
            positive divisor of ``num_attention_heads``.
    """
    heads = _config_int(config, "num_attention_heads", path)
    head_dim = _head_dim(config, heads, path)
    blocks = config["block_configs"]
    if not isinstance(blocks, list):
        raise ValueError(f'{path}: "block_configs" must be a list')
    kv_heads_per_layer: list[int] = []
    for i, block in enumerate(blocks):
        attention = block.get("attention") if isinstance(block, dict) else None
        if not isinstance(attention, dict):
            raise ValueError(f"{path}: block_configs[{i}] has no attention object")
        skip = False
        for flag in ("no_op", "replace_with_linear"):
            value = attention.get(flag)
            if value is not None and not isinstance(value, bool):
                raise ValueError(
                    f"{path}: block_configs[{i}].attention.{flag} must be a boolean"
                )
            skip = skip or bool(value)
        if skip:
            continue
        group_size = attention.get("n_heads_in_group")
        if not isinstance(group_size, int) or group_size <= 0:
            raise ValueError(
                f"{path}: block_configs[{i}].attention.n_heads_in_group "
                "must be a positive integer"
            )
        if heads % group_size != 0:
            raise ValueError(
                f"{path}: block_configs[{i}].attention.n_heads_in_group "
                f"{group_size} does not divide num_attention_heads {heads}"
            )
        kv_heads_per_layer.append(heads // group_size)
    if not kv_heads_per_layer:
        raise ValueError(f"{path}: no block has real attention")
    return ModelShape(kv_heads_per_layer=tuple(kv_heads_per_layer), head_dim=head_dim)


def _from_llama_config(config: dict[str, Any], path: Path) -> ModelShape:
    """Parse a standard llama-style config with uniform layers.

    Args:
        config: Parsed ``config.json``.
        path: Source path, for error messages.

    Returns:
        The parsed uniform shape.

    Raises:
        ValueError: If required fields are missing or inconsistent.
    """
    layers = _config_int(config, "num_hidden_layers", path)
    kv_heads = _config_int(config, "num_key_value_heads", path)
    heads = _config_int(config, "num_attention_heads", path)
    return ModelShape.uniform(
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
        ``head_dim`` if present and valid, otherwise
        ``hidden_size // num_heads`` after validating exact divisibility.

    Raises:
        ValueError: If a present ``head_dim`` is not a positive integer
            (a present-but-invalid value is rejected, never silently
            replaced by the fallback), or ``hidden_size`` is missing or
            not an exact multiple of ``num_heads``.
    """
    head_dim = config.get("head_dim")
    if head_dim is not None:
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f'{path}: "head_dim" must be a positive integer')
        return head_dim
    hidden = _config_int(config, "hidden_size", path)
    if hidden % num_heads != 0:
        raise ValueError(
            f'{path}: "hidden_size" {hidden} is not divisible by '
            f'"num_attention_heads" {num_heads}'
        )
    return hidden // num_heads
