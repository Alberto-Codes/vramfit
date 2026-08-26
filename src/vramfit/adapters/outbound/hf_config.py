"""Hugging Face ``config.json`` adapter: model file → `ModelShape`.

Handles three config families. DeciLM-style NAS configs (the
north-star target) carry per-block ``block_configs`` where attention
can be deleted (``no_op``) or replaced with a linear layer
(``replace_with_linear``) — both are excluded from KV accounting.
Standard llama-style configs carry uniform layers. Composite configs
(Gemma 4, #420) nest the decoder under ``text_config``. Invalid
geometry (non-divisible GQA group sizes, non-divisible head dimensions)
is rejected rather than silently truncated.

The uniform parse models one geometry: every layer global, storing
full K and V at one head width. A decoder that declares more refuses
instead of parsing as uniform, at the top level and inside
``text_config`` alike (#420). A uniform read of a windowed stack
prices a wrong KV cache with no report. Modeling the declared
geometry stays #421.

The model publisher owns this file. vramfit reads it and never writes
it, and it still refuses a file that defines one key twice (#283). The
alternative keeps the last value, so a repeated ``num_hidden_layers``
would give a wrong `ModelShape` and a wrong weight budget with no
report.

Every integer the reader admits fits the signed 64-bit range. Every
parse failure names the file. ADR-0008's 2026-08-16 amendment gives a
reader the format bound and the domain the meaning bound. So this
module answers whether a value is representable. It never answers
whether a layer count is plausible (#314, #287).

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
from typing import Any, Final

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.domain.budget import ModelShape

# The largest integer this reader admits. ADR-0008's 2026-08-16
# amendment gives the reader the format bound. The signed 64-bit range
# is that bound, and the four artifact readers already apply it (#260).
# Without it a declared count reaches `ModelShape` and raises
# `OverflowError` past the error root (#314).
_INT_MAX: Final[int] = 2**63 - 1


def shape_from_config_json(path: Path) -> ModelShape:
    """Build a `ModelShape` from a Hugging Face ``config.json``.

    The publisher owns this file, so vramfit reads it and never writes
    it. A repeated key still refuses (#283). `json.loads` would keep the
    last value, and a repeated ``num_hidden_layers`` would then give a
    wrong `ModelShape` and a wrong weight budget, with no report.

    Dispatch order: a ``text_config`` container first (#420), then a
    DeciLM ``block_configs`` file, then a top-level llama-style config.
    A file that mixes the container with top-level decoder fields is
    ambiguous and refuses.

    Args:
        path: Path to the model's ``config.json``.

    Returns:
        The parsed shape.

    Raises:
        ValueError: If the file is not UTF-8, is not valid JSON, defines
            the same key twice, declares an integer outside the signed
            64-bit range, required fields are missing, the attention
            geometry is inconsistent, the decoder container is
            ambiguous, or the decoder declares KV geometry the uniform
            parse does not model (#420). Every message names ``path``.

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
    except ValueError as exc:
        # An integer literal past `sys.get_int_max_str_digits` (4300 by
        # default) fails here, before any extractor sees it (#287). The
        # clause sits below the two ValueError subclasses above, which
        # carry their own messages. `DuplicateKeyError` is no
        # `ValueError`, so the structural refusal cannot land here.
        raise ValueError(f"{path}: cannot parse JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if "text_config" in config:
        return _from_text_config(config, path)
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
            a boolean, an integer exceeds the largest signed 64-bit
            integer, or a block's ``n_heads_in_group`` is not a positive
            divisor of ``num_attention_heads``. A boolean
            ``n_heads_in_group`` refuses as a non-integer, and the bound
            runs before the divisor message renders the value.
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
        # `bool` subclasses `int`, so `true` read as one head group and
        # reported a shape. The two sibling extractors already guard it
        # (#348).
        if (
            isinstance(group_size, bool)
            or not isinstance(group_size, int)
            or group_size <= 0
        ):
            raise ValueError(
                f"{path}: block_configs[{i}].attention.n_heads_in_group "
                "must be a positive integer"
            )
        # Bound before the divisibility check below renders the value.
        # That message prints `group_size`. `json.loads` admits 4300
        # digits and refuses at 4301, so a 4000-digit literal reaches
        # the message and fills a terminal.
        group_size = _bounded(
            group_size, f"block_configs[{i}].attention.n_heads_in_group", path
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


def _from_text_config(config: dict[str, Any], path: Path) -> ModelShape:
    """Select the nested decoder config inside a composite model file.

    Composite configs (Gemma 4) wrap the decoder geometry in a
    ``text_config`` object beside ``vision_config`` and
    ``audio_config``. This selects that object and hands it to the
    llama-style parser, which guards the geometry it can model. An
    ambiguous container refuses: ``num_hidden_layers`` anchors the
    llama parse and ``block_configs`` anchors the DeciLM parse, so a
    file that carries either beside ``text_config`` declares two
    decoders, and this reader does not pick one.

    Args:
        config: Parsed ``config.json`` containing ``text_config``.
        path: Source path, for error messages.

    Returns:
        The parsed uniform shape of the nested decoder.

    Raises:
        ValueError: If ``text_config`` is not an object, the top level
            also declares a decoder (``block_configs`` or
            ``num_hidden_layers``), or the nested decoder refuses in
            the llama-style parser.
    """
    decoder = config["text_config"]
    if not isinstance(decoder, dict):
        raise ValueError(f'{path}: "text_config" must be a JSON object')
    for key in ("block_configs", "num_hidden_layers"):
        if key in config:
            raise ValueError(
                f'{path}: "text_config" and top-level "{key}" are both '
                "present, so the decoder config is ambiguous"
            )
    return _from_llama_config(decoder, path, prefix="text_config")


def _refuse_unmodeled_geometry(
    config: dict[str, Any], layers: int, path: Path, prefix: str = ""
) -> None:
    """Refuse a decoder whose KV geometry the uniform parse misstates.

    The uniform parse stores full-context K and V for every layer at
    one head width. A Gemma-family decoder declares more, and reading
    it as uniform prices a wrong KV cache with no report (#420).
    Modeling the declared geometry stays #421. No message renders a
    publisher-controlled value (#363).

    ``layer_types`` is admitted only when it proves uniformity: one
    string per hidden layer, every entry ``full_attention``. A recent
    transformers dump serializes the key for a plain uniform stack, so
    refusing the key outright would refuse those files.

    A declared window counts as active unless ``use_sliding_window``
    is the boolean ``false`` — Qwen-family configs carry the window
    value with the switch off, and those stacks are uniform. A null
    ``sliding_window`` also parses, because Mistral-family configs use
    null for no window.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        layers: The decoder's hidden layer count, already parsed.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Raises:
        ValueError: If ``layer_types`` is malformed, misses a layer,
            or declares a layer other than ``full_attention``, an
            active ``sliding_window`` is declared, ``attention_k_eq_v``
            is true, ``num_kv_shared_layers`` is above zero, a split
            local/global geometry key carries a value, or a key above
            carries a type it cannot mean. ``bool`` subclasses ``int``,
            so a boolean window or share count refuses as a non-integer
            (#348).
    """
    _refuse_unmodeled_layer_pattern(config, layers, path, prefix)
    _refuse_unmodeled_kv_storage(config, path, prefix)


def _refuse_unmodeled_layer_pattern(
    config: dict[str, Any], layers: int, path: Path, prefix: str
) -> None:
    """Refuse a per-layer attention pattern or an active window.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        layers: The decoder's hidden layer count, already parsed.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Raises:
        ValueError: Per `_refuse_unmodeled_geometry`.
    """
    layer_types = config.get("layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list) or not all(
            isinstance(t, str) for t in layer_types
        ):
            raise ValueError(
                f"{path}: {_label('layer_types', prefix)} must be a list of strings"
            )
        if len(layer_types) != layers:
            raise ValueError(
                f"{path}: {_label('layer_types', prefix)} does not list "
                "one type per hidden layer"
            )
        if any(t != "full_attention" for t in layer_types):
            raise ValueError(
                f"{path}: {_label('layer_types', prefix)} declares "
                "per-layer attention this reader does not model"
            )
    window = config.get("sliding_window")
    if window is not None and (isinstance(window, bool) or not isinstance(window, int)):
        raise ValueError(
            f"{path}: {_label('sliding_window', prefix)} must be an integer or null"
        )
    if (
        isinstance(window, int)
        and window > 0
        and config.get("use_sliding_window") is not False
    ):
        raise ValueError(
            f"{path}: {_label('sliding_window', prefix)} declares "
            "windowed attention this reader does not model"
        )


def _refuse_unmodeled_kv_storage(
    config: dict[str, Any], path: Path, prefix: str
) -> None:
    """Refuse K=V storage, KV sharing, or split local/global geometry.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Raises:
        ValueError: Per `_refuse_unmodeled_geometry`.
    """
    k_eq_v = config.get("attention_k_eq_v")
    if k_eq_v is not None and not isinstance(k_eq_v, bool):
        raise ValueError(
            f"{path}: {_label('attention_k_eq_v', prefix)} must be a boolean or null"
        )
    if k_eq_v:
        raise ValueError(
            f"{path}: {_label('attention_k_eq_v', prefix)} declares "
            "K=V storage this reader does not model"
        )
    shared = config.get("num_kv_shared_layers")
    if shared is not None:
        if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
            raise ValueError(
                f"{path}: {_label('num_kv_shared_layers', prefix)} "
                "must be a non-negative integer or null"
            )
        if shared > 0:
            raise ValueError(
                f"{path}: {_label('num_kv_shared_layers', prefix)} declares "
                "KV sharing this reader does not model"
            )
    for key in ("global_head_dim", "num_global_key_value_heads"):
        if config.get(key) is not None:
            raise ValueError(
                f"{path}: {_label(key, prefix)} declares split local/global "
                "attention this reader does not model"
            )


def _from_llama_config(
    config: dict[str, Any], path: Path, prefix: str = ""
) -> ModelShape:
    """Parse a standard llama-style config with uniform layers.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level. Refusals name each field through it.

    Returns:
        The parsed uniform shape.

    Raises:
        ValueError: If required fields are missing or inconsistent, or
            the decoder declares KV geometry the uniform parse does
            not model (`_refuse_unmodeled_geometry`).
    """
    layers = _config_int(config, "num_hidden_layers", path, prefix)
    _refuse_unmodeled_geometry(config, layers, path, prefix)
    kv_heads = _config_int(config, "num_key_value_heads", path, prefix)
    heads = _config_int(config, "num_attention_heads", path, prefix)
    return ModelShape.uniform(
        attn_layers=layers,
        kv_heads=kv_heads,
        head_dim=_head_dim(config, heads, path, prefix),
    )


def _label(key: str, prefix: str) -> str:
    """Name a field in a refusal message.

    A top-level key renders quoted. A nested key renders as a bare
    path expression, matching the ``block_configs[i]`` messages.

    Args:
        key: Field name.
        prefix: JSON path of the containing object, empty at the top
            level.

    Returns:
        The label for the field's messages.
    """
    return f"{prefix}.{key}" if prefix else f'"{key}"'


def _config_int(config: dict[str, Any], key: str, path: Path, prefix: str = "") -> int:
    """Read a required positive integer from a model config.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        key: Field to read.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        The integer value.

    Raises:
        ValueError: If the field is missing, is not a positive integer,
            or is outside the signed 64-bit range.
    """
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}: {_label(key, prefix)} must be a positive integer")
    return _bounded(value, _label(key, prefix), path)


def _bounded(value: int, label: str, path: Path) -> int:
    """Refuse an integer the wire format cannot carry.

    Each caller rejects a boolean and a value at or below zero first.
    So this checks the upper bound alone. A publisher's ``config.json``
    is an input vramfit never writes, and Python integers carry
    unlimited precision. Without the bound ``num_hidden_layers`` at
    10^30 reached `ModelShape.uniform`. Its repeat expression raised
    `OverflowError`, which the CLI's ``(OSError, ValueError)`` clause
    does not catch (#314).

    The bound answers representability alone. It does not answer
    whether a layer count is plausible. ADR-0008's 2026-08-16 amendment
    gives that question to the domain, and `ModelShape.uniform` still
    repeats a tuple by the layer count. So a count below this bound
    raises `MemoryError` and escapes the root the same way (#314).

    Args:
        value: A positive integer the caller has already type-checked.
        label: How the field names itself in a refusal. Match that
            field's other messages, which quote a top-level key and
            leave a path expression bare.
        path: Source path, for error messages.

    Returns:
        The integer value.

    Raises:
        ValueError: If the value exceeds the largest signed 64-bit
            integer.
    """
    if value > _INT_MAX:
        raise ValueError(
            f"{path}: {label} exceeds {_INT_MAX}, "
            "the largest integer this format carries"
        )
    return value


def _head_dim(
    config: dict[str, Any], num_heads: int, path: Path, prefix: str = ""
) -> int:
    """Derive the attention head dimension from a model config.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        num_heads: The model's attention head count.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        ``head_dim`` if present and valid, otherwise
        ``hidden_size // num_heads`` after validating exact divisibility.

    Raises:
        ValueError: If a present ``head_dim`` is not a positive integer
            inside the signed 64-bit range (a present-but-invalid value
            is rejected, never silently replaced by the fallback), or
            ``hidden_size`` is missing or not an exact multiple of
            ``num_heads``.
    """
    head_dim = config.get("head_dim")
    if head_dim is not None:
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(
                f"{path}: {_label('head_dim', prefix)} must be a positive integer"
            )
        return _bounded(head_dim, _label("head_dim", prefix), path)
    hidden = _config_int(config, "hidden_size", path, prefix)
    if hidden % num_heads != 0:
        raise ValueError(
            f"{path}: {_label('hidden_size', prefix)} {hidden} is not divisible "
            f"by {_label('num_attention_heads', prefix)} {num_heads}"
        )
    return hidden // num_heads
