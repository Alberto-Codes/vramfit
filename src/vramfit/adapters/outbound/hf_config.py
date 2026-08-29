"""Hugging Face ``config.json`` adapter: model file → `ModelShape`.

Handles three config families. DeciLM-style NAS configs (the
north-star target) carry per-block ``block_configs`` where attention
can be deleted (``no_op``) or replaced with a linear layer
(``replace_with_linear``) — both are excluded from KV accounting.
Standard llama-style configs carry uniform layers. Composite configs
(Gemma 4, #420) nest the decoder under ``text_config``. Invalid
geometry (non-divisible GQA group sizes, non-divisible head dimensions)
is rejected rather than silently truncated.

The llama-style parse models declared heterogeneous KV geometry
(#421): a ``layer_types`` list of global and sliding layers, an
active ``sliding_window``, split local/global head widths and
KV-head counts (``attention_k_eq_v`` gates the KV-head override,
#431), and a shared-KV tail (``num_kv_shared_layers``). The geometry readers live in
[vramfit.adapters.outbound.hf_kv_geometry][], which also carries the
shared field-label and integer-bound helpers. The parse prices only
what the file declares. A decoder that declares geometry past that
set refuses instead of parsing as uniform, at the top level and
inside ``text_config`` alike (#420). A uniform read of a windowed
stack prices a wrong KV cache with no report.

The model publisher owns this file. vramfit reads it and never writes
it, and it still refuses a file that defines one key twice (#283). The
alternative keeps the last value, so a repeated ``num_hidden_layers``
would give a wrong `ModelShape` and a wrong weight budget with no
report. `config_claims_vision` reads the same file through the same
strict loader and reports whether it declares ``vision_config`` —
the claim that gates the vision line in the weight budget (ADR-0030
decision 3).

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
from typing import Any

from vramfit.adapters.outbound.hf_kv_geometry import (
    bounded_int,
    field_label,
    kv_layers_from_decoder,
    refuse_decilm_geometry,
)
from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)
from vramfit.domain.budget import KVLayer, ModelShape


def shape_from_config_json(path: Path) -> ModelShape:
    """Build a `ModelShape` from a Hugging Face ``config.json``.

    The publisher owns this file, so vramfit reads it and never writes
    it. The strict load lives in `_load_config`, shared with
    `config_claims_vision`: a repeated key still refuses (#283), since
    `json.loads` would keep the last value, and a repeated
    ``num_hidden_layers`` would then give a wrong `ModelShape` and a
    wrong weight budget, with no report.

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
            ambiguous, or the decoder declares KV geometry this
            reader does not model (#420, #421). Every message names
            ``path``.

    Examples:
        Standard llama-style configs parse to uniform layers:

        ```python
        shape = shape_from_config_json(Path("config.json"))
        print(len(shape.kv_layers))
        ```
    """
    config = _load_config(path)
    if "text_config" in config:
        return _from_text_config(config, path)
    if "block_configs" in config:
        return _from_decilm_config(config, path)
    return _from_llama_config(config, path)


def config_claims_vision(path: Path) -> bool:
    """Report whether the model card claims vision.

    The claim is mechanical: the top level declares a
    ``vision_config`` JSON object. Composite files (Gemma 4) carry
    it beside ``text_config``, and no admitted config nests it
    deeper. A ``vision_config`` that is not an object — ``null``
    included — claims nothing. The claim gates the vision line in
    the weight budget (ADR-0030 decision 3) — it prices nothing
    itself.

    Args:
        path: Path to the model's ``config.json``.

    Returns:
        True when the file declares a top-level ``vision_config``
        object.

    Raises:
        ValueError: If the file is not UTF-8, is not valid JSON,
            defines the same key twice, or is not a JSON object. The
            same refusals as `shape_from_config_json`, so the two
            reads of one file cannot disagree on validity.

    Examples:
        A text-only config claims no vision:

        ```python
        claims = config_claims_vision(Path("config.json"))
        ```
    """
    return isinstance(_load_config(path).get("vision_config"), dict)


def _load_config(path: Path) -> dict[str, Any]:
    """Read and parse a ``config.json`` into a validated object.

    Args:
        path: Path to the model's ``config.json``.

    Returns:
        The parsed top-level object.

    Raises:
        ValueError: If the file is not UTF-8, is not valid JSON,
            defines the same key twice (#283), declares an integer
            past the parser's digit bound (#287), or is not a JSON
            object. Every message names ``path``.
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
    return config


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
            integer, a block's ``n_heads_in_group`` is not a positive
            divisor of ``num_attention_heads``, or a llama-geometry
            key beside ``block_configs`` declares a window, K=V
            storage, KV sharing, a split local/global key, or a
            ``layer_types`` list (`refuse_decilm_geometry`, #426). A
            boolean ``n_heads_in_group`` refuses as a non-integer, and
            the bound runs before the divisor message renders the
            value.
    """
    refuse_decilm_geometry(config, path)
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
        group_size = bounded_int(
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
    return ModelShape(
        kv_layers=tuple(
            KVLayer(kv_heads=kv_heads, head_dim=head_dim)
            for kv_heads in kv_heads_per_layer
        )
    )


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
            ``num_hidden_layers``), the nested decoder carries
            ``block_configs`` (a NAS decoder inside a composite file,
            which the llama-style parser would flatten to a wrong KV
            price, #426), or the nested decoder refuses in the
            llama-style parser.
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
    if "block_configs" in decoder:
        raise ValueError(
            f"{path}: text_config.block_configs declares a NAS decoder "
            "this reader does not model"
        )
    return _from_llama_config(decoder, path, prefix="text_config")


def _from_llama_config(
    config: dict[str, Any], path: Path, prefix: str = ""
) -> ModelShape:
    """Parse a llama-style config into per-layer KV geometry.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level. Refusals name each field through it.

    Returns:
        The parsed shape, uniform when the file declares no per-layer
        geometry.

    Raises:
        ValueError: If required fields are missing or inconsistent, or
            the decoder declares KV geometry this reader does not
            model
            (`vramfit.adapters.outbound.hf_kv_geometry.kv_layers_from_decoder`).
    """
    layers = _config_int(config, "num_hidden_layers", path, prefix)
    kv_heads = _config_int(config, "num_key_value_heads", path, prefix)
    heads = _config_int(config, "num_attention_heads", path, prefix)
    head_dim = _head_dim(config, heads, path, prefix)
    return ModelShape(
        kv_layers=kv_layers_from_decoder(
            config, layers, kv_heads, head_dim, path, prefix
        )
    )


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
            or is outside the signed 64-bit range. The message names
            the field through `field_label`, and `bounded_int` applies
            the range.
    """
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{path}: {field_label(key, prefix)} must be a positive integer"
        )
    return bounded_int(value, field_label(key, prefix), path)


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
            ``num_heads``. `bounded_int` applies the range.
    """
    head_dim = config.get("head_dim")
    if head_dim is not None:
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(
                f"{path}: {field_label('head_dim', prefix)} must be a positive integer"
            )
        return bounded_int(head_dim, field_label("head_dim", prefix), path)
    hidden = _config_int(config, "hidden_size", path, prefix)
    if hidden % num_heads != 0:
        raise ValueError(
            f"{path}: {field_label('hidden_size', prefix)} {hidden} is not divisible "
            f"by {field_label('num_attention_heads', prefix)} {num_heads}"
        )
    return hidden // num_heads
