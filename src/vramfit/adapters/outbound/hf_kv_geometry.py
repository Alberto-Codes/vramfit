"""Per-layer KV geometry readers for the HF ``config.json`` adapter.

Builds the `KVLayer` tuple a llama-style decoder config declares
(#421): a ``layer_types`` list of global and sliding layers, an active
``sliding_window``, split local/global head widths and KV-head counts
(``attention_k_eq_v`` gates the KV-head override, #431), and a
shared-KV tail (``num_kv_shared_layers``), and a hybrid
``layers_block_type`` stack whose non-attention blocks store no KV
(#427). A ``hybrid_override_pattern`` string expands into the same
stack, one block per letter, since published Nemotron-H files declare
the string and omit the list. The module also carries the field-label and integer-bound
helpers the whole adapter shares, and the refusal that keeps these
keys off the DeciLM path (#426). It also defines `HfConfigError`,
the refusal class both modules raise under the `VramfitError` root
(#474).

Split from [vramfit.adapters.outbound.hf_config][] to hold the
300-code-line cap. That module owns dispatch and the container rules.
This one owns what a decoder's declared KV geometry means in bytes.

Examples:
    Build the per-layer geometry for a nested decoder:

    ```python
    from pathlib import Path

    from vramfit.adapters.outbound.hf_kv_geometry import kv_layers_from_decoder

    layers = kv_layers_from_decoder(
        decoder,
        layers=60,
        kv_heads=16,
        head_dim=256,
        path=Path("config.json"),
        prefix="text_config",
    )
    ```

See Also:
    - [vramfit.adapters.outbound.hf_config][]: Dispatches config
      families and calls this module.
    - [vramfit.domain.budget][]: Prices the `KVLayer` tuple.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from vramfit.domain.budget import KVLayer
from vramfit.domain.errors import VramfitError

# The largest integer this reader admits. ADR-0008's 2026-08-16
# amendment gives the reader the format bound. The signed 64-bit range
# is that bound, and the four artifact readers already apply it (#260).
# Without it a declared count reaches `ModelShape` and raises
# `OverflowError` past the error root (#314).
INT_MAX: Final[int] = 2**63 - 1

# The letters a `hybrid_override_pattern` string uses for each block
# type, per the transformers `configuration_nemotron_h.py`
# `_pattern_to_list` helper (#427).
BLOCK_LETTERS: Final[dict[str, str]] = {
    "M": "mamba",
    "E": "moe",
    "*": "attention",
    "-": "mlp",
}


class HfConfigError(VramfitError, ValueError):
    """A ``config.json`` value the HF config reader refuses.

    Both `vramfit.adapters.outbound.hf_config` and this module raise
    it: for a file that is not UTF-8 or not JSON, a repeated key, a
    missing or malformed field, an integer past `INT_MAX`, and a
    geometry key on a config family that cannot carry it. Every
    message names the file.

    The class sits under the `VramfitError` root per ADR-0011 decision
    5. It keeps `ValueError` as a base, so the `ModelShapeSource`
    contract and every caller that catches the historical type still
    hold. Before #474 both modules raised a plain `ValueError`, which
    escaped the root. The class lives here because `hf_config` imports
    this module and not the reverse.

    Examples:
        Catch the refusal through the root:

        ```python
        from vramfit.domain.errors import VramfitError

        try:
            shape_from_config_json(path)
        except VramfitError as exc:
            print(f"error: {exc}")
        ```
    """


def kv_layers_from_decoder(
    config: dict[str, Any],
    layers: int,
    kv_heads: int,
    head_dim: int,
    path: Path,
    prefix: str,
) -> tuple[KVLayer, ...]:
    """Build per-layer KV geometry from a llama-style decoder config.

    Prices only declared geometry. A sliding layer keeps the base
    ``num_key_value_heads`` and ``head_dim``. A global layer takes
    ``global_head_dim`` when the file declares one, and takes
    ``num_global_key_value_heads`` only when ``attention_k_eq_v`` is
    true. The transformers Gemma 4 config class
    (``configuration_gemma4.py``, the ``per_layer_config`` block)
    gates the KV-head override on that flag. The flag changes no
    layer's price: the ruled runtime allocates the K and V caches
    even where it fills V with K, so every layer prices two KV
    tensors (#431). The last ``num_kv_shared_layers`` layers allocate
    no fresh KV. A hybrid ``layers_block_type`` stack (Nemotron-H,
    #427) keeps a `KVLayer` for each ``attention`` entry alone. A
    ``mamba``, ``moe``, or ``mlp`` block stores no KV, the way the
    DeciLM path skips a ``no_op`` block. A hybrid stack beside a
    shared-KV tail refuses: this reader does not model which
    attention block a shared tail reuses. Published Nemotron-H files
    declare the same stack as a ``hybrid_override_pattern`` string
    and omit the list. This reader expands the string the way the
    transformers ``_pattern_to_list`` helper does: ``M`` is
    ``mamba``, ``E`` is ``moe``, ``*`` is ``attention``, and ``-``
    is ``mlp``. When both keys come, the list is authoritative and
    the string is not read.

    The same class also synthesizes defaults this reader does not
    mirror: an absent ``global_head_dim`` defaults to 512, an absent
    ``layer_types`` synthesizes a 5:1 sliding pattern, and a sliding
    last layer is forced global at load. Every published family
    config declares the keys, so this reader prices declared values
    alone.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        layers: The decoder's hidden layer count, already parsed.
        kv_heads: The base KV head count, already parsed.
        head_dim: The base head dimension, already parsed.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Raises:
        HfConfigError: If ``layer_types`` is malformed, misses a layer,
            or declares a type this reader does not model, a sliding
            layer has no active window, an active window comes with
            no ``layer_types`` list, ``use_bidirectional_attention``
            carries a value other than ``"vision"`` or null, a
            declared ``num_global_key_value_heads`` disagrees with
            the base count while ``attention_k_eq_v`` disables it,
            ``num_kv_shared_layers`` leaves no layer that stores KV
            or leaves a shared layer with no earlier layer of its
            type, ``layers_block_type`` is malformed, misses a layer,
            names a block type this reader does not model, lists no
            ``attention`` block, comes beside ``layer_types``, or comes
            beside a ``num_kv_shared_layers`` above zero, a
            ``hybrid_override_pattern`` with no ``layers_block_type``
            list carries a letter outside the four above, misses a
            layer, or lists no ``attention`` block, or a geometry key
            carries a type it cannot mean.
            ``bool`` subclasses ``int``, so a boolean count refuses
            as a non-integer (#348). No message renders a
            publisher-controlled value (#363).

    Returns:
        One `KVLayer` per hidden layer that stores KV.
    """
    layer_types = _layer_types(config, layers, path, prefix)
    stack_key, block_types = _hybrid_stack(config, layers, path, prefix)
    if layer_types is not None and block_types is not None:
        raise HfConfigError(
            f"{path}: {field_label(stack_key, prefix)} beside "
            f"{field_label('layer_types', prefix)} declares two per-layer "
            "patterns this reader does not combine"
        )
    window = _active_window(config, path, prefix)
    if window is not None and layer_types is None:
        raise HfConfigError(
            f"{path}: {field_label('sliding_window', prefix)} declares "
            "windowed attention this reader does not model"
        )
    types = layer_types or ("full_attention",) * layers
    if "sliding_attention" in types and window is None:
        raise HfConfigError(
            f"{path}: {field_label('layer_types', prefix)} declares sliding "
            "layers with no active sliding window"
        )
    k_eq_v = _k_eq_v(config, path, prefix)
    global_dim = _optional_int(config, "global_head_dim", path, prefix)
    if global_dim is None:
        global_dim = head_dim
    global_heads = _optional_int(config, "num_global_key_value_heads", path, prefix)
    if global_heads is None:
        global_heads = kv_heads
    elif not k_eq_v and global_heads != kv_heads:
        # The transformers loader discards the override in this case.
        # This reader never silently discards a declared geometry
        # value, so the disagreement refuses instead.
        raise HfConfigError(
            f"{path}: {field_label('num_global_key_value_heads', prefix)} declares "
            "a KV-head override its attention_k_eq_v setting disables, "
            "which this reader does not model"
        )
    shared = _shared_layers(config, layers, path, prefix)
    if block_types is not None and shared > 0:
        raise HfConfigError(
            f"{path}: {field_label(stack_key, prefix)} beside "
            f"{field_label('num_kv_shared_layers', prefix)} declares a "
            "shared-KV tail over a hybrid stack this reader does not model"
        )
    fresh = layers - shared
    if not set(types[fresh:]) <= set(types[:fresh]):
        raise HfConfigError(
            f"{path}: {field_label('num_kv_shared_layers', prefix)} leaves a "
            "shared layer with no earlier layer of its type"
        )
    stores_kv = block_types or ("attention",) * layers
    return tuple(
        KVLayer(
            kv_heads=kv_heads if t == "sliding_attention" else global_heads,
            head_dim=head_dim if t == "sliding_attention" else global_dim,
            window=window if t == "sliding_attention" else None,
            shares_kv=i >= fresh,
        )
        for i, t in enumerate(types)
        if stores_kv[i] == "attention"
    )


def _layer_types(
    config: dict[str, Any], layers: int, path: Path, prefix: str
) -> tuple[str, ...] | None:
    """Read and validate a declared ``layer_types`` list.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        layers: The decoder's hidden layer count, already parsed.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        One type string per hidden layer, or ``None`` when the file
        declares none.

    Raises:
        HfConfigError: If the list is not a list of strings, misses a
            layer, or names a type other than ``full_attention`` or
            ``sliding_attention``.
    """
    layer_types = config.get("layer_types")
    if layer_types is None:
        return None
    if not isinstance(layer_types, list) or not all(
        isinstance(t, str) for t in layer_types
    ):
        raise HfConfigError(
            f"{path}: {field_label('layer_types', prefix)} must be a list of strings"
        )
    if len(layer_types) != layers:
        raise HfConfigError(
            f"{path}: {field_label('layer_types', prefix)} does not list "
            "one type per hidden layer"
        )
    if any(t not in ("full_attention", "sliding_attention") for t in layer_types):
        raise HfConfigError(
            f"{path}: {field_label('layer_types', prefix)} declares "
            "a layer type this reader does not model"
        )
    return tuple(layer_types)


def _hybrid_stack(
    config: dict[str, Any], layers: int, path: Path, prefix: str
) -> tuple[str, tuple[str, ...] | None]:
    """Read the hybrid stack from either key that declares it (#427).

    ``layers_block_type`` is authoritative when both keys come, and
    the ``hybrid_override_pattern`` beside it is not read.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        layers: The decoder's hidden layer count, already parsed.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        The key that supplied the stack and one block type per hidden
        layer, or ``("layers_block_type", None)`` when the file
        declares neither key.

    Raises:
        HfConfigError: If the key that supplies the stack is malformed.
    """
    block_types = _block_types(config, layers, path, prefix)
    if block_types is not None:
        return "layers_block_type", block_types
    pattern_types = _pattern_block_types(config, layers, path, prefix)
    if pattern_types is not None:
        return "hybrid_override_pattern", pattern_types
    return "layers_block_type", None


def _block_types(
    config: dict[str, Any], layers: int, path: Path, prefix: str
) -> tuple[str, ...] | None:
    """Read and validate a declared ``layers_block_type`` list (#427).

    Nemotron-H configs mark each hidden layer as ``attention``,
    ``mamba``, ``moe``, or ``mlp``. Only an ``attention`` block stores
    KV. The Nemotron 3.5 Lightning 30B-A3B file lists 52 entries with
    6 ``attention`` blocks, so a uniform read over-counts its KV cache
    by 8.7x. An unknown block type refuses, since it could carry a
    cache this reader does not price. The length and attention-count
    checks run in `_checked_stack`, shared with the pattern reader.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        layers: The decoder's hidden layer count, already parsed.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        One block type per hidden layer, or ``None`` when the file
        declares none.

    Raises:
        HfConfigError: If the list is not a list of strings, misses a
            layer, names a block type outside `BLOCK_LETTERS`, or
            lists no ``attention`` block.
    """
    block_types = config.get("layers_block_type")
    if block_types is None:
        return None
    label = field_label("layers_block_type", prefix)
    if not isinstance(block_types, list) or not all(
        isinstance(t, str) for t in block_types
    ):
        raise HfConfigError(f"{path}: {label} must be a list of strings")
    if any(t not in BLOCK_LETTERS.values() for t in block_types):
        raise HfConfigError(
            f"{path}: {label} declares a block type this reader does not model"
        )
    return _checked_stack(tuple(block_types), layers, label, path)


def _pattern_block_types(
    config: dict[str, Any], layers: int, path: Path, prefix: str
) -> tuple[str, ...] | None:
    """Expand a declared ``hybrid_override_pattern`` string (#427).

    The transformers Nemotron-H config class expands the string into
    ``layers_block_type`` with its ``_pattern_to_list`` helper, one
    block per letter. Published Nemotron-H files, such as the
    Nemotron 3 Nano 30B-A3B and Nemotron-H 8B files, declare the
    string and omit the list. This reader applies the same four
    letters (`BLOCK_LETTERS`) and refuses any other.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        layers: The decoder's hidden layer count, already parsed.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        One block type per hidden layer, or ``None`` when the key is
        absent or null.

    Raises:
        HfConfigError: If the value is not a string or null, carries
            a letter outside the four, misses a layer, or lists no
            ``attention`` block. An empty string misses every layer.
    """
    pattern = config.get("hybrid_override_pattern")
    if pattern is None:
        return None
    label = field_label("hybrid_override_pattern", prefix)
    if not isinstance(pattern, str):
        raise HfConfigError(f"{path}: {label} must be a string or null")
    if any(c not in BLOCK_LETTERS for c in pattern):
        raise HfConfigError(
            f"{path}: {label} declares a block letter this reader does not model"
        )
    stack = tuple(BLOCK_LETTERS[c] for c in pattern)
    return _checked_stack(stack, layers, label, path)


def _checked_stack(
    stack: tuple[str, ...], layers: int, label: str, path: Path
) -> tuple[str, ...]:
    """Check a hybrid stack's length and its attention count.

    Args:
        stack: One block type per declared entry.
        layers: The decoder's hidden layer count, already parsed.
        label: The rendered key name, for error messages.
        path: Source path, for error messages.

    Returns:
        ``stack`` unchanged.

    Raises:
        HfConfigError: If the stack misses a layer or lists no
            ``attention`` block.
    """
    if len(stack) != layers:
        raise HfConfigError(f"{path}: {label} does not list one type per hidden layer")
    if "attention" not in stack:
        raise HfConfigError(f"{path}: {label} lists no attention block")
    return stack


def _active_window(config: dict[str, Any], path: Path, prefix: str) -> int | None:
    """Read the sliding window, honoring the enable switch.

    A declared window counts as active unless ``use_sliding_window``
    is the boolean ``false`` — Qwen-family configs carry the window
    value with the switch off, and those stacks are uniform. A null
    window, a zero window, and a null switch mean unset. A negative
    or non-integer window refuses as a type error, not as declared
    windowing (#425 review).

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        The active window in tokens, or ``None``.

    Raises:
        HfConfigError: If ``sliding_window`` is not a non-negative
            integer or null, ``use_sliding_window`` is not a boolean
            or null, or ``use_bidirectional_attention`` carries a
            value other than ``"vision"`` or null. On ``"all"`` the
            runtime rescales the stored window at load, so this
            reader cannot price it. Any other value is unknown, and
            an unknown value could rescale the same way.
    """
    bidirectional = config.get("use_bidirectional_attention")
    if bidirectional is not None and bidirectional != "vision":
        raise HfConfigError(
            f"{path}: {field_label('use_bidirectional_attention', prefix)} declares "
            "bidirectional attention this reader does not model"
        )
    window = config.get("sliding_window")
    if window is not None and (
        isinstance(window, bool) or not isinstance(window, int) or window < 0
    ):
        raise HfConfigError(
            f"{path}: {field_label('sliding_window', prefix)} must be a "
            "non-negative integer or null"
        )
    switch = config.get("use_sliding_window")
    if switch is not None and not isinstance(switch, bool):
        raise HfConfigError(
            f"{path}: {field_label('use_sliding_window', prefix)} must be a boolean or null"
        )
    if isinstance(window, int) and window > 0 and switch is not False:
        return bounded_int(window, field_label("sliding_window", prefix), path)
    return None


def _k_eq_v(config: dict[str, Any], path: Path, prefix: str) -> bool:
    """Read the ``attention_k_eq_v`` flag.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        True when the file declares K=V storage. Absent and null mean
        false, matching the transformers Gemma 4 default.

    Raises:
        HfConfigError: If the value is not a boolean or null.
    """
    k_eq_v = config.get("attention_k_eq_v")
    if k_eq_v is not None and not isinstance(k_eq_v, bool):
        raise HfConfigError(
            f"{path}: {field_label('attention_k_eq_v', prefix)} must be a boolean or null"
        )
    return bool(k_eq_v)


def _shared_layers(config: dict[str, Any], layers: int, path: Path, prefix: str) -> int:
    """Read the ``num_kv_shared_layers`` count.

    The count marks the last N layers as reusing an earlier layer's
    cache, matching the transformers Gemma 4 loader.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        layers: The decoder's hidden layer count, already parsed.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        The shared-layer count, zero when absent or null.

    Raises:
        HfConfigError: If the value is not a non-negative integer or
            null, or the count leaves no layer that stores KV.
    """
    shared = _shared_count(config, path, prefix)
    if shared >= layers:
        raise HfConfigError(
            f"{path}: {field_label('num_kv_shared_layers', prefix)} "
            "leaves no layer that stores KV"
        )
    return shared


def _shared_count(config: dict[str, Any], path: Path, prefix: str) -> int:
    """Read and type-check the raw ``num_kv_shared_layers`` value.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        The declared count, zero when absent or null.

    Raises:
        HfConfigError: If the value is not a non-negative integer or
            null, or is outside the signed 64-bit range.
    """
    shared = config.get("num_kv_shared_layers")
    if shared is None:
        return 0
    if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
        raise HfConfigError(
            f"{path}: {field_label('num_kv_shared_layers', prefix)} "
            "must be a non-negative integer or null"
        )
    return bounded_int(shared, field_label("num_kv_shared_layers", prefix), path)


def _optional_int(
    config: dict[str, Any], key: str, path: Path, prefix: str
) -> int | None:
    """Read an optional positive integer from a model config.

    Args:
        config: Parsed ``config.json``, or a nested decoder object.
        key: Field to read.
        path: Source path, for error messages.
        prefix: JSON path of ``config`` inside the file, empty at the
            top level.

    Returns:
        The integer value, or ``None`` when absent or null.

    Raises:
        HfConfigError: If a present value is not a positive integer or
            null, or is outside the signed 64-bit range.
    """
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HfConfigError(
            f"{path}: {field_label(key, prefix)} must be a positive integer or null"
        )
    return bounded_int(value, field_label(key, prefix), path)


def refuse_decilm_geometry(config: dict[str, Any], path: Path) -> None:
    """Refuse llama-geometry keys beside ``block_configs`` (#426).

    The DeciLM parse prices every kept block as a global K and V
    pair. A window, KV sharing, a split local/global key, a
    ``layer_types`` or ``layers_block_type`` list, or a
    ``hybrid_override_pattern`` string beside ``block_configs``
    would silently misprice that read.
    ``attention_k_eq_v`` no longer changes a price (#431), but it
    marks a geometry family this parse does not model. Each key
    refuses.

    Args:
        config: Parsed ``config.json`` containing ``block_configs``.
        path: Source path, for error messages.

    Raises:
        HfConfigError: If a geometry key above carries an active value,
            or carries a type it cannot mean.
    """
    for key in ("layer_types", "layers_block_type", "hybrid_override_pattern"):
        if config.get(key) is not None:
            raise HfConfigError(
                f'{path}: "{key}" beside "block_configs" declares '
                "per-layer attention this reader does not model"
            )
    if _active_window(config, path, "") is not None:
        raise HfConfigError(
            f'{path}: "sliding_window" declares windowed attention '
            "this reader does not model"
        )
    if _k_eq_v(config, path, ""):
        raise HfConfigError(
            f'{path}: "attention_k_eq_v" declares K=V storage '
            "this reader does not model"
        )
    if _shared_count(config, path, "") > 0:
        raise HfConfigError(
            f'{path}: "num_kv_shared_layers" declares KV sharing '
            "this reader does not model"
        )
    for key in ("global_head_dim", "num_global_key_value_heads"):
        if _optional_int(config, key, path, "") is not None:
            raise HfConfigError(
                f'{path}: "{key}" declares split local/global '
                "attention this reader does not model"
            )


def field_label(key: str, prefix: str) -> str:
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


def bounded_int(value: int, label: str, path: Path) -> int:
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
        HfConfigError: If the value exceeds the largest signed 64-bit
            integer.
    """
    if value > INT_MAX:
        raise HfConfigError(
            f"{path}: {label} exceeds {INT_MAX}, "
            "the largest integer this format carries"
        )
    return value
