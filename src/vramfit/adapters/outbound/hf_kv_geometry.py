"""Per-layer KV geometry readers for the HF ``config.json`` adapter.

Builds the `KVLayer` tuple a llama-style decoder config declares
(#421): a ``layer_types`` list of global and sliding layers, an active
``sliding_window``, split local/global head widths and KV-head counts
(``attention_k_eq_v`` gates the KV-head override, #431), and a
shared-KV tail (``num_kv_shared_layers``). The module also carries
the field-label and integer-bound helpers the whole adapter shares,
and the refusal that keeps these keys off the DeciLM path (#426).

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

# The largest integer this reader admits. ADR-0008's 2026-08-16
# amendment gives the reader the format bound. The signed 64-bit range
# is that bound, and the four artifact readers already apply it (#260).
# Without it a declared count reaches `ModelShape` and raises
# `OverflowError` past the error root (#314).
INT_MAX: Final[int] = 2**63 - 1


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
    no fresh KV.

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
        ValueError: If ``layer_types`` is malformed, misses a layer,
            or declares a type this reader does not model, a sliding
            layer has no active window, an active window comes with
            no ``layer_types`` list, ``use_bidirectional_attention``
            carries a value other than ``"vision"`` or null, a
            declared ``num_global_key_value_heads`` disagrees with
            the base count while ``attention_k_eq_v`` disables it,
            ``num_kv_shared_layers`` leaves no layer that stores KV
            or leaves a shared layer with no earlier layer of its
            type, or a geometry key carries a type it cannot mean.
            ``bool`` subclasses ``int``, so a boolean count refuses
            as a non-integer (#348). No message renders a
            publisher-controlled value (#363).

    Returns:
        One `KVLayer` per hidden layer.
    """
    layer_types = _layer_types(config, layers, path, prefix)
    window = _active_window(config, path, prefix)
    if window is not None and layer_types is None:
        raise ValueError(
            f"{path}: {field_label('sliding_window', prefix)} declares "
            "windowed attention this reader does not model"
        )
    types = layer_types or ("full_attention",) * layers
    if "sliding_attention" in types and window is None:
        raise ValueError(
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
        raise ValueError(
            f"{path}: {field_label('num_global_key_value_heads', prefix)} declares "
            "a KV-head override its attention_k_eq_v setting disables, "
            "which this reader does not model"
        )
    fresh = layers - _shared_layers(config, layers, path, prefix)
    if not set(types[fresh:]) <= set(types[:fresh]):
        raise ValueError(
            f"{path}: {field_label('num_kv_shared_layers', prefix)} leaves a "
            "shared layer with no earlier layer of its type"
        )
    return tuple(
        KVLayer(
            kv_heads=kv_heads if t == "sliding_attention" else global_heads,
            head_dim=head_dim if t == "sliding_attention" else global_dim,
            window=window if t == "sliding_attention" else None,
            shares_kv=i >= fresh,
        )
        for i, t in enumerate(types)
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
        ValueError: If the list is not a list of strings, misses a
            layer, or names a type other than ``full_attention`` or
            ``sliding_attention``.
    """
    layer_types = config.get("layer_types")
    if layer_types is None:
        return None
    if not isinstance(layer_types, list) or not all(
        isinstance(t, str) for t in layer_types
    ):
        raise ValueError(
            f"{path}: {field_label('layer_types', prefix)} must be a list of strings"
        )
    if len(layer_types) != layers:
        raise ValueError(
            f"{path}: {field_label('layer_types', prefix)} does not list "
            "one type per hidden layer"
        )
    if any(t not in ("full_attention", "sliding_attention") for t in layer_types):
        raise ValueError(
            f"{path}: {field_label('layer_types', prefix)} declares "
            "a layer type this reader does not model"
        )
    return tuple(layer_types)


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
        ValueError: If ``sliding_window`` is not a non-negative
            integer or null, ``use_sliding_window`` is not a boolean
            or null, or ``use_bidirectional_attention`` carries a
            value other than ``"vision"`` or null. On ``"all"`` the
            runtime rescales the stored window at load, so this
            reader cannot price it. Any other value is unknown, and
            an unknown value could rescale the same way.
    """
    bidirectional = config.get("use_bidirectional_attention")
    if bidirectional is not None and bidirectional != "vision":
        raise ValueError(
            f"{path}: {field_label('use_bidirectional_attention', prefix)} declares "
            "bidirectional attention this reader does not model"
        )
    window = config.get("sliding_window")
    if window is not None and (
        isinstance(window, bool) or not isinstance(window, int) or window < 0
    ):
        raise ValueError(
            f"{path}: {field_label('sliding_window', prefix)} must be a "
            "non-negative integer or null"
        )
    switch = config.get("use_sliding_window")
    if switch is not None and not isinstance(switch, bool):
        raise ValueError(
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
        ValueError: If the value is not a boolean or null.
    """
    k_eq_v = config.get("attention_k_eq_v")
    if k_eq_v is not None and not isinstance(k_eq_v, bool):
        raise ValueError(
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
        ValueError: If the value is not a non-negative integer or
            null, or the count leaves no layer that stores KV.
    """
    shared = _shared_count(config, path, prefix)
    if shared >= layers:
        raise ValueError(
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
        ValueError: If the value is not a non-negative integer or
            null, or is outside the signed 64-bit range.
    """
    shared = config.get("num_kv_shared_layers")
    if shared is None:
        return 0
    if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
        raise ValueError(
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
        ValueError: If a present value is not a positive integer or
            null, or is outside the signed 64-bit range.
    """
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{path}: {field_label(key, prefix)} must be a positive integer or null"
        )
    return bounded_int(value, field_label(key, prefix), path)


def refuse_decilm_geometry(config: dict[str, Any], path: Path) -> None:
    """Refuse llama-geometry keys beside ``block_configs`` (#426).

    The DeciLM parse prices every kept block as a global K and V
    pair. A window, KV sharing, a split local/global key, or a
    ``layer_types`` list beside ``block_configs`` would silently
    misprice that read. ``attention_k_eq_v`` no longer changes a
    price (#431), but it marks a geometry family this parse does
    not model. Each key refuses.

    Args:
        config: Parsed ``config.json`` containing ``block_configs``.
        path: Source path, for error messages.

    Raises:
        ValueError: If a geometry key above carries an active value,
            or carries a type it cannot mean.
    """
    if config.get("layer_types") is not None:
        raise ValueError(
            f'{path}: "layer_types" beside "block_configs" declares '
            "per-layer attention this reader does not model"
        )
    if _active_window(config, path, "") is not None:
        raise ValueError(
            f'{path}: "sliding_window" declares windowed attention '
            "this reader does not model"
        )
    if _k_eq_v(config, path, ""):
        raise ValueError(
            f'{path}: "attention_k_eq_v" declares K=V storage '
            "this reader does not model"
        )
    if _shared_count(config, path, "") > 0:
        raise ValueError(
            f'{path}: "num_kv_shared_layers" declares KV sharing '
            "this reader does not model"
        )
    for key in ("global_head_dim", "num_global_key_value_heads"):
        if _optional_int(config, key, path, "") is not None:
            raise ValueError(
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
        ValueError: If the value exceeds the largest signed 64-bit
            integer.
    """
    if value > INT_MAX:
        raise ValueError(
            f"{path}: {label} exceeds {INT_MAX}, "
            "the largest integer this format carries"
        )
    return value
