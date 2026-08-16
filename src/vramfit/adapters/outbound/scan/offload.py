"""Offloaded-group support for the meter (ADR-0015).

``auto`` sharding under a GPU cap moves overflow modules to host RAM
and replaces their parameters with meta tensors. The real values live
in each module's ``AlignDevicesHook.weights_map``, and the forward
hooks stream them to the GPU every pass. This module resolves each
meta parameter to that backing CPU tensor, so the meter can perturb
and restore offloaded groups in place.

Resolution verifies behavior, not accelerate versions: the backing
tensor must be a real CPU tensor of the parameter's shape, and two
map reads must return one storage. A parameter that fails any check
keeps the honest refusal — the meter degrades to "cannot measure",
never to zero damage. Tied names that alias one storage collapse to
one group (`dedupe_aliased_groups`), so a capped tied model keeps
the uncapped group set instead of perturbing one tensor twice.

`ShardReader` is the second half: a whole-recipe pass cannot stage
~93 GB of original clones in host RAM at 49B scale, so offloaded
originals restore from the model's safetensors shards instead.

The model publisher owns the shard index. vramfit reads it and never
writes it, and it still refuses an index that defines one key twice
(#283). The alternative keeps the last value, so a repeated tensor name
in ``weight_map`` would restore the wrong shard with no report.

Examples:
    Resolve a sharded model's offloaded parameters:

    ```python
    backing = resolve_offloaded_params(model, groups)
    ```

See Also:
    - [vramfit.adapters.outbound.scan.meter][]: The meter both halves
      serve.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from vramfit.adapters.outbound.json_duplicate_key import (
    DuplicateKeyError,
    object_from_pairs,
)

_INDEX_FILE = "model.safetensors.index.json"
_SINGLE_FILE = "model.safetensors"


def resolve_offloaded_params(
    model: torch.nn.Module, groups: dict[str, list[str]]
) -> dict[str, torch.Tensor]:
    """Map every offloaded group parameter to its backing CPU tensor.

    Args:
        model: The loaded model.
        groups: Discovered group membership.

    Returns:
        Backing CPU tensors keyed by parameter name. Empty when no
        group parameter is offloaded.

    Raises:
        ValueError: If any group holds a meta parameter without a
            verified backing tensor. The message counts affected
            groups, names the first three, and names both causes —
            disk spill, or an accelerate layout this adapter does not
            recognize.
    """
    params = dict(model.named_parameters())
    backing: dict[str, torch.Tensor] = {}
    unreachable: list[str] = []
    for group_name, members in groups.items():
        resolved: dict[str, torch.Tensor] = {}
        for member in members:
            if not params[member].is_meta:
                continue
            tensor = _backing_tensor(model, member, params[member].shape)
            if tensor is None:
                unreachable.append(group_name)
                break
            resolved[member] = tensor
        else:
            backing.update(resolved)
    if unreachable:
        shown = ", ".join(unreachable[:3])
        raise ValueError(
            f"{len(unreachable)} of {len(groups)} groups hold offloaded "
            f"weights the meter cannot reach (first: {shown}) — disk spill "
            "or an unrecognized accelerate layout. Raise --gpu-memory, free "
            "host RAM, or use a smaller model"
        )
    return backing


def _backing_tensor(
    model: torch.nn.Module, name: str, shape: torch.Size
) -> torch.Tensor | None:
    """Find and verify one meta parameter's backing CPU tensor.

    Args:
        model: The loaded model.
        name: The parameter's dotted name.
        shape: The meta parameter's shape.

    Returns:
        The verified backing tensor, or None when any check fails:
        no hook, no weights map, no entry, a non-CPU or wrong-shape
        tensor, or two reads returning different storages. A defect
        raised by custom modeling code propagates as itself.
    """
    # get_submodule cannot fail here: the name came from
    # named_parameters, so a raise is a model-code defect that must
    # surface as itself, not as an offload refusal.
    module = model.get_submodule(name.rpartition(".")[0])
    hook = getattr(module, "_hf_hook", None)
    weights_map = None
    for candidate in [hook, *getattr(hook, "hooks", ())]:
        weights_map = getattr(candidate, "weights_map", None)
        if weights_map is not None:
            break
    if weights_map is None:
        return None
    # PrefixedDataset wraps the loader that keys on full dotted names.
    dataset = getattr(weights_map, "dataset", weights_map)
    try:
        first, second = dataset[name], dataset[name]
    except KeyError:
        return None
    if (
        not isinstance(first, torch.Tensor)
        or first.is_meta
        or first.device.type != "cpu"
        or first.shape != shape
        or first.data_ptr() != second.data_ptr()
    ):
        return None
    return first


def dedupe_aliased_groups(
    groups: dict[str, list[str]], backing: dict[str, torch.Tensor]
) -> tuple[dict[str, list[str]], dict[str, torch.Tensor]]:
    """Collapse offloaded parameters that share one backing storage.

    Uncapped, ``named_parameters`` deduplicates tied weights by
    identity — one name survives. Under a cap, transformers installs
    a separate meta tensor per tied name, and every alias resolves to
    the same CPU storage. Measuring aliases as separate groups would
    perturb one tensor per alias, capture a quantized "original", and
    restore corrupt weights without poisoning. The first name in
    group order keeps the storage. A group left without members is
    dropped, which matches the uncapped group set.

    Args:
        groups: Discovered group membership.
        backing: Backing tensors from `resolve_offloaded_params`.

    Returns:
        The groups and backing map with aliases removed.
    """
    seen: set[int] = set()
    kept_groups: dict[str, list[str]] = {}
    kept_backing: dict[str, torch.Tensor] = {}
    for group_name, members in groups.items():
        kept: list[str] = []
        for member in members:
            tensor = backing.get(member)
            if tensor is None:
                # A real parameter — named_parameters already
                # deduplicated identity ties among these.
                kept.append(member)
                continue
            pointer = tensor.data_ptr()
            if pointer in seen:
                continue
            seen.add(pointer)
            kept.append(member)
            kept_backing[member] = tensor
        if kept:
            kept_groups[group_name] = kept
    return kept_groups, kept_backing


class ShardReader:
    """Reads original tensors back from a model's safetensors shards.

    The restore source for offloaded originals in a whole-recipe pass
    (ADR-0015): offloaded tensors load from the shards without dtype
    conversion, so the files on disk already hold their originals.

    Attributes:
        index (dict[str, Path]): Shard file per tensor name.

    Examples:
        Verify against the live tensor, then restore after a
        measurement:

        ```python
        reader = open_shard_reader("./model")
        assert reader.verify({"model.layers.0.mlp.w": live}) is None
        reader.read_into({"model.layers.0.mlp.w": live})
        ```
    """

    def __init__(self, index: dict[str, Path]) -> None:
        """Wrap a resolved tensor-name-to-shard-file index.

        Args:
            index: Shard file per tensor name.
        """
        self.index = index

    def _by_file(self, names: list[str]) -> dict[Path, list[str]]:
        """Group tensor names by shard file, opening each file once.

        Args:
            names: Tensor names to group.

        Returns:
            Names per shard file.
        """
        by_file: dict[Path, list[str]] = {}
        for name in names:
            by_file.setdefault(self.index[name], []).append(name)
        return by_file

    def verify(self, live: Mapping[str, torch.Tensor]) -> str | None:
        """Check that every tensor is restorable before weights change.

        Beyond name and shape, the first row of each shard entry must
        equal the live tensor's — a value drift between the files and
        the loaded model would otherwise restore a wrong baseline
        without any error.

        Args:
            live: The loaded tensor per tensor name, unperturbed.

        Returns:
            None when every tensor resolves to a shard entry with the
            live shape and matching sample values, otherwise the first
            mismatch, named.

        Raises:
            OSError: If a shard file cannot be read.
        """
        missing = sorted(set(live) - set(self.index))
        if missing:
            return f'no shard entry for "{missing[0]}"'
        for file, names in self._by_file(sorted(live)).items():
            with safe_open(file, framework="pt", device="cpu") as shard:
                for name in names:
                    if name not in shard.keys():  # noqa: SIM118 - not a dict
                        return f'"{name}" missing from {file.name}'
                    entry = shard.get_slice(name)
                    found = tuple(entry.get_shape())
                    if found != tuple(live[name].shape):
                        return (
                            f'"{name}" has shape {found} in {file.name}, '
                            f"expected {tuple(live[name].shape)}"
                        )
                    sample = entry[0:1].to(live[name].dtype)
                    if not torch.equal(sample, live[name][0:1].cpu()):
                        return (
                            f'"{name}" differs from the loaded model in '
                            f"{file.name} — the files changed since the "
                            "model loaded"
                        )
        return None

    def read_into(self, targets: Mapping[str, torch.Tensor]) -> None:
        """Copy each named original from its shard into a live tensor.

        One tensor loads at a time, so peak extra memory is one
        tensor's bytes. ``copy_`` casts when the model loaded at a
        different dtype than the shards store.

        Args:
            targets: Destination tensor per tensor name.

        Raises:
            KeyError: If a name has no shard entry — `verify` runs
                first on every path that reaches here.
            OSError: If a shard file cannot be read.
        """
        with torch.no_grad():
            for file, names in self._by_file(sorted(targets)).items():
                with safe_open(file, framework="pt", device="cpu") as shard:
                    for name in names:
                        targets[name].copy_(shard.get_tensor(name))


def open_shard_reader(model_id: str) -> ShardReader | None:
    """Locate the safetensors shards behind a local model path.

    The publisher owns the index file, so vramfit reads it and never
    writes it. A repeated key still refuses (#283). `json.loads` would
    keep the last value, and a tensor name repeated in ``weight_map``
    would then point the reader at one shard and drop the other. The
    restored original would be the wrong tensor, and the damage figure
    would be wrong with no report.

    Args:
        model_id: Hugging Face model id or local checkpoint path.

    Returns:
        A reader over the sharded or single-file layout, or None when
        ``model_id`` is not a local directory with safetensors files.

    Raises:
        OSError: If an index or shard file exists but cannot be read.
        ValueError: If the index file is not valid JSON, defines the
            same key twice, or holds no ``weight_map`` object.
    """
    directory = Path(model_id)
    if not directory.is_dir():
        return None
    index_path = directory / _INDEX_FILE
    if index_path.is_file():
        try:
            index = json.loads(
                index_path.read_text(encoding="utf-8"),
                object_pairs_hook=object_from_pairs,
            )
        except DuplicateKeyError as exc:
            raise ValueError(f"{index_path}: {exc.message}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path} has no weight_map object")
        return ShardReader(
            {name: directory / file for name, file in weight_map.items()}
        )
    single = directory / _SINGLE_FILE
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as shard:
            return ShardReader(dict.fromkeys(shard.keys(), single))
    return None
