"""Weights-map resolution and shard-restore behavior on stubs (ADR-0015).

Hermetic pins of the refusal logic: no accelerate, no model loads, no
CUDA — stub modules and tmp safetensors files only. The module skips
where the scan extra is absent (CI), and runs in the default suite
wherever torch is installed, so a refusal-check regression cannot ride
a green fast suite onto the reference box.
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")

from vramfit.adapters.outbound.scan import offload
from vramfit.adapters.outbound.scan.offload import (
    ShardPathError,
    ShardReader,
    dedupe_aliased_groups,
    open_shard_reader,
    resolve_offloaded_params,
)

pytestmark = pytest.mark.unit


class _Hook:
    def __init__(self, weights_map) -> None:
        self.weights_map = weights_map


class _CloningMap(dict):
    """A weights map whose reads return fresh copies — unstable storage."""

    def __getitem__(self, key):
        return super().__getitem__(key).clone()


def _ghost_model(weights_map) -> torch.nn.Module:
    """A stub model with one real module and one offloaded ("ghost") module."""
    model = torch.nn.Module()
    model.resident = torch.nn.Linear(4, 4)
    model.ghost = torch.nn.Module()
    model.ghost.weight = torch.nn.Parameter(torch.empty(4, 4, device="meta"))
    if weights_map is not None:
        model.ghost._hf_hook = _Hook(weights_map)
    return model


GHOST_GROUPS = {"resident": ["resident.weight"], "ghost": ["ghost.weight"]}


class TestResolveOffloadedParams:
    def test_model_without_meta_params_resolves_to_nothing(self) -> None:
        model = torch.nn.Module()
        model.resident = torch.nn.Linear(4, 4)

        backing = resolve_offloaded_params(model, {"resident": ["resident.weight"]})

        assert backing == {}

    def test_meta_param_with_stable_weights_map_resolves(self) -> None:
        tensor = torch.randn(4, 4)
        model = _ghost_model({"ghost.weight": tensor})

        backing = resolve_offloaded_params(model, GHOST_GROUPS)

        assert backing == {"ghost.weight": tensor}
        assert backing["ghost.weight"].data_ptr() == tensor.data_ptr()

    def test_meta_param_without_hook_is_refused(self) -> None:
        model = _ghost_model(None)

        with pytest.raises(ValueError, match=r"cannot reach.*ghost"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_unstable_weights_map_storage_is_refused(self) -> None:
        model = _ghost_model(_CloningMap({"ghost.weight": torch.randn(4, 4)}))

        with pytest.raises(ValueError, match="cannot reach"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_wrong_shape_backing_tensor_is_refused(self) -> None:
        model = _ghost_model({"ghost.weight": torch.randn(2, 2)})

        with pytest.raises(ValueError, match="cannot reach"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_missing_weights_map_entry_is_refused(self) -> None:
        model = _ghost_model({"other.weight": torch.randn(4, 4)})

        with pytest.raises(ValueError, match="cannot reach"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_meta_backing_tensor_is_refused(self) -> None:
        ghost = torch.empty(4, 4, device="meta")
        model = _ghost_model({"ghost.weight": ghost})

        with pytest.raises(ValueError, match="cannot reach"):
            resolve_offloaded_params(model, GHOST_GROUPS)

    def test_group_with_late_unresolvable_member_is_refused_whole(self) -> None:
        # A group whose first member resolves and second does not must
        # refuse as one unit — no partial resolution may leak out.
        good = torch.randn(4, 4)
        model = _ghost_model({"ghost.weight": good})
        model.ghost.bias2 = torch.nn.Parameter(torch.empty(4, 4, device="meta"))
        groups = {"ghost": ["ghost.weight", "ghost.bias2"]}

        with pytest.raises(ValueError, match=r"cannot reach.*ghost"):
            resolve_offloaded_params(model, groups)


class TestDedupeAliasedGroups:
    def test_two_names_on_one_storage_collapse_to_the_first(self) -> None:
        shared = torch.randn(4, 4)
        groups = {"embed": ["embed.weight"], "head": ["head.weight"]}
        backing = {"embed.weight": shared, "head.weight": shared}

        kept_groups, kept_backing = dedupe_aliased_groups(groups, backing)

        assert kept_groups == {"embed": ["embed.weight"]}
        assert kept_backing == {"embed.weight": shared}

    def test_distinct_storages_pass_through_unchanged(self) -> None:
        groups = {"a": ["a.weight"], "b": ["b.weight"]}
        backing = {"a.weight": torch.randn(4, 4), "b.weight": torch.randn(4, 4)}

        kept_groups, kept_backing = dedupe_aliased_groups(groups, backing)

        assert kept_groups == groups
        assert kept_backing == backing

    def test_real_members_are_kept_without_storage_tracking(self) -> None:
        groups = {"a": ["a.weight", "a.other"]}
        backing = {"a.weight": torch.randn(4, 4)}

        kept_groups, _ = dedupe_aliased_groups(groups, backing)

        assert kept_groups == {"a": ["a.weight", "a.other"]}


class TestShardReader:
    def _saved(self, tmp_path, tensors) -> ShardReader:
        from safetensors.torch import save_file

        save_file(tensors, tmp_path / "model.safetensors")
        reader = open_shard_reader(str(tmp_path))
        assert reader is not None
        return reader

    def test_open_on_single_file_layout_finds_every_tensor(self, tmp_path) -> None:
        live = torch.randn(8, 4)
        reader = self._saved(tmp_path, {"model.embed.weight": live})

        assert reader.verify({"model.embed.weight": live}) is None

    def test_open_on_indexed_layout_maps_tensors_to_shards(self, tmp_path) -> None:
        from safetensors.torch import save_file

        a, b = torch.randn(2, 2), torch.randn(3, 3)
        save_file({"a": a}, tmp_path / "m-00001.safetensors")
        save_file({"b": b}, tmp_path / "m-00002.safetensors")
        index = {"weight_map": {"a": "m-00001.safetensors", "b": "m-00002.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        reader = open_shard_reader(str(tmp_path))

        assert reader is not None
        assert reader.verify({"a": a, "b": b}) is None

    def test_open_on_a_hub_id_returns_none(self) -> None:
        assert open_shard_reader("org/model-that-is-not-a-path") is None

    def test_open_on_a_directory_without_safetensors_returns_none(
        self, tmp_path
    ) -> None:
        assert open_shard_reader(str(tmp_path)) is None

    def test_open_on_an_index_without_weight_map_raises(self, tmp_path) -> None:
        (tmp_path / "model.safetensors.index.json").write_text('{"metadata": {}}')

        with pytest.raises(ValueError, match="weight_map"):
            open_shard_reader(str(tmp_path))

    def test_open_on_an_index_repeating_a_tensor_name_raises(self, tmp_path) -> None:
        # The publisher owns this file. Taking the last value would
        # restore "a" from the wrong shard, and the damage figure would
        # be wrong with no report (#283).
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "m-00001.safetensors", "a": "m-00002.safetensors"}}'
        )

        with pytest.raises(ValueError, match=r'index\.json: duplicate key "a"'):
            open_shard_reader(str(tmp_path))

    def test_open_on_an_index_repeating_a_top_level_key_raises(self, tmp_path) -> None:
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "m-00001.safetensors"}, "weight_map": {}}'
        )

        with pytest.raises(
            ValueError, match=r'index\.json: duplicate key "weight_map"'
        ):
            open_shard_reader(str(tmp_path))

    def test_open_on_an_index_that_is_not_json_raises_naming_the_file(
        self, tmp_path
    ) -> None:
        # A bare `JSONDecodeError` reported the column and no file. A
        # scan reads several files from one model directory (#287).
        (tmp_path / "model.safetensors.index.json").write_text("{oops")

        with pytest.raises(ValueError, match=r"index\.json: invalid JSON"):
            open_shard_reader(str(tmp_path))

    def test_open_on_an_index_that_is_not_utf8_raises_naming_the_file(
        self, tmp_path
    ) -> None:
        (tmp_path / "model.safetensors.index.json").write_bytes(b"\xff\xfe{}")

        with pytest.raises(ValueError, match=r"index\.json: not valid UTF-8"):
            open_shard_reader(str(tmp_path))

    def test_open_on_an_index_past_the_digit_limit_raises_naming_the_file(
        self, tmp_path
    ) -> None:
        # `json.loads` raises a plain `ValueError` above
        # `sys.get_int_max_str_digits`, which escaped the named clause
        # with CPython's remedy and no file (#287).
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "m.safetensors"}, "n": ' + "9" * 5000 + "}"
        )

        with pytest.raises(ValueError, match=r"index\.json: cannot parse JSON"):
            open_shard_reader(str(tmp_path))

    def test_open_on_an_index_nested_past_the_recursion_limit_raises_naming_the_file(
        self, tmp_path
    ) -> None:
        # Deep nesting exhausts the decoder's stack. `RecursionError` is
        # no `ValueError`, so it escaped every caller (#287).
        (tmp_path / "model.safetensors.index.json").write_text(
            "[" * 100_000 + "]" * 100_000
        )

        with pytest.raises(ValueError, match=r"index\.json: JSON nests too deeply"):
            open_shard_reader(str(tmp_path))

    def test_open_on_an_index_that_is_not_an_object_raises_naming_the_file(
        self, tmp_path
    ) -> None:
        # A JSON array reached `index.get` and raised `AttributeError`,
        # outside every clause a caller catches (#287).
        (tmp_path / "model.safetensors.index.json").write_text("[]")

        with pytest.raises(ValueError, match=r"index\.json: expected a JSON object"):
            open_shard_reader(str(tmp_path))

    def test_verify_names_a_missing_tensor(self, tmp_path) -> None:
        reader = self._saved(tmp_path, {"a": torch.randn(2, 2)})

        problem = reader.verify({"gone": torch.randn(2, 2)})

        assert problem is not None
        assert "gone" in problem

    def test_verify_names_a_shape_mismatch(self, tmp_path) -> None:
        reader = self._saved(tmp_path, {"a": torch.randn(2, 2)})

        problem = reader.verify({"a": torch.randn(4, 4)})

        assert problem is not None
        assert "shape" in problem

    def test_verify_names_a_value_drift(self, tmp_path) -> None:
        # Same name, same shape, different values: the files changed
        # since the model loaded, and a restore would install a wrong
        # baseline without any error.
        saved = torch.randn(4, 4)
        reader = self._saved(tmp_path, {"a": saved})

        problem = reader.verify({"a": saved + 1.0})

        assert problem is not None
        assert "differs" in problem

    def test_read_into_restores_a_mutated_tensor(self, tmp_path) -> None:
        original = torch.randn(4, 4)
        reader = self._saved(tmp_path, {"a": original})
        live = original.clone()
        live.zero_()

        reader.read_into({"a": live})

        assert torch.equal(live, original)

    def test_read_into_casts_to_the_live_dtype(self, tmp_path) -> None:
        original = torch.randn(4, 4)
        reader = self._saved(tmp_path, {"a": original})
        live = torch.zeros(4, 4, dtype=torch.bfloat16)

        reader.read_into({"a": live})

        assert live.dtype == torch.bfloat16
        assert torch.equal(live, original.to(torch.bfloat16))


class TestShardPathContainment:
    """The publisher owns ``weight_map``, so every entry is untrusted.

    accelerate carries the same defect through
    ``load_checkpoint_in_model`` (GHSA-4j2p-28q2-5m79), and its
    maintainer declined the fix. vramfit never calls that entry point.
    It reached the same exposure through its own index read, so it
    closes the hole in its own code.
    """

    @staticmethod
    def _planted(tmp_path):
        """Plant a target outside the model directory, inside tmp_path."""
        from safetensors.torch import save_file

        outside = tmp_path / "outside.safetensors"
        save_file({"planted": torch.tensor([42.0])}, str(outside))
        model = tmp_path / "model"
        model.mkdir()
        save_file({"a": torch.randn(2, 2)}, str(model / "m-00001.safetensors"))
        return outside, model

    @staticmethod
    def _write_index(model, entry) -> None:
        index = {"weight_map": {"a": "m-00001.safetensors", "planted": entry}}
        (model / "model.safetensors.index.json").write_text(json.dumps(index))

    def test_open_on_a_relative_traversal_entry_refuses(self, tmp_path) -> None:
        _, model = self._planted(tmp_path)
        self._write_index(model, "../outside.safetensors")

        with pytest.raises(ShardPathError, match="outside the model directory"):
            open_shard_reader(str(model))

    def test_open_on_an_absolute_entry_refuses(self, tmp_path) -> None:
        # `pathlib` discards the left side on an absolute right side, so
        # the join alone never leaves the entry under the model directory.
        outside, model = self._planted(tmp_path)
        self._write_index(model, str(outside))

        with pytest.raises(ShardPathError, match="outside the model directory"):
            open_shard_reader(str(model))

    def test_refusal_names_the_entry_and_the_tensor(self, tmp_path) -> None:
        _, model = self._planted(tmp_path)
        self._write_index(model, "../outside.safetensors")

        with pytest.raises(ShardPathError) as caught:
            open_shard_reader(str(model))

        assert caught.value.name == "planted"
        assert caught.value.entry == "../outside.safetensors"
        assert "index.json" in str(caught.value)

    @pytest.mark.parametrize("entry", ["../outside.safetensors", "/etc/hostname"])
    def test_refusal_reads_no_file_outside_the_model_directory(
        self, tmp_path, entry, monkeypatch
    ) -> None:
        # The refusal has to land before any read. A path that only
        # fails at `safe_open` would already have opened the file.
        # Spy on both read boundaries the adapter uses: `Path.read_text`
        # for the index, and the `safe_open` bound in the module for
        # every shard. `builtins.open` catches neither.
        _, model = self._planted(tmp_path)
        self._write_index(model, entry)
        read: list[str] = []

        real_read_text = Path.read_text
        real_safe_open = offload.safe_open

        def _read_text_spy(self, *args, **kwargs):
            read.append(os.path.abspath(self))
            return real_read_text(self, *args, **kwargs)

        def _safe_open_spy(filename, *args, **kwargs):
            read.append(os.path.abspath(filename))
            return real_safe_open(filename, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _read_text_spy)
        monkeypatch.setattr(offload, "safe_open", _safe_open_spy)

        with pytest.raises(ShardPathError):
            open_shard_reader(str(model))

        # The index read proves the spy sits on the real boundary. An
        # empty list would pass the containment assertion vacuously.
        base = os.path.abspath(model)
        assert read == [os.path.join(base, "model.safetensors.index.json")]
        assert [p for p in read if not p.startswith(base + os.sep)] == []

    @pytest.mark.parametrize(
        "entry", [None, 123, ["m-00001.safetensors"], {"file": "m-00001.safetensors"}]
    )
    def test_open_on_a_non_string_entry_refuses(self, tmp_path, entry) -> None:
        # A non-string value names no file. The join would raise
        # `TypeError`, which escapes the reader's `ValueError` contract
        # and the scan-loop catch that reports a halted scan.
        _, model = self._planted(tmp_path)
        self._write_index(model, entry)

        with pytest.raises(ValueError, match="no shard file name") as caught:
            open_shard_reader(str(model))

        assert "index.json" in str(caught.value)
        assert "'planted'" in str(caught.value)

    def test_open_accepts_a_shard_symlinked_out_of_the_directory(
        self, tmp_path
    ) -> None:
        # The Hugging Face hub cache stores every shard under
        # `snapshots/<revision>/` as a symlink into `blobs/`. Containment
        # that resolved symlinks would refuse a correct checkpoint, so
        # the check compares lexically normalized paths instead.
        from safetensors.torch import save_file

        blobs = tmp_path / "blobs"
        blobs.mkdir()
        live = torch.randn(3, 3)
        save_file({"a": live}, str(blobs / "sha256-deadbeef"))
        snapshot = tmp_path / "snapshots" / "main"
        snapshot.mkdir(parents=True)
        (snapshot / "m-00001.safetensors").symlink_to(blobs / "sha256-deadbeef")
        index = {"weight_map": {"a": "m-00001.safetensors"}}
        (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))

        reader = open_shard_reader(str(snapshot))

        assert reader is not None
        assert reader.verify({"a": live}) is None
