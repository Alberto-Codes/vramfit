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

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.offload import (
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
