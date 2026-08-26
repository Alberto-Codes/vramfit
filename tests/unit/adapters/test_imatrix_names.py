"""Checks that vramfit's GGUF name table agrees with ``gguf``'s own.

`_SUFFIX_TO_GGUF` restates a mapping llama.cpp already owns. The
adapter cannot call `gguf.TensorNameMap` at run time. That map keys
on a model architecture the scan step has not resolved, and it fuses
no expert stack. So the table is a copy, and a copy drifts. These
checks read `gguf.TensorNameMap` directly and hold every row of the
table against it.

The checks run in two directions. Each row of the table must reach
the same GGUF tensor `gguf` reaches, which catches a wrong row. Every
dense name the target checkpoint holds must reach some row, which
catches a missing one. #186 was a missing row, not a wrong one.

The table stays deliberately narrower than `gguf`. That map carries
287 block names under the "model" and "backbone" decoder roots at
`gguf` 0.19.0, and the table covers 16 of them. The rest belong to families vramfit has
never scanned. So an unmapped name is not a defect on its own, and no
check here asserts the reverse direction over the whole map.

The suite is hermetic: `gguf` ships the map as data, so nothing here
reads a model or a network. It skips where the scan extra is absent,
because the base install carries no `gguf` and no torch (ADR-0005).

These are unit checks. ADR-0009 reserves the `contract` marker for a
verified fake over a port, and this table is a helper inside the torch
meter adapter with no port and no fake.
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import json
from pathlib import Path

import pytest

gguf = pytest.importorskip("gguf", reason="scan extra not installed")
pytest.importorskip("torch", reason="scan extra not installed")

from vramfit.adapters.outbound.scan.imatrix import _SUFFIX_TO_GGUF, gguf_tensor_name

pytestmark = pytest.mark.unit

LAYER = 3
# The decoder roots the adapter supports (#177, #423). The table keys
# on the module suffix, so each suffix is checked under every root.
# `gguf` carries no "model.language_model" template — its converter
# strips the nesting before it maps — so every pair under that root
# is unbacked by construction.
ROOTS = ("model", "backbone", "model.language_model")
# Every module suffix the adapter's table carries, and the GGUF stem
# it claims. Restated here rather than imported: a test that reads the
# table under test would pass whatever the table says.
MAPPED_SUFFIXES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "mixer.q_proj": "attn_q",
    "mixer.k_proj": "attn_k",
    "mixer.v_proj": "attn_v",
    "mixer.o_proj": "attn_output",
    "mixer.in_proj": "ssm_in",
    "mixer.out_proj": "ssm_out",
    "mixer.gate": "ffn_gate_inp",
    "mixer.shared_experts.up_proj": "ffn_up_shexp",
    "mixer.shared_experts.down_proj": "ffn_down_shexp",
}
# The Nemotron-H rows (#186). Every one spells a distinct module
# under one "mixer." prefix.
MIXER_SUFFIXES = tuple(s for s in MAPPED_SUFFIXES if s.startswith("mixer."))
LLAMA_SUFFIXES = tuple(s for s in MAPPED_SUFFIXES if not s.startswith("mixer."))
# The (root, suffix) pairs the adapter answers and `gguf` does not
# carry. The adapter keys on the suffix alone, so every row answers
# under every root, while `gguf` carries each family's names under the
# one root that family uses — and none at all under the nested
# "model.language_model" root, which its converter flattens away. So
# the adapter reaches 32 names no checkpoint spells, 16 of them under
# the nested root. Naming them here rather than skipping them keeps
# the suite honest: a pair that leaves this set fails, and a pair that
# joins it fails too.
UNBACKED_PAIRS = frozenset(
    [("model", suffix) for suffix in MIXER_SUFFIXES]
    + [("backbone", suffix) for suffix in LLAMA_SUFFIXES]
    + [("model.language_model", suffix) for suffix in MAPPED_SUFFIXES]
)
# Names in this same family that the table omits, and the stem `gguf`
# gives each. A dense-MLP Nemotron-H spells its MLP the first two
# ways, and Nemotron 3 Super carries the latent projections. No target
# holds one yet (#186). Adding a row here without a checkpoint to
# prove it would price against a guess.
OMITTED_FAMILY_SUFFIXES = (
    "mixer.up_proj",
    "mixer.down_proj",
    "mixer.fc1_latent_proj",
    "mixer.fc2_latent_proj",
)
# Every dense parameter family under the checkpoint's decoder root,
# read from `model.safetensors.index.json` of
# nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, with the count
# the index holds. Three sets are excluded. The routed experts:
# `gguf` fuses them in its converter rather than its name map, and
# #177 already covers them. The norms, biases, and SSM state
# parameters: no imatrix entry exists for a tensor the quantizer never
# touches. The MTP head's 8 dense linears and its expert stack: they
# root at `mtp.layers.`, which `_LAYER_PARAM` rejects, and ADR-0026's
# measurement found no imatrix entry for any of them.
CHECKPOINT_DENSE_FAMILIES = (
    ("mixer.in_proj", 23),
    ("mixer.out_proj", 23),
    ("mixer.gate", 23),
    ("mixer.shared_experts.up_proj", 23),
    ("mixer.shared_experts.down_proj", 23),
    ("mixer.q_proj", 6),
    ("mixer.k_proj", 6),
    ("mixer.v_proj", 6),
    ("mixer.o_proj", 6),
)


def _gguf_stems(template: str) -> set[str]:
    """Every GGUF stem ``gguf`` assigns to one templated HF name.

    A name may serve more than one architecture. RWKV reuses
    ``self_attn.q_proj`` for ``time_mix_receptance``, so the answer is
    a set and the table must name one member of it.

    The arch overrides are read too. A name that moved into one would
    otherwise read as a name ``gguf`` no longer carries.
    """
    name_map = gguf.tensor_mapping.TensorNameMap
    sources = [name_map.block_mappings_cfg]
    sources.extend(name_map.arch_block_mappings_cfg.values())
    stems = set()
    for source in sources:
        for tensor, templates in source.items():
            if template in templates:
                stems.add(gguf.TENSOR_NAMES[tensor].format(bid=LAYER))
    return stems


def test_the_restated_table_names_every_row_the_adapter_carries() -> None:
    # Every check below iterates the restated copy. A row the adapter
    # gained and the copy did not would be invisible to all of them,
    # so an added row could price against any tensor it liked. Keys
    # only: the stems stay restated, or the correctness checks would
    # assert the table against itself.
    assert set(MAPPED_SUFFIXES) == set(_SUFFIX_TO_GGUF)


@pytest.mark.parametrize("root", ROOTS)
@pytest.mark.parametrize("suffix", sorted(MAPPED_SUFFIXES), ids=sorted(MAPPED_SUFFIXES))
def test_a_mapped_suffix_matches_the_gguf_name_map(root: str, suffix: str) -> None:
    stems = _gguf_stems(f"{root}.layers.{{bid}}.{suffix}")
    ours = gguf_tensor_name(f"{root}.layers.{LAYER}.{suffix}.weight")

    assert ours == f"blk.{LAYER}.{MAPPED_SUFFIXES[suffix]}.weight"
    if (root, suffix) in UNBACKED_PAIRS:
        # `gguf` carries no such name, so it can confirm nothing. The
        # adapter still answers, and no checkpoint spells the name.
        assert stems == set()
    else:
        assert ours.removesuffix(".weight") in stems


def test_the_unbacked_pairs_are_exactly_the_ones_named() -> None:
    # A skip would hide this. `gguf` confirms 16 of the 48 (root,
    # suffix) pairs the adapter answers, and the other 32 are named in
    # UNBACKED_PAIRS. A release that moves a pair either way fails
    # here rather than passing green on part of the table.
    unbacked = {
        (root, suffix)
        for root in ROOTS
        for suffix in MAPPED_SUFFIXES
        if not _gguf_stems(f"{root}.layers.{{bid}}.{suffix}")
    }

    assert unbacked == UNBACKED_PAIRS
    assert len(unbacked) == 32


@pytest.mark.parametrize("suffix", OMITTED_FAMILY_SUFFIXES)
def test_an_omitted_nemotron_h_name_stays_unmapped(suffix: str) -> None:
    # `gguf` claims each of these under the backbone root, and the
    # table carries none of them. That is deliberate (#186), and this
    # pins it so a later reader sees a decision rather than an
    # oversight. A checkpoint that holds one needs a row and a
    # measurement, not a guess.
    assert _gguf_stems(f"backbone.layers.{{bid}}.{suffix}") != set()
    assert gguf_tensor_name(f"backbone.layers.{LAYER}.{suffix}.weight") is None


@pytest.mark.parametrize(
    ("suffix", "count"),
    CHECKPOINT_DENSE_FAMILIES,
    ids=[suffix for suffix, _ in CHECKPOINT_DENSE_FAMILIES],
)
def test_every_dense_family_in_the_checkpoint_resolves(suffix: str, count: int) -> None:
    # The completeness half. #186 opened because none of these
    # resolved, which reported the imatrix as the wrong file.
    stem = MAPPED_SUFFIXES[suffix]
    names = [f"backbone.layers.{layer}.{suffix}.weight" for layer in range(count)]

    resolved = [gguf_tensor_name(name) for name in names]

    assert resolved == [f"blk.{layer}.{stem}.weight" for layer in range(count)]


def test_all_one_hundred_thirty_nine_dense_parameters_reach_a_distinct_tensor() -> None:
    # 139 is the dense count bartowski's imatrix reports for this
    # model's decoder (ADR-0026). The MTP head's dense linears sit
    # outside it. Each parameter must reach its own GGUF tensor: a
    # table that collapsed two families onto one stem would price the
    # second against the first's columns.
    names = [
        f"backbone.layers.{layer}.{suffix}.weight"
        for suffix, count in CHECKPOINT_DENSE_FAMILIES
        for layer in range(count)
    ]

    resolved = [gguf_tensor_name(name) for name in names]

    assert len(names) == 139
    assert len(set(resolved)) == 139


def _gemma_dense_names() -> list[str]:
    """Every quantizable decoder parameter Gemma 4 31B nests.

    Built from the committed fixture config: 60 decoder layers under
    "model.language_model.layers.", where a "full_attention" layer
    carries no v_proj (``attention_k_eq_v``, #421). The count must
    land on 410, the quantizable 2-d decoder surface the #423 fast
    gate enumerated from the GGUF header (410 tensors beside
    ``token_embd``).
    """
    config = json.loads(
        Path("tests/data/gemma-4-31b/config.json").read_text(encoding="utf-8")
    )
    layer_types = config["text_config"]["layer_types"]
    names = []
    for layer, layer_type in enumerate(layer_types):
        for suffix in LLAMA_SUFFIXES:
            if suffix == "self_attn.v_proj" and layer_type == "full_attention":
                continue
            names.append(f"model.language_model.layers.{layer}.{suffix}.weight")
    return names


def test_every_gemma_dense_parameter_resolves_distinctly() -> None:
    # The completeness half for the nested root (#423). Before the
    # root landed, all 410 reported uncovered and a kquant-imx scan
    # priced every cell unassisted.
    names = _gemma_dense_names()

    resolved = [gguf_tensor_name(name) for name in names]

    assert len(names) == 410
    assert None not in resolved
    assert len(set(resolved)) == 410


def test_the_nested_embedding_resolves_to_token_embd() -> None:
    # Gemma 4 ties its head to the embedding, so the loaded model
    # reports one name for both. The direct table answers it under
    # the nested root the same way it answers the flat one.
    name = gguf_tensor_name("model.language_model.embed_tokens.weight")

    assert name == "token_embd.weight"


def test_a_vision_tower_layer_stays_unmapped() -> None:
    # The closed alternation exists so a tower's "layers.5" never
    # prices against the decoder's "blk.5" columns. The name is the
    # loaded model's own spelling — the tower wraps each projection
    # in a "linear" module.
    name = "model.vision_tower.encoder.layers.5.self_attn.q_proj.linear.weight"

    assert gguf_tensor_name(name) is None


@pytest.mark.parametrize("suffix", MIXER_SUFFIXES)
def test_a_mixer_suffix_carries_one_meaning_under_the_supported_roots(
    suffix: str,
) -> None:
    # phi2 and jina both spell "mixer.out_proj" for their attention
    # output, and plamo2 spells "mixer.in_proj" — each under a root
    # the adapter's pattern rejects. The table keys on the suffix
    # alone, so a family that reached one of these names under a
    # supported root would mis-price silently. Exactly one claim means
    # no such family exists.
    claimed = {
        stem
        for root in ROOTS
        for stem in _gguf_stems(f"{root}.layers.{{bid}}.{suffix}")
    }

    assert claimed == {f"blk.{LAYER}.{MAPPED_SUFFIXES[suffix]}"}
