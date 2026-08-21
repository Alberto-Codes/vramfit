"""Checks that the pack backend's class table agrees with ``gguf``'s.

`GGUF_SUFFIX_BY_HF` restates a mapping llama.cpp already owns. The
backend cannot call `gguf.TensorNameMap` at run time: that map keys
on a model architecture the pack step has not resolved. So the table
is a copy, and a copy drifts. These checks read `gguf.TensorNameMap`
directly and hold every row of the table against it — the #186
technique, applied to the pack side after the 2026-08-20 ADR-0012
amendment added the nine Nemotron-H rows.

Each family's rows are checked under the root that family's
checkpoints use: `model` for the llama rows, `backbone` for the
Nemotron-H rows. The table itself keys on the suffix under a free
prefix, so the root here only decides which side ``gguf`` can
confirm — the scan-side suite (`test_imatrix_names.py`) pins the
full root-by-suffix matrix.

The suite is hermetic: `gguf` ships the map as data, so nothing here
reads a model or a network. It skips where gguf-py is absent, because
the base install carries no gguf (ADR-0005).

These are unit checks. ADR-0009 reserves the `contract` marker for a
verified fake over a port, and this table is a helper inside the GGUF
backend with no port of its own.
"""

# ruff: noqa: E402 - the importorskip guard must run before adapter imports

from __future__ import annotations

import pytest

gguf = pytest.importorskip("gguf", reason="pack extra not installed")

from vramfit.adapters.outbound.gguf.types import GGUF_SUFFIX_BY_HF

pytestmark = pytest.mark.unit

LAYER = 3
# Every row the backend's class table carries, and the stem it
# claims, under the root whose family `gguf` confirms. Restated
# rather than imported: a test that reads the table under test would
# pass whatever the table says.
MAPPED_ROWS = {
    ("model", "self_attn.q_proj"): "attn_q",
    ("model", "self_attn.k_proj"): "attn_k",
    ("model", "self_attn.v_proj"): "attn_v",
    ("model", "self_attn.o_proj"): "attn_output",
    ("model", "mlp.gate_proj"): "ffn_gate",
    ("model", "mlp.up_proj"): "ffn_up",
    ("model", "mlp.down_proj"): "ffn_down",
    ("backbone", "mixer.in_proj"): "ssm_in",
    ("backbone", "mixer.out_proj"): "ssm_out",
    ("backbone", "mixer.gate"): "ffn_gate_inp",
    ("backbone", "mixer.shared_experts.up_proj"): "ffn_up_shexp",
    ("backbone", "mixer.shared_experts.down_proj"): "ffn_down_shexp",
    ("backbone", "mixer.q_proj"): "attn_q",
    ("backbone", "mixer.k_proj"): "attn_k",
    ("backbone", "mixer.v_proj"): "attn_v",
    ("backbone", "mixer.o_proj"): "attn_output",
}


def _gguf_stems(template: str) -> set[str]:
    """Every GGUF stem ``gguf`` assigns to one templated HF name.

    A name may serve more than one architecture, so the answer is a
    set and the table must name one member of it. The arch overrides
    are read too — a name that moved into one would otherwise read
    as a name ``gguf`` no longer carries.
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


def test_the_restated_rows_name_every_row_the_backend_carries() -> None:
    # Every check below iterates the restated copy. A row the backend
    # gained and the copy did not would be invisible to all of them.
    assert {suffix for _, suffix in MAPPED_ROWS} == set(GGUF_SUFFIX_BY_HF)


def test_the_restated_stems_match_the_backend_table() -> None:
    for (_, suffix), stem in MAPPED_ROWS.items():
        assert GGUF_SUFFIX_BY_HF[suffix] == stem


@pytest.mark.parametrize(
    ("root", "suffix"),
    sorted(MAPPED_ROWS),
    ids=[f"{root}-{suffix}" for root, suffix in sorted(MAPPED_ROWS)],
)
def test_a_mapped_row_matches_the_gguf_name_map(root: str, suffix: str) -> None:
    stems = _gguf_stems(f"{root}.layers.{{bid}}.{suffix}")

    assert f"blk.{LAYER}.{MAPPED_ROWS[(root, suffix)]}" in stems


def test_the_conv1d_class_stays_out_of_the_table_deliberately() -> None:
    # `gguf` maps the Mamba convolution, and the table omits it: the
    # class pins at the F16 passthrough and emits no override, so it
    # needs no row (the 2026-08-20 amendment). This pins the omission
    # as a decision rather than an oversight.
    assert _gguf_stems("backbone.layers.{bid}.mixer.conv1d") != set()
    assert "mixer.conv1d" not in GGUF_SUFFIX_BY_HF
