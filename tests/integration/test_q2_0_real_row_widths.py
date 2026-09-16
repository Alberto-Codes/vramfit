"""The assisted ``Q2_0`` pack path at the 30B target's own row widths.

``test_q2_0_stock_toolchain`` proves the path on a 64-wide expert
stack. The 30B acceptance target carries no such row: its routed
stacks are shaped ``(128, 1856, 2688)`` and ``(128, 2688, 1856)``,
so they hold rows of 2688 and 1856 (ADR-0026, ADR-0028). This module
drives `LlamaCppPacker.pack` at those two widths before a rented card
is booked. It keeps two experts, because the encoder fits each row
alone and the expert count only multiplies the row count.

The recipe assigns nominal 2, so the ADR-0028 routing decides the
tensor type from the measured row width. Nothing here names ``q2_0``.
The stock runtime then loads the packed file and runs a forward pass,
which is what reads the blocks back at 29 and 42 blocks per row.

The meter and the encoder share one fit, so their agreement alone
would hold on a degenerate result. `_weighted_error` bounds the fit
instead: the assisted decode must cost less imatrix-weighted squared
error than the unassisted ``q0`` reference at each width.

The tools come from ``VRAMFIT_LLAMA_CPP_BIN``, a directory holding
stock ``llama-quantize`` and ``llama-bench``. The suite skips with
that reason when the variable is unset, when a tool is missing, or
when the scan extra is absent (ADR-0009). A skip is the honest
outcome, and it never reports a pass. The variable names the build,
so the module cannot pin one. ``docs/adr/evidence/0032`` records the
run that discharged ADR-0032's row-width consequence.
"""

# ruff: noqa: E402 - the importorskip guards must run before adapter imports
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")
pytest.importorskip("gguf", reason="scan extra not installed")

from gguf import GGMLQuantizationType

from tests.integration.q2_0_fixture import (
    DOWN,
    DOWN_GROUP,
    GATE,
    GATE_GROUP,
    UP,
    UP_GROUP,
    payload,
    tool,
    write_imatrix,
    write_model,
)
from vramfit.adapters.outbound.gguf.header import read_header
from vramfit.adapters.outbound.gguf.pack import LlamaCppPacker
from vramfit.adapters.outbound.gguf.q2_0_blocks import (
    Q2_0_TYPE_ID,
    dequantize_q2_0,
)
from vramfit.adapters.outbound.scan.imatrix import load_imatrix
from vramfit.adapters.outbound.scan.q2_0_assisted import Q2_0_ENCODER_REVISION
from vramfit.adapters.outbound.scan.within_group import perturb
from vramfit.domain.model import (
    Q0_IMX2_METHOD,
    Assignment,
    PlanMeta,
    Recipe,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# The 30B target's own widths. The gate and up stacks hold rows of
# N_EMBD, the down stack rows of N_FF. Neither divides by 256, so
# both take the ADR-0028 table.
N_EMBD = 2688
N_FF = 1856
# The two stacks the recipe assigns nominal 2, and the row width each
# carries. Together they cover both target widths.
ENCODED = ((DOWN, DOWN_GROUP, N_FF), (GATE, GATE_GROUP, N_EMBD))
# The routed peer: nominal 4 at 2688, which the same table maps to
# q4_0 and the stock pass quantizes beside the pre-encoded stacks.
PEER_BITS = 4
# The dense attention classes. Their 2688-wide rows refuse the 256
# super-block too, so the recipe must assign them rather than leave
# them to the Q2_K floor the nominal-2 stacks set. The quantizer
# rewrites a k-quant it cannot block at that width, and the pack
# halts on the warning pair (ADR-0028 decision 3).
ATTENTION_GROUPS = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
)
# The embedding takes `--token-embedding-type` from the ADR-0012
# k-quant table, which no row width reaches. 8 is the one precision
# whose block divides 2688. Nominal 4 or 2 sends a 256-block k-quant
# to a 2688-wide row, and the quantizer aborts on a ggml block-size
# assertion at tensor 2 of 12, which is
# [issue #608](https://github.com/Alberto-Codes/vramfit/issues/608).
EMBEDDING_GROUP = "model.embed_tokens"
EMBEDDING_BITS = 8


def _recipe(imatrix: Path) -> Recipe:
    return Recipe(
        model_id="fixture",
        plan=PlanMeta(
            vram_budget_bytes=10**9,
            kv_headroom_bytes=10**6,
            weight_budget_bytes=10**9,
            predicted_total_bytes=10**5,
            predicted_damage=0.1,
            solver="greedy-damage-per-byte",
            pins={},
            protections={},
            format_overhead=0.05,
            trace=(),
        ),
        assignments=(
            Assignment(group=DOWN_GROUP, bits=2, bytes=1_000, damage=0.01),
            Assignment(group=GATE_GROUP, bits=2, bytes=1_000, damage=0.01),
            Assignment(group=UP_GROUP, bits=PEER_BITS, bytes=1_000, damage=0.01),
            Assignment(
                group=EMBEDDING_GROUP,
                bits=EMBEDDING_BITS,
                bytes=1_000,
                damage=0.01,
            ),
            *(
                Assignment(group=group, bits=PEER_BITS, bytes=1_000, damage=0.01)
                for group in ATTENTION_GROUPS
            ),
        ),
        runtime="llama.cpp",
        within_group=Q0_IMX2_METHOD,
        imatrix=str(imatrix),
        protected_tensors=(),
    )


def _weighted_error(
    weight: torch.Tensor, fitted: torch.Tensor, column_weights: torch.Tensor
) -> float:
    """State one fit's imatrix-weighted squared error on a stack.

    Each element weighs by its expert's imatrix column weight, the
    per-expert mapping the encoder applies (ADR-0026). The quantity
    lives in weight space, so the glossary rules it reconstruction
    error and not damage. Issue #302 measured a weight-space term and
    measured damage ordering apart on the 30B target. It serves one
    purpose here: a fit that beats the unassisted reference is not
    degenerate.

    Args:
        weight: The original stack, shape ``(experts, rows, row)``
            or flat.
        fitted: The round-tripped values, same element count.
        column_weights: Imatrix weights, shape ``(experts, row)``.

    Returns:
        The summed weighted squared error.
    """
    experts, row = column_weights.shape
    shaped = weight.reshape(experts, -1, row)
    diff = shaped - fitted.reshape(experts, -1, row)
    return float((column_weights[:, None, :] * diff * diff).sum())


def _packer(
    tmp_path: Path, base: Path, imatrix: Path, quantize: Path
) -> LlamaCppPacker:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    return LlamaCppPacker(
        model_dir=model_dir,
        base_gguf=base,
        out_path=tmp_path / "packed.gguf",
        convert_script=base,
        quantize_bin=quantize,
        python_bin=Path(sys.executable),
        threads=2,
        imatrix=imatrix,
        row_widths={
            DOWN_GROUP: N_FF,
            UP_GROUP: N_EMBD,
            GATE_GROUP: N_EMBD,
            **dict.fromkeys(ATTENTION_GROUPS, N_EMBD),
        },
    )


class TestRealRowWidths:
    """The pack path and the stock runtime at 2688 and 1856."""

    def test_pack_pre_encodes_both_target_widths_and_the_runtime_loads_the_file(
        self, tmp_path: Path
    ) -> None:
        quantize = tool("llama-quantize")
        bench = tool("llama-bench")
        rng = np.random.default_rng(2688)
        base = tmp_path / "base-f16.gguf"
        tensors = write_model(
            base,
            rng,
            n_embd=N_EMBD,
            n_ff=N_FF,
            name="vramfit q2_0 real-row-width fixture",
        )
        imatrix = tmp_path / "imatrix.gguf"
        write_imatrix(imatrix, rng, n_embd=N_EMBD, n_ff=N_FF, zero_count_expert=False)
        header = read_header(base)
        by_name = {t.name: t for t in header.tensors}
        assert by_name[DOWN].dims[0] == N_FF
        assert by_name[GATE].dims[0] == N_EMBD
        packer = _packer(tmp_path, base, imatrix, quantize)

        result = packer.pack(_recipe(imatrix))

        # The ADR-0028 routing chose q2_0 from the measured widths.
        assert set(result.pre_encoded) == {DOWN, GATE}
        assert result.q2_0_encoder == Q2_0_ENCODER_REVISION
        by_type = {t.name: t.type_id for t in read_header(packer.out_path).tensors}
        assert by_type[DOWN] == Q2_0_TYPE_ID
        assert by_type[GATE] == Q2_0_TYPE_ID
        # The routed peer at 2688 still quantizes beside them.
        assert by_type[UP] == GGMLQuantizationType.Q4_0
        assert not packer.mixed_gguf.exists()
        assert not packer.pre_encode_dir.exists()

        # `pack` runs `verify_pre_encoded` on the quantizer's output,
        # so it already refused a payload the stock pass changed. The
        # check below is the independent one: the scan meter fits each
        # stack again and decodes the packed bytes to those exact
        # values, which a stock requantize could not produce.
        entries = load_imatrix(imatrix)
        for name, group, width in ENCODED:
            packed_bytes = payload(packer.out_path, name)
            weight = torch.from_numpy(tensors[name].astype(np.float32))
            priced = perturb(
                weight, 2, group, "q0-imx2", 32, entries[name].column_weights
            )
            decoded = dequantize_q2_0(packed_bytes, weight.numel())
            assert np.array_equal(decoded, priced.reshape(-1).numpy()), (
                f"the scan meter and the pack disagree on {width}-wide {name}"
            )
            # Both sides of that equality run the one encoder, so it
            # holds on a degenerate fit too — all-zero blocks satisfy
            # it. The bound below does not: the assisted fit must beat
            # the unassisted reference on the metric the imatrix
            # defines, which no degenerate fit does.
            column_weights = entries[name].column_weights.to(torch.float32)
            reference = perturb(weight, 2, group, "q0", 32, None)
            assisted_error = _weighted_error(
                weight, torch.from_numpy(decoded), column_weights
            )
            reference_error = _weighted_error(weight, reference, column_weights)
            print(
                f"weighted squared error at {width}-wide {name}: "
                f"assisted {assisted_error:.6e}, "
                f"unassisted reference {reference_error:.6e}"
            )
            assert assisted_error < reference_error, (
                f"the assisted fit does not beat the unassisted reference on "
                f"{width}-wide {name}: {assisted_error:.6e} versus "
                f"{reference_error:.6e}"
            )

        # The stock runtime reads the blocks back at 29 and 42 blocks
        # per row, which no byte comparison covers.
        run = subprocess.run(  # noqa: S603 - fixed tool path, test-controlled args
            [
                str(bench),
                "-m",
                str(packer.out_path),
                "-p",
                "8",
                "-n",
                "0",
                "-ngl",
                "0",
                "-r",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        assert run.returncode == 0, run.stdout + run.stderr
