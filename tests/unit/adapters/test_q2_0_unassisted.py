"""Checks of the unassisted ``Q2_0`` encoder path at the target's row widths.

ADR-0018's 2026-09-17 amendment gives the unassisted fit its own
``within_group`` token, ``q0-fit2``. The encoder program is the one
the pack drives, so this module runs that program at the 30B
target's own row widths of 2688 and 1856 (ADR-0026, ADR-0028), once
with an importance matrix and once without.

The two runs must differ, or the token would name a distinction the
bytes do not carry. The unassisted payload must also decode to the
shipped fit's weight-None output, or the scan would price cells the
pack does not emit.

The program runs in process here. `test_q2_0_stock_toolchain` drives
it as the separate program the pack launches.
"""

# ruff: noqa: E402 - the importorskip guards must run before adapter imports
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="scan extra not installed")
np = pytest.importorskip("numpy", reason="scan extra not installed")
pytest.importorskip("gguf", reason="scan extra not installed")

from gguf import GGUFWriter

from vramfit.adapters.outbound.gguf.q2_0_blocks import dequantize_q2_0
from vramfit.adapters.outbound.scan.q2_0_assisted import (
    q2_0_assisted_quantize_dequantize,
)
from vramfit.adapters.outbound.scan.q2_0_program import encode
from vramfit.adapters.outbound.scan.within_group import perturb

pytestmark = pytest.mark.unit

# The 30B target's two routed-expert row widths, against the stacks
# that carry them (ADR-0026, ADR-0028): the up stack holds rows of
# n_embd 2688, and the down stack rows of n_ff 1856, the same mapping
# `tests.integration.q2_0_fixture` writes.
UP = "blk.0.ffn_up_exps.weight"
DOWN = "blk.0.ffn_down_exps.weight"
WIDTHS = {UP: 2688, DOWN: 1856}
ROWS = 2


def write_base(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write the float base the encoder reads."""
    writer = GGUFWriter(path, "llama")
    writer.add_block_count(1)
    writer.add_file_type(1)
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_imatrix(path: Path, rng: np.random.Generator) -> None:
    """Write an imatrix covering both tensors with one matrix each."""
    writer = GGUFWriter(path, "imatrix")
    writer.add_type("imatrix")
    writer.add_uint32("imatrix.chunk_count", 4)
    for name, columns in WIDTHS.items():
        sums = rng.uniform(0.5, 4.0, size=(1, columns)).astype(np.float32)
        writer.add_tensor(f"{name}.in_sum2", sums)
        writer.add_tensor(f"{name}.counts", np.array([4.0], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def run(base: Path, out_dir: Path, imatrix: Path | None) -> dict[str, bytes]:
    """Encode both tensors and read back the payload bytes."""
    report = encode(
        argparse.Namespace(
            base_gguf=base,
            imatrix=imatrix,
            out_dir=out_dir,
            threads=1,
            tensors=list(WIDTHS),
        )
    )
    return {
        name: Path(entry["payload"]).read_bytes()
        for name, entry in report["tensors"].items()
    }


@dataclass(frozen=True)
class Workspace:
    """The base, its matrix, the float tensors, and the working root."""

    base: Path
    imatrix: Path
    tensors: dict[str, np.ndarray]
    root: Path

    def weight(self, name: str) -> torch.Tensor:
        """Read one tensor back as float32 torch rows."""
        return torch.from_numpy(self.tensors[name].astype(np.float32))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    rng = np.random.default_rng(11)
    tensors = {
        name: (rng.standard_normal((ROWS, columns)) * 0.05).astype(np.float16)
        for name, columns in WIDTHS.items()
    }
    base = tmp_path / "base.gguf"
    write_base(base, tensors)
    imatrix = tmp_path / "m.gguf"
    write_imatrix(imatrix, rng)
    return Workspace(base=base, imatrix=imatrix, tensors=tensors, root=tmp_path)


class TestMatrixFreeEncoder:
    def test_unassisted_payloads_differ_from_the_assisted_ones(
        self, workspace: Workspace
    ) -> None:
        assisted = run(workspace.base, workspace.root / "assisted", workspace.imatrix)
        free = run(workspace.base, workspace.root / "free", None)

        for name, columns in WIDTHS.items():
            assert len(free[name]) == len(assisted[name])
            assert free[name] != assisted[name], f"{name} at {columns} columns"

    def test_unassisted_payload_decodes_to_the_shipped_weight_none_fit(
        self, workspace: Workspace
    ) -> None:
        payloads = run(workspace.base, workspace.root / "free", None)

        for name, columns in WIDTHS.items():
            weight = workspace.weight(name)
            decoded = torch.from_numpy(
                dequantize_q2_0(payloads[name], ROWS * columns).astype(np.float32)
            ).reshape(ROWS, columns)
            expected = q2_0_assisted_quantize_dequantize(weight, None)
            assert torch.equal(decoded, expected), f"{name} at {columns} columns"

    def test_the_unassisted_method_prices_that_same_fit(
        self, workspace: Workspace
    ) -> None:
        # The scan must price what the pack emits, so the method's
        # nominal-2 cell is the encoder's weight-None output.
        for name, columns in WIDTHS.items():
            weight = workspace.weight(name)

            priced = perturb(weight, 2, name, "q0-fit2", 32, None)

            assert torch.equal(
                priced, q2_0_assisted_quantize_dequantize(weight, None)
            ), f"{name} at {columns} columns"
