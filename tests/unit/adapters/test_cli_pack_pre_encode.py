"""Checks of what ``vramfit pack`` records and echoes for a pre-encoding pack (ADR-0032)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fakes import MemoryImatrixCountSource
from tests.unit.adapters.test_cli_pack import make_recipe
from vramfit.adapters.inbound import cli_pack, cli_pack_imatrix
from vramfit.adapters.inbound.cli import app
from vramfit.adapters.outbound.recipe_json import save_recipe
from vramfit.adapters.outbound.run_log_jsonl import read_run_log
from vramfit.domain.model import Q0_IMX2_METHOD, Recipe
from vramfit.domain.pack import PackResult, TypeOverride

pytestmark = pytest.mark.unit

runner = CliRunner()


class _PreEncodingPacker:
    """A packer whose result carries the pre-encoding record."""

    def __init__(self, out: Path) -> None:
        self.out = out

    def convert(self) -> int:
        return 1_000

    def pack(self, recipe: Recipe) -> PackResult:
        self.out.write_bytes(b"G" * 500)
        return PackResult(
            packed_bytes=500,
            base_type="Q2_K",
            token_embedding_type=None,
            output_tensor_type=None,
            overrides=(TypeOverride(r"blk\.0\.ffn_down_exps\.", "q2_0"),),
            imatrix_path=recipe.imatrix,
            file_type="Q2_0",
            pre_encoded=("blk.0.ffn_down_exps.weight",),
            q2_0_encoder="vramfit-q2_0-assisted-1",
        )


@pytest.fixture
def llama_cpp_dir(tmp_path: Path) -> Path:
    checkout = tmp_path / "llama.cpp"
    (checkout / "build" / "bin").mkdir(parents=True)
    (checkout / "convert_hf_to_gguf.py").touch()
    (checkout / "build" / "bin" / "llama-quantize").touch()
    (checkout / "build" / "bin" / "llama-perplexity").touch()
    return checkout


def test_pack_records_and_echoes_the_pre_encoding_stage(
    tmp_path: Path, llama_cpp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    imatrix = tmp_path / "m.gguf"
    imatrix.write_bytes(b"not read: the count source is faked")
    recipe = make_recipe(
        str(model_dir), within_group=Q0_IMX2_METHOD, imatrix=str(imatrix)
    )
    recipe_path = tmp_path / "recipe.json"
    save_recipe(recipe, recipe_path)
    out = tmp_path / "packed.gguf"
    monkeypatch.setattr(
        cli_pack, "_build_packer", lambda *args: _PreEncodingPacker(out)
    )
    monkeypatch.setattr(
        cli_pack_imatrix,
        "_build_count_source",
        lambda *args: MemoryImatrixCountSource(),
    )

    result = runner.invoke(
        app,
        [
            "pack",
            str(recipe_path),
            "--llama-cpp",
            str(llama_cpp_dir),
            "--out",
            str(out),
            "--imatrix",
            str(imatrix),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        "pre-encoded 1 tensor with the assisted Q2_0 encoder "
        "vramfit-q2_0-assisted-1 (ADR-0032)"
    ) in result.output
    packed = next(
        line
        for line in read_run_log(out.with_name(out.stem + ".runlog.jsonl"))
        if line["event"] == "model_packed"
    )
    assert packed["pre_encoded"] == ["blk.0.ffn_down_exps.weight"]
    assert packed["q2_0_encoder"] == "vramfit-q2_0-assisted-1"
    assert "pre_encode_cost" not in packed
