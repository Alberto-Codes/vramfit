"""Checks of the pre-encoding stage's selection, program driver, and verification.

The selection follows the quantizer's first-match rule and refuses
before anything is written (ADR-0032). The program driver runs a
stub encoder here, so the suite stays torch-free; the real program
is proven in the scan-tier and stock-toolchain suites.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from vramfit.adapters.outbound.gguf.header import GgufHeader, TensorInfo
from vramfit.adapters.outbound.gguf.pre_encode import (
    PreEncodeTarget,
    quantizer_skips,
    run_encoder,
    select_pre_encode_targets,
    verify_pre_encoded,
)
from vramfit.adapters.outbound.gguf.q2_0_blocks import Q2_0_TYPE_ID, q2_0_payload_bytes
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.pack import TypeOverride

pytestmark = pytest.mark.unit

STACK = "blk.0.ffn_down_exps.weight"
UP = "blk.0.ffn_up_exps.weight"
NORM = "blk.0.attn_norm.weight"
ROUTER = "blk.0.ffn_gate_inp.weight"


def header(*infos: TensorInfo) -> GgufHeader:
    return GgufHeader(
        version=3,
        alignment=32,
        file_type_offset=None,
        file_type=0,
        infos_offset=0,
        data_start=0,
        tensors=infos,
    )


def base_header() -> GgufHeader:
    return header(
        TensorInfo(STACK, (128, 4, 2), 1, 0),
        TensorInfo(UP, (64, 8, 2), 1, 2048),
        TensorInfo(NORM, (64,), 0, 4096),
        TensorInfo(ROUTER, (64, 2), 0, 4352),
        TensorInfo("blk.0.attn_q.weight", (64, 16), 1, 4864),
    )


Q2_0_STACK = TypeOverride(r"blk\.0\.ffn_down_exps\.", "q2_0")
Q2_0_UP = TypeOverride(r"blk\.0\.ffn_up_exps\.", "q2_0")
LAYER = TypeOverride(r"blk\.0\.", "q4_k")


class TestSelectPreEncodeTargets:
    def test_covered_q2_0_stacks_select_in_header_order(self) -> None:
        targets = select_pre_encode_targets(
            (Q2_0_STACK, Q2_0_UP, LAYER),
            base_header(),
            covered={STACK, UP},
            excluded=(),
        )
        assert targets == (PreEncodeTarget(STACK, 1024), PreEncodeTarget(UP, 1024))

    def test_uncovered_and_excluded_tensors_pack_unassisted(self) -> None:
        targets = select_pre_encode_targets(
            (Q2_0_STACK, Q2_0_UP), base_header(), covered={STACK, UP}, excluded=(UP,)
        )
        assert targets == (PreEncodeTarget(STACK, 1024),)
        assert select_pre_encode_targets(
            (Q2_0_STACK, Q2_0_UP), base_header(), covered={UP}, excluded=()
        ) == (PreEncodeTarget(UP, 1024),)

    def test_no_q2_0_override_selects_nothing(self) -> None:
        assert (
            select_pre_encode_targets(
                (LAYER,), base_header(), covered={STACK}, excluded=()
            )
            == ()
        )

    def test_a_layer_pattern_at_q2_0_skips_the_norm_and_the_router(self) -> None:
        layer_q2_0 = TypeOverride(r"blk\.0\.", "q2_0")
        names = {STACK, UP, NORM, ROUTER, "blk.0.attn_q.weight"}
        targets = select_pre_encode_targets(
            (layer_q2_0,), base_header(), covered=names, excluded=()
        )
        assert [t.name for t in targets] == [STACK, UP, "blk.0.attn_q.weight"]

    def test_shadowing_override_refuses_before_writing(self) -> None:
        protection = TypeOverride(r"blk\.0\.ffn_down_exps\.weight", "q8_0")
        with pytest.raises(
            PackError, match="applies the first matching pattern"
        ) as info:
            select_pre_encode_targets(
                (protection, Q2_0_STACK), base_header(), covered={STACK}, excluded=()
            )
        assert "refuses before the preprocessor writes" in str(info.value)
        assert STACK in str(info.value)

    def test_a_shadowed_uncovered_tensor_packs_stock_without_refusing(self) -> None:
        # The preprocessor never writes it, so the quantizer never
        # meets it as Q2_0 (ADR-0032 decision 3).
        protection = TypeOverride(r"blk\.0\.ffn_down_exps\.weight", "q8_0")
        assert (
            select_pre_encode_targets(
                (protection, Q2_0_STACK), base_header(), covered=(), excluded=()
            )
            == ()
        )

    def test_an_excluded_tensor_with_odd_rows_packs_stock_without_refusing(
        self,
    ) -> None:
        odd = header(TensorInfo(STACK, (96, 4, 2), 1, 0))
        assert (
            select_pre_encode_targets(
                (Q2_0_STACK,), odd, covered={STACK}, excluded=(STACK,)
            )
            == ()
        )

    def test_rows_outside_the_block_refuse_before_writing(self) -> None:
        odd = header(TensorInfo(STACK, (96, 4, 2), 1, 0))
        with pytest.raises(PackError, match="rows of 96") as info:
            select_pre_encode_targets((Q2_0_STACK,), odd, covered={STACK}, excluded=())
        assert "refuses before the preprocessor writes" in str(info.value)

    def test_patterns_match_lower_cased_and_unanchored(self) -> None:
        upper = TypeOverride(r"FFN_DOWN_EXPS", "q2_0")
        targets = select_pre_encode_targets(
            (upper,), base_header(), covered={STACK}, excluded=()
        )
        assert targets == (PreEncodeTarget(STACK, 1024),)

    def test_dedicated_flags_keep_their_tensors_out(self) -> None:
        embedding = header(TensorInfo("token_embd.weight", (64, 32), 1, 0))
        catch_all = TypeOverride("weight", "q2_0")
        assert (
            select_pre_encode_targets(
                (catch_all,),
                embedding,
                covered={"token_embd.weight"},
                excluded=(),
                embedding_flag=True,
            )
            == ()
        )
        assert select_pre_encode_targets(
            (catch_all,), embedding, covered={"token_embd.weight"}, excluded=()
        ) == (PreEncodeTarget("token_embd.weight", 2048),)


class TestQuantizerSkips:
    @pytest.mark.parametrize(
        ("info", "skipped"),
        [
            (TensorInfo(NORM, (64,), 0, 0), True),
            (TensorInfo("blk.0.attn_norm.weight", (64, 2), 0, 0), True),
            (TensorInfo(ROUTER, (64, 2), 0, 0), True),
            (TensorInfo("blk.0.attn_q.bias", (64, 2), 0, 0), True),
            (TensorInfo("output.weight", (64, 2), 1, 0), False),
            (TensorInfo(STACK, (128, 4, 2), 1, 0), False),
        ],
    )
    def test_models_the_quantizer_skip_rules(
        self, info: TensorInfo, skipped: bool
    ) -> None:
        assert quantizer_skips(info, embedding_flag=False, output_flag=False) is skipped

    def test_output_flag_binds_the_head_before_any_pattern(self) -> None:
        head = TensorInfo("output.weight", (64, 2), 1, 0)
        assert quantizer_skips(head, embedding_flag=False, output_flag=True)


STUB_ENCODER = """\
import json, sys
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1]
names = [args[i + 1] for i, a in enumerate(args) if a == "--tensor"]
out = opt("--out-dir")
import os, hashlib
os.makedirs(out, exist_ok=True)
sizes = json.loads(os.environ["STUB_SIZES"])
report = {"encoder": "stub-encoder", "tensors": {}}
for i, name in enumerate(names):
    path = os.path.join(out, f"{i}.q2_0")
    payload = bytes([i + 1]) * sizes[name]
    with open(path, "wb") as h:
        h.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    report["tensors"][name] = {"payload": path, "bytes": len(payload), "sha256": digest}
with open(opt("--report"), "w") as h:
    json.dump(report, h)
with open(os.environ["STUB_ARGV"], "w") as h:
    json.dump(args, h)
"""


def stub_command(tmp_path: Path) -> tuple[str, ...]:
    script = tmp_path / "stub_encoder.py"
    script.write_text(STUB_ENCODER)
    return (sys.executable, str(script))


class TestRunEncoder:
    def test_passes_every_input_and_reads_the_report_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        argv_log = tmp_path / "argv.json"
        monkeypatch.setenv("STUB_ARGV", str(argv_log))
        monkeypatch.setenv("STUB_SIZES", json.dumps({STACK: 288, UP: 288}))
        targets = (PreEncodeTarget(STACK, 1024), PreEncodeTarget(UP, 1024))
        report = run_encoder(
            stub_command(tmp_path),
            base_gguf=tmp_path / "base.gguf",
            imatrix=tmp_path / "m.gguf",
            targets=targets,
            work_dir=tmp_path / "work",
            threads=3,
        )
        argv = json.loads(argv_log.read_text())
        assert argv[: argv.index("--tensor")] == [
            "--base-gguf",
            str(tmp_path / "base.gguf"),
            "--imatrix",
            str(tmp_path / "m.gguf"),
            "--out-dir",
            str(tmp_path / "work"),
            "--report",
            str(tmp_path / "work" / "report.json"),
            "--threads",
            "3",
        ]
        assert argv[argv.index("--tensor") :] == ["--tensor", STACK, "--tensor", UP]
        assert report.encoder == "stub-encoder"
        assert [t.name for t in report.tensors] == [STACK, UP]
        assert report.tensors[0].size == q2_0_payload_bytes(1024)
        assert report.tensors[0].payload.read_bytes() == b"\x01" * 288

    def test_wrong_payload_size_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STUB_ARGV", str(tmp_path / "argv.json"))
        monkeypatch.setenv("STUB_SIZES", json.dumps({STACK: 100}))
        with pytest.raises(PackError, match="reported 100 bytes"):
            run_encoder(
                stub_command(tmp_path),
                base_gguf=tmp_path / "b",
                imatrix=tmp_path / "m",
                targets=(PreEncodeTarget(STACK, 1024),),
                work_dir=tmp_path / "work",
                threads=1,
            )

    def test_failing_program_refuses_with_its_last_lines(self, tmp_path: Path) -> None:
        script = tmp_path / "broken.py"
        script.write_text(
            "import sys; print('error: no such tensor', file=sys.stderr); sys.exit(1)"
        )
        with pytest.raises(
            PackError, match="pre-encode failed with exit code 1"
        ) as info:
            run_encoder(
                (sys.executable, str(script)),
                base_gguf=tmp_path / "b",
                imatrix=tmp_path / "m",
                targets=(PreEncodeTarget(STACK, 1024),),
                work_dir=tmp_path / "work",
                threads=1,
            )
        assert "no such tensor" in str(info.value)

    def test_missing_report_refuses(self, tmp_path: Path) -> None:
        script = tmp_path / "silent.py"
        script.write_text("pass")
        with pytest.raises(PackError, match="cannot read the encoder report"):
            run_encoder(
                (sys.executable, str(script)),
                base_gguf=tmp_path / "b",
                imatrix=tmp_path / "m",
                targets=(PreEncodeTarget(STACK, 1024),),
                work_dir=tmp_path / "work",
                threads=1,
            )

    def test_malformed_report_refuses(self, tmp_path: Path) -> None:
        script = tmp_path / "odd.py"
        script.write_text(
            "import sys, json; a = sys.argv[1:]; "
            "open(a[a.index('--report') + 1], 'w').write(json.dumps({'encoder': 'x'}))"
        )
        with pytest.raises(PackError, match="is malformed"):
            run_encoder(
                (sys.executable, str(script)),
                base_gguf=tmp_path / "b",
                imatrix=tmp_path / "m",
                targets=(PreEncodeTarget(STACK, 1024),),
                work_dir=tmp_path / "work",
                threads=1,
            )


def write_packed(path: Path, payload: bytes, type_id: int = Q2_0_TYPE_ID) -> None:
    """Write a minimal GGUF holding one 1024-element tensor."""
    import struct

    body = bytearray(b"GGUF")
    body += struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 0)
    name = STACK.encode()
    body += struct.pack("<Q", len(name)) + name + struct.pack("<I", 3)
    for dim in (128, 4, 2):
        body += struct.pack("<Q", dim)
    body += struct.pack("<I", type_id) + struct.pack("<Q", 0)
    body += bytes(-len(body) % 32)
    body += payload
    path.write_bytes(bytes(body))


class TestVerifyPreEncoded:
    def test_matching_bytes_pass(self, tmp_path: Path) -> None:
        payload = bytes(range(256)) + bytes(32)
        packed = tmp_path / "out.gguf"
        write_packed(packed, payload)
        verify_pre_encoded(packed, {STACK: hashlib.sha256(payload).hexdigest()})

    def test_changed_bytes_refuse_and_name_the_tensor(self, tmp_path: Path) -> None:
        packed = tmp_path / "out.gguf"
        write_packed(packed, bytes(288))
        with pytest.raises(PackError, match="holds different bytes") as info:
            verify_pre_encoded(packed, {STACK: hashlib.sha256(b"x" * 288).hexdigest()})
        assert STACK in str(info.value)
        assert "kept for inspection" in str(info.value)

    def test_another_type_refuses(self, tmp_path: Path) -> None:
        packed = tmp_path / "out.gguf"
        write_packed(packed, bytes(1024 // 32 * 34), type_id=8)
        with pytest.raises(PackError, match="type id 8, not Q2_0"):
            verify_pre_encoded(packed, {STACK: "0" * 64})

    def test_missing_tensor_refuses(self, tmp_path: Path) -> None:
        packed = tmp_path / "out.gguf"
        write_packed(packed, bytes(288))
        with pytest.raises(PackError, match=r'carries no tensor "blk\.9\.x"'):
            verify_pre_encoded(packed, {"blk.9.x": "0" * 64})
