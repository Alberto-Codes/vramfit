"""RuntimeDivergenceMeter contract: the llama.cpp adapter and the fake agree.

The real side runs the true adapter — argument construction,
subprocess handling, per-chunk recovery — against a stub tool standing
in for ``llama-perplexity``, so the suite stays hermetic (ADR-0009).
The stub records its argv, so the real-only tests pin the exact
command line and the running-mean recovery the fake cannot reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fakes import MemoryRuntimeDivergenceMeter
from vramfit.adapters.outbound.gguf.divergence import LlamaCppDivergenceMeter
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.ports.outbound import RuntimeDivergenceMeter

pytestmark = pytest.mark.contract

# Per-chunk divergences, and the running means a tool would print for
# them. Chunk n's running mean is the mean of the first n values.
CHUNKS = (0.30, 0.10, 0.20, 0.40)
RUNNING = (0.30, 0.20, 0.20, 0.25)

_TOOL_STUB = """\
#!/usr/bin/env python3
import json, sys

with open({argv_log!r}, "w") as log:
    json.dump(sys.argv[1:], log)
print("kl_divergence: computing over 4 chunks, n_ctx=512")
print("chunk   PPL   ln(PPL(Q)/PPL(base))   KL Divergence   dp RMS   Same top p")
for i, mean in enumerate({running!r}, start=1):
    print(
        "%4d   6.0014 ±  1.1066   0.21574 ±  0.07030   "
        "%.5f ±  0.04850   16.873 ± 1.797 %%   83.529 ± 2.327 %%"
        % (i, mean)
    )
print("Mean    KLD:   0.250000 ±   0.001159")
"""

_FAILING_STUB = """\
#!/usr/bin/env python3
import sys

sys.stderr.write("divergence stub exploded\\n")
sys.exit(3)
"""

_NO_ROWS_STUB = """\
#!/usr/bin/env python3
print("kl_divergence: computing over 4 chunks, n_ctx=512")
"""

_GAP_STUB = """\
#!/usr/bin/env python3
for i in (1, 2, 4):
    print(
        "%4d   6.0014 ±  1.1066   0.21574 ±  0.07030   "
        "0.25000 ±  0.04850   16.873 ± 1.797 %%   83.529 ± 2.327 %%"
        % i
    )
"""


def _write_stub(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o700)
    return path


def _real_meter(
    tmp_path: Path,
    running: tuple[float, ...] = RUNNING,
    fail: bool = False,
) -> RuntimeDivergenceMeter:
    body = (
        _FAILING_STUB
        if fail
        else _TOOL_STUB.format(
            argv_log=str(tmp_path / "divergence-argv.json"), running=list(running)
        )
    )
    return LlamaCppDivergenceMeter(
        perplexity_bin=_write_stub(tmp_path / "llama-perplexity", body),
        text_path=tmp_path / "wiki.test.raw",
        base_logits=tmp_path / "base.logits",
        threads=1,
    )


def _fake_meter(
    tmp_path: Path,
    running: tuple[float, ...] = RUNNING,
    fail: bool = False,
) -> RuntimeDivergenceMeter:
    del running
    return MemoryRuntimeDivergenceMeter(default=CHUNKS, fail=fail)


@pytest.mark.parametrize(
    "build", [_real_meter, _fake_meter], ids=["real-subprocess", "fake-memory"]
)
class TestRuntimeDivergenceMeterContract:
    def test_measure_reports_one_value_per_chunk(self, build, tmp_path) -> None:
        meter: RuntimeDivergenceMeter = build(tmp_path)

        assert meter.measure(str(tmp_path / "arm.gguf")) == pytest.approx(CHUNKS)

    def test_measure_returns_the_chunks_in_order(self, build, tmp_path) -> None:
        meter: RuntimeDivergenceMeter = build(tmp_path)

        measured = meter.measure(str(tmp_path / "arm.gguf"))

        assert measured[0] == pytest.approx(0.30)
        assert measured[-1] == pytest.approx(0.40)

    def test_measure_never_returns_an_empty_series(self, build, tmp_path) -> None:
        meter: RuntimeDivergenceMeter = build(tmp_path)

        assert meter.measure(str(tmp_path / "arm.gguf"))

    def test_measure_tool_failure_raises_pack_error_with_exit_code(
        self, build, tmp_path
    ) -> None:
        meter: RuntimeDivergenceMeter = build(tmp_path, fail=True)

        with pytest.raises(PackError, match="divergence failed with exit code 3"):
            meter.measure(str(tmp_path / "arm.gguf"))


class TestLlamaCppDivergenceCommandLine:
    """Real-adapter argv and parsing contracts the fake cannot cover."""

    def test_measure_argv_carries_model_text_base_and_the_divergence_flag(
        self, tmp_path
    ) -> None:
        meter = _real_meter(tmp_path)

        meter.measure(str(tmp_path / "arm.gguf"))

        argv = json.loads((tmp_path / "divergence-argv.json").read_text())
        assert argv[argv.index("-m") + 1] == str(tmp_path / "arm.gguf")
        assert argv[argv.index("-f") + 1] == str(tmp_path / "wiki.test.raw")
        assert argv[argv.index("--kl-divergence-base") + 1] == str(
            tmp_path / "base.logits"
        )
        assert "--kl-divergence" in argv
        # Offload stays on, unlike the smoke test: a full-window pass
        # over hundreds of chunks is the refinement pass's whole cost.
        assert argv[argv.index("-ngl") + 1] == "99"

    def test_measure_recovers_chunk_values_from_running_means(self, tmp_path) -> None:
        meter = _real_meter(tmp_path)

        assert meter.measure(str(tmp_path / "arm.gguf")) == pytest.approx(CHUNKS)

    def test_measure_without_chunk_rows_raises_pack_error(self, tmp_path) -> None:
        meter = LlamaCppDivergenceMeter(
            perplexity_bin=_write_stub(tmp_path / "llama-perplexity", _NO_ROWS_STUB),
            text_path=tmp_path / "wiki.test.raw",
            base_logits=tmp_path / "base.logits",
        )

        with pytest.raises(PackError, match="without per-chunk rows"):
            meter.measure(str(tmp_path / "arm.gguf"))

    def test_measure_with_a_skipped_chunk_raises_pack_error(self, tmp_path) -> None:
        # A gap would silently shift every later chunk against the
        # other arms, which is exactly what pairing must not tolerate.
        meter = LlamaCppDivergenceMeter(
            perplexity_bin=_write_stub(tmp_path / "llama-perplexity", _GAP_STUB),
            text_path=tmp_path / "wiki.test.raw",
            base_logits=tmp_path / "base.logits",
        )

        with pytest.raises(PackError, match="skips a chunk"):
            meter.measure(str(tmp_path / "arm.gguf"))

    def test_measure_failure_tail_carries_the_merged_stderr(self, tmp_path) -> None:
        meter = _real_meter(tmp_path, fail=True)

        with pytest.raises(PackError, match="divergence stub exploded"):
            meter.measure(str(tmp_path / "arm.gguf"))

    def test_measure_with_missing_tool_raises_pack_error(self, tmp_path) -> None:
        meter = LlamaCppDivergenceMeter(
            perplexity_bin=tmp_path / "absent-tool",
            text_path=tmp_path / "wiki.test.raw",
            base_logits=tmp_path / "base.logits",
        )

        with pytest.raises(PackError, match="cannot run"):
            meter.measure(str(tmp_path / "arm.gguf"))
