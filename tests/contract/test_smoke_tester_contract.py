"""SmokeTester contract: the llama.cpp adapter and the memory fake agree.

The real side runs the true adapter — argument construction,
subprocess handling, output parsing — against a stub tool standing in
for ``llama-perplexity``, so the suite stays hermetic (ADR-0009). The
stub records its argv, so the real-only tests pin the exact command
line — the seam the verified fake cannot reach.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from quantfit.adapters.outbound.gguf.smoke import LlamaCppSmokeTester
from quantfit.adapters.outbound.gguf.types import PackError
from quantfit.ports.outbound import SmokeTester
from tests.fakes import MemorySmokeTester

pytestmark = pytest.mark.contract

WORKING_PPL = 9.9174

_TOOL_STUB = """\
#!/usr/bin/env python3
import json, sys

with open({argv_log!r}, "w") as log:
    json.dump(sys.argv[1:], log)
print("perplexity: tokenizing the input ..")
print("Final estimate: PPL = {estimate} +/- 0.07536")
"""

_FAILING_STUB = """\
#!/usr/bin/env python3
import sys

sys.stderr.write("smoke stub exploded\\n")
sys.exit(3)
"""

_NO_ESTIMATE_STUB = """\
#!/usr/bin/env python3
print("perplexity: tokenizing the input ..")
"""


def _write_stub(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o700)
    return path


def _real_tester(
    tmp_path: Path,
    estimate: str = str(WORKING_PPL),
    fail: bool = False,
) -> SmokeTester:
    body = (
        _FAILING_STUB
        if fail
        else _TOOL_STUB.format(
            argv_log=str(tmp_path / "smoke-argv.json"), estimate=estimate
        )
    )
    return LlamaCppSmokeTester(
        perplexity_bin=_write_stub(tmp_path / "llama-perplexity", body),
        model_path=tmp_path / "packed.gguf",
        text_path=tmp_path / "smoke.txt",
        chunks=2,
        threads=1,
    )


def _fake_tester(
    tmp_path: Path,
    estimate: str = str(WORKING_PPL),
    fail: bool = False,
) -> SmokeTester:
    return MemorySmokeTester(perplexity=float(estimate), fail=fail)


@pytest.mark.parametrize(
    "build", [_real_tester, _fake_tester], ids=["real-subprocess", "fake-memory"]
)
class TestSmokeTesterContract:
    def test_smoke_reports_the_final_estimate(self, build, tmp_path) -> None:
        tester: SmokeTester = build(tmp_path)

        assert tester.smoke() == WORKING_PPL

    def test_smoke_with_destroyed_artifact_reports_the_huge_estimate(
        self, build, tmp_path
    ) -> None:
        tester: SmokeTester = build(tmp_path, estimate="1020627.8662")

        assert tester.smoke() == 1020627.8662

    def test_smoke_with_nan_estimate_reports_nan(self, build, tmp_path) -> None:
        tester: SmokeTester = build(tmp_path, estimate="nan")

        assert math.isnan(tester.smoke())

    def test_smoke_with_inf_estimate_reports_inf(self, build, tmp_path) -> None:
        tester: SmokeTester = build(tmp_path, estimate="inf")

        assert tester.smoke() == float("inf")

    def test_smoke_tool_failure_raises_pack_error_with_exit_code(
        self, build, tmp_path
    ) -> None:
        tester: SmokeTester = build(tmp_path, fail=True)

        with pytest.raises(PackError, match="smoke failed with exit code 3"):
            tester.smoke()


class TestLlamaCppSmokeCommandLine:
    """Real-adapter argv and parsing contracts the fake cannot cover."""

    def test_smoke_argv_carries_model_text_chunks_and_threads(self, tmp_path) -> None:
        tester = _real_tester(tmp_path)

        tester.smoke()

        argv = json.loads((tmp_path / "smoke-argv.json").read_text())
        assert argv[argv.index("-m") + 1] == str(tmp_path / "packed.gguf")
        assert argv[argv.index("-f") + 1] == str(tmp_path / "smoke.txt")
        assert argv[argv.index("--chunks") + 1] == "2"
        assert argv[argv.index("-t") + 1] == "1"
        # Layer offload must be off — the smoke test never contends
        # for the GPU (ADR-0017).
        assert argv[argv.index("-ngl") + 1] == "0"

    def test_smoke_without_final_estimate_raises_pack_error(self, tmp_path) -> None:
        tester = LlamaCppSmokeTester(
            perplexity_bin=_write_stub(
                tmp_path / "llama-perplexity", _NO_ESTIMATE_STUB
            ),
            model_path=tmp_path / "packed.gguf",
            text_path=tmp_path / "smoke.txt",
        )

        with pytest.raises(PackError, match="without a final perplexity estimate"):
            tester.smoke()

    def test_smoke_with_unreadable_estimate_raises_pack_error(self, tmp_path) -> None:
        tester = _real_tester(tmp_path, estimate="garbage")

        with pytest.raises(PackError, match="unreadable estimate"):
            tester.smoke()

    def test_smoke_failure_tail_carries_the_merged_stderr(self, tmp_path) -> None:
        # The adapter merges stderr into stdout so the failure tail is
        # the tool's real last words — the operator's debugging
        # lifeline (ADR-0017).
        tester = _real_tester(tmp_path, fail=True)

        with pytest.raises(PackError, match="smoke stub exploded"):
            tester.smoke()

    def test_smoke_with_missing_tool_raises_pack_error(self, tmp_path) -> None:
        tester = LlamaCppSmokeTester(
            perplexity_bin=tmp_path / "absent-tool",
            model_path=tmp_path / "packed.gguf",
            text_path=tmp_path / "smoke.txt",
        )

        with pytest.raises(PackError, match="cannot run"):
            tester.smoke()
