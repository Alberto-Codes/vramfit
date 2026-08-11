"""llama.cpp smoke-test adapter: a few perplexity chunks by subprocess.

Implements the `SmokeTester` port for the GGUF serving path
(ADR-0017). One `smoke` call runs ``llama-perplexity`` over the
packed model for a handful of chunks — layer offload disabled, with
a timeout, through the shared toolchain plumbing
([vramfit.adapters.outbound.gguf.toolrun][]) — and parses the
tool's final estimate. The measurement comes back verbatim — NaN included — and
the caller judges it against the ceiling
(`vramfit.domain.pack.smoke_passed`). A tool that cannot start,
exits nonzero, or reports no final estimate raises `PackError` with
its last output lines (ADR-0011).

Examples:
    Smoke-test a packed model:

    ```python
    tester = LlamaCppSmokeTester(
        perplexity_bin=Path("llama.cpp/build/bin/llama-perplexity"),
        model_path=Path("packed.gguf"),
        text_path=Path("wiki.test.raw"),
    )
    perplexity = tester.smoke()
    ```

See Also:
    - [vramfit.ports.outbound][]: `SmokeTester`, which this
      satisfies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.gguf.toolrun import run_tool, tail_of
from vramfit.adapters.outbound.gguf.types import PackError

_FINAL_ESTIMATE: Final[re.Pattern[str]] = re.compile(r"Final estimate: PPL = (\S+)")


@dataclass(frozen=True, slots=True)
class LlamaCppSmokeTester:
    """`SmokeTester` adapter driving ``llama-perplexity``.

    Attributes:
        perplexity_bin (Path): ``llama-perplexity`` path. The run
            disables layer offload (``-ngl 0``) — the smoke test
            must not contend for the GPU (ADR-0017).
        model_path (Path): The packed model to prove.
        text_path (Path): Text the chunks run over.
        chunks (int): Chunk count. Destroyed and working artifacts
            sit 5 orders of magnitude apart, so 2 chunks decide.
        threads (int): Tool thread count.
        timeout_seconds (float): Kill the tool after this long. The
            smoke test has a natural bound — minutes at 49B scale —
            so a wedged tool must not hang the pack forever.

    Examples:
        The composition root wires the paths:

        ```python
        tester: SmokeTester = LlamaCppSmokeTester(perplexity_bin, model_path, text_path)
        ```
    """

    perplexity_bin: Path
    model_path: Path
    text_path: Path
    chunks: int = 2
    threads: int = 8
    timeout_seconds: float = 3600.0

    def smoke(self) -> float:
        """Run the smoke chunks and report the final perplexity.

        Returns:
            The tool's final perplexity estimate, non-finite values
            included — the ceiling rejects them downstream.

        Raises:
            PackError: If the tool cannot start, exits nonzero, dies
                to a signal, exceeds the timeout, or reports no final
                estimate. The message carries the tool's last output
                lines.
        """
        command = [
            str(self.perplexity_bin),
            "-m",
            str(self.model_path),
            "-f",
            str(self.text_path),
            "--chunks",
            str(self.chunks),
            "-t",
            str(self.threads),
            # Layer offload off: the smoke test must not contend for
            # the GPU, whatever backend the binary was built with
            # (ADR-0017).
            "-ngl",
            "0",
        ]
        output = run_tool(command, stage="smoke", timeout_seconds=self.timeout_seconds)
        match = _FINAL_ESTIMATE.search(output)
        if match is None:
            raise PackError(
                "smoke exited 0 without a final perplexity estimate:\n"
                f"{tail_of(output)}"
            )
        try:
            return float(match.group(1))
        except ValueError as exc:
            raise PackError(
                f'smoke reported an unreadable estimate "{match.group(1)}"'
            ) from exc
