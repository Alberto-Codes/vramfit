"""llama.cpp smoke-test adapter: a few perplexity chunks by subprocess.

Implements the `SmokeTester` port for the GGUF serving path
(ADR-0017). One `smoke` call runs ``llama-perplexity`` over the
packed model for a handful of chunks and parses the tool's final
estimate. The measurement comes back verbatim — NaN included — and
the caller judges it against the ceiling
(`quantfit.domain.pack.smoke_passed`). A tool that cannot start,
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
    - [quantfit.ports.outbound][]: `SmokeTester`, which this
      satisfies.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from quantfit.adapters.outbound.gguf.types import PackError

_TAIL_LINES: Final[int] = 15

_FINAL_ESTIMATE: Final[re.Pattern[str]] = re.compile(r"Final estimate: PPL = (\S+)")


@dataclass(frozen=True, slots=True)
class LlamaCppSmokeTester:
    """`SmokeTester` adapter driving ``llama-perplexity``.

    Attributes:
        perplexity_bin (Path): ``llama-perplexity`` path. The CPU
            build on purpose — the smoke test must not contend for
            the GPU (ADR-0017).
        model_path (Path): The packed model to prove.
        text_path (Path): Text the chunks run over.
        chunks (int): Chunk count. Destroyed and working artifacts
            sit 5 orders of magnitude apart, so 2 chunks decide.
        threads (int): Tool thread count.

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

    def smoke(self) -> float:
        """Run the smoke chunks and report the final perplexity.

        Returns:
            The tool's final perplexity estimate, non-finite values
            included — the ceiling rejects them downstream.

        Raises:
            PackError: If the tool cannot start, exits nonzero, dies
                to a signal, or reports no final estimate. The
                message carries the tool's last output lines.
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
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - fixed tool path from the composition root, no shell
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise PackError(f"smoke: cannot run {command[0]}: {exc}") from exc
        output = completed.stdout or ""
        tail = "\n".join(output.splitlines()[-_TAIL_LINES:])
        if completed.returncode != 0:
            raise PackError(
                f"smoke failed with exit code {completed.returncode}:\n"
                f"{tail or '(no output captured)'}"
            )
        match = _FINAL_ESTIMATE.search(output)
        if match is None:
            raise PackError(
                "smoke exited 0 without a final perplexity estimate:\n"
                f"{tail or '(no output captured)'}"
            )
        try:
            return float(match.group(1))
        except ValueError as exc:
            raise PackError(
                f'smoke reported an unreadable estimate "{match.group(1)}"'
            ) from exc
