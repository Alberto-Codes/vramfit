"""llama.cpp divergence adapter: per-chunk KLD against a stored base.

Implements the `RuntimeDivergenceMeter` port for the GGUF serving
path (ADR-0031 decision 2). One `measure` call runs
``llama-perplexity`` with ``--kl-divergence`` over a packed model,
against logits the caller already stored from the reference build,
and returns one divergence per chunk.

The tool prints a running mean after each chunk, not the chunk's own
value. The adapter recovers the per-chunk series with
`vramfit.domain.paired.per_chunk`, because a paired test needs the
individual values and cannot run on an aggregate.

Layer offload stays on here, unlike the smoke test. A full-window
divergence pass over hundreds of chunks is the refinement pass's
whole cost, and running it on CPU would take the stage out of reach.

Examples:
    Measure one packed arm:

    ```python
    meter = LlamaCppDivergenceMeter(
        perplexity_bin=Path("llama.cpp/build/bin/llama-perplexity"),
        text_path=Path("wiki.test.raw"),
        base_logits=Path("base.logits"),
    )
    divergences = meter.measure("arm11.gguf")
    ```

See Also:
    - [vramfit.ports.outbound][]: `RuntimeDivergenceMeter`, which
      this satisfies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vramfit.adapters.outbound.gguf.toolrun import run_tool, tail_of
from vramfit.adapters.outbound.gguf.types import PackError
from vramfit.domain.paired import per_chunk

# One per-chunk row of the tool's divergence table. The columns are
# chunk, PPL, ln(PPL(Q)/PPL(base)), KL divergence, dp RMS, same top p,
# each value followed by its own standard error. The third value pair
# is the running mean divergence this adapter reads.
_CHUNK_ROW: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d+)\s+[\d.]+\s+±\s+[\d.]+"
    r"\s+-?[\d.]+\s+±\s+[\d.]+"
    r"\s+(-?[\d.]+)\s+±",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class LlamaCppDivergenceMeter:
    """`RuntimeDivergenceMeter` adapter driving ``llama-perplexity``.

    Attributes:
        perplexity_bin (Path): ``llama-perplexity`` path.
        text_path (Path): Evaluation text the chunks run over.
        base_logits (Path): Logits stored from the reference build,
            which every arm measures against. The same file for
            every arm in one pass, or the arms do not compare.
        context (int): Context length per chunk.
        batch (int): Batch size.
        gpu_layers (int): Layers offloaded to the card.
        threads (int): Tool thread count.
        timeout_seconds (float): Kill the tool after this long.

    Examples:
        The composition root wires the paths:

        ```python
        meter: RuntimeDivergenceMeter = LlamaCppDivergenceMeter(
            perplexity_bin, text_path, base_logits
        )
        ```
    """

    perplexity_bin: Path
    text_path: Path
    base_logits: Path
    context: int = 512
    batch: int = 512
    gpu_layers: int = 99
    threads: int = 8
    timeout_seconds: float = 7200.0

    def measure(self, packed: str) -> tuple[float, ...]:
        """Measure one packed model against the stored base logits.

        Args:
            packed: Path of the packed model to evaluate.

        Returns:
            One divergence per chunk, in chunk order.

        Raises:
            PackError: If the tool cannot start, exits nonzero, dies
                to a signal, exceeds the timeout, reports no chunk
                rows, or skips a chunk. The message carries the
                tool's last output lines.
        """
        command = [
            str(self.perplexity_bin),
            "-m",
            packed,
            "-f",
            str(self.text_path),
            "--kl-divergence-base",
            str(self.base_logits),
            "--kl-divergence",
            "-ngl",
            str(self.gpu_layers),
            "-c",
            str(self.context),
            "-b",
            str(self.batch),
            "-t",
            str(self.threads),
        ]
        output = run_tool(
            command, stage="divergence", timeout_seconds=self.timeout_seconds
        )
        return _running_means(output)


def _running_means(output: str) -> tuple[float, ...]:
    """Recover the per-chunk divergences from the tool's output.

    Args:
        output: Everything the tool printed.

    Returns:
        One divergence per chunk, in chunk order.

    Raises:
        PackError: If the output holds no chunk rows, or the chunk
            numbers are not 1..n without a gap. A gap would silently
            shift every later chunk against the other arms.
    """
    rows = [(int(m.group(1)), float(m.group(2))) for m in _CHUNK_ROW.finditer(output)]
    if not rows:
        raise PackError(
            f"divergence exited 0 without per-chunk rows:\n{tail_of(output)}"
        )
    numbers = [n for n, _ in rows]
    if numbers != list(range(1, len(rows) + 1)):
        raise PackError(
            f"divergence reported {len(rows)} chunk rows numbered "
            f"{numbers[0]}..{numbers[-1]}, which skips a chunk:\n"
            f"{tail_of(output)}"
        )
    return per_chunk([mean for _, mean in rows])
