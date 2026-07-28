"""Calibration text loading and batching for the scan.

Calibration arrives as a plain UTF-8 text file — no dataset framework
in the dependency tree (ADR-0005). The choice of text is an open
question recorded in ``docs/how-to/scan-a-model.md``. The file path is
recorded in the map's ``scan.calibration`` provenance.

Examples:
    Tokenize a workload sample into scan batches:

    ```python
    batches, n_tokens = load_calibration(path, tokenizer, max_tokens=4096)
    ```

See Also:
    - [quantfit.adapters.outbound.scan.meter][]: Runs these batches per
      measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

DEFAULT_SEQUENCE_LENGTH = 2048


def load_calibration(
    path: Path,
    tokenizer: Any,
    max_tokens: int,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[list[torch.Tensor], int]:
    """Read, tokenize, and batch a calibration text file.

    The text is tokenized once, truncated to ``max_tokens``, and split
    into ``[1, sequence_length]`` batches. A final partial batch is
    kept when it holds at least two tokens (one next-token
    distribution).

    Args:
        path: UTF-8 text file to read.
        tokenizer: The model's tokenizer.
        max_tokens: Upper bound on calibration tokens.
        sequence_length: Tokens per forward pass.

    Returns:
        The batches and the total token count they cover.

    Raises:
        ValueError: If ``max_tokens`` is not positive,
            ``sequence_length`` is below 2, or the file yields fewer
            than two tokens.
        OSError: If the file cannot be read.

    Examples:
        Token counts never exceed the request:

        ```python
        batches, n_tokens = load_calibration(path, tokenizer, max_tokens=64)
        assert n_tokens <= 64
        ```
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if sequence_length < 2:  # noqa: PLR2004 - one next-token prediction needs two
        raise ValueError("sequence_length must be at least 2")
    text = path.read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    ids = ids[:max_tokens]
    if ids.numel() < 2:  # noqa: PLR2004 - one next-token prediction needs two tokens
        raise ValueError(
            f"{path}: calibration text yields {ids.numel()} tokens — need at least 2"
        )
    batches = [
        chunk.unsqueeze(0)
        for chunk in ids.split(sequence_length)
        if chunk.numel() >= 2  # noqa: PLR2004 - drop a trailing single token
    ]
    return batches, sum(b.numel() for b in batches)
