"""Mean KL divergence between reference and perturbed logits.

The scan metric fixed by ADR-0006: divergence of next-token
distributions at the final logits, averaged over calibration tokens.
KL is computed in float32 regardless of model dtype.

Examples:
    Identical logits diverge by zero:

    ```python
    import torch

    logits = torch.randn(1, 8, 32)
    assert mean_kl(logits, logits) == 0.0
    ```

See Also:
    - [quantfit.adapters.outbound.scan.meter][]: Accumulates this over
      calibration batches.
"""

from __future__ import annotations

import torch


def reference_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """Compute the reference distribution to cache between measurements.

    Args:
        logits: Reference logits, shape ``[batch, tokens, vocab]``.

    Returns:
        Log-probabilities in float16, on the CPU. Roughly
        ``2 x tokens x vocab`` bytes per batch — ~0.25 GiB per 1024
        tokens at a 128k vocabulary.
    """
    return torch.log_softmax(logits.to(torch.float32), dim=-1).to(torch.float16).cpu()


def mean_kl(reference: torch.Tensor, perturbed_logits: torch.Tensor) -> float:
    """Compute mean per-token KL(reference ‖ perturbed).

    Args:
        reference: Cached reference log-probabilities, as produced by
            `reference_log_probs`.
        perturbed_logits: The perturbed model's logits for the same
            batch.

    Returns:
        The mean KL divergence over all tokens, clamped at zero —
        float16 caching can produce a negligible negative residue on
        near-identical distributions.

    Examples:
        Damage is positive once logits differ:

        ```python
        import torch

        ref = reference_log_probs(torch.randn(1, 8, 32))
        assert mean_kl(ref, torch.randn(1, 8, 32)) > 0.0
        ```
    """
    ref = reference.to(torch.float32)
    pert = torch.log_softmax(perturbed_logits.to(torch.float32).to(ref.device), dim=-1)
    kl = (ref.exp() * (ref - pert)).sum(dim=-1).mean().item()
    return max(kl, 0.0)
