"""The pure core: artifact types, budget math, and the recipe solver.

Domain modules hold no IO — no file access, no JSON, no CLI framework.
They take plain data in and return plain data out, which keeps the whole
layer unit-testable without touching a GPU, a file, or a network. The
import-linter contracts enforce this mechanically (see ADR-0008).

Attributes:
    model: Artifact dataclasses (`SensitivityMap`, `Recipe`, and parts).
    solver: The greedy damage-per-byte recipe solver.
    budget: Size parsing, KV-cache math, and the weight-budget ledger.

Examples:
    Solve a recipe entirely in memory:

    ```python
    from vramfit.domain.solver import solve

    recipe = solve(
        map_,
        weight_budget_bytes=20 * 2**30,
        vram_budget_bytes=24 * 2**30,
        kv_headroom_bytes=4 * 2**30,
    )
    ```

See Also:
    - [vramfit.adapters][]: IO implementations that feed this layer.
"""

from __future__ import annotations
