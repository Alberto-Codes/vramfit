# Nemotron Super 49B v1.5 config

`config.json` is the unmodified model config of
[nvidia/Llama-3_3-Nemotron-Super-49B-v1_5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5),
fetched 2026-07-28. It is the north-star target's real geometry
([ADR-0003](../../../docs/adr/0003-north-star-benchmark.md)): 80 blocks,
49 with attention, 64 heads, hidden size 8192.

The worked-example integration test
(`tests/integration/test_worked_example.py`) checks the stable numbers
in [VRAM budget math](../../../docs/explanation/vram-budget.md) against
this file. Re-fetch only if NVIDIA revises the checkpoint, and update
the docs in the same change.
