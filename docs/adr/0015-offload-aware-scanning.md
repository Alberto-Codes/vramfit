# ADR-0015: The meter perturbs offloaded groups through accelerate's weights map

- **Status:** Accepted
- **Date:** 2026-07-28 (accepted 2026-07-28)

## Context

The north-star 49B carries ~93 GB of bf16 weights and does not fit a
24 GiB card. Under a `--gpu-memory` cap, `auto` sharding offloads
overflow modules to host RAM. transformers then exposes those
parameters as meta tensors. An in-place write to a meta tensor
silently no-ops, so a perturbation there measures zero damage. Since
PR #13 the meter refuses such models at construction. The refusal
blocks the first real 49B scan (#8): at a 17 GiB cap, 72 of 82 groups
land off the GPU.

Issue #16 held two candidate designs:

1. **Weights-map perturbation.** Quantize the CPU-resident tensors
   that accelerate's forward hooks load from.
2. **Group streaming.** Keep the model on host RAM and move one
   decoder layer to the GPU per forward slice.

A probe on the real 49B fixed the facts (2026-07-28, transformers
5.14.1, accelerate 1.14.0, 17 GiB cap, RTX 4090):

- Offloaded weights live in `AlignDevicesHook.weights_map`, backed by
  `OffloadedWeightsLoader.state_dict` — real CPU bf16 tensors, no
  disk spill (`save_folder=None`).
- Repeated map access returns the same storage. In-place mutation is
  possible and survives across forward passes.
- The forward pass reads the mutated values: zeroing one tensor moved
  final logits by 7.4 (max absolute difference). Copying the saved
  original back restored the logits bit-exactly.
- One 2048-token forward takes ~9 s. A 32,768-token measurement is
  16 forwards, ~145 s per cell.

The validation pass adds a second constraint. `measure_recipe` stages
every original on host RAM today. At 49B scale that is ~93 GB of
clones next to ~76 GB of offloaded weights — over the reference box's
124 GB. The offloaded tensors load from the safetensors shards without
dtype conversion, so the shards on disk already hold their originals.

## Decision

1. **The meter perturbs offloaded groups through the weights map.**
   At construction it resolves every meta parameter to its backing
   CPU tensor in the owning module's `AlignDevicesHook.weights_map`.
   Perturb and restore write those tensors in place. The forward
   hooks stream the current values to the GPU each pass, unchanged.
2. **The meter verifies behavior, not versions.** For each resolved
   tensor it checks: the hook exists, the tensor is a real CPU tensor
   of the parameter's shape, and two map reads return one storage.
   A parameter that fails verification keeps the honest refusal from
   PR #13 — the meter degrades to "cannot measure", never to zero
   damage. Disk-offloaded weights stay refused.
3. **`measure_recipe` restores offloaded originals from the model's
   safetensors shards.** GPU-resident originals keep the existing CPU
   staging (~17 GB at 49B scale). Offloaded originals are not staged
   at all — the restore reads them back from the shard files. Before
   perturbing anything, the meter verifies every offloaded tensor
   resolves to a shard entry with the recorded shape. Without a local
   safetensors checkpoint directory, `measure_recipe` refuses
   offloaded groups with a clear error. `measure` is untouched: one
   group's originals clone into RAM as before.
4. **Group streaming stays a throughput optimization.** It is tracked
   in #8 and does not gate correctness. This ADR changes which
   devices the meter accepts, not the scan loop.

Acceptance evidence (2026-07-28, the first 49B scan): at a 15 GiB
cap, 73 of 82 groups offloaded, and `model.layers.9` — previously an
unmeasurable meta tensor — measured 0.00028 damage at 8 bits over
8,192 calibration tokens, at ~37 s per cell. Two operational facts
from the same runs: a 17 GiB cap leaves too little contiguous
workspace for the 2.1 GB embedding copy-back, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` corrupts memory
near out-of-memory pressure on torch 2.13 — the scan measured NaN
where the default allocator measures 0.0045, twice, bit-identically.
Launch 49B scans at a 15 GiB cap with the default allocator.

## Consequences

- The first real 49B scan ran the night this ADR landed: 328 cells at
  8,192 tokens, 3 h 42 m, mean 40.7 s per cell (acceptance note
  below). A 32,768-token convergence re-scan measures ~155 s per
  cell, close to the probe's ~145 s estimate.
- The meter couples to two accelerate names: `_hf_hook` on offloaded
  modules and `weights_map` on the hook. The behavior checks in
  decision 2 turn an accelerate layout change into a construction
  refusal, not a corrupt map. No version pin is added.
- Damage numbers for offloaded groups are exact, not approximate:
  the perturbed CPU tensor is byte-identical to what the hook ships
  to the GPU each forward.
- `quantfit validate` at 49B scale requires the model as a local
  safetensors directory. Hub-cached models with offloaded groups can
  scan but not validate. The refusal message names the fix.
- The scan's host-RAM floor at 49B scale is the offloaded weights
  plus the reference distributions — ~76 GB plus ~8 GB at 32,768
  tokens. The reference box's 124 GB holds it.
