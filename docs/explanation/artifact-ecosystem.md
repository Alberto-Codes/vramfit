---
status: sketch
---

# The artifact ecosystem: how this work could outlive the tool

> **Status: sketch** — strategy thinking recorded 2026-07-28, before the
> 49B benchmark ran. Nothing here is committed work. The
> triggers that would activate each phase are tracked in issue #11.

## The honest competitive picture

Non-uniform per-layer quantization is not new, and a 2026-07-28
re-survey shows the field actively converging on measure-then-mix.
EXL2 measured per-layer bitrates years ago. Unsloth's Dynamic 2.0
GGUFs measure per-layer sensitivity and assign every layer its own
type, shipped at scale with leading (self-reported) KL benchmarks.
NVIDIA's Model
Optimizer searches per-layer formats under an effective-bits
constraint. llama.cpp itself now carries an
[auto-adaptive mixed-precision effort](https://github.com/ggml-org/llama.cpp/discussions/18531)
— per-tensor error measurement plus a Lagrangian solver against a
target size, inside the runtime we pack for. Convergence this broad
validates the approach and closes the "only measure-then-solve tool"
window for good.

What none of them publish is the measurement itself — every one
consumes a sensitivity proxy internally and discards it, and none
carries a portable recipe with provenance. quantfit's differentiated assets
(fuller argument in
[why selective quantization](why-selective-quantization.md)):

1. **Telemetry** — the sensitivity map as a standalone, versioned
   artifact: end-to-end damage curves as first-class, inspectable
   data, run logs beside them.
2. **Budget-first solving** — an explicit VRAM + KV budget derived
   from the user's intended serving shape, not an average-bits or
   file-size target (issue #29 sketches the budget command that
   deepens this).
3. **Portability** — recipes as runtime-agnostic artifacts with
   provenance and trace, retargetable through the capability table.
4. **The falsifiable benchmark** — measured against named baselines,
   negative result publishable
   ([ADR-0003](../adr/0003-north-star-benchmark.md),
   [ADR-0010](../adr/0010-sub-4-bit-serving-path.md)), with the
   validation pass checking our own additivity assumption on the way.

The core experiment can still fail: heuristic recipes may prove
near-optimal, and the additivity assumption may leak badly. If so, the
published measurement infrastructure and curves remain the durable
contribution.

## The thesis: make the artifact the standard, not the tool

Tools get replaced. Formats persist. The long-term win condition is
other tools *consuming* sensitivity maps — even tools that never run
our scanner. Concrete implications:

- Treat the map schema as a public spec (it is already versioned via
  `quantfit_schema`).
- Ship a converter from any sensitivity map to `llama-quantize
  --tensor-type` flags, so llama.cpp users benefit without adopting
  quantfit.
- Publish maps for models we did not pack. The map is the product.

## Phased socialization plan

Each phase is cheap and gated on the one before it. Do not reorder.

1. **The result** — the 49B benchmark writeup with artifacts attached.
   Nothing else starts until this exists.
2. **Ride existing rails** — sensitivity maps as Hugging Face datasets,
   packed models as ordinary model repos whose cards embed the damage
   curves and recipe. A `quantfit-maps` git registry where submission is
   a PR and CI validates schema, fingerprint, and provenance.
3. **The browser demo** — `quantfit plan` is torch-free pure Python, so
   a Hugging Face Space can re-solve recipes live against published
   maps (VRAM slider, watch assignments move).
4. **Verification as ritual** — the per-cell checkpoint design lets
   anyone re-measure a random subset of a published map's cells. A
   "verified" badge means independently spot-checked, at a fraction of
   a full scan's cost.
5. **A dedicated site, only if traction demands it** — the honest
   leaderboard is "recipes for the same model at the same budget,
   ranked by measured quality of the packed result". That needs eval
   compute and referee credibility. Do not build it speculatively.

## Power user #0: publish measured packed models, small models first

Recorded 2026-07-28, the day the first completable scan (Qwen2.5-3B)
ran. Nobody adopts an artifact standard from a spec — they adopt it
because artifacts they already want carry it. So the most credible
socialization move is to be the ecosystem's first power user: publish
packed models to Hugging Face that people download for their own sake,
with the sensitivity map, recipe, and run log riding along beside the
weights.

What makes the model card unwritable by anyone else: the damage
curves (the 2026-07-28 Qwen2.5-3B scan measured layer 1 as ~490× more
fragile at 3-bit than layer 0 — damage 6.28 vs 0.013 — so layer 1
keeps 4 bits), the explicit budget arithmetic, the solver trace, and a
one-command reproduction line. The closing move on every card:
"Disagree? Re-run the scan — the map is right there." That is the
invitation that turns downloaders into publishers.

This amends the phase ordering in one way: small models do not wait
for the 49B result. The Qwen2.5-3B scan runs at ~23 s per cell on the
reference box (148 cells ≈ one hour), so a measured Qwen-class packed
model can be publication number one while the 49B pipeline finishes —
offload-aware scanning landed
([ADR-0015](../adr/0015-offload-aware-scanning.md)) and the first 49B
scan runs at ~42 s per cell at 8,192 tokens. The 49B writeup remains the gate for
everything *else* in the phase list.

Hard gates before any publication:

1. `quantfit pack` exists (the GGUF backend of
   [ADR-0010](../adr/0010-sub-4-bit-serving-path.md)).
2. The whole-recipe validation pass exists — publishing a recipe whose
   additivity assumption was never checked is the exact sin the
   project criticizes.
3. The packed model measurably beats the size-matched heuristic GGUF,
   judged per [evaluating packed models](evaluating-packed-models.md).
   One bad debut kills a "measured beats folklore" brand permanently.
   If it loses, publish the negative result in the writeup instead of
   the model.

Conventions to settle at publication time: a `quantfit` HF tag, the
budget in the repo name (e.g. `-fit24gib`), and which artifacts sit
beside the weights in the repo.

## What a score may never claim

Damage values are calibration-relative. A cross-model or
cross-calibration damage leaderboard would be statistically dishonest —
the [glossary](../reference/glossary.md) already says maps are only
comparable within one calibration set. Rankings must compare packed-model quality at a fixed (model,
budget) pair, never raw damage across scans.

## Ideas parked for later

- **Distributed scanning**: the fingerprint plus cell grid makes
  sharding one scan across community GPUs plausible. Turns "scan the
  top 20 models" into a participatory event.
- **Content evidence in the fingerprint** (already in issue #8) becomes
  load-bearing the moment third parties submit maps.

## The window

Three trends make the work valuable now: open models improve fast,
consumer VRAM is nearly frozen, and top open models keep growing. The
squeeze widens every quarter. The countervailing risk: cheap
high-VRAM hardware (unified memory, a future affordable 48 GiB card)
relieves the pressure. Plan for a multi-year window, not a permanent
one.
