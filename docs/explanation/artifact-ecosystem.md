---
status: draft
---

# The artifact ecosystem: how this work could outlive the tool

> **Status: draft** — strategy thinking recorded 2026-07-28, before the
> 49B benchmark ran. The benchmark ran 2026-07-29 and **lost** — see
> [the fourth data point](evaluating-packed-models.md) and the gate
> notes below.
> Promoted to `draft` 2026-08-12: the publication gates on this page
> ran for real. Gate 3 ruled GO on #80, and publication #1 shipped
> through the procedure. The phases beyond publication remain
> uncommitted, and their triggers stay tracked in issue #11.
>
> Note 2026-08-22: gate 3 ruled GO a second time, for publication #2
> — the 30B falsifier arm (chart #158). Publication #2 shipped the
> same day, through the procedure. The identity followed the #401
> identity grammar, every uploaded file matched its ledger hash,
> and both cards published byte-verbatim. The
> [#404 closing comment](https://github.com/Alberto-Codes/vramfit/issues/404#issuecomment-5382801425)
> carries the ship record, and v0.3.0 released after it.
>
> Note 2026-09-04: gate 3 ran unnamed for publication #3, and the
> #500 record sits in the gate notes below.

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

What none of them publish is the measurement *as evidence*.
exllamav3 persists per-group KL to a reusable file (the strongest
exception — see [prior art](../reference/prior-art.md)), but it is
runtime-locked and carries no provenance; the rest consume a
sensitivity proxy internally and discard it. None carries a
portable recipe with recorded provenance. vramfit's differentiated assets
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

The core experiment ran, and the competitive question closed. The additivity worry has
been measured seven times and points both ways: six passes
sub-additive (2.05×, 2.94×, 1.6×, 2.0×, 1.87×, 4.87×), one
super-additive by 11.9× on a 2-bit-heavy recipe — caught before
packing, which is the pass doing its job. The competitive worry
resolved on 2026-08-09: after five losing head-to-heads mapped the
gap (imatrix, 2-bit membership, frame transfer, granularity), the
fifteenth data point's end-to-end pack beat the size-matched
heuristic-plus-imatrix quant on full-window KL divergence at 7.8σ.
The measurement infrastructure and curves remain a durable
contribution beside the win.

## The thesis: make the artifact the standard, not the tool

Tools get replaced. Formats persist. The long-term win condition is
other tools *consuming* sensitivity maps — even tools that never run
our scanner. Concrete implications:

- Treat the map schema as a public spec (it is already versioned via
  `vramfit_schema`).
- Ship a converter from any sensitivity map to `llama-quantize
  --tensor-type` flags, so llama.cpp users benefit without adopting
  vramfit.
- Publish maps for models we did not pack. The map is the product.

## Phased socialization plan

Each phase is cheap and gated on the one before it. Do not reorder.

1. **The result** — the 49B benchmark writeup with artifacts attached.
   Nothing else starts until this exists.
2. **Ride existing rails** — sensitivity maps as Hugging Face datasets,
   packed models as ordinary model repos whose cards embed the damage
   curves and recipe. A `vramfit-maps` git registry where submission is
   a PR and CI validates schema, fingerprint, and provenance.
3. **The browser demo** — `vramfit plan` is torch-free pure Python, so
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
for the 49B result. The Qwen2.5-3B scan ran at ~23 s per cell on the
reference box (148 cells ≈ one hour), and the first 49B scan ran at
~41 s per cell at 8,192 tokens
([ADR-0015](../adr/0015-offload-aware-scanning.md)). The 49B writeup
remains the gate for everything *else* in the phase list.

Hard gates before any publication, with their current status:

1. `vramfit pack` exists (the GGUF backend of
   [ADR-0010](../adr/0010-sub-4-bit-serving-path.md)). **Satisfied** —
   it packed the 49B, 169.7 MiB under budget, first try.
2. The whole-recipe validation pass exists — publishing a recipe whose
   additivity assumption was never checked is the exact sin the
   project criticizes. **Satisfied** — `vramfit validate` has seven
   measurements across both directions, including the
   super-additive one that stopped a bad recipe before packing.
3. The packed model measurably beats the size-matched heuristic GGUF,
   judged per [evaluating packed models](evaluating-packed-models.md).
   One bad debut kills a "measured beats folklore" brand permanently.
   If it loses, publish the negative result in the writeup instead of
   the model. **Satisfied on 2026-08-09**: after five published
   losses mapped the gap, the fifteenth data point's pipeline pack
   beats the imatrix Q3_K_S on full-window KL divergence at 7.8σ
   with the best nominal perplexity in the lane. The remaining
   publication step is the task-eval tier. **GO ruled 2026-08-10**
   (#80): the tier-3 slice certified five statistical ties
   (sixteenth data point), the negative-result branch has no
   trigger, and publication #1 ships the model. The i-quant
   comparison (#90) rides the card to answer the weak-baseline
   objection.

**GO ruled 2026-08-22 for publication #2**, the 30B falsifier arm on
chart #158. Gate 3 ran a second time. Nothing smaller than the arm
is published, so no size-matched build exists, and #393 ruled
bartowski's `IQ2_XXS`, the smallest published GGUF of the target,
the bar — with the tier-3 slice run before publication to gate
go/no-go, as #80 gated publication #1. (Correction 2026-08-31,
#415: the three size readings in that sentence were false at
ruling time. A Hub-wide query on 2026-08-22 found eight other
publishers' full-model builds below the arm's 15.76 GiB. Two sit
within 0.24 GiB of it, so the no-size-matched-build clause fails
too. The selection had read five publishers' repositories, not
the Hub. #416 measures those builds. The measured win over
`IQ2_XXS` stands.) Tier 2 already ranks the arm
ahead of that build on both ruled metrics (nineteenth data point).
The tier-3 slice ran on #400 and certifies the rank: per ADR-0024
decision 4, the arm leads on four of five tasks (MMLU 16.1σ,
HellaSwag 6.7σ, GSM8K 1.3σ, Winogrande 1.04σ) and ties ARC-Challenge
(0.4σ), on full splits with zero truncations. The arm's pack,
rebuilt on the reference box, measures 16,922,476,480 B against the
published build's 18,838,022,112 B, which is 1.78 GiB smaller
(packed byte size moves across machines with the stored imatrix
path, #300). #400 records the run and the scores. The
negative-result branch has no trigger, and publication #2 ships the
model. Per #164, no tokens-per-second figure from a capped 4090
publishes. #279 carries the 16 GiB fit claim's open counter, CPU
offload.

**Publication #3 shipped 2026-08-29 with gate 3 run unnamed**, the
Gemma 4 31B fit24gib pack on chart #441. The tier-3 slice ran on #423
before publication: four ties and MMLU +1.15 against the vendor's
QAT `Q4_0` comparator. The
[#446 ruling](https://github.com/Alberto-Codes/vramfit/issues/446#issuecomment-5465455247)
of 2026-08-29 said "publish as packaged" and named no gate. No record
names a gate-3 ruling for this publication, and this page did not
carry a note until 2026-09-04. The maintainer ruled on that day that
the #446 ruling beside the #423 results is the closest record, and
that this note closes
[#500](https://github.com/Alberto-Codes/vramfit/issues/500). The
sidecar swap of 2026-08-31 changed the published projector and not
the gate.

The naming conventions are settled. #79 ruled the `vramfit` HF
tag, the budget in the repo name, and publication #1's artifact
set
([the Hugging Face conventions](evaluating-packed-models.md#the-hugging-face-conventions)).
#401 ruled the reusable identity grammar from publication #2
forward on 2026-08-22
([the identity grammar](evaluating-packed-models.md#the-identity-grammar-from-publication-2)).
The maintainer confirmed publication #2's artifact set the same
day: the #79 split carries over unchanged
([#404 confirmation](https://github.com/Alberto-Codes/vramfit/issues/404#issuecomment-5382579749)).

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
