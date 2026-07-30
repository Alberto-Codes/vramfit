# ADR-0017: A packed model proves it emits language before anything trusts it

- **Status:** Proposed
- **Date:** 2026-07-29

## Context

On 2026-07-29 a 3-bit-heavy 49B recipe passed every gate the
pipeline had — solver, validation pass, pack size re-check — and
produced a destroyed artifact: PPL ~10⁶ on two backends, every
payload finite
([the fourth data point](../explanation/evaluating-packed-models.md)).
The scan-frame damage model predicted 1.44 for it. Nothing between
`plan` and the evaluation tier catches this class. ADR-0012's second
amendment records the requirement and leaves the placement open:
inside `quantfit pack` behind a flag, or first step of the
evaluation tier. Issue #36 tracks it.

Two facts size the gate:

- Destroyed and working artifacts sit 5 orders of magnitude apart.
  Working 49B artifacts measure PPL 8–10. The destroyed one
  measured ~10⁶. The gate needs no precision.
- A few perplexity chunks through the CPU build cost minutes at 49B
  scale — noise against a ~19-minute quantize.

## Decision

1. **The smoke test lives inside `quantfit pack`.** The pack step is
   the last stage that owns the artifact. Placed in the evaluation
   tier, `pack` could still exit 0 on a destroyed file, and every
   downstream consumer would have to know that. The gate belongs to
   the producer.
2. **`--smoke-text <file>` enables it.** After the size re-check,
   pack runs `llama-perplexity` from the same `--llama-cpp` checkout
   over the packed file: `--smoke-chunks` chunks (default 2) of the
   given text. The CPU binary is deliberate — the smoke test must
   not contend for the GPU.
3. **The verdict is a perplexity ceiling.** `--smoke-threshold`
   defaults to 1000. A finite result below the ceiling passes.
   Anything else — non-finite, unparsable, or above — fails: pack
   exits 1 and keeps the file, mirroring the size re-check.
4. **Without `--smoke-text`, pack warns once.** The line names the
   flag. The artifact is then packed but unproven, and the human
   channel says so.
5. **A new `SmokeTester` port carries the measurement.** The
   llama.cpp adapter drives `llama-perplexity` by subprocess and
   parses the final estimate. A verified-fake contract suite covers
   both sides (ADR-0009). The run log gains one event,
   `smoke_tested` (perplexity, threshold, chunks, passed), and the
   halt stage `smoke`.

## Consequences

- The destroyed-artifact class dies at the producer: a file like the
  2026-07-29 diagnostic cannot exit `pack` with code 0 when the
  smoke test runs.
- Pack's toolchain requirement grows by one binary:
  `llama-perplexity` beside `llama-quantize`. The how-to gains it in
  the build target list.
- The smoke test proves "emits language", nothing more. A ceiling of
  1000 accepts artifacts far too damaged to publish — quality
  judgment stays with the evaluation tiers.
- One more subprocess seam to keep hermetic in tests: the contract
  suite stubs the perplexity tool the same way it stubs the
  quantizer.

## Open questions

- Whether the smoke test becomes a required input once every caller
  passes text anyway.
- Whether the threshold should scale with the recipe's predicted
  damage instead of a constant.
