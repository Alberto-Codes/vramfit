# ADR-0014: The solver predicts sizes from per-type effective bits

- **Status:** Accepted
- **Date:** 2026-07-28 (accepted 2026-07-28)

## Context

The solver predicted every group's size from nominal bits plus one
scalar: `bytes_fp16 × bits / 16 × (1 + format_overhead)`. ADR-0012
and ADR-0013 both carried the resulting open question — whether the
solver should consume per-type effective-bit tables instead of one
fraction.

The evidence arrived with the 6/5/4 mix
([evaluating packed models](../explanation/evaluating-packed-models.md)).
The scalar that fit the {8, 4} recipe with 17 MiB to spare overflowed
the next recipe by 4.5 MiB. The mechanism: GGUF types drift from
nominal by different amounts (+6.25 % for `Q8_0` up to +31.25 % for
`Q2_K`), so one scalar has to match the mix-weighted drift of
whatever types the solver picks. It cannot track a moving mix.

Tensor-level inspection of the packed artifacts (2026-07-28, both
Qwen2.5-3B packs) shows the real cost has two parts:

1. **Per-type effective bits are exact constants.** Every quantized
   tensor spends precisely its K-quant block layout: `Q8_0` 8.5,
   `Q6_K` 6.5625, `Q5_K` 5.5, `Q4_K` 4.5 bits/weight — byte-for-byte
   across 2.1 GiB files. `Q3_K` (3.4375) and `Q2_K` (2.625) follow
   the same block math.
2. **A small residual is not per-type.** The packed file adds
   5,956,096 bytes of GGUF metadata and 966,656 bytes of F32 norm
   tensors — 6,922,752 bytes total, identical in both packs, 0.32 %
   of the weight bytes. Scan groups exclude norms (only 2-D floating
   tensors join groups), so no per-group prediction can see this
   part.

## Decision

1. **The domain records effective bits per runtime.**
   `vramfit.domain.runtime.EFFECTIVE_BITS` maps a runtime name to a
   nominal-bits → effective-bits table:

   | Nominal bits | llama.cpp type | Effective bits/weight |
   |--------------|----------------|----------------------|
   | 8 | `Q8_0` | 8.5 |
   | 6 | `Q6_K` | 6.5625 |
   | 5 | `Q5_K` | 5.5 |
   | 4 | `Q4_K` | 4.5 |
   | 3 | `Q3_K` | 3.4375 |
   | 2 | `Q2_K` | 2.625 |

   The values are exact block-layout constants, verified against
   packed files. Only runtimes with a measured pack path get a
   table — vLLM has none until a vLLM pack backend exists. A table
   must cover its runtime's full capability set (ADR-0013), and a
   test enforces the invariant. This amends the ADR-0013 note that
   effective bits stay a pack concern: the pack tables (ADR-0012)
   keep the type names, the domain now keeps the byte costs.
2. **The solver prices candidates at effective bits.** When the
   target runtime has a table, a group's predicted size is
   `ceil(bytes_fp16 × effective_bits / 16 × (1 + format_overhead))`.
   Without a table (no runtime, or vLLM), the nominal-bits formula
   stands unchanged.
3. **`format_overhead` shrinks to a residual.** With a table, the
   overhead fraction covers only what per-type bits cannot see:
   unquantized tensors and file metadata. Its default drops from
   0.05 to 0.005 — 1.5× the measured 0.32 % residual, conservative
   for larger models where the near-constant metadata shrinks
   relative to weights. `solve` takes `format_overhead=None` as
   "default for the size model", and the recipe records the resolved
   value. An explicit `--format-overhead` always wins.
4. **The recipe schema does not change.** `plan.format_overhead`
   keeps its name, type, and role — the fraction used for size
   predictions. The assignments' `bytes` values embed the size
   model, and the recorded `runtime` plus the vramfit version name
   the table that produced them. Schema stays at version 2 as of
   this record. (Since bumped to 3, 4, then 5 — ADR-0022, ADR-0023,
   issue #59.)

## Consequences

- Predicted bytes now track the packed file to the residual, for any
  mix the solver picks. The 10 %-fit-then-overflow failure mode —
  re-planning at hand-tuned overheads until pack stops refusing — is
  gone. The pack re-check (ADR-0012) stays as the honest backstop.
- The greedy ratio now ranks downgrades by real bytes freed. Under
  the scalar, a 3→2 step appeared to free 1 bit/weight; it really
  frees 0.8125 (19 % less), while a 4→3 step frees 1.0625 (6 %
  more). Recipes planned for llama.cpp can differ from scalar-era
  recipes even before the budget line moves.
- A fractional residual over-reserves on large models: at the 49B
  target's ~20 GiB weight budget, 0.5 % reserves ~102 MiB against
  ~10-20 MiB of real extras. Safe direction, some waste. **Open
  question:** whether the residual becomes an absolute allowance
  once pack feeds measured metadata bytes back into planning.
- Resolves the per-type open question of ADR-0012 and ADR-0013. The
  i-quant table stays open (ADR-0012) — when it lands, its types
  bring their own effective-bits rows.
- A new runtime pack path must land its effective-bits table in the
  same change that adds its capability row, or its plans silently
  fall back to the scalar path.
