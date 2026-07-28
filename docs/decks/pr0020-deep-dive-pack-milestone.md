---
marp: true
theme: quantfit
paginate: true
footer: quantfit · PR #20
---

<!-- _class: lead -->

# quantfit — pack step deep dive

Milestone 3: GGUF backend, ADR-0012 type mapping, first end-to-end proof

**PR [#20](https://github.com/Alberto-Codes/quantfit/pull/20)** · branch `feat/pack-gguf-backend` · 2026-07-28

---

## Context — the pipeline closes

```
scan ──► sensitivity.json ──► plan ──► recipe.json ──► pack ──► packed.gguf
(PR #6)      artifact         (PR #1)     artifact      NEW      servable
```

- Until now a recipe was a claim on paper. `quantfit pack` makes it a
  model llama.cpp serves (ADR-0010: sub-4-bit runs through GGUF).
- Mechanism: `llama-quantize --tensor-type` per-tensor overrides —
  verified against llama.cpp source (`e9fa078`) before building.
- First target: the Qwen2.5-3B pilot map from the 2026-07-28 scan,
  recipe `recipe-4GiB.json` (2 GiB weight budget).

---

## Architecture — where the new pieces sit

| Layer | New | Contents |
|---|---|---|
| `domain/pack.py` | ✚ | `TypeOverride`, `PackResult`, `weight_budget_margin` — pure |
| `ports/outbound.py` | +1 | `RecipePacker`: `convert()` / `pack(recipe)` |
| `adapters/outbound/gguf/types.py` | ✚ | bits→type tables, group→pattern mapping — pure, torch-free |
| `adapters/outbound/gguf/pack.py` | ✚ | `LlamaCppPacker`: two subprocesses, failure translation |
| `adapters/inbound/cli_pack.py` | ✚ | `quantfit pack` — composition root, run-log events |
| `adapters/inbound/run_log.py` | ✚ | `SafeRunLog` hoisted out of `scan_events` for reuse |

The package imports neither torch nor gguf — the convert script's
*interpreter* carries them (`quantfit[pack]` extra, ADR-0005).

---

## Contract — the port splits at the stage boundary

```python
class RecipePacker(Protocol):
    def convert(self) -> int: ...          # f16 base GGUF, reused if present
    def pack(self, recipe: Recipe) -> PackResult: ...
```

```python
@dataclass(frozen=True, slots=True)
class PackResult:
    packed_bytes: int                   # real file size, stat'd
    base_type: str                      # e.g. "Q4_K_S"
    token_embedding_type: str | None    # e.g. "q8_0"
    overrides: tuple[TypeOverride, ...] # unique patterns, recipe order
```

Two methods because the composition root logs each stage (ADR-0011):
a 49B convert is minutes and cacheable, quantize is separate work.

---

## ADR-0012 — nominal bits to K-quant types

| Nominal | Tensor type | Effective bits | Drift |
|---|---|---|---|
| 8 | `Q8_0` | 8.50 | +6.25 % |
| 4 | `Q4_K` | 4.50 | +12.5 % |
| 3 | `Q3_K` | 3.44 | +14.6 % |
| 2 | `Q2_K` | 2.63 | +31.25 % |

- K-quants, not i-quants: no importance matrix needed — the scan does
  not emit one yet (open question stays open for IQ types).
- Second table for the quantizer's positional ftype argument
  (8→`Q8_0`, 4→`Q4_K_S`, 3→`Q3_K_S`, 2→`Q2_K`): `Q4_K` as an ftype
  aliases `Q4_K_M`, and `Q3_K` is not an ftype at all.

---

## The command line pack drives

```
llama-quantize --pure \
  --token-embedding-type q8_0 --output-tensor-type q8_0 \
  --tensor-type 'blk\.0\.=q4_k' ... (one per layer group) \
  base-f16.gguf out.gguf Q4_K_S 14
```

- Patterns are lower-cased ECMAScript regexes, `regex_search`, first
  match wins → dots must be escaped or `blk.1.` also matches `blk.11.`.
- A matched override beats the built-in heuristics; `--pure` stops
  heuristic mixing for anything uncovered → **recipe-driven, never
  heuristic-driven**.
- `--output-tensor-type` mirrors the embedding assignment: on untied
  models the head would otherwise fall to the floor type silently
  (caught in review — Qwen ties, the 49B target does not).

---

## Real bytes over nominal bits

- Plan predicts sizes as nominal bits × (1 + `format_overhead`),
  default 0.05. GGUF's effective bits make that undershoot.
- Pack stats the artifact and re-checks `plan.weight_budget_bytes`:

```json
{"event": "size_checked", "packed_bytes": 2206491136,
 "weight_budget_bytes": 2147483648,
 "margin_bytes": -59007488, "fits": false}
```

- That event is from the **first real pack**: 56 MiB over, exit 1,
  file kept. Re-planning at `--format-overhead 0.10` → 1.98 GiB,
  17 MiB under, exit 0. Measured overhead of this Q8_0/Q4_K mix ≈ 10 %.

---

## Measured results — tier 1 + tier 2 (reference box)

Full WikiText-2 perplexity; whole-model KL vs f16 over 100 chunks
(51,200 tokens). llama.cpp b10172, Vulkan on the 4090.

| Model | Size | Fits 2 GiB | PPL ↓ | Mean KLD ↓ | Same top ↑ |
|---|---|---|---|---|---|
| f16 reference | 5.75 GiB | no | 8.422 | — | — |
| **recipe pack** | **1.98 GiB** | **yes** | **8.661** | **0.0382** | **90.5 %** |
| Q4_K_M | 1.80 GiB | yes | 8.790 | 0.0494 | 88.9 % |
| Q5_K_S | 2.02 GiB | no (+21 MiB) | 8.520 | 0.0161 | 93.3 % |

In budget: **35 % less** of the f16→quant PPL climb, **23 % lower**
mean KL than the heuristic. Q5_K_S wins quality, loses the budget.

---

## Verification — what the file actually contains

Read back via `gguf-py` (`GGUFReader`), packed file vs recipe:

```
type counts: {'F32': 181, 'Q8_0': 57, 'Q4_K': 196}
8-bit layers: [1, 2, 27, 30, 31, 33, 34, 35]   ← == recipe's 8-bit groups
token_embd:  Q8_0                              ← == embed assignment
```

- 8 layers × 7 tensors + `token_embd` = 57 Q8_0 ✓ · 28 × 7 = 196 Q4_K ✓
- Norms stay F32 (excluded from quantization) ✓
- Re-pack after the review-round flag change: **byte-identical** —
  pack is deterministic, eval numbers survive the fix.

---

## Test pyramid (ADR-0009)

- **Contract**: `MemoryRecipePacker` vs the real `LlamaCppPacker`
  against argv-recording stub tools — hermetic, both sides raise the
  same `PackError` shapes. Fake shares the real mapping functions →
  cannot drift from the ADR-0012 tables.
- **Argv contract** (real-only): pins `--pure`, both embedding flags,
  every `pattern=type` pair, the positional tail, `--outtype f16`.
- **Unit**: mapping tables, escaping (`blk\.1\.` ∤ `blk.11.`),
  CLI event sequences, halt stages, domain invariants.
- **e2e**: console script with a stub toolchain.
- Gates: 323 fast / 327 push-stage, coverage 97 %, docvet 100 %.
- **Deliberately uncovered**: the real llama.cpp binaries in CI —
  proven by the reference-box run instead (this deck's numbers).

---

## Risks / open questions

- **ADR-0012 is Proposed** — this PR builds on it; accept or amend at
  merge (flagged in the PR).
- Successful tool output is discarded: a `--tensor-type` pattern that
  matches nothing warns only in swallowed stdout (ADR-0012 open
  question — likely a sidecar artifact).
- `format_overhead` stays one scalar; per-type effective-bit tables
  in the solver would kill the re-plan loop (open question).
- Fingerprint still proves provenance, not content (issue #8).
- Tensor-level recipes are rejected in v1 — layer + embedding only.

---

## Next steps

1. **Runtime-capability table in `plan`** (ADR-0010): llama.cpp allows
   {8, 6, 5, 4, 3, 2} — 6/5-bit candidates are exactly the ground
   Q5_K_S occupies. The scan grid gains two columns.
2. **Whole-recipe validation pass** (ADR-0006): replay the recipe in
   the scan's frame, check the additivity assumption pre-pack.
3. **First real 49B scan** on the downloaded checkpoint, then the
   north-star pack: 49B on the 4090 through this exact path.
4. Tier 3 (lm-evaluation-harness slice) on the publication winner.

---

## Links

- PR: [#20](https://github.com/Alberto-Codes/quantfit/pull/20) · issue [#8](https://github.com/Alberto-Codes/quantfit/issues/8)
- ADRs: [0012](../adr/0012-gguf-type-mapping.md) type mapping (Proposed) ·
  [0010](../adr/0010-sub-4-bit-serving-path.md) GGUF path ·
  [0011](../adr/0011-run-logs-and-error-root.md) run logs
- Evidence: [evaluating packed models](../explanation/evaluating-packed-models.md) —
  the tier-1/2 table with method notes
- How-to: [pack a recipe](../how-to/pack-a-recipe.md)
- Key files: `domain/pack.py` · `adapters/outbound/gguf/{types,pack}.py` ·
  `adapters/inbound/cli_pack.py` · `tests/contract/test_recipe_packer_contract.py`
