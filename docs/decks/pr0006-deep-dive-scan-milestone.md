---
marp: true
theme: quantfit
paginate: true
footer: quantfit · PR #6
---

<!-- _class: lead -->

# quantfit — scan step deep dive

Milestone 2: damage meter, checkpointed resume, torch behind an extra

**PR [#6](https://github.com/Alberto-Codes/quantfit/pull/6)** · merged to main as `0490417` · 2026-07-28

---

## Context — the science step lands

```
scan  ──►  sensitivity.json  ──►  plan  ──►  recipe.json  ──►  pack
 NEW           artifact          (PR #1)       artifact       (next)
```

- `plan` shipped first (PR #1): pure math, torch-free, testable in ms.
- `scan` produces the numbers `plan` consumes: per (group × precision),
  the **damage** — mean final-logits KL vs the bf16 reference (ADR-0006).
- Decided just before this PR (ADR-0010): candidate set is **{8, 4, 3, 2}**,
  runtime-independent — the scan simulates quantization, no kernels needed.
- A 49B scan is hours of GPU time → **crash-safe resume is a core
  requirement**, not a nicety (issue #2).

---

## Architecture — where the new pieces sit

| Layer | New | Contents |
|---|---|---|
| `domain/scan.py` | ✚ | grid planning, resume filtering, fingerprint, map assembly — pure |
| `ports/outbound.py` | +3 | `DamageMeter`, `ScanCheckpointStore`, `SensitivityMapSink` |
| `adapters/outbound/scan_checkpoint_json.py` | ✚ | checkpoint file, own schema version |
| `adapters/outbound/scan/` | ✚ | torch meter package (`meter`, `quantize`, `kl`, `calibration`) |
| `adapters/inbound/cli_scan.py` | ✚ | `quantfit scan` — composition root + loop |

---

## Architecture — torch stays quarantined

- torch lives **only** in `adapters/outbound/scan/`, behind the new
  `quantfit[scan]` extra.
- Three scoped carve-outs, each commented with its ADR rationale:
  import-linter `ignore_imports`, a ty `unresolved-import` override,
  and a coverage `omit`.
- Base install stays typer-only (ADR-0005); the CLI imports the meter
  lazily inside the command body.
- CI never installs torch — the hermetic suites prove the orchestration
  against the verified fake.

---

## Contract — the expensive port

```python
class DamageMeter(Protocol):
    def groups(self) -> tuple[GroupSpec, ...]: ...  # discovery
    def calibration_tokens(self) -> int: ...
    def measure(self, group: str, bits: int) -> float:  # one cell =
        ...  # one calibration pass
```

- `measure` returns **finite, non-negative damage** — the port contract
  forbids recording unstable numbers (they raise instead).
- Cell-by-cell shape is what makes resume possible: the orchestrator
  owns the loop, the meter owns one measurement.
- Verified fake (`MemoryDamageMeter`) proven equivalent by a dual-run
  contract suite; the real-torch side carries `integration`+`slow`
  markers and runs where the extra is installed (ADR-0009).

---

## Contract — checkpoint + fingerprint

```json
{ "quantfit_schema": 1,
  "fingerprint": "…|kl_divergence|calib.txt|32768|layer|8,4,3,2|rtn-block32",
  "measurements": [ {"group": "model.layers.0", "bits": 4, "damage": 1.8e-06} ] }
```

- Written after **every cell**, atomic replace — a crash costs at most
  one measurement.
- Fingerprint = model, metric, calibration path, token count, grouping,
  precisions, **method**. Separator-escaped (injection cannot collide two
  scans). A mismatch refuses the checkpoint with a `--no-resume` hint.
- Own schema version — a breaking artifact-schema bump cannot strand an
  in-flight scan.
- **Documented limit:** provenance, not content. Swapped weights under an
  unchanged path defeat it (open question in #8).

---

## Core loop — perturb, measure, restore

```python
param.copy_(rtn_quantize_dequantize(param, bits, block=32))  # perturb
damage = mean_kl(cached_ref_logprobs, model(batch).logits)  # measure
param.copy_(original)  # restore (finally)
```

- **RTN, symmetric, 32-elem blocks** (ADR-0006 v1): approximates pack
  formats without depending on any; GPTQ/AWQ later = new scan, same schema.
- Reference log-probs computed **once**, cached fp16 on CPU:
  0.25 GiB / 1024 tokens at 128k vocab. Rejected alternative: recompute
  per cell — doubles a multi-hour scan.
- Restore failure **poisons the meter** — it refuses further cells rather
  than measure against corrupt weights; root-cause exception keeps the stage.

---

## Failure design — reject, never normalize

The worst bug class: a wrong number that *looks* healthy feeding the solver.

- `mean_kl` **raises** on non-finite KL and on negative KL beyond fp16
  residue (`> 1e-3`) — a NaN forward pass halts the scan, not a
  `damage = 0.0` that tells the solver a fragile layer is free.
- Every failure is a clean `error:` line + exit 1 naming the cell and the
  surviving checkpoint size — model load, measurement, checkpoint write,
  map write.
- Fail-fast before the model loads: bad `--out` dir, sub-2-bit
  precisions, duplicate/foreign checkpoints.
- `Measurement.__post_init__` re-validates on checkpoint reload — a
  hand-edited file hits the same invariants.

---

## Measured — real damage curves (tiny model, RTX 4090)

16-cell scan, 128 calibration tokens, CUDA:

| group | 8-bit | 4-bit | 3-bit | 2-bit |
|---|---|---|---|---|
| `model.embed_tokens` | 0.0 | 5.9e-05 | 3.2e-04 | 2.3e-03 |
| `model.layers.0` | 0.0 | 1.8e-06 | 4.8e-05 | 4.0e-04 |
| `model.layers.1` | 0.0 | 6.0e-07 | 2.2e-05 | 2.4e-04 |
| `lm_head` | 0.0 | 5.4e-05 | 3.3e-04 | 2.8e-03 |

- Monotone in bits for every group; embeddings ~50× more fragile than
  decoder layers — the heterogeneity the whole project bets on, visible
  even in a random-weight toy.
- Same run verified on CPU; resume verified end-to-end (kill → rerun →
  continues at first unmeasured cell).

---

## Verification — the pyramid holds

- **240 hermetic** tests (unit + contract), <2 s; **98% coverage** on the
  push gate (floor 90).
- **25 torch-tier** (integration/slow): built-in-test tiny Llama with a
  trained BPE tokenizer — no network, CPU-sufficient, CUDA-verified.
- Every new port has a **verified-fake contract suite**; the meter's
  real side runs the same suite where torch exists.
- Hypothesis properties: todo ∪ done partitions the grid, order
  preserved, assembly order-invariant.
- Deliberately not covered: real-model scan cost/quality (starts now),
  power-loss durability (atomic-replace only, no fsync — documented).

---

## Review cycle — what it caught

Six specialist agents + Copilot, ~45 findings triaged, 8 declined with
written rationale. The ones that mattered:

- `max(nan, 0.0) == nan` — NaN slipped the KL clamp *and* crashed outside
  the try. Two reviewers, independently.
- Fingerprint separator injection; missing method token.
- `sys.exc_info()` inside an except handler made restore-failure
  detection dead code (Copilot).
- Generator `precisions` silently truncated the grid after group one.
- transformers <4.56 ignores `dtype=` silently → fp32 at 2× memory.
  Floor raised.
- CI would have gone red on merge (ty/ruff drift) — caught pre-merge by
  the cross-PR review pass.

---

## Risks / open questions

- **Fingerprint is content-blind** — weights/calibration swapped under an
  unchanged path resume silently. Options (digest vs mtime manifest)
  costed in #8 after the first real scan.
- **Memory at defaults doesn't fit the 49B**: 32 GiB reference cache +
  ~98 GiB bf16 weights > 124 GB box. How-to prescribes
  `--max-tokens 32768` (~8 GiB); enforcement is documentation only.
- **`device_map=auto` throughput** on a model larger than VRAM is
  unmeasured — streaming groups to GPU is the planned fix, not built.
- **Additivity assumption** unguarded until the whole-recipe validation
  pass (next milestone, #8).

---

## Next steps (issue #8)

1. **First real scan** — checkpoint downloading to the reference box;
   `--max-tokens 32768 --trust-remote-code --device auto`; report KL at
   1/4, 1/2, full calibration (ADR-0006's convergence question).
2. **Whole-recipe validation pass** — the additivity check.
3. **Runtime-capability table in `plan`** — llama.cpp {8,6,5,4,3,2},
   vLLM {8,4} (ADR-0010).
4. **Pack: GGUF backend** via `llama-quantize --tensor-type`.

---

## Links

- PR: [#6](https://github.com/Alberto-Codes/quantfit/pull/6) · merged `0490417` · roadmap: [#8](https://github.com/Alberto-Codes/quantfit/issues/8)
- ADRs: [0005](../adr/0005-heavy-deps-as-extras.md) extras ·
  [0006](../adr/0006-sensitivity-metric.md) KL metric ·
  [0009](../adr/0009-testing-strategy.md) verified fakes ·
  [0010](../adr/0010-sub-4-bit-serving-path.md) precision set / GGUF path
- Key files: `domain/scan.py` · `ports/outbound.py` ·
  `adapters/outbound/scan/meter.py` · `adapters/inbound/cli_scan.py`
- Docs: [scan-a-model](../how-to/scan-a-model.md) ·
  [sensitivity-scanning](../explanation/sensitivity-scanning.md) ·
  [sensitivity-map](../reference/sensitivity-map.md)
