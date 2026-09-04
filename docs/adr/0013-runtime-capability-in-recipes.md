# ADR-0013: Recipes record their target runtime

- **Status:** Accepted, amended by
  [ADR-0014](0014-per-type-effective-bits.md) and
  [ADR-0029](0029-plan-independent-size-source.md)
- **Date:** 2026-07-28 (accepted 2026-07-28)
- **Amendment (2026-07-28):** decision 1's note that effective bits
  stay a pack concern is revised — the domain now records effective
  bits per runtime, and the solver prices sizes from them. The
  capability table and everything else stand.
- **Amendment (2026-08-11, issue #161):** decision 3 states that
  schema versions advance per artifact. It implied one readable
  version per adapter, which the loader enforced by equality. That
  rule is now narrower: **an adapter reads one version unless the
  bump only widened what a document may say.** The sensitivity map
  writes 3 and reads 2 and 3, because version 3 only added the
  `stack` value to `group_by`. Every version-2 map is already a
  valid version-3 document, and the published maps dataset ships
  version 2. Adapters name the older versions through
  `_check_schema_version(also_reads=...)`, which defaults to empty —
  a widening is stated, never assumed. A bump that changes a
  field's meaning still reads one version. Decision 3 says "the
  sensitivity map stays at 1". Read that as version 3: #118's
  envelope rename took it to 2, and this ruling takes it to 3.
- **Amendment (2026-08-16, issue #261):** decision 3 governs the
  version an artifact declares. It said nothing about a **field** the
  reader does not know, so every reader accepted one and dropped it.
  A load followed by a save deleted it. The rule is now stated:
  **an artifact reader reports a field it does not know, and loads
  the document.** See [Unknown fields](#unknown-fields) below.
- **Note (2026-09-04, issue #359):** decision 1's `llama.cpp` row
  carries nominal 16. [ADR-0029](0029-plan-independent-size-source.md)
  decision 4 put the F16 passthrough in `EFFECTIVE_BITS` at 16.0 bits
  per weight, and two pinned invariants carry it into
  `RUNTIME_CAPABILITIES`: `test_every_table_covers_its_runtime_capability_exactly`
  and `test_type_table_covers_the_llama_cpp_capability_set`. The
  widening is forced rather than chosen. 16 is a passthrough that
  holds a group unquantized. It is not a quantization type the scan
  measures. One consequence: a map that scanned 16 can carry a
  `--pin "*=16"` through `servable_precisions`. No map does today.

## Context

ADR-0010 split serving by precision: llama.cpp below 4-bit, vLLM at
4-bit and above. Its consequences named a follow-up — the plan step
gains a runtime-capability input. ADR-0007 deferred the same
constraint from the solver. Until now, nothing enforced it: the
solver drew candidates from `scan.precisions` alone, and a recipe
did not say which runtime it was planned for. A recipe assigning
3-bit could reach a vLLM pack path with no warning.

The scan measures more than any one runtime serves. Sensitivity is a
property of the model (ADR-0010), so one map should feed plans for
several runtimes. The constraint belongs at plan time, not scan
time.

## Decision

1. **A capability table maps runtime names to servable nominal
   precisions.** It lives in the domain
   (`vramfit.domain.runtime.RUNTIME_CAPABILITIES`):

   | Runtime | Servable precisions |
   |---------|---------------------|
   | `llama.cpp` | 16, 8, 6, 5, 4, 3, 2 |
   | `vllm` | 8, 4 |

   The 16 entry dates from the 2026-09-04 note (#359). It is the F16
   passthrough of ADR-0029 decision 4, not a measured precision.
   The table records nominal bits only. Effective bits per type stay
   a pack concern (ADR-0012).
2. **The solver filters candidates through the table.** `solve`
   takes an optional `runtime`. When given, the candidate set is the
   scanned precisions intersected with the runtime's capability,
   order preserved. An unknown runtime, or an empty intersection,
   raises `RuntimeCapabilityError` under the `VramfitError` root.
   Pins are validated against the filtered set. When absent, the
   solver behaves as before — the domain accepts unconstrained
   plans. The narrowing is never silent: the CLI reports the dropped
   precisions, and an infeasible budget names the precisions the
   runtime removed from the floor.
3. **The recipe records the runtime.** A new top-level `runtime`
   field, string or null. `vramfit plan` always sets it
   (`--runtime`, default `llama.cpp`), so real artifacts always
   carry it. The recipe schema version bumps to 2 — a version-1
   reader would silently drop the constraint. (Since bumped to 3,
   4, then 5 — ADR-0022, ADR-0023, issue #59.) Schema versions now
   advance per artifact: the sensitivity map stays at 1. The loader
   enforces the cross-field invariant for runtimes it knows: an
   assignment precision the recorded runtime cannot serve is a
   schema violation. An unknown runtime name loads untouched, so a
   newer vramfit's recipe stays readable and pack backends judge
   it at use.
4. **Pack backends reject foreign runtimes.** The GGUF backend
   refuses a recipe whose `runtime` is neither null nor `llama.cpp`.
   The refusal is a `PackError` before any toolchain work starts.
5. **The ADR-0012 type tables extend to the full llama.cpp
   capability set.** This amends ADR-0012 decision 1 and decision 3:

   | Nominal bits | Tensor type | Effective bits/weight | Drift |
   |--------------|-------------|----------------------|-------|
   | 6 | `Q6_K` | 6.5625 | +9.4 % |
   | 5 | `Q5_K` | 5.50 | +10.0 % |

   The base-ftype table gains 6→`Q6_K` and 5→`Q5_K_S` (`Q5_K` as an
   ftype aliases `Q5_K_M`, matching the `Q4_K` case; `Q6_K` is its
   own ftype). All other ADR-0012 clauses stand.

## Unknown fields

Added by the 2026-08-16 amendment (issue #261). Decision 3 rules on
the version an artifact declares. This section rules on a field the
reader does not know.

1. **An artifact reader reports the field, then loads the document.**
   The report names the JSON path. It states that a save drops the
   field. The reader refuses nothing on this account.
2. **One level reaches every reader and every caller.** The
   sensitivity map, the recipe, the scan checkpoint, and the evals
   sidecar behave alike. A reader does not report differently for the
   CLI than for a port caller.
3. **Rejected alternative — refuse the load.** This project adds
   fields inside a version. `imatrix_counts` landed at map version 3
   (#217). `derived` landed at map version 3 (#254). Neither bumped.
   A reader that refused an unknown field could not read the next
   such addition. Decision 3 already tolerates an unknown `runtime`
   value for the same reason.
4. **Rejected alternative — preserve the field.** No domain type
   carries it. Preservation needs a store on 17 frozen dataclasses
   across `vramfit.domain.model` and `vramfit.domain.evals`. It puts
   unparsed JSON in the domain, which ADR-0008 keeps in the adapters.
   Every writer also rebuilds its document field by field. A stored
   field would need a merge at each of 18 objects.
5. **An object whose keys the schema does not fix never reports.** The
   map's `sensitivity` and `tensor_bytes`, and the recipe's `pins` and
   `protections`, key on precisions, tensor names, and patterns. Their
   own rules check those keys. A group's `imatrix_counts` fixes three
   keys exactly (ADR-0026), so it keeps refusing.
6. **The report travels on the human channel.** ADR-0011 decision 1
   keeps two channels apart. A reader holds no run log, so it reports
   through the stdlib `warnings` module. The CLI installs its own
   reporter and prints one `warning:` line per report. That matches
   the run-log failure line of ADR-0011 decision 2. The reader names
   no source line, because the raising line sits inside the adapter
   and says nothing about the artifact. The JSON path in the message
   is the locator.
7. **A reader of a file the model publisher owns accepts an unknown
   field and reports nothing.** `hf_config.py` and `scan/offload.py`
   read documents another project writes, where an unknown key is
   normal. Only a vramfit artifact reader reports. Issues #286 and
   #287 cover the remaining parse sites.

Measured on 2026-08-16, before the build. The 8 published sensitivity
maps carry no field their readers do not know. Neither do the 5
published evals sidecars, nor the published recipe. No report fires on
the published set today.

## Consequences

- One sensitivity map now plans for several runtimes: scan once at
  the union, plan per target.
- The 5- and 6-bit candidates open the ground Q5_K_S occupies — the
  evaluation page's over-budget quality reference becomes reachable
  by a measured recipe.
- The single `format_overhead` scalar stretches thinner: with six
  precisions its per-type error spread widens (drift ranges +6.25 %
  to +31.25 %). The per-type effective-bit open question of ADR-0012
  gains urgency but stays open.
- Old version-1 recipe files no longer load. They are regenerated
  from their maps in minutes, and no published artifact carries the
  old schema.
- A load followed by a save still deletes a field the reader does not
  know. The 2026-08-16 amendment reports the loss instead of fixing
  it. Anyone who hand-extends a published artifact must keep the
  source, not re-save the loaded copy.

## Open questions

- Whether the capability table should version per runtime release
  (llama.cpp gains types; vLLM kernels evolve). Today it tracks what
  the current serving path exercises.
- ~~Whether the solver should consume per-type effective-bit tables
  instead of one `format_overhead` fraction (carried from ADR-0012,
  ties to this table).~~ Resolved by
  [ADR-0014](0014-per-type-effective-bits.md).
