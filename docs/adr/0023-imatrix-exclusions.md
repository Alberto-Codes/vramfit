# ADR-0023: Imatrix exclusions carry the fit-collapse remedy in the recipe

- **Status:** Accepted
- **Date:** 2026-08-09
- **Extends:** [ADR-0022](0022-within-layer-protections.md) decision 6
  (the reconstruction gate gains a second remedy) and
  [ADR-0016](0016-imatrix-in-the-pack-path.md) (the imatrix pass-through
  gains per-tensor exclusions).
- **Amendment (2026-08-09, issue #59):** the recipe schema is 5, not
  the 4 decision 3 records — no-op protection pairs stopped
  resolving (ADR-0022's issue-#59 amendment). Decision 1's refusal
  surface also grew: an exclusion pattern whose every pair drops as
  a per-tensor no-op refuses, because nothing survives to ride.

## Context

The fourteenth data point (in
[evaluating packed models](../explanation/evaluating-packed-models.md))
isolated the fit-collapse mechanism. Imatrix rows with extreme column
dynamic range destabilize the weighted `Q4_K`/`Q5_K` super-block fit.
The unweighted fit reconstructs the same tensors 5.8–14.7× cleaner.
The remedy is per-tensor: `llama-quantize --exclude-weights` deletes
one row from the loaded matrix, and every other tensor keeps its
assisted fit. Probe G1c applied it by hand to `blk.{1,2,3,5}.attn_v`
and produced the first all-green reconstruction check, the best
in-budget 100-window KLD on record (0.1509 against the baseline's
0.1584), and
returned the two instability chunks to baseline order — per-chunk
KLD 0.120/0.122 against the baseline's 0.120/0.107, from G1's
5.96/6.86.

The pipeline cannot reproduce G1c. ADR-0022's only remedy for a
collapsed tensor is to drop its protection and re-plan — which
forfeits the promotion the protection exists to make. The exclusion
keeps the promotion and drops only the imatrix row. Issue #57 tracks
this gap.

One mechanical fact shapes the design. `--exclude-weights` matches by
*substring* against imatrix entry names. A full GGUF tensor name
(`blk.1.attn_v.weight`) matches exactly one row. A partial name
over-deletes — `attn_v` alone would strip every layer's row. The
quantizer then reports each deleted row with the same warning it
uses for genuine coverage gaps.

## Decision

1. **An imatrix exclusion rides a protection.** The recipe marks
   resolved protected pairs with a required `exclude_imatrix`
   boolean, and the plan records the `--exclude-imatrix` globs
   verbatim in `plan.imatrix_exclusions`. Only a protected tensor
   can drop its row — every known collapse lives in a promotion
   under an imatrix. A glob that matches no protected tensor is
   refused, naming the first unprotected match when there is one.
2. **The exclusion swaps the fit, not the type.** Solver sizes and
   predicted damage do not change. The excluded tensor still packs
   at its resolved precision.
3. **The recipe schema bumps to 4.** A version-3 reader would drop
   the exclusion record and pack collapsed tensors silently —
   ADR-0013 ruled the silent-drop case breaking.
4. **Pack emits `--exclude-weights` with full GGUF tensor names,
   only under an imatrix.** Without a matrix the exclusions are
   inert, and the command says so. The quantizer reports each
   deleted row as a coverage miss — the adapter files those as
   `imatrix_excluded`, keeping `imatrix_uncovered` an honest record
   of *unintentional* gaps (ADR-0016).
5. **The reconstruction gate still measures excluded tensors, and
   its refusal suggests the exclusion flags.** The gate already
   names the collapsed tensors — it now prints the exact
   `--exclude-imatrix` flags for the re-plan, and only for tensors
   not already excluded. A tensor that collapses *with* its
   exclusion has exhausted this remedy — the refusal then offers
   only ADR-0022's: drop the protection. The revision stays the
   user's (ADR-0012 decision 3).

## Consequences

- G1c becomes reproducible from the CLI: `plan --protect
  --exclude-imatrix` plus `pack --imatrix`. The winning artifact
  gets a recipe file with provenance — that replication is the
  next lane. **Measured (2026-08-09, the fifteenth data point):
  the loop ran end-to-end and the gate passed all-green.** The
  solver kept its format-overhead margin and demoted blk.3 one
  step below G1c's hand layout. The sibling artifact set the
  best full-window KLD on record and holds the scoreboard's
  first spike-free chunk profile. The mechanism replicates. The
  exact layout does not, and no measurement argues to force it.
- The fit-collapse revision loop shortens: the gate's refusal is a
  copy-paste re-plan instead of a forfeited promotion.
- Schema 4 breaks version-3 readers — the loader reads exactly one
  version per quantfit, as with the 2→3 bump.
- An exclusion is user-directed, like the protection it rides. The
  gate proposes; the user disposes.
- The exclusion list is a pack-layout fact, not a universal one:
  the thirteenth data point showed the collapse signature is
  (tensor, type, allocation)-dependent. A recipe's exclusions are
  evidence-driven per plan, never a default.
