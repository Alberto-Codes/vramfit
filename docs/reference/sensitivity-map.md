---
status: stable
---

# Sensitivity map format

> **Status: stable** — implemented in
> `vramfit.adapters.outbound.sensitivity_map_json`, whose loader enforces
> everything described here. `vramfit scan` produces these files, and
> real maps (Qwen2.5-3B, the 49B target) drove the full loop.

The sensitivity map is the output of `vramfit scan` and the input to
`vramfit plan`: JSON, one entry per (layer group × candidate precision).

```json
{
  "vramfit_schema": 3,
  "model_id": "nvidia/Nemotron-Super-49B",
  "scan": {
    "metric": "kl_divergence",
    "calibration": "wikitext",
    "calibration_tokens": 131072,
    "precisions": [8, 4, 3, 2],
    "group_by": "layer",
    "started_at": "2026-07-27T00:00:00Z",
    "within_group": "rtn-block32",
    "imatrix": null
  },
  "groups": [
    {
      "name": "model.layers.0.self_attn",
      "tensors": ["q_proj", "k_proj", "v_proj", "o_proj"],
      "bytes_fp16": 100000000,
      "sensitivity": {
        "8": 0.0001,
        "4": 0.0042,
        "3": 0.0311,
        "2": 0.4170
      },
      "tensor_bytes": {
        "q_proj": 40000000,
        "k_proj": 10000000,
        "v_proj": 10000000,
        "o_proj": 40000000
      }
    }
  ]
}
```

## Field notes

[ADR-0021](../adr/0021-runtime-frame-measurement.md) supersedes
[ADR-0019](../adr/0019-kquant-priced-maps.md) and
[ADR-0020](../adr/0020-imatrix-assisted-pricing.md): the fields
below remain, the sub-4-bit pricing claims do not.

- **`vramfit_schema`** — the writer emits 3 since `group_by` gained
  the `stack` value (#161). The reader accepts 2 and 3, because
  version 3 only widened that enum: every version-2 map is already a
  valid version-3 document, and the
  [published maps dataset](https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps)
  ships version 2. Version 2 dates from the envelope key rename with
  the tool (#118). The reader accepts only the new key. A schema-1
  map migrates with a key rename plus a version bump, or a re-scan.
  The #134 ruling froze
  the 49B run root as a pre-rename archive. Migrate a copy. Never
  edit the archive. A #134 load check read all nine of its schema-1
  maps on 2026-08-11: 82 groups each
  ([card ledger](../../publication/model-card/card-ledger.md)).

  A scan checkpoint carries the same envelope and migrates the same
  way, at its own version 2. The reader then compares `fingerprint`
  against the running scan. That fingerprint stores the model,
  calibration, and imatrix paths as the invocation spelled them, so
  a resume must reproduce the original command line. A rename of any
  of those paths invalidates every checkpoint that names them.
- **`sensitivity`** — divergence of the perturbed model's output from the
  full-precision reference, measured per
  [ADR-0006](../adr/0006-sensitivity-metric.md) (mean final-logits KL).
  Higher = more damage. Values are comparable *within* a scan, not across
  scans with different calibration sets. In-frame low-bit prices do not
  predict the packed artifact
  ([ADR-0021](../adr/0021-runtime-frame-measurement.md)) — current
  practice plans on a map copy without the 2-bit column.
- **`bytes_fp16`** — group size at reference precision. The solver derives
  per-precision sizes from this — at the runtime's per-type effective
  bits when it has a table ([ADR-0014](../adr/0014-per-type-effective-bits.md)),
  at nominal bits plus the overhead fraction otherwise.
- **`tensor_bytes`** — each member tensor's bytes at reference
  precision ([ADR-0022](../adr/0022-within-layer-protections.md)).
  Protections price against these, and `vramfit plan` refuses a
  `--protect` rule on a group without them. The field is additive
  and informational, so it forced no schema bump: the loader accepts an
  absent field as unknown, and a present field must cover exactly
  the group's tensors with positive sizes summing to
  `bytes_fp16`. New scans record it. For
  older maps, `scripts/backfill_tensor_sizes.py` reads the
  checkpoint's safetensors headers — a JSON parse, no torch — and
  writes an annotated map copy.
- **`imatrix_counts`** — the group's pooled imatrix count
  distribution: `{"min": ..., "median": ..., "max": ...}`
  ([ADR-0026](../adr/0026-moe-expert-pricing.md) decision 4, scoped
  by the 2026-08-13 #201 amendment). An assisted scan reads each
  fused expert stack's count vector through
  `resolve_imatrix_counts` and pools the group's vectors into three
  numbers. Provenance, not a gate. A scalar chunk tally never
  enters the reduction, so the router, the shared experts, and
  every dense member stay out. The field is all-or-nothing per
  group: it appears only when every expert-stack member resolved
  its full count vector, and a group without an expert stack never
  carries it. `median` is always a float. The field is additive and
  informational, so the schema holds at 3. The loader accepts an
  absent field as no summary. A present field must hold exactly the
  three keys, with values ordered `min <= median <= max`. An absent
  field leaves a dense-only group and an unresolved group alike —
  #194 owns the map's coverage record.
- **`scan.within_group`** — the within-group method token
  ([ADR-0018](../adr/0018-kquant-within-group-method.md)):
  `rtn-block32` (round-to-nearest, the v1 default), `kquant-ref`
  (the ported llama.cpp reference quantizers), or `kquant-imx`
  (the same port with assisted pricing,
  [ADR-0020](../adr/0020-imatrix-assisted-pricing.md)). The writer
  always records it. The loader accepts an absent field as
  `rtn-block32` — every map written before the field existed
  measured with that method. Damage values are only comparable
  between maps with the same token.
- **`scan.imatrix`** — the path of the imatrix that assisted the
  scan, or null for an unassisted scan
  ([ADR-0020](../adr/0020-imatrix-assisted-pricing.md)). The field
  pairs with the `kquant-imx` token: the loader rejects a map that
  claims assistance without naming its imatrix, or the reverse. An
  assisted map is only comparable to a pack that consumed the same
  imatrix file. The loader accepts an absent field as null.
- **`derived`** — why this map is not a scan artifact: the edit that
  produced it and what it is for (#136). `vramfit scan` never writes
  the field. The author of a hand-made copy adds it. Two published maps
  carry it: `sensitivity-64k-kquant-imx-no2.json` and the
  `-no2-sized.json` copy `vramfit plan` solved the published recipe
  from. Both read:

    ```json
    "derived": "Derived from sensitivity-64k-kquant-imx.json by removing the 2-bit column. Not a scan artifact. Diagnostic for the 2-bit-specific frame-transfer hypothesis (eleventh data point)."
    ```

    The field is additive and informational, so the schema holds at 3.
    A reader that ignores it stays correct. The loader accepts an
    absent field as a scan artifact. A present field must be a
    non-empty string. The writer omits the field when the map carries
    no note and never writes null, so an explicit null is a hand-edit
    — rejected, not normalized. A load then save preserves the note.
    A load then save deleted the note before #136.
- **`groups`** — granularity is set by `--group-by`. Marginal (one group at a
  time) measurement is assumed — interaction effects between groups are a
  known blind spot recorded in ADR-0006. Group names must be unique, and
  every group's `sensitivity` keys must equal `scan.precisions` exactly
  (the v1 loader rejects partially-scanned groups).
- **`scan.group_by`** — `layer`, `tensor`, or `stack`.

    | Value | One group per | Dense model | Nemotron 3.5 Lightning 30B-A3B backbone |
    |-------|---------------|-------------|--------------------------------|
    | `layer` | decoder layer | 1 per layer | 52 layers plus the embeddings |
    | `stack` | pack-addressable stack | 1 per weight | 46 expert stacks plus the rest |
    | `tensor` | checkpoint weight | 1 per weight | 5888 expert weights plus the rest |

    Counts are backbone-only (#160). Scanning the MTP block adds 256
    expert weights, which is 2 more expert stacks.

    `stack` keys on the unit a pack assigns a precision to (#161). It
    collapses a mixture-of-experts layer's routed experts into one
    group per projection, and keeps every other weight separate. On a
    dense model it matches `tensor`, because a pack addresses each of
    those weights alone.

    Pick `stack` when a finer key would buy nothing. llama.cpp fuses
    each layer's experts into one tensor that carries one quantization
    type, which gives 46 addressable expert slots on the Nemotron
    target (#159). vLLM, TensorRT-LLM, and SGLang each resolve one
    algorithm per mixture-of-experts module, which gives 23 (#166). No
    surveyed runtime serves a per-expert precision, so a
    `tensor`-keyed map of that model prices 5888 distinctions no pack
    can express.

    !!! warning "A `stack` scan packs its expert stacks, not every group"

        The GGUF backend maps two group shapes (ADR-0012 decision 2,
        amended 2026-08-12). A layer group becomes `blk.<n>.` across
        the three naming families above — both `model.layers.<n>`
        and the Nemotron 3.5 Lightning target's
        `backbone.layers.<n>`. A routed-expert stack becomes its
        fused tensor: `blk.<n>.ffn_up_exps.`,
        `blk.<n>.ffn_down_exps.`, or `blk.<n>.ffn_gate_exps.`.

        Every other `stack` group still raises a `PackError` that
        names it. On the Nemotron target that covers the Mamba
        `in_proj`, `out_proj`, and `conv1d`, the attention
        projections, the router, and the shared experts. So a
        `layer`-keyed recipe packs today and a whole-model
        `stack`-keyed recipe does not. Issue #183 carries the
        remaining classes.

        The backend also refuses a recipe naming two layer stacks.
        GGUF numbers one stack `blk.<n>.`, so the target's
        `mtp.layers.<n>` and a multimodal checkpoint's vision tower
        each collide with the backbone. Scan one stack at a time.

## Unknown fields

The loader reports a field it does not know, then loads the map
(ADR-0013, the 2026-08-16 amendment, issue #261). The report names the
JSON path and states that a save drops the field. The rule covers the
map root, `scan`, and each entry of `groups`.

Three objects never report. `sensitivity` keys on precisions,
`tensor_bytes` keys on tensor names, and each has its own rule. A
group's `imatrix_counts` fixes `min`, `median`, and `max` exactly
(ADR-0026), so it refuses instead.

A load then save still deletes the field. Keep the source. Never
re-save a hand-extended copy over itself.
