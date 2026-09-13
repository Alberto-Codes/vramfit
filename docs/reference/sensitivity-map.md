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
  "vramfit_schema": 4,
  "model_id": "nvidia/Nemotron-Super-49B",
  "scan": {
    "metric": "kl_divergence",
    "calibration": "/work/calibration.txt",
    "calibration_tokens": 131072,
    "calibration_sha256": "74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806",
    "calibration_bytes": 772386,
    "precisions": [8, 4, 3, 2],
    "group_by": "layer",
    "started_at": "2026-07-27T00:00:00Z",
    "within_group": "rtn-block32",
    "imatrix": null
  },
  "groups": [
    {
      "name": "model.layers.0",
      "tensors": [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight"
      ],
      "bytes_fp16": 100000000,
      "sensitivity": {
        "8": 0.0001,
        "4": 0.0042,
        "3": 0.0311,
        "2": 0.4170
      },
      "tensor_bytes": {
        "model.layers.0.self_attn.q_proj.weight": 40000000,
        "model.layers.0.self_attn.k_proj.weight": 10000000,
        "model.layers.0.self_attn.v_proj.weight": 10000000,
        "model.layers.0.self_attn.o_proj.weight": 40000000
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

- **`vramfit_schema`** — the writer emits 4 since `scan` gained the
  calibration file's content identity. The reader accepts 2, 3, and
  4, because each bump only added: version 3 widened `group_by` with
  the `stack` value (#161), and version 4 added two optional fields.
  Every older map is already a valid version-4 document that records
  no content identity, and the
  [published maps dataset](https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps)
  ships version 2, which records neither. Version 2 dates from the
  envelope key rename with
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

    The fingerprint also stores the calibration file's SHA-256 and
    byte count. Every checkpoint written before that change refuses
    to resume — pass `--no-resume` to discard it and start over.
    From that change on, re-issued calibration bytes behind an
    unchanged path refuse the old checkpoint too, which is the
    point. The digest pins the corpus, not the chunking: the
    tokenizer stays unpinned, so the same corpus through two
    tokenizers still measures two `calibration_tokens` counts.
- **`model_id`** — the scanned model as the `vramfit scan` invocation
  spelled its `MODEL` argument: a Hub id or a local path. The loader
  requires a non-empty string and reads nothing else from it. The
  published maps record the reference box's absolute checkpoint path,
  so the value identifies the scan and does not resolve elsewhere.
  `vramfit plan` copies it into the recipe.
- **`scan.metric`** — the damage metric's name. The writer records
  `kl_divergence`, the mean final-logits KL of
  [ADR-0006](../adr/0006-sensitivity-metric.md). The loader requires a
  non-empty string. The checkpoint fingerprint includes it.
- **`scan.calibration`** — the calibration text's path as the
  invocation spelled it. `vramfit scan` takes a file, so the field
  records a path, never a corpus nickname — a nickname names no
  bytes. Damage is relative to this text, so two maps compare only
  when the field matches. The loader requires a non-empty string.
  The published maps record the reference box's absolute path.
- **`scan.calibration_tokens`** — the count of calibration tokens the
  meter measured, which the `--max-tokens` budget caps. The meter
  tokenizes the whole file and truncates to that cap, so this count
  bounds the prefix the scan measured. The loader
  requires a positive integer. The published dataset's file names
  carry the same count in short form (`64k` is 65,536).
- **`scan.calibration_sha256`** and **`scan.calibration_bytes`** —
  the SHA-256 hex digest of the calibration file's bytes, and the
  file's size in bytes. A path proves which file a scan named. These
  two prove which file it read, so a reader reproduces a damage
  value against the same corpus rather than against whatever now
  carries that name. They do not prove which part of that file the
  scan measured. The meter truncates to `--max-tokens`, so the scan
  measures a prefix bounded by `scan.calibration_tokens`. Two maps
  that share a digest and record different token counts measured
  different text. The published calibration text is
  `74f2665d…3777806` at 772,386 bytes, and the
  [maps dataset](https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps)
  ships the file itself. The two fields pair: the loader requires
  both or neither, 64 lowercase hex digits for the digest, and a
  positive integer for the count. The fields are additive, so the
  loader accepts them absent — see NOT RECORDED below. The
  checkpoint fingerprint includes both.

    They pin the corpus, not the chunking. The tokenizer stays
    unpinned. One corpus through two tokenizers measures two
    `calibration_tokens` counts, and a digest match does not promise
    the same count.

    !!! warning "NOT RECORDED is the honest record"

        Absent or null means the scan recorded no content identity.
        Read that as NOT RECORDED, never as "the bytes were the
        ones that file carries today". Hashing a file now records
        today's bytes. It proves nothing about a run from months
        ago. Leave the record NOT RECORDED, whether or not the run's
        own calibration file survives.
        Never compute a digest from a fresh download
        and write it into a map as that run's input. `vramfit`
        itself never back-fills these fields: the loader does not
        hash, and a save writes null rather than inventing a
        value.

- **`scan.precisions`** — the candidate bit-widths the scan measured,
  as `--precisions` listed them. The loader requires a non-empty list
  of distinct positive integers in strictly descending order, and it
  never reorders one. Every group's `sensitivity` keys must equal this
  list exactly. A hand-made copy that drops a column edits both, and
  records the edit under `derived`.
- **`scan.started_at`** — the UTC start of the invocation that wrote
  the map, as an ISO-8601 timestamp (`2026-08-04T21:07:22Z`). A
  resumed scan is a new invocation, so the field records the last
  resume of a halted scan, and the run log carries every earlier
  attempt. The fingerprint excludes it for that reason. The loader
  requires a non-empty string and checks no format.
- **`groups[].name`** — the group's key under `scan.group_by`: a
  layer prefix such as `model.layers.0` for `layer`, a tensor name
  without its `.weight` suffix for `tensor`, and a pack-addressable
  stack for `stack`. The name carries the naming root the loaded
  model's module tree names, because the scan normalizes none. A
  `backbone.`-rooted module tree writes `backbone.`-rooted names here
  ([ADR-0029](../adr/0029-plan-independent-size-source.md) decision 7,
  amended 2026-09-06). A `backbone.`-rooted map that keeps a root-less
  group such as `lm_head` currently trips #564 in `plan`. Names are
  unique across the map, and the loader refuses a duplicate.
  `vramfit plan` keys assignments on this name, and `--pin` matches
  against it.
- **`groups[].tensors`** — the full names of the loaded parameters
  the group quantizes together, as the scan discovered them
  (`model.layers.0.self_attn.v_proj.weight`). They carry the same
  naming root as `groups[].name`. The loader requires a
  list of strings. `tensor_bytes` keys on these names, and
  `--protect` and `--exclude-imatrix` match against them.
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
  A tensor of zero elements has no positive size, so the map cannot
  record it. The safetensors format permits such a tensor, and the
  backfill refuses one rather than writing a size this field rejects
  (maintainer ruling 2026-08-19 on #335). The operator learns at the
  backfill instead of at the next read.
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
  (the ported llama.cpp reference quantizers), `kquant-imx`
  (the same port with assisted pricing,
  [ADR-0020](../adr/0020-imatrix-assisted-pricing.md)),
  `q0-ref` (the ported block quantizers `Q2_0`, `Q4_0`, and
  `Q8_0`, which reach the rows no K-quant tiles), or `q0-imx`
  (the same port with the imatrix weighting the nominal-4 fit,
  [ADR-0018](../adr/0018-kquant-within-group-method.md)'s
  2026-08-21 amendment). The writer
  always records it. The loader accepts an absent field as
  `rtn-block32` — every map written before the field existed
  measured with that method. Damage values are only comparable
  between maps with the same token.
- **`scan.imatrix`** — the path of the imatrix that assisted the
  scan, or null for an unassisted scan
  ([ADR-0020](../adr/0020-imatrix-assisted-pricing.md)). The field
  pairs with the assisted tokens, `kquant-imx` and `q0-imx`: the
  loader rejects a map that
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

    | Value | One group per | Dense model | Nemotron 3.5 Lightning 30B-A3B, native load |
    |-------|---------------|-------------|--------------------------------|
    | `layer` | decoder layer | 1 per layer | 54: 52 layers, the embeddings, the output head |
    | `stack` | pack-addressable stack | 1 per weight | 164: 46 expert stacks plus 118 other groups |
    | `tensor` | loaded parameter | 1 per weight | 164: the same set `stack` gives |

    The counts come from a native Transformers load (#571). Those
    groups root at `model.`, apart from `lm_head`. Discovery keeps a
    floating-point parameter of two or more dimensions. It then
    drops every class a quantizer refuses, which is `mixer.gate` and
    `mixer.conv1d` on this target (#204). Those two filters, not
    expert fusion, are why discovery reports fewer groups than the
    load reports parameters. The load fuses each projection's routed
    experts into one parameter, so `tensor` reaches no finer key
    than `stack` here. It loads no MTP parameters, so no count above
    covers the MTP block. The on-disk checkpoint carries that block
    at the `mtp` root (#571).

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
    surveyed runtime serves a per-expert precision. The native load
    fuses the routed experts, so no granularity reaches one anyway.

    !!! warning "A `stack` scan packs only the groups the backend maps"

        The GGUF backend maps layer groups, routed-expert stacks,
        and layer-class groups, beside the dedicated embedding and
        output-head flags (ADR-0012 decision 2, as amended). Two
        shapes matter here. A layer group becomes `blk.<n>.` across
        the three naming families above and any prefix —
        `model.layers.<n>`, `backbone.layers.<n>`, and Gemma 4's
        nested `model.language_model.layers.<n>`.
        Group names follow the loaded module tree. Nemotron 3.5
        Lightning's checkpoint parameter names use `backbone.`.
        Native Transformers converts them to `model.` module paths
        before discovery, so that target's groups use `model.`.
        A routed-expert stack becomes its fused tensor:
        `blk.<n>.ffn_up_exps.`, `blk.<n>.ffn_down_exps.`, or
        `blk.<n>.ffn_gate_exps.`.

        Every other `stack` group still raises a `PackError` that
        names it. The Nemotron target reaches no such group. Its
        164 native groups all map:

        - 23 `mixer.in_proj` and 23 `mixer.out_proj`.
        - 46 routed-expert stacks.
        - 46 shared-expert groups.
        - 6 each of `mixer.q_proj`, `mixer.k_proj`, `mixer.v_proj`,
          and `mixer.o_proj`.
        - `model.embeddings` and `lm_head`.

        Mapping does not by itself guarantee every precision. A
        group whose rows refuse the 256 super-block prices through
        the ADR-0028 table. That table holds no type between 2.25
        and 4.25 bits per weight. So such a group raises a
        `PackError` at nominal 3. The measured width decides, never
        the class name (ADR-0028, #515).
        [ADR-0012](../adr/0012-gguf-type-mapping.md)'s 2026-08-20
        amendment rules the class table, and #368 landed it.

        The backend also refuses a recipe naming two layer stacks.
        GGUF numbers one stack `blk.<n>.`, so a multimodal
        checkpoint's vision tower collides with the decoder stack.
        Scan one stack at a time.

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
