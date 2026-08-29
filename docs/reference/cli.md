---
status: stable
---

# CLI reference

> **Status: stable** — `version`, `budget`, `plan`, `scan`, `pack`,
> `validate`, and `capacity` are implemented, and the flags and
> behaviors below match the built commands (audited 2026-08-14 on
> #149, promoted with the #228 build; `capacity` added 2026-08-26
> on #422). `pack` covers the GGUF backend only (ADR-0010).

## `vramfit version`

Implemented. Prints the installed package version.

```console
$ vramfit version
vramfit 0.1.0
```

## `vramfit budget`

Implemented. Prints the VRAM budget breakdown. The `--kv-headroom` value
for `plan` is the sum of the KV-cache and runtime-overhead lines. The attention shape comes from
exactly one source: `--model-config` (a Hugging Face `config.json`) or
the manual triple. The reader handles DeciLM NAS configs with
skipped-attention blocks and composite configs that nest the decoder
under `text_config`. It prices declared per-layer KV geometry (#421).
That covers a `layer_types` sliding/global pattern with its window,
split local/global head geometry, the `attention_k_eq_v` KV-head
override (#431), and shared-KV layers that allocate no cache. An active window without
`layer_types`, an unknown layer type, or a llama-geometry key beside
`block_configs` refuses rather than guessing.

```
vramfit budget
  --vram SIZE            Total VRAM  [default: 24GiB]
  --context INT          Context length in tokens  [default: 16384]
  --kv-dtype TEXT        fp16 | bf16 | fp8  [default: fp16]
  --sequences INT        Concurrent sequences  [default: 1]
  --overhead SIZE        Runtime overhead reservation  [default: 2GiB]
  --vision-line SIZE     Measured vision line, subtracted when the
                         card claims vision (ADR-0030)  [default: none]
  --model-config PATH    Model config.json to derive the shape from
  --attn-layers INT      Attention layer count (manual shape)
  --kv-heads INT         KV heads per layer (manual shape)
  --head-dim INT         Head dimension (manual shape)
```

Exits 1 when nothing is left for weights, and 2 on conflicting or missing
shape sources or a `--vision-line` with the manual shape triple.

```console
$ vramfit budget --model-config config.json --vram 24GiB --kv-dtype fp8
attention layers      49  (KV grows 100352 bytes/token, fp8)
VRAM total            24.00 GiB
- KV cache            1.53 GiB  (16384 tokens x 1 seq)
- runtime overhead    2.00 GiB
vision                none claimed — nothing subtracted
= weight budget       20.47 GiB
```

On a mixed sliding/global stack the first line adds the saturated
window pool, e.g. `(KV grows 81920 bytes/token, fp16, + 1.17 GiB
window pool per sequence)`. Each concurrent sequence pays its own
pool. The KV-cache line sums both terms at the given context and
`--sequences`.

The ledger subtracts `--vision-line` only when the model card
claims vision — a top-level `vision_config` object in
`--model-config`
([ADR-0030](../adr/0030-vision-budget-sidecar.md) decision 3). The
line is a serving measurement, 1,600 MiB on the Gemma 4 31B target,
never the mmproj file size. A card that claims no vision subtracts
nothing and states the absence, as above — a supplied
`--vision-line` does not apply there, and the note says so. The
manual shape triple
carries no card, so it admits no `--vision-line`. A vision-claiming
card with no supplied line subtracts nothing and states the gap —
whether that case should warn or refuse is ADR-0030's open
question.

## `vramfit capacity`

Implemented. Prints the capacity readout for a packed recipe (#422):
the budget ledger run in reverse. The KV headroom is the card minus
the recipe's predicted weight bytes, minus `--overhead`, minus the
`--vision-line` the card's claim licenses. The command
solves the headroom against the per-layer KV arithmetic itself.
Sliding terms saturate while global terms grow, so the readout
stays exact on a mixed stack. The attention shape comes from
exactly one source, as in `budget`. `--vram` defaults to the
VRAM budget the recipe records. The weights line uses the recipe's
predicted bytes — the packed file can exceed them (#307), and
`vramfit pack` re-checks the real bytes.

```
vramfit capacity RECIPE
  --vram SIZE            Total VRAM  [default: the recipe's record]
  --context INT          Fixed context — adds the sequence-capacity line
  --kv-dtype TEXT        fp16 | bf16 | fp8  [default: fp16]
  --sequences INT        Concurrent sequences for the context line  [default: 1]
  --tokens-per-image INT Measured image token cost — adds the image-capacity line
  --overhead SIZE        Runtime overhead reservation  [default: 2GiB]
  --vision-line SIZE     Measured vision line, subtracted when the
                         card claims vision (ADR-0030)  [default: none]
  --model-config PATH    Model config.json to derive the shape from
  --attn-layers INT      Attention layer count (manual shape)
  --kv-heads INT         KV heads per layer (manual shape)
  --head-dim INT         Head dimension (manual shape)
```

Exits 1 when the recipe leaves nothing for KV cache or the recipe
or config cannot be read, and 2 on conflicting or missing shape
sources or a `--vision-line` with the manual shape triple.

```console
$ vramfit capacity recipe.json --model-config config.json \
    --context 32768 --tokens-per-image 256
attention layers      60  (KV grows 81920 bytes/token, fp16, + 1.17 GiB window pool per sequence)
VRAM total            24.00 GiB
- weights (recipe)    13.33 GiB
- runtime overhead    2.00 GiB
vision                claimed — no --vision-line supplied, nothing subtracted
= KV headroom         8.67 GiB
max context           98304 tokens  (1 sequence)
max sequences         2  (at 32768 tokens)
image capacity        384 images  (256 tokens per image, 1 sequence)
```

The context and image lines print `unbounded` when the KV cache
stops growing inside the headroom — an all-sliding stack past its
padded windows. The reading is then not KV-limited. Both lines read per
the `--sequences` split and say so. The sequence line goes
`unbounded` only for a shape that allocates no KV, which no
admitted config produces. The image line divides the context
capacity by the `--tokens-per-image` cost the caller supplies.
The caller takes that cost from a measurement, not a config claim
([ADR-0030](../adr/0030-vision-budget-sidecar.md) decision 4) —
256 tokens per 768×768 image on the Gemma 4 31B target, against
the config's 280.
ADR-0030 rules the multimodal VRAM ledger, and its 2026-08-29
amendment records the measured vision-quality bound.

`--vision-line` follows the `budget` rules (ADR-0030 decision 3).
The headroom subtracts the line only when the card claims vision.
A card that claims no vision subtracts nothing, and the ledger
states the absence. The example above shows the vision-claiming
card with no measured line supplied.

## `vramfit plan`

Implemented. Solves a sensitivity map into a recipe under a VRAM budget.

```
vramfit plan SENSITIVITY_MAP
  --vram SIZE            Hard VRAM ceiling (e.g. 24GiB)  [required]
  --checkpoint PATH      Checkpoint the map was scanned from. Its
                         safetensors headers price every group
                         (ADR-0029)
  --kv-headroom SIZE     Reserved for KV cache + runtime  [default: 4GiB]
  --pin TEXT             Pin groups to a precision, repeatable (glob=bits)
  --protect TEXT         Hold tensors at a precision floor inside
                         their groups, repeatable (glob=bits)
  --exclude-imatrix TEXT Quantize matched protected tensors without
                         their imatrix rows, repeatable (glob)
  --out PATH             Output recipe  [default: recipe.json]
  --runtime TEXT         Target runtime the recipe is planned for
                         [default: llama.cpp]
  --format-overhead F    Overhead fraction on top of the size model
                         [default: 0.005 with an effective-bits
                         table, 0.05 without]
```

Size predictions follow
[ADR-0014](../adr/0014-per-type-effective-bits.md): a runtime with an
effective-bits table prices each precision at its real per-weight
cost (llama.cpp: Q4_K spends 4.5 bits, not 4), and the overhead
fraction covers only unquantized tensors and file metadata. A
runtime without a table (vLLM, or `--runtime` omitted via the API)
keeps the nominal-bits prediction and the 0.05 scalar. A
routed-expert-stack group prices through the expert-stack type
table instead ([ADR-0028](../adr/0028-expert-stack-type-table.md)):
2.25 bits at nominal 2, not Q2_K's 2.625. A stack precision without
a table row (3, 5, 6) keeps its dense entry — pack refuses it, and
the plan-time refusal stays an open question in ADR-0028.

Size source ([ADR-0029](../adr/0029-plan-independent-size-source.md)):
`--checkpoint` reads each safetensors shard header, which is a JSON
parse and needs no torch. It sums the tensors into the groups the map
names. The checkpoint roots at `backbone.` and the maps root at
`model.`, so a domain table reconciles the two. The table is explicit
and carries no prefix wildcard. A checkpoint rooted at neither name
refuses, rather than pricing one stack against another (#177). The MTP
block stays out, because a GGUF numbers one layer stack and backbone
and MTP cannot pack together.

A group the checkpoint holds and the map does not measure is
*uncovered*. It prices at reference precision, and the recipe assigns
it there at nominal 16, the F16 passthrough. Both halves matter.
`pack` runs `llama-quantize --pure` at the recipe's precision floor. So
a group the recipe leaves unnamed reaches the artifact at that floor,
and not at the reference bytes the plan reserved.

An uncovered group carries no damage curve, so the solver never
downgrades it. `--pin` reaches it (the 2026-08-22
[ADR-0007](../adr/0007-recipe-solver-strategy.md) amendment): a
pinned uncovered group prices at the pinned width instead of holding
at reference, and it never enters the downgrade loop.

The command refuses a target runtime that cannot serve reference
precision, when any uncovered group holds at reference. It refuses a
checkpoint carrying none of the map's groups. It warns and continues
when the checkpoint carries only some of them.

`pack` maps decoder-layer groups, routed-expert stacks, and the
layer classes in the
[ADR-0012](../adr/0012-gguf-type-mapping.md) class table, as amended
2026-08-20. It refuses every other `stack` group by name. A group of
an unquantizable class — the router's `mixer.gate`, the Mamba
`mixer.conv1d` — packs at the F16 passthrough, and `pack` refuses a
recipe that assigns it a lower width.

Pin semantics: patterns are case-sensitive `fnmatch` globs matched
against the full group name (`--pin "model.layers.0.*=8"`). With
`--checkpoint` the match universe is every discovered group, and
without it the map's groups (the 2026-08-22 ADR-0007 amendment). A
pin may name any width the target runtime serves, beyond the map's
candidates, and a width the map never measured records 0.0 damage. A
pattern that matches no group is an error (typo detection). Later
pins override earlier ones for overlapping groups — repeating a
pattern moves it to the last position. Pins are recorded in the
recipe in their effective order.

Protection semantics
([ADR-0022](../adr/0022-within-layer-protections.md)): patterns are
the same `fnmatch` language, matched against full *tensor* names
(`--protect "*.self_attn.v_proj.weight=5"`). A protected tensor
packs at the higher of its group's assignment and the floor, priced
by size only — predicted damage stays the group-level sum. The
command rejects a floor the target runtime cannot serve, a pattern
that matches no tensor, a protection on a single-tensor group (use
`--pin` — the embedding and output head hold one tensor each), and a
map without per-tensor sizes. A rule whose floor every matched
group's assignment already meets draws a no-op warning. The recipe
records the rules verbatim and the resolved (tensor, precision)
pairs. A pair exists only where the floor exceeds the group's
assignment (issue #59). A matched tensor whose assignment already
meets the floor draws a per-tensor warning, and the recipe drops
that pair — it would quantize identically to the unprotected
reference and falsely fail the reconstruction check. A dropped
pair's `--exclude-imatrix` mark drops with it, and the warning says
so. An `--exclude-imatrix` pattern left with no surviving pair is
rejected — the pack would keep the imatrix rows the pattern names.
A glob that also matches unprotected tensors draws a warning
naming what it did not cover. The unprotected rows stay (ADR-0023).

Exit codes: 1 when the map is invalid, the output is unwritable, no
recipe fits the budget (the gap is reported), or a `--protect` or
`--exclude-imatrix` rule is rejected. Exit 2 on malformed options (`--pin` or `--protect` not of
the form `pattern=bits` with positive bits, unparseable sizes, a
negative, NaN, or infinite `--format-overhead`, or a `--runtime`
outside the capability table).

## `vramfit scan`

Implemented. Measures per-group damage and writes a sensitivity map.
Requires the scan extra (`uv pip install "vramfit[scan]"`) — without
it the command exits 1 with the install hint.

```
vramfit scan MODEL
  --calibration PATH     Calibration text file (UTF-8)  [required]
  --out PATH             Output sensitivity map  [default: sensitivity.json]
  --precisions TEXT      Candidate bit-widths, strictly descending CSV,
                         2-bit floor, default per ADR-0010
                         [default: 8,4,3,2]
  --group-by TEXT        Grouping granularity (layer | tensor | stack)  [default: layer]
  --groups TEXT          Restrict the run to these group names (CSV).
                         Names must be keys --group-by produces
                         [default: every discovered group]
  --max-tokens INT       Calibration token budget  [default: 131072]
  --device TEXT          Device map: auto | cpu | cuda  [default: auto]
  --trust-remote-code    Allow model repos with custom code (the
                         north-star target needs this)
  --resume / --no-resume Continue from the checkpoint file  [default: resume]
  --within-group TEXT    Within-group method: rtn | kquant | q0
                         (ADR-0018). kquant prices cells with the
                         ported K-quant reference quantizers (8, 4,
                         3, 2). q0 prices them with the ported
                         block quantizers Q2_0, Q4_0, and Q8_0 (8,
                         4, 2), which reach the rows no K-quant
                         tiles. Each pairs only with precisions its
                         port covers  [default: rtn]
  --imatrix PATH         GGUF imatrix for assisted K-quant pricing
                         (ADR-0020). Requires --within-group kquant.
                         Use the file the pack step will consume
                         [default: none]
  --runlog PATH          Run-log path (JSONL)
                         [default: <stem>.runlog.jsonl]
  --gpu-memory SIZE      Byte cap on GPU 0 model shards (e.g. 17GiB),
                         parsed with the same grammar as --vram.
                         Requires --device auto. Keeps workspace free
                         for activations and quantization
                         [default: none]
```

Every run appends machine-readable events to a run log
(`<stem>.runlog.jsonl`, ADR-0011): scan_started, meter_built,
resume_loaded, one cell_measured per cell with damage, seconds, and
the RSS high-water mark, then scan_finished or scan_halted. Every
finished (group x precision) cell lands in a checkpoint file next
to `--out` (`<stem>.checkpoint.json`). A rerun of the same scan resumes
from it. The checkpoint carries the scan's fingerprint (model, metric,
calibration, token count, grouping, precisions, method, imatrix
path) — a rerun with
any of those changed refuses the checkpoint instead of mixing numbers.
The fingerprint identifies provenance, not content: do not swap weights
or calibration text under an unchanged path between resumes.
`--no-resume` deletes the checkpoint first and says so.

`--groups` restricts the run to named groups. A caller that wants 46 of
210 groups pays for 46. The map then carries the selected groups alone.
A name that matches no discovered group halts the run. The halt runs
after the model loads and before any cell measures. It names every
unmatched name at once and records the stage `group_select`.

The selection stays out of the fingerprint, because a group subset is
not provenance. So a narrow run and a wide run share one checkpoint on
purpose. A narrowed run reuses the wide run's cells for the groups it
selects. The cells in deselected groups stay in the file, and the run
reports how many it ignored.

The checkpoint still validates against the whole model first. A cell
outside the full grid, or a cell that repeats, halts the run whatever
the selection. A selection narrows what a run measures and never what
it checks.

A narrowed map measures a subset of the model. `vramfit plan
--checkpoint` prices the rest: it reads the checkpoint's safetensors
headers and holds every unmeasured group at reference precision
(ADR-0029). Without that option the map still defines the model, and
the command says so.

Groups that `auto` sharding offloads to host RAM measure through
accelerate's weights map (ADR-0015) — `meter_built` reports the count
as `offloaded_groups`. The scan refuses a model whose weights fall
beyond host RAM (disk spill) — an unperturbable weight would record
zero damage. Raise `--gpu-memory`, free host RAM, or use a smaller
model.

With `--imatrix` the kquant fit weighs every covered tensor with the
pack's importance matrix (assisted pricing, ADR-0020). The map records
the token `kquant-imx` and the imatrix path in `scan.imatrix`. The
command echoes the coverage split, and `meter_built` records it as
`imatrix_covered` and `imatrix_uncovered` — uncovered tensors price
unassisted, the `llama-quantize` fallback (`token_embd` is the
expected miss). A covered tensor whose rows do not divide into
256-element super-blocks joins the uncovered set instead of refusing
the scan. A scan is only comparable to a pack that consumes the
same imatrix file — the CLI resolves the path, and the map records
the resolved spelling. An assisted scan also reads each fused
expert stack's count vector and pools each group's vectors into the
`imatrix_counts` count minimum, median, and maximum
([ADR-0026](../adr/0026-moe-expert-pricing.md) decision 4). A group
without a resolved expert stack records none — see the
[sensitivity map format](../reference/sensitivity-map.md).

Exit codes: 1 when the scan extra is missing, the model or calibration
cannot load, sharding offloaded a quantizable group beyond host RAM,
a `--groups` name matches no discovered group, the checkpoint
belongs to a different scan, a measurement fails (the checkpoint keeps
completed cells), a checkpoint write fails, or the map cannot be
written. Exit 2 on malformed `--precisions`, `--group-by`, `--groups`,
`--within-group`, or `--gpu-memory`, a `--gpu-memory` without
`--device auto`, a `--within-group kquant` or `q0` combined with
precisions the port does not cover, an `--imatrix` without
`--within-group kquant` or naming a missing file, or a missing
`--out` or `--runlog` directory.

A `kquant` scan also refuses a cell whose mapped type cannot tile the
tensor's rows, and the message names the parameter, the type, the
block size, and the row length. `Q2_K`, `Q3_K`, and `Q4_K` block 256
elements, which divides neither 2688 nor 1856 — the routed-expert row
lengths on the 30B target. `llama-quantize` never applies those types
there. `tensor_type_fallback` warns and substitutes `Q4_0` or `Q5_0`,
so pricing the requested type records a frame the pack cannot apply.
`Q8_0` blocks 32 elements and reaches both rows, so nominal 8 never
refuses. Use `--within-group q0` for those rows.

## `vramfit validate`

Implemented. Runs the whole-recipe validation pass (ADR-0006). The
command quantizes every group to its assigned precision in one
calibration pass. The pass uses the scan's own quantization,
selected with `--within-group` and `--imatrix`. The pass only checks
additivity when its frame matches the map that priced the recipe
(ADR-0019) — a recipe that records its map's method resolves the
frame by itself, and the command refuses flags that contradict the
record. Recipes without the record leave the pairing to the caller,
with a warning. The
command reports the measured damage next to the recipe's summed
marginal damages. The gap is the additivity assumption leaking.
Requires the scan extra — without it the command exits 1 with the
install hint.

```
vramfit validate RECIPE
  --calibration PATH     Calibration text file (UTF-8)  [required]
  --model TEXT           Model id or checkpoint path
                         [default: the recipe's model_id]
  --max-tokens INT       Calibration token budget  [default: 131072]
  --group-by TEXT        Grouping granularity (layer | tensor | stack)  [default: layer]
  --device TEXT          Device map: auto | cpu | cuda  [default: auto]
  --trust-remote-code    Allow model repos with custom code
  --gpu-memory SIZE      Byte cap on GPU 0 model shards (e.g. 17GiB).
                         Requires --device auto  [default: none]
  --within-group TEXT    Within-group method: rtn | kquant | q0
                         (ADR-0018)  [default: the recipe's recorded
                         method, or rtn without a record]
  --imatrix PATH         GGUF imatrix for assisted K-quant measurement
                         (ADR-0020). Required when the recipe was
                         priced on an assisted map — use the map's
                         imatrix file  [default: none]
  --runlog PATH          Run-log path (JSONL)
                         [default: <recipe stem>.validation.runlog.jsonl]
```

With offloaded groups, the whole-recipe pass restores their originals
from the model's safetensors shards (ADR-0015), so `--model` must
point at a local safetensors directory — a bare hub id is refused
before any weight changes.

Use the scan's calibration file and token budget — damage values are
only comparable within one calibration set. The command refuses a
recipe whose groups do not match the model's discovered groups (wrong
model or wrong `--group-by`). A `--model` that differs from the
recipe's `model_id` prints a warning — the comparison assumes the
scanned model. An `--imatrix` that differs from the recipe's recorded
imatrix path also prints a warning — a different file contaminates
the additivity comparison. The command echoes the imatrix coverage
split like the scan does. The command reports the gap and does not
gate on it: the invalidation threshold is an open question in
ADR-0006 until measured gaps exist.

Every run appends events to a run log: validation_started,
meter_built, then validation_finished with predicted_damage,
measured_damage, gap, and ratio — or validation_halted (stage:
meter_build, group_match, or measure).

```console
$ vramfit validate recipe.json --calibration calib.txt --max-tokens 32768
validated 37 groups over 32768 tokens
summed marginal damage (predicted)  0.066107
whole-recipe damage (measured)      0.032240
gap -0.033867 (-51.2 % of predicted)
```

Exit codes: 1 when the recipe is invalid, the scan extra is missing,
the model or calibration cannot load, the recipe's groups do not match
the model's, or the measurement fails. Exit 2 on a malformed
`--group-by`, `--within-group`, or `--gpu-memory`, a `--gpu-memory`
without `--device auto`, an `--imatrix` without the kquant method or
naming a missing file, a frame that contradicts the recipe's
recorded method, a `--within-group kquant` or `q0` that meets
recipe assignments the ported quantizers do not cover, or a missing
`--runlog` directory.

## `vramfit pack`

Implemented for the GGUF backend (ADR-0010, ADR-0012). Applies a
recipe through llama.cpp's quantizer: one f16 base GGUF conversion
(reused when present), then `llama-quantize` with one type override
per layer group. The embedding assignment binds
`--token-embedding-type`. An `lm_head` group binds
`--output-tensor-type` with its own assignment — without one, the
embedding assignment pins an untied head (ADR-0012 as amended). The
base type is the recipe's precision floor, applied with `--pure`, so
no heuristic mixing leaks in.

A routed-expert-stack group maps through its own type table
([ADR-0028](../adr/0028-expert-stack-type-table.md)): 8 to `Q8_0`, 4
to `Q4_0`, 2 to `Q2_0`. K-quant super-blocks do not divide the
stack rows, so the dense table cannot reach them. The backend
refuses every stack precision without a table row. Nominal 3 draws
the dedicated refusal: it names the group, the empty 2.25–4.25
bits-per-weight gap, and both neighboring table entries. Nominal 6
and 5 draw the table-bounds refusal — whether the table gains those
rows is #232's question.

```
vramfit pack RECIPE
  --llama-cpp PATH       llama.cpp checkout with convert_hf_to_gguf.py
                         and build/bin/llama-quantize  [required]
  --model PATH           Model checkpoint directory
                         [default: the recipe's model_id]
  --out PATH             Packed model path  [default: packed.gguf]
  --base-gguf PATH       f16 base GGUF, reused when present
                         [default: <model name>-f16.gguf beside --out]
  --python-bin PATH      Interpreter for the convert script — install
                         vramfit[pack] to provision it
                         [default: current]
  --threads INT          Thread count for the quantizer and the
                         smoke test  [default: 8]
  --imatrix PATH         Importance matrix for the quantizer
                         (ADR-0016)  [default: none]
  --mmproj PATH          Vendor mmproj shipped beside --out as the
                         projector sidecar (ADR-0030)  [default: none]
  --smoke-text PATH      Text for the post-pack smoke test (ADR-0017)
                         [default: none — pack warns]
  --smoke-chunks INT     Smoke-test chunk count  [default: 2]
  --smoke-threshold F    Perplexity ceiling a passing smoke test
                         stays under  [default: 1000]
  --runlog PATH          Run-log path (JSONL)
                         [default: <stem>.runlog.jsonl]
```

A recipe priced on an assisted map records its imatrix — the command
warns when `--imatrix` is absent or names a different file, because
the pack would not match the map's frame (ADR-0020). A warning, not
a refusal: packing itself works either way.

Every pack holds the recipe's overrides against the base GGUF's
tensor names before the quantizer runs. An override no tensor matches
refuses the pack (#303).

That read opens one file. The pack step refuses a base GGUF that
declares itself one shard of a split file, before the override check
runs. The message
names the shard, the shard count, and `llama-gguf-split --merge`. A
shard carries part of the model, so holding a whole recipe against it
reports a correct recipe as wrong (#308). `llama-quantize` follows
the chain from a first shard, so vramfit accepts less than the tool
at that one input. Every later shard the tool refuses itself. Issue
#351 carries whether the read should follow the chain too.

The same read holds the two dedicated flags against the exact tensors
they bind. `--token-embedding-type` reaches `token_embd.weight` or
`per_layer_token_embd.weight`, and `--output-tensor-type` reaches
`output.weight`. The base GGUF may carry none of a flag's target
tensors. The flag then binds nothing and the quantizer exits 0, so
that tensor takes the floor while the record states the recipe's
type. The command refuses a recipe whose `lm_head` group meets a base
GGUF with no `output.weight`. It refuses an embedding group the same
way, against a file carrying neither embedding tensor (#306).

A recipe with no `lm_head` group does not refuse. Its output flag
carries the embedding's assignment, and ADR-0012 decision 2 rules that
flag a no-op on a model that ties embeddings. The head is the
embedding tensor there, which the embedding flag already typed.

A layer the file numbers that no override
reaches does not refuse: it packs at the base-type floor, which
ADR-0012 decision 3 describes. The quantizer prints nothing for it,
so a console warning states the count and names the layers, and the
`model_packed` event carries them under `floored_layers` — a report,
never a gate (#307). Those layers carry no assignment, so the packed
file exceeds `plan.predicted_total_bytes` by their cost. Issue #320
carries whether the case should refuse instead.

An `--imatrix` pack also reads the matrix's `.counts` tensors
against the base GGUF, between the convert and quantize stages
([ADR-0026](../adr/0026-moe-expert-pricing.md) decision 5). The
quantizer fills a zero-count expert's row with ones and prints no
warning, so only this read finds the case. A console warning states
the zero-count pair total (#226). Each pair lands in the
`model_packed` event under `imatrix_zero_count_experts`, as
two-element `[stack, expert]` arrays — a report, never a gate. A matrix the reader cannot vouch
for halts the pack before the quantizer runs. The refusals form a
closed list: not an imatrix, no counts, an unknown tensor suffix,
a sums tensor without its counts twin, a count that is negative or
not finite, or a count length that contradicts the base tensor.
The read needs gguf-py, which the pack extra provisions — a
matrix-less pack touches neither the library nor the file.

A protected recipe drives its resolved (tensor, precision) pairs as
extra overrides, placed *before* the group overrides — the quantizer
applies the first matching pattern
([ADR-0022](../adr/0022-within-layer-protections.md)).

The command scans the quantizer's merged output for the
type-fallback warning pair
([ADR-0028](../adr/0028-expert-stack-type-table.md) decision 3). A
match means the quantizer substituted a type on a zero exit, so the
packed file no longer carries the recipe. The pack halts with exit
1 and keeps the file. The `pack_halted` event carries stage
`type_fallback`, every rewritten tensor, and the requested and
substituted types. The ADR-0016 imatrix-miss scan records and
continues — this scan halts.

The imatrix-miss scan halts on one input. A miss whose tensor name
carries U+FFFD means `run_tool` could not decode that name, so the
command refuses to record it as coverage (#252). The halt reports
stage `quantize`, keeps the packed file, and names every
undecodable tensor. A pack without `--imatrix` runs no such scan.

After quantizing, the command re-checks the packed file's real bytes
against `plan.weight_budget_bytes` — nominal-bit predictions
undershoot GGUF's effective bits (ADR-0012). On a protected pack made
with `--imatrix`, the reconstruction check then runs, mandatory
([ADR-0022](../adr/0022-within-layer-protections.md)): the command
packs the same recipe with its protections stripped as the
reference, dequantizes every protected tensor from both files with
gguf-py, and compares each against the f16 base. The reference file
is deleted after measurement. A tensor that does not reconstruct
strictly closer to f16 than its unprotected type is collapsed — the
command names it, keeps the file, and exits 1. The remedy is the
user's: re-plan with `--exclude-imatrix` for the named tensors
(ADR-0023) — the refusal prints the exact flags. A
protected pack without `--imatrix` skips the check with a note —
every known fit collapse involved a promotion under one. A recipe
that records protections but resolved no pairs also skips with a
note: every floor was a per-tensor no-op at plan time (issue #59).

A recipe with imatrix exclusions drives one `--exclude-weights`
flag per marked pair when the pack runs with `--imatrix`
([ADR-0023](../adr/0023-imatrix-exclusions.md)). The excluded
tensors keep their promoted types and quantize without their
imatrix rows. The command reads the matrix's entry names first and
refuses an exclusion that reaches none of them. The quantizer erases
a row by substring, so such a name erases nothing and exits 0. The
tensor would keep the fit the recipe asked to drop, and the record
would state an exclusion that never applied. Packing such a recipe
without `--imatrix` warns that the exclusions change nothing.

`--mmproj` ships the vendor mmproj beside `--out` as the
unquantized projector sidecar
([ADR-0030](../adr/0030-vision-budget-sidecar.md) decision 2). The
stage runs after the size check and the reconstruction gate pass:
a byte-identical copy under
the vendor file name, proven by SHA-256 of the source and the copy.
A stale file at the destination is replaced. A symlink there
refuses, and a mismatch removes the copy — each halts with the
decoder kept. The `sidecar_shipped` event
carries the path, the bytes, and the digest. The sidecar never
enters the weight budget — the vision line is a serving
measurement, not the file size (decision 3). The command refuses an
`--mmproj` whose copy would land on a run-owned path — the decoder,
the base GGUF, the run log, or the reconstruction reference —
before any tool runs. An empty `--mmproj` refuses the same way.

With `--smoke-text` the command runs the smoke test: `--smoke-chunks` perplexity chunks through
`build/bin/llama-perplexity`, gated by the `--smoke-threshold`
ceiling (ADR-0017). Without the flag the command warns that the
packed model is unproven. Every run appends the pack events to the
run log: pack_started, gguf_converted (with `reused`), model_packed
(real bytes, base type, embedding and output tensor types, override
count, imatrix, uncovered tensors, excluded tensors, zero-count
experts, floored layers), size_checked (margin and
`fits`), reconstruction_checked when the gate ran
(per-tensor
protected and reference RMSE, `collapsed`, `passed`),
sidecar_shipped when `--mmproj` shipped (mmproj, path,
bytes, sha256), smoke_tested
when the smoke test ran (perplexity — null
when non-finite, with a text copy — threshold, chunks, `passed`),
then pack_finished (with `smoked`) or pack_halted (stage:
convert, imatrix_counts, quantize, type_fallback, size_check,
reconstruction, sidecar, or smoke).

Exit codes: 1 when the recipe is invalid, the model directory does
not exist, a toolchain stage fails, the quantizer substitutes a
type (the file is kept), the packed model exceeds the
weight budget, the reconstruction check finds a collapsed tensor
(the file is kept), the sidecar copy fails or does not match its
source, or the smoke test fails (the file is kept).
Exit 2 when the llama.cpp checkout misses a needed tool,
`--imatrix`, `--mmproj`, or `--smoke-text` is not a file,
`--mmproj` is empty or its copy would land on a run-owned path,
`--smoke-threshold` is
not positive, or the `--out`/`--runlog` directory does not exist.
