# Changelog

## [0.5.0](https://github.com/Alberto-Codes/vramfit/compare/v0.4.0...v0.5.0) (2026-09-05)


### Features

* **adapters:** parse hybrid_override_pattern into block types instead of refusing ([#541](https://github.com/Alberto-Codes/vramfit/issues/541)) ([3e18dae](https://github.com/Alberto-Codes/vramfit/commit/3e18dae4872d55e7c457e1b5af096e78e899acc2))
* **docs:** add the card upload-and-verify script and align the reproduce blocks ([#538](https://github.com/Alberto-Codes/vramfit/issues/538)) ([80c2306](https://github.com/Alberto-Codes/vramfit/commit/80c2306b7e34fb9bd1e3754b8a1531feb9f6dcae))
* **pack:** add a thin gguf extra and fold test-gguf into the test job ([eca8eaf](https://github.com/Alberto-Codes/vramfit/commit/eca8eaf237f49121c2d7dbe0468de0a8e50ec507)), closes [#310](https://github.com/Alberto-Codes/vramfit/issues/310) [#311](https://github.com/Alberto-Codes/vramfit/issues/311)
* **pack:** add the 5- and 6-bit rows to the expert-stack type table ([#534](https://github.com/Alberto-Codes/vramfit/issues/534)) ([0090020](https://github.com/Alberto-Codes/vramfit/commit/0090020f4eb51f498e45bf3ae7f42b73db602b73))
* **pack:** compare packed bytes against the recipe's prediction at pack time ([#543](https://github.com/Alberto-Codes/vramfit/issues/543)) ([818ac32](https://github.com/Alberto-Codes/vramfit/commit/818ac327efa71eb5a10e1d285200f5030d372de0))


### Bug Fixes

* **adapters:** apply the reader's integer bound to the backfill write ([#476](https://github.com/Alberto-Codes/vramfit/issues/476)) ([d6301da](https://github.com/Alberto-Codes/vramfit/commit/d6301da3ab5c2e036f2ef9cfd454354df6a9eb7a))
* **adapters:** close the run-log handle per emit and raise RunLogError for a refused field ([#522](https://github.com/Alberto-Codes/vramfit/issues/522)) ([d8d3806](https://github.com/Alberto-Codes/vramfit/commit/d8d3806f003de5ea0ff84d73d53db145d0c07950))
* **adapters:** halt on repeated group names before the checkpoint loads ([#471](https://github.com/Alberto-Codes/vramfit/issues/471)) ([4058cee](https://github.com/Alberto-Codes/vramfit/commit/4058ceef879d387c7b242e934a0d42feb07da29c))
* **adapters:** name the map when the backfill refuses --out over it ([#479](https://github.com/Alberto-Codes/vramfit/issues/479)) ([18c398b](https://github.com/Alberto-Codes/vramfit/commit/18c398bee9b2ba37d9e5efb2059da4d873ad148e))
* **adapters:** name the shard index for every parse refusal ([#477](https://github.com/Alberto-Codes/vramfit/issues/477)) ([7dfcce8](https://github.com/Alberto-Codes/vramfit/commit/7dfcce8d6622131807eefb087cb87ff3ed0b3492))
* **adapters:** raise HfConfigError under the VramfitError root from the HF config readers ([#523](https://github.com/Alberto-Codes/vramfit/issues/523)) ([511a8af](https://github.com/Alberto-Codes/vramfit/commit/511a8af6be2ce8910297eb48c1eeb76c3178d4be))
* **adapters:** raise run-log refusals under the VramfitError root ([#472](https://github.com/Alberto-Codes/vramfit/issues/472)) ([a585026](https://github.com/Alberto-Codes/vramfit/commit/a58502689fb5356514360ef11095b57c9bd0c38d))
* **cli:** plain-text the cli_shape module name in budget and plan help ([#490](https://github.com/Alberto-Codes/vramfit/issues/490)) ([#520](https://github.com/Alberto-Codes/vramfit/issues/520)) ([350fbe6](https://github.com/Alberto-Codes/vramfit/commit/350fbe6741b7bff11bbc8fc73f393635512fa19d))
* **pack:** refuse a recipe rooted outside the scan name table ([#208](https://github.com/Alberto-Codes/vramfit/issues/208)) ([#536](https://github.com/Alberto-Codes/vramfit/issues/536)) ([0b4a291](https://github.com/Alberto-Codes/vramfit/commit/0b4a2914a98dcfd5e0817b8296e8441a60767336))
* **plan:** price the F16 passthrough from the convert dtype and skip unquantizable classes at scan ([#533](https://github.com/Alberto-Codes/vramfit/issues/533)) ([95bd4ea](https://github.com/Alberto-Codes/vramfit/commit/95bd4ea9a4361c52282e5ba98b3eef50dbca969e))
* **plan:** skip held groups under a multi-group pin pattern and warn ([#535](https://github.com/Alberto-Codes/vramfit/issues/535)) ([04960a2](https://github.com/Alberto-Codes/vramfit/commit/04960a2412a3126d21788de8ad84a98ff152663d))
* **plan:** warn when a pre-skip layer map folds a class the checkpoint holds by itself ([#542](https://github.com/Alberto-Codes/vramfit/issues/542)) ([4206db4](https://github.com/Alberto-Codes/vramfit/commit/4206db402b2da8c55efd085bd1b0b71900166c56))

## [0.4.0](https://github.com/Alberto-Codes/vramfit/compare/v0.3.0...v0.4.0) (2026-09-02)


### Features

* **cli:** build the ADR-0030 vision clauses ([#443](https://github.com/Alberto-Codes/vramfit/issues/443)) ([2638a3d](https://github.com/Alberto-Codes/vramfit/commit/2638a3d2d75a738533cba9720771ae20bd950493)), closes [#442](https://github.com/Alberto-Codes/vramfit/issues/442)
* **domain:** represent heterogeneous KV geometry in ModelShape ([#428](https://github.com/Alberto-Codes/vramfit/issues/428)) ([9e512f6](https://github.com/Alberto-Codes/vramfit/commit/9e512f614eafbf40ac9660c294f91f2b716856c5)), closes [#421](https://github.com/Alberto-Codes/vramfit/issues/421) [#426](https://github.com/Alberto-Codes/vramfit/issues/426)
* **domain:** solve capacity readouts from a packed recipe ([#430](https://github.com/Alberto-Codes/vramfit/issues/430)) ([da15fe3](https://github.com/Alberto-Codes/vramfit/commit/da15fe3681d7def7eaff6e3d65e989f11544f394))
* **scan:** frame calibration text for channel-locked targets ([#437](https://github.com/Alberto-Codes/vramfit/issues/437)) ([f4cbe2d](https://github.com/Alberto-Codes/vramfit/commit/f4cbe2d1ead2d41fed385c7fdb4c5f6b101dd094))


### Bug Fixes

* **adapters:** refuse discarded geometry and split the KV readers ([9e512f6](https://github.com/Alberto-Codes/vramfit/commit/9e512f614eafbf40ac9660c294f91f2b716856c5))
* **adapters:** select the nested text_config decoder in composite configs ([#425](https://github.com/Alberto-Codes/vramfit/issues/425)) ([c1a48eb](https://github.com/Alberto-Codes/vramfit/commit/c1a48ebc57340c4f7159ab8cb734b70cfefc7aaf))
* **domain:** price KV at the runtime's measured allocation ([#432](https://github.com/Alberto-Codes/vramfit/issues/432)) ([71f587a](https://github.com/Alberto-Codes/vramfit/commit/71f587a48be124cbca5c7b7c9bc996ce4ad27e78))
* **pack:** map the Gemma 4 multimodal embedding name ([#438](https://github.com/Alberto-Codes/vramfit/issues/438)) ([082a66b](https://github.com/Alberto-Codes/vramfit/commit/082a66b5022e15b599db4510b465c7be1ad99d46))
* **scan:** map the nested language_model decoder root to imatrix names ([#435](https://github.com/Alberto-Codes/vramfit/issues/435)) ([07a8379](https://github.com/Alberto-Codes/vramfit/commit/07a8379312588e8e66d3d776541d0c7605149aec))

## [0.3.0](https://github.com/Alberto-Codes/vramfit/compare/v0.2.0...v0.3.0) (2026-08-22)


### Features

* **domain:** build the spread placement rule ([#376](https://github.com/Alberto-Codes/vramfit/issues/376)) ([903aa5c](https://github.com/Alberto-Codes/vramfit/commit/903aa5c392ad9864bde379da16055958f49d91ea))
* **domain:** widen the pin surface to runtime widths and discovered groups ([#397](https://github.com/Alberto-Codes/vramfit/issues/397)) ([8d21570](https://github.com/Alberto-Codes/vramfit/commit/8d21570ff81e7658348341b65a5cedda7f0f1f9c))
* **pack:** map the Nemotron-H classes and pin unquantizable ones at F16 ([#370](https://github.com/Alberto-Codes/vramfit/issues/370)) ([f3a07ac](https://github.com/Alberto-Codes/vramfit/commit/f3a07ac25fd2736050b29e5229e305416a9878bb))
* **plan:** read tensor sizes from the checkpoint, not the map ([5bfc8b5](https://github.com/Alberto-Codes/vramfit/commit/5bfc8b5cbdf61d8af55bc38d3218ea298f1194ee)), closes [#358](https://github.com/Alberto-Codes/vramfit/issues/358)
* **scan:** add the gguf-ref within-group method and the kquant refusal ([#330](https://github.com/Alberto-Codes/vramfit/issues/330)) ([b619733](https://github.com/Alberto-Codes/vramfit/commit/b61973394c5cf4b5bfb61c559e284c6c73a4a161)), closes [#327](https://github.com/Alberto-Codes/vramfit/issues/327)
* **scan:** build the q0-imx assisted meter ([#385](https://github.com/Alberto-Codes/vramfit/issues/385)) ([74ac9d6](https://github.com/Alberto-Codes/vramfit/commit/74ac9d6bcdda24583caf14a3a021ecc2fc5fb6a6))
* **scan:** restrict a run to a subset of groups ([#336](https://github.com/Alberto-Codes/vramfit/issues/336)) ([60651a8](https://github.com/Alberto-Codes/vramfit/commit/60651a8bcea41f2a06ac4727ec5f7d7d63780494)), closes [#282](https://github.com/Alberto-Codes/vramfit/issues/282)


### Bug Fixes

* **adapters:** bound a model config integer to what the format carries ([#347](https://github.com/Alberto-Codes/vramfit/issues/347)) ([7c273f0](https://github.com/Alberto-Codes/vramfit/commit/7c273f021b97c72316fbe270efd57c72e926faa5)), closes [#314](https://github.com/Alberto-Codes/vramfit/issues/314) [#287](https://github.com/Alberto-Codes/vramfit/issues/287) [#348](https://github.com/Alberto-Codes/vramfit/issues/348)
* **adapters:** bound an artifact integer to what the format can carry ([#313](https://github.com/Alberto-Codes/vramfit/issues/313)) ([eee53a2](https://github.com/Alberto-Codes/vramfit/commit/eee53a20c6077fe8a663d363e5afdfedfa7d8803))
* **adapters:** map a protection under a free prefix ([#366](https://github.com/Alberto-Codes/vramfit/issues/366)) ([1b4d480](https://github.com/Alberto-Codes/vramfit/commit/1b4d480732ff12116aa7cf1c55e3aee2468bb78d)), closes [#365](https://github.com/Alberto-Codes/vramfit/issues/365)
* **adapters:** refuse a base GGUF that is one shard of a split file ([#352](https://github.com/Alberto-Codes/vramfit/issues/352)) ([047f1b6](https://github.com/Alberto-Codes/vramfit/commit/047f1b6b1efa8fa8078eeb99dab02776cdc3bd0d)), closes [#308](https://github.com/Alberto-Codes/vramfit/issues/308)
* **adapters:** refuse a dedicated flag that reaches no base-GGUF tensor ([#329](https://github.com/Alberto-Codes/vramfit/issues/329)) ([35f924a](https://github.com/Alberto-Codes/vramfit/commit/35f924a2ef9bec4f4f98aca237e5cf3bd1fa5f44))
* **adapters:** refuse a duplicate JSON key at the load step ([#281](https://github.com/Alberto-Codes/vramfit/issues/281)) ([4aed979](https://github.com/Alberto-Codes/vramfit/commit/4aed97916ef03332b8a116534a8ce37a84f8ad0d)), closes [#262](https://github.com/Alberto-Codes/vramfit/issues/262) [#283](https://github.com/Alberto-Codes/vramfit/issues/283)
* **adapters:** refuse a duplicate JSON key outside _load_json ([#285](https://github.com/Alberto-Codes/vramfit/issues/285)) ([c193263](https://github.com/Alberto-Codes/vramfit/commit/c193263d1021cb5d8b29dd63947f861613f98591)), closes [#283](https://github.com/Alberto-Codes/vramfit/issues/283) [#286](https://github.com/Alberto-Codes/vramfit/issues/286) [#287](https://github.com/Alberto-Codes/vramfit/issues/287)
* **adapters:** refuse a duplicate key in the backfill script ([#296](https://github.com/Alberto-Codes/vramfit/issues/296)) ([b36c178](https://github.com/Alberto-Codes/vramfit/commit/b36c1782eab368985f96b9b7b1a487ac806ce74a)), closes [#286](https://github.com/Alberto-Codes/vramfit/issues/286)
* **adapters:** refuse a non-map input and a repeated shard tensor ([#334](https://github.com/Alberto-Codes/vramfit/issues/334)) ([817bb77](https://github.com/Alberto-Codes/vramfit/commit/817bb77855e1c0c1018b61756b6b842dbc1a0b02))
* **adapters:** refuse a protection under a second root ([1b4d480](https://github.com/Alberto-Codes/vramfit/commit/1b4d480732ff12116aa7cf1c55e3aee2468bb78d))
* **adapters:** refuse an exclusion that reaches no imatrix row ([#323](https://github.com/Alberto-Codes/vramfit/issues/323)) ([5c5c872](https://github.com/Alberto-Codes/vramfit/commit/5c5c872e86989ba7806f28e19f8fbc39f46dbe38)), closes [#309](https://github.com/Alberto-Codes/vramfit/issues/309)
* **adapters:** refuse an imatrix-miss name the reader could not decode ([#272](https://github.com/Alberto-Codes/vramfit/issues/272)) ([7e2fee8](https://github.com/Alberto-Codes/vramfit/commit/7e2fee8bfa28ac1beb3ff1757da55e27957de94d)), closes [#252](https://github.com/Alberto-Codes/vramfit/issues/252)
* **adapters:** refuse an override that matches no base-GGUF tensor ([#304](https://github.com/Alberto-Codes/vramfit/issues/304)) ([8bb119c](https://github.com/Alberto-Codes/vramfit/commit/8bb119c7e9fa1eeeb789f4b04e68bc2b2e0a2901))
* **adapters:** report a base-GGUF layer no override reaches ([#318](https://github.com/Alberto-Codes/vramfit/issues/318)) ([022a4f8](https://github.com/Alberto-Codes/vramfit/commit/022a4f81cfd988ad9406e410d3539c52d9df52ba))
* **adapters:** report an artifact field the reader does not know ([#290](https://github.com/Alberto-Codes/vramfit/issues/290)) ([82eafa9](https://github.com/Alberto-Codes/vramfit/commit/82eafa98e91b1b931b64f18debb7b1757e0495f5))
* **adapters:** report every backfill refusal instead of a traceback ([#361](https://github.com/Alberto-Codes/vramfit/issues/361)) ([1dfb804](https://github.com/Alberto-Codes/vramfit/commit/1dfb804ce18d0e26757830274846a95c34860c64))
* **adapters:** scope the run-log final-line drop by kind ([#344](https://github.com/Alberto-Codes/vramfit/issues/344)) ([b001646](https://github.com/Alberto-Codes/vramfit/commit/b001646b194cbaabaec904ac50acba9535115ae2)), closes [#315](https://github.com/Alberto-Codes/vramfit/issues/315)
* **config:** contain the e2e run log to a temp directory ([#343](https://github.com/Alberto-Codes/vramfit/issues/343)) ([f012124](https://github.com/Alberto-Codes/vramfit/commit/f0121249fa0c1d0604b89e57087b922de7857b27))

## [0.2.0](https://github.com/Alberto-Codes/vramfit/compare/v0.1.0...v0.2.0) (2026-08-15)


### Features

* **adapters:** read an evals sidecar back through a port ([#259](https://github.com/Alberto-Codes/vramfit/issues/259)) ([a68390b](https://github.com/Alberto-Codes/vramfit/commit/a68390bbb716dd652f4fbf1951bc04006e774715))
* **cli:** state coverage counts in the pack echo, not every name ([#226](https://github.com/Alberto-Codes/vramfit/issues/226)) ([9d4401a](https://github.com/Alberto-Codes/vramfit/commit/9d4401af700208d9f9a334d1a773578644284479))
* **cli:** state the uncovered imatrix count instead of every name ([#224](https://github.com/Alberto-Codes/vramfit/issues/224)) ([b8ae709](https://github.com/Alberto-Codes/vramfit/commit/b8ae7098b96b0518ec50732fcecc78fdb231d8da))
* **config:** match guard rules on tokens rather than raw text ([04e544b](https://github.com/Alberto-Codes/vramfit/commit/04e544b25746a25dce755461b1e7d24f28918f55))
* **config:** route each guard rule by whether its check is mechanical ([#269](https://github.com/Alberto-Codes/vramfit/issues/269)) ([04e544b](https://github.com/Alberto-Codes/vramfit/commit/04e544b25746a25dce755461b1e7d24f28918f55)), closes [#246](https://github.com/Alberto-Codes/vramfit/issues/246)
* **pack:** map expert stacks and every layer naming family ([979bf40](https://github.com/Alberto-Codes/vramfit/commit/979bf40fdcdbc7b5e75dad342a15d01548216582)), closes [#180](https://github.com/Alberto-Codes/vramfit/issues/180)
* **pack:** map expert stacks through the ADR-0028 type table ([#231](https://github.com/Alberto-Codes/vramfit/issues/231)) ([a415cf9](https://github.com/Alberto-Codes/vramfit/commit/a415cf9400d444484f7aecc891870cfbc51e47fb)), closes [#228](https://github.com/Alberto-Codes/vramfit/issues/228)
* **pack:** report zero-count experts from the imatrix ([#218](https://github.com/Alberto-Codes/vramfit/issues/218)) ([9ed0c7b](https://github.com/Alberto-Codes/vramfit/commit/9ed0c7b9817081821e29d0d217507c24117acddf)), closes [#179](https://github.com/Alberto-Codes/vramfit/issues/179)
* **scan:** add the slice perturbation path to the meter ([#222](https://github.com/Alberto-Codes/vramfit/issues/222)) ([8788d4c](https://github.com/Alberto-Codes/vramfit/commit/8788d4cfb0559e9d9228cd40b39e037669800135))
* **scan:** key the sensitivity map on the pack-addressable stack ([#181](https://github.com/Alberto-Codes/vramfit/issues/181)) ([48ad52c](https://github.com/Alberto-Codes/vramfit/commit/48ad52cec39cce9d7219f5ced3f65322b6b4aa69))
* **scan:** map Nemotron-H dense tensor names to their imatrix entries ([8a6f374](https://github.com/Alberto-Codes/vramfit/commit/8a6f3741453ef690f11d7898380aef8bdd6b2a10)), closes [#186](https://github.com/Alberto-Codes/vramfit/issues/186)
* **scan:** read a fused expert stack's counts as one vector ([#214](https://github.com/Alberto-Codes/vramfit/issues/214)) ([fe8f478](https://github.com/Alberto-Codes/vramfit/commit/fe8f47841ef8b95848042b0ec43cee271b1b8806))
* **scan:** read an expert stack's imatrix rows ([#187](https://github.com/Alberto-Codes/vramfit/issues/187)) ([a3c7b92](https://github.com/Alberto-Codes/vramfit/commit/a3c7b92346bf57e3cfab694615e34c67a6263099)), closes [#177](https://github.com/Alberto-Codes/vramfit/issues/177)
* **scan:** record per-stack imatrix count summaries on the map ([#217](https://github.com/Alberto-Codes/vramfit/issues/217)) ([315d790](https://github.com/Alberto-Codes/vramfit/commit/315d7904dbdfb6fcbea65aaba5cf0798360321f8)), closes [#179](https://github.com/Alberto-Codes/vramfit/issues/179)


### Bug Fixes

* **adapters:** keep the map's derived note across a save ([#254](https://github.com/Alberto-Codes/vramfit/issues/254)) ([be37d61](https://github.com/Alberto-Codes/vramfit/commit/be37d615a4bab13a6e8855e96806a37912a2293a)), closes [#136](https://github.com/Alberto-Codes/vramfit/issues/136)
* **pack:** decode toolchain output with a replacing handler ([#250](https://github.com/Alberto-Codes/vramfit/issues/250)) ([de2590e](https://github.com/Alberto-Codes/vramfit/commit/de2590ea346e709e85413a7930f21e40bb504671)), closes [#247](https://github.com/Alberto-Codes/vramfit/issues/247)
* **pack:** report an unnamed signal by number, not a ValueError ([#258](https://github.com/Alberto-Codes/vramfit/issues/258)) ([d079c04](https://github.com/Alberto-Codes/vramfit/commit/d079c04fd9e1112fc4ce6dcb231f3a8e4243e81c))
* **scan:** close the imatrix reader's silent-mispricing paths ([a3c7b92](https://github.com/Alberto-Codes/vramfit/commit/a3c7b92346bf57e3cfab694615e34c67a6263099))

## 0.1.0 (2026-08-11)

First published release.

vramfit fits a large open model on one GPU. It measures how much damage
each layer group takes at each candidate precision, then solves for a
mixed-precision recipe that fits a hard VRAM ceiling. Most quantized
models pick precision by heuristic. vramfit measures first.

### The pipeline

* **`vramfit scan`** — quantize one layer group at a time. Measure output
  divergence against the full-precision reference. Write a sensitivity map.
* **`vramfit plan`** — solve for a recipe under a VRAM ceiling. The solver
  spends bits where the sensitivity map says they matter. The plan step
  runs without torch.
* **`vramfit validate`** — replay a full recipe in one pass. Compare the
  measured recipe damage against the solver prediction.
* **`vramfit pack`** — apply a recipe and write a GGUF checkpoint. Pack
  takes an importance matrix through `--imatrix`. Given one, it guards
  protected tensors with a per-tensor reconstruction check. `--smoke-text`
  runs a smoke test on the packed model. Without those flags pack warns
  that the artifact is unproven.
* **`vramfit budget`** — report the VRAM arithmetic for a model shape,
  context length, and KV-cache dtype.

### The acceptance test

Nemotron Super 49B serves on a 24 GiB RTX 4090. On 2026-08-09 an
end-to-end pack beat the size-matched community imatrix quant on
full-window KL divergence: 0.2873 against 0.2959, 7.8σ paired. Perplexity
reads 8.517 against 8.532, which the interval calls a tie. The baseline
holds its one clear lead, top-token agreement at 83.4 % against 82.9 %.
The packed model sits 112 MiB under the weight budget. Tier 3 ran five
task benchmarks on 2026-08-10 and returned five statistical ties, none
past 0.8σ.

Hugging Face hosts both artifacts: the
[packed model](https://huggingface.co/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-fit24gib-GGUF)
and the
[sensitivity-map dataset](https://huggingface.co/datasets/Alberto-Codes/Llama-3_3-Nemotron-Super-49B-v1_5-sensitivity-maps).
The
[evidence ledger](https://github.com/Alberto-Codes/vramfit/blob/main/docs/explanation/evaluating-packed-models.md)
records all sixteen data points behind those numbers.

### Install

Python 3.12 or later.

```bash
pip install vramfit          # plan and budget
pip install "vramfit[scan]"  # adds the torch stack for scan and validate
pip install "vramfit[pack]"  # the scan stack plus the GGUF converter deps
```

The base install carries typer and structlog only. torch and transformers
stay behind the extras, so the plan step installs without a GPU stack
([ADR-0005](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0005-heavy-deps-as-extras.md)).
Two consequences follow. `vramfit validate` builds the same torch-backed
meter as `vramfit scan`, so it needs the scan extra. `vramfit pack` needs
a llama.cpp checkout with built tools, which you supply with
`--llama-cpp`
([ADR-0012](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0012-gguf-type-mapping.md)).

### Limits of this release

* Pack targets GGUF and llama.cpp. vramfit has no vLLM backend for 4-bit
  and wider recipes yet
  ([ADR-0010](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0010-sub-4-bit-serving-path.md)).
* The solver does not buy 2-bit until a runtime-frame price exists. It
  solves on a copy of the sensitivity map with the 2-bit column removed
  ([ADR-0021](https://github.com/Alberto-Codes/vramfit/blob/main/docs/adr/0021-runtime-frame-measurement.md)).
* Artifacts carry the `vramfit_schema` envelope key. Readers reject the
  pre-rename key by name, so a pre-rename artifact does not load.
