# ADR-0032 evidence: stock Q2_0 passthrough

- **Status:** draft
- **Recorded:** 2026-09-14
- **Decision:** [ADR-0032](../../0032-assisted-q2-encoder-home.md)

## Result

Stock `llama-quantize` b10362 preserves an already-Q2_0 tensor when
its resolved output type is Q2_0. `--pure` and a Q2_K base type do
not override an explicit matching tensor override. Supplying an
imatrix does not change that result.

| Probe | Exit | Q2_0 payload | Float peer |
| --- | ---: | --- | --- |
| Matching overrides | 0 | 36,864 B, unchanged | F16 → Q4_0 |
| Matching overrides plus imatrix | 0 | 36,864 B, unchanged | F16 → Q4_0 |
| Q4_0 override on existing Q2_0 | 1 | Refused requantization | Converted before refusal |

The input and both successful output Q2_0 payloads have SHA-256:

```text
12e50e60f457343c4d88658257ce3aa153d67690446e7941adec635a00440541
```

These are payload hashes, not whole-file hashes. The quantizer adds
metadata and changes the float tensor. The negative probe leaves a
partial output, which is never an accepted artifact.

## Files and exact invocation

- [probe.py.txt](probe.py.txt): executed Python fixture builder and verifier,
  archived as text. It creates finite Q2_0 blocks directly. It contains
  no assisted encoder and uses no model download.
- [transcript.txt](transcript.txt): complete stdout and stderr for all
  three calls, including each argv, exit code, tensor sizes, and hashes.
- [initial-missing-metadata.txt](initial-missing-metadata.txt): selected stderr from the first attempt,
  which failed before quantization because `llama.context_length` was absent.
  The corrected fixture adds context length and RMS normalization metadata.

The archived transcripts normalize one path root and change nothing
else. `<repo>` replaces the session's working directory in
[transcript.txt](transcript.txt) and
[initial-missing-metadata.txt](initial-missing-metadata.txt). Every
command, argument, exit code, byte count, and hash in them reads as the
session produced it.

This page substitutes a second root below. `<llama.cpp>` stands for the
local llama.cpp checkout in the reproduce command. Neither transcript
carries that root.

The session copied the archived stock toolchain into `scratch/stock-b10362`
and repeated the successful probes there. The committed transcript is that
repeat, with its output unchanged apart from that one root. The session
executed:

```bash
PYTHONPATH=<llama.cpp>/gguf-py \
  python scratch/q2-passthrough/probe.py \
  scratch/stock-b10362/llama-quantize \
  > scratch/q2-passthrough/local-transcript.txt 2>&1
```

To reproduce, copy `probe.py.txt` to `scratch/q2-passthrough/probe.py`.
Provide a Python environment with NumPy and gguf-py supporting Q2_0.
Pass the stock quantizer path as its sole argument. The script writes
fixtures beside itself, runs each command, and asserts the results.

The first successful quantizer command has this shape:

```bash
llama-quantize --pure \
  --tensor-type 'blk\.0\.ffn_down_exps\.weight=q2_0' \
  --tensor-type 'blk\.0\.attn_q\.weight=q4_0' \
  mixed.gguf matching.gguf Q2_K 1
```

The second adds `--imatrix imatrix.gguf`. The negative control changes
only the expert override to Q4_0 and uses a separate output path.
None supplies `--allow-requantize`.

Selected output, verbatim:

```text
llama_print_build_info: build = 10362 (4801e3c56)
[   2/   3] blk.0.attn_q.weight                  - [   256,    256,      1,      1], type =    f16, converting to q4_0 .. size =     0.12 MiB ->     0.04 MiB
[   3/   3] blk.0.ffn_down_exps.weight           - [   256,    256,      2,      1], type =   q2_0, size =    0.035 MiB
```

The negative control prints `requantizing from type q2_0 is disabled`.
The full transcript carries the payload comparisons after GGUF reading,
so the conclusion does not depend on interpreting a progress line.

## Instrument and limits

- Quantizer reports build `10362 (4801e3c56)`, GNU 11.4.0, Linux x86_64.
- [toolchain-sha256.txt](toolchain-sha256.txt) pins the executable and loaded libraries.
- Executable SHA-256:
  `7ee236b4fb67c0924a636baba94f9be3f5d56b9a32a29cc0c7ef03688bccf61a`.
- Python 3.14.7, NumPy 2.5.2.
- gguf-py comes from the clean local llama.cpp checkout
  `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0`.
  The Python writer's checkout differs from the quantizer's build.
- The binary loads its Vulkan backends during startup. The transcript
  records that discovery. No inference or GPU measurement ran.
- The fixture has a two-expert stack with logical shape `[256, 256, 2]`,
  one F16 attention tensor, and one F32 norm. It is not a runnable model.
- This probe establishes passthrough on this build. Production integration
  still needs real row widths, complete metadata preservation, and load tests.

## Rebuy evidence

[rebuy-results.txt](rebuy-results.txt) copies `results.txt` from `nemotron-30b-a3b/rebuy-2026-09-05/` beneath the frozen run root
identified in the [publication ledger](../../../../publication/nemotron-30b-a3b-fit16gib/card-ledger.md).
The copy removes only the final blank line.
The original archive file has SHA-256
`2af983ad9337e36182031a48c4b9bf4a9389b3110aabdfb1f1ddddb07024b8c8`.
These are archived measurements, not evaluations performed for this ADR.

Every figure below names the archive block it comes from.

| Quantity | Control `ctl-repro` | Assisted `c2-assisted-q2_0` |
| --- | ---: | ---: |
| File bytes | 16,922,476,352 | 16,922,476,352 |
| Mean KLD | 0.204318 ± 0.001160 | 0.071428 ± 0.000432 |
| Same-top agreement | 83.127 ± 0.096 % | 89.441 ± 0.079 % |
| PPL ratio | 1.161096 ± 0.002417 | 1.036637 ± 0.001263 |

The comparison above uses those two blocks and no other.

The archive's `REPORT.md`, sections 0 and 7.4, specifies the measurement
frame: stock b10362, CUDA, H100 SXM, full 594-chunk WikiText-2 split,
context 512, and the same f16 reference. It reports a 65.0 % KLD
reduction and a 6.31-point same-top increase. It states that C2 used
the patched encoder and the stock evaluator.

`REPORT.md` SHA-256:
`d5f7ba40b5722c0cab32f594cc62b52b13c84c3827de1c2b678089070f5614a6`.
The report calls the equal-size files “byte-identical.” Their recorded
hashes differ. This ADR uses the precise claim: identical byte counts.

The archive holds two control blocks, and they name two files.
`ctl-published` names 16,922,476,480 B with SHA-256 `85ed06fa…c062`. It
records mean KLD 0.204322, same top 83.146 ± 0.096 %, and PPL
7.917417 ± 0.054002 (lines 1–17). `ctl-repro` names 16,922,476,352 B
with SHA-256 `5f63a831…618ec`. It records mean KLD 0.204318, same top
83.127 ± 0.096 %, and PPL 7.917699 ± 0.054005 (lines 19–35).

The [publication ledger](../../../../publication/nemotron-30b-a3b-fit16gib/card-ledger.md)
binds the `ctl-published` byte count and hash to the `ctl-repro`
figures. [Issue #598](https://github.com/Alberto-Codes/vramfit/issues/598)
carries that provenance question to the maintainer. This change reads
the published files and edits none of them.

The [reference patch](https://huggingface.co/vcruz305/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/blob/main/q2_0_weighted_optimizer.patch)
is identified by SHA-256
`db6fe2f276b6b854650ea7dc5673b4c44d67f8df14c9d12c9879089309a50bec`.
The hash identifies the measured specification even if that URL changes.
The [prior findings](https://github.com/Alberto-Codes/vramfit/issues/461#issuecomment-5486773582)
record its algorithm and unchanged block format. This repository carries
neither that patch nor a patched binary.
