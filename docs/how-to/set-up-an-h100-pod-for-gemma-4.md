---
status: draft
---

# How to set up an H100 pod for Gemma 4 31B

> **Status: draft** — three rented pods ran this setup on 2026-09-01
> for the #462 and #463 evaluation runs. The sizes, times, and costs
> below come from those run records. The pod parameters come from
> the create calls. Rules marked with an earlier ticket number come
> from pods before #462. The page describes what worked. It does
> not describe a plan.

## Goal

Stand up a rented H100 that serves three Gemma 4 31B decoders
through llama.cpp: the fit24gib packed model, the vendor QAT
baseline, and a BF16 reference. When the pod is up, run any
evaluation you want against them.

This pod is the [instrument](../reference/glossary.md) behind the
fit24gib card's divergence and accuracy numbers. Match it and your
numbers compare with the record. Change the GPU or the llama.cpp release and you
have a new instrument
([ADR-0027](../adr/0027-instrument-frame-matching.md)).

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with an API key and
  about $3.29 per hour of budget.
- An SSH key pair. The pod takes the public key at creation.
- A Hugging Face token. The `google/gemma-4-*` repos are gated.
  Accept the Gemma license on huggingface.co once with that account.
- About 100 GB of pod disk: 36 GB for the two GGUF pairs, 61.4 GB
  for the BF16 reference.

## Step 1: Create the pod

Create the pod from the RunPod console, the REST API, or the
`create-pod` MCP tool. The runs used these parameters:

| Parameter | Value |
|---|---|
| GPU | `NVIDIA H100 80GB HBM3` (H100 SXM), 1 GPU |
| Cloud type | Secure |
| Data center | `AP-IN-1` (the #463 pod ran in `EUR-IS-3`) |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Container disk | 40 GB |
| Volume | 150 GB, mounted at `/workspace` |
| Ports | `22/tcp` |
| Environment | `HF_TOKEN=<your token>` |
| SSH public key | the full contents of `~/.ssh/id_ed25519.pub` |

The 80 GB comes from the BF16 reference arm. The decoder is 61.4 GB.
Add the 1.2 GB BF16 projector sidecar, the KV cache, and the encode
workspace, and the arm needs about 72 GB. A smaller GPU serves the
packed model and the QAT baseline only.

Three rules earlier pods learned:

- **Pass the whole public key** (#328). A truncated key boots a pod
  with no working sshd. The fix is `update-pod` and `restart-pod`,
  not a delete.
- **The API refuses a create call without a container disk size**
  (#387). Set it.
- **A plain volume is host-local** (#372). A stopped pod with a
  host-local volume waits for that one host to free a GPU. In August
  2026 the wait ran 48 min once and 75 min the next time, with no
  upper bound. Create a network volume instead if you plan to stop
  and restart the pod. Do the same to keep the 61.4 GB BF16 file
  between pods.

## Step 2: Connect over SSH

Read the direct SSH host and port from the pod's `runtime.ports`
(`get-pod`, or the console's Connect panel). The port changes on
every report (#328). Read it again after every restart.

```bash
ssh -o StrictHostKeyChecking=accept-new -p <port> root@<host>
```

Use the direct TCP port. The proxy SSH endpoint authenticates and
then refuses non-PTY exec.

## Step 3: Install the toolchain

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl cmake build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
uv tool install "huggingface_hub[cli]"
```

This step took 33 s. The image's `pip` is PEP 668 managed. Install
extra packages with `uv`, or pass `--break-system-packages` to
`python3 -m pip install`. A silenced `pip install` hides the refusal
and costs a relaunch.

## Step 4: Build llama.cpp b10362 with CUDA

```bash
cd /workspace
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch --tags origin
git checkout 4801e3c56    # tag b10362
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
build/bin/llama-server --version
```

The last command prints `version: 10362 (4801e3c56)`. The build took
429 s. Run it in the background while the downloads proceed.

Four pods reproduced results bit-exactly on this release (the
#462 instrument note). A different release is a different
instrument. Record any change in your run notes.

## Step 5: Download the packed model and the QAT baseline

```bash
mkdir -p /workspace/models
cd /workspace/models
hf download Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF \
  gemma-4-31B-it-fit24gib.gguf gemma-4-31B-it-mmproj-q4km.gguf \
  --local-dir .
hf download google/gemma-4-31B-it-qat-q4_0-gguf \
  gemma-4-31B_q4_0-it.gguf gemma-4-31B-it-mmproj.gguf \
  --local-dir .
```

| File | Size |
|---|---|
| `gemma-4-31B-it-fit24gib.gguf` | 16.0 GB |
| `gemma-4-31B-it-mmproj-q4km.gguf` | 0.66 GB |
| `gemma-4-31B_q4_0-it.gguf` | 17.65 GB |
| `gemma-4-31B-it-mmproj.gguf` | 1.20 GB |

The four downloads took 200 s. The full run skipped the q4km
sidecar, and its three files (34.9 GB) took 114 s.

## Step 6: Build the BF16 reference decoder (optional)

Skip this step when your evaluation needs no BF16 reference. It
costs 50 to 57 min of pod time, $2.72 to $3.14.

The converter runs inside the vramfit `scan` extra at a pinned
commit. The lockfile gives every pod the same torch and
transformers stack. The `--remote` flag streams the safetensors from
Hugging Face, so the pod never stores the 61 GB checkpoint.

```bash
cd /workspace
git clone https://github.com/Alberto-Codes/vramfit
cd vramfit
git checkout 4dea483aaa086493ecde5ded7d6525b8fec2fc68
uv sync --extra scan
PYTHONPATH=/workspace/llama.cpp/gguf-py uv run --no-sync python \
  /workspace/llama.cpp/convert_hf_to_gguf.py \
  google/gemma-4-31B-it-qat-q4_0-unquantized --remote \
  --outfile /workspace/models/gemma-4-31b-bf16.gguf --outtype bf16
```

The source repo is `google/gemma-4-31B-it-qat-q4_0-unquantized`, not
`google/gemma-4-31B-it`. The box's `convert-bf16.log` is the recipe
of record and names the former. The output is 61,413,171,264 bytes.
The convert took 2,978 s on one pod and 3,440 s on another.

Keep the file on a network volume. Each fresh pod otherwise pays the
convert again.

## Step 7: Serve an arm

```bash
BIN=/workspace/llama.cpp/build/bin
M=/workspace/models
"$BIN/llama-server" -m "$M/gemma-4-31B-it-fit24gib.gguf" \
  --mmproj "$M/gemma-4-31B-it-mmproj.gguf" \
  -c 8192 -ngl 99 -np 1 --port 8991 > server-kv9.log 2>&1 &
```

Wait for `model loaded` in the log. The b10362 CUDA server never
prints `all slots are idle`. A readiness check on that line times
out with the server healthy.

The runs served four arms:

| Arm | `-m` | `--mmproj` |
|---|---|---|
| BF16 reference | `gemma-4-31b-bf16.gguf` | `gemma-4-31B-it-mmproj.gguf` |
| fit24gib pack (kv9) | `gemma-4-31B-it-fit24gib.gguf` | `gemma-4-31B-it-mmproj.gguf` |
| QAT baseline | `gemma-4-31B_q4_0-it.gguf` | `gemma-4-31B-it-mmproj.gguf` |
| Shipped pair | `gemma-4-31B-it-fit24gib.gguf` | `gemma-4-31B-it-mmproj-q4km.gguf` |

The first three arms share the BF16 projector sidecar, so the
decoder is the only variable between them. The fourth arm is the
pair the card ships, used for timing. The runs used context 8,192
for single-image requests. The frame ladders used 61,440 on the
pack and 40,960 on the QAT baseline, the windows
[ADR-0030](../adr/0030-vision-budget-sidecar.md) names.

Stop an arm before the next load. Kill the server, then wait until
`nvidia-smi` reports more than 70,000 MiB free:

```bash
kill "$(pgrep -f llama-server)"
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

## Step 8: Run unattended, then delete the pod

Launch a driver with `nohup`. Environment variables set at pod
creation do not reach a non-login SSH exec, so pass the token on the
command line:

```bash
ssh -p <port> root@<host> \
  'cd /workspace && nohup env HF_TOKEN=<token> bash driver.sh \
   > driver.log 2>&1 < /dev/null &'
```

Make the driver's last act write a marker, for example
`date > /workspace/DONE`, and poll for it over SSH. Then copy the
results back with `scp` and delete the pod. An idle pod bills at
the full rate. One probe paid $0.62 for a 680 s gap between two
driver runs.

## Measured setup times

| Stage | Wall | Cost at $3.29/hr |
|---|---|---|
| Toolchain install | 33 s | $0.03 |
| Four downloads, 35.5 GB | 200 s | $0.18 |
| llama.cpp CUDA build | 429 s | $0.39 |
| BF16 convert | 2,978 s | $2.72 |

Setup to the first serve takes about 8 min without the convert and
about 54 min with it. The build runs beside the downloads and the
convert when you start it first, so its row overlaps the others.

## Related

- [Evaluating packed models](../explanation/evaluating-packed-models.md)
  records what the runs on this pod measured.
- [ADR-0027](../adr/0027-instrument-frame-matching.md) defines the
  instrument and the frame-match rule.
- [ADR-0030](../adr/0030-vision-budget-sidecar.md) defines the
  projector sidecar and the context windows above.
