---
status: draft
---

# How to serve the Gemma 4 fit24gib pack on a rented H100

> **Status: draft** — three rented pods ran this setup on 2026-09-01
> for the #462 and #463 evaluation runs. The sizes, times, and costs
> below come from those run records. The pod parameters come from
> the create calls. Rules dated before 2026-09-01 come from earlier
> pods. The page describes what worked. It does not describe a plan.

## Goal

Rent an H100 pod that serves the Gemma 4 31B fit24gib
[packed model](../reference/glossary.md) through llama.cpp. The pod
then serves an HTTP API for your own evaluation.

This GPU and llama.cpp release form the
[instrument](../reference/glossary.md) behind the fit24gib model
card's PSAI divergence and task-identification numbers. The card's
held-out benchmark table ran on other hardware. Change the GPU or
the llama.cpp release and you have a new instrument
([ADR-0027](../adr/0027-instrument-frame-matching.md)).

One frame difference matters. The recorded runs served the vendor's
BF16 projector. This page serves the Q4_K_M projector sidecar the
pack ships. Expect your divergence numbers to differ from the
record.

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with an API key and
  about $3.29 per hour of budget.
- An SSH key pair. The pod takes the public key at creation.
- The weights carry the Gemma 4 license. Read it before you serve
  them.

## Step 1: Create the pod

Create the pod from the RunPod console or the RunPod API. The runs
used these parameters:

| Parameter | Value |
|---|---|
| GPU | `NVIDIA H100 80GB HBM3` (H100 SXM), 1 GPU |
| Cloud type | Secure |
| Data center | any with H100 SXM stock (the runs used `AP-IN-1` and `EUR-IS-3`) |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Container disk | 40 GB |
| Volume | 150 GB, mounted at `/workspace` |
| Ports | `22/tcp` |
| SSH public key | the full contents of `~/.ssh/id_ed25519.pub` |

The volume held more than this page downloads. The pack and its
sidecar take 15.5 GiB, and the llama.cpp build a few GiB more.

Three rules come from earlier pods:

- **Pass the whole public key** (2026-08-18, #328). A truncated key
  gave a pod with no SSH access. Updating the pod's key and
  restarting the pod repaired it.
- **The API refuses a create call without a container disk size.**
  Set it.
- **A plain volume is host-local** (2026-08-21, #321). A stopped pod
  with a host-local volume waits for that one host to free a GPU.
  One wait ran 48 min. The next hit a 75 min retry cap, and the work
  moved to a fresh pod. Create a network volume instead if you plan
  to stop and restart the pod.

## Step 2: Connect over SSH

Read the direct SSH host and port from the pod's `runtime.ports`
field in the API, or from the console's Connect panel. The reported port
changes without a restart (#328). Read it again before every
connection.

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
echo 'export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
uv tool install "huggingface_hub[cli]"
```

This step took 33 s. The `.bashrc` line keeps `hf` and `nvcc` on
the path after a reconnect. The image's `pip` is PEP 668 managed.
Install extra packages with `uv`, or pass `--break-system-packages`
to `python3 -m pip install`. A silenced `pip install` hides the
refusal and costs a relaunch.

## Step 4: Build llama.cpp b10362 with CUDA

```bash
cd /workspace
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch --tags origin
git checkout 4801e3c56    # the commit tag b10362 points at
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/bin/llama-server --version
```

The last command prints `version: 10362 (4801e3c56)`. The build took
429 s. Open a second SSH session and run step 5 while the build
runs.

A different llama.cpp release is a different instrument
([ADR-0027](../adr/0027-instrument-frame-matching.md)). Record any
change in your run notes.

## Step 5: Download the packed model

```bash
mkdir -p /workspace/models
cd /workspace/models
hf download Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF \
  gemma-4-31B-it-fit24gib.gguf gemma-4-31B-it-mmproj-q4km.gguf \
  --local-dir .
```

| File | Bytes | Size |
|---|---|---|
| `gemma-4-31B-it-fit24gib.gguf` | 16,015,862,144 | 14.92 GiB |
| `gemma-4-31B-it-mmproj-q4km.gguf` | 659,537,504 | 0.61 GiB |

The repo is public. You need no Hugging Face token.

## Step 6: Serve the packed model

```bash
BIN=/workspace/llama.cpp/build/bin
M=/workspace/models
"$BIN/llama-server" -m "$M/gemma-4-31B-it-fit24gib.gguf" \
  --mmproj "$M/gemma-4-31B-it-mmproj-q4km.gguf" \
  -c 8192 -ngl 99 -np 1 --port 8991 > server.log 2>&1 &
```

Wait for `model loaded` in the log. The b10362 CUDA server never
prints `all slots are idle`. A readiness check on that line times
out with the server healthy. Then confirm the server answers:

```bash
curl -s localhost:8991/health
```

The reply is `{"status":"ok"}`.

The runs used context 8,192 for single-image requests. Multi-image
requests work on this pod with no extra flags (#462). The card's
24 GiB serve boundaries and its encode-batch flag belong to the
RTX 4090 frame, not to this pod.

The server listens inside the pod. Reach it from your own machine
through an SSH tunnel:

```bash
ssh -p <port> -N -L 8991:localhost:8991 root@<host>
```

Your evaluation then talks to `http://localhost:8991`.

Stop the server before the next load. Kill it, then wait until
`nvidia-smi` reports more than 70,000 MiB free:

```bash
pkill -f llama-server
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -gt 70000 ]; do
  sleep 2
done
```

## Step 7: Delete the pod when you finish

An idle pod bills at the full rate. A 680 s idle gap costs $0.62 at
this rate. Copy results back with `scp`, then delete the pod.

One trap for unattended runs: environment variables set at pod
creation do not reach a non-login SSH exec (#462). Set what you
need inside the command you send.

## Measured setup times

| Stage | Wall | Cost at $3.29/hr |
|---|---|---|
| Toolchain install | 33 s | $0.03 |
| llama.cpp CUDA build | 429 s | $0.39 |

Setup to the first serve takes about 8 min when the download
finishes inside the build.

## Related

- [Evaluating packed models](../explanation/evaluating-packed-models.md)
  records what the runs on this pod measured.
- [ADR-0027](../adr/0027-instrument-frame-matching.md) defines the
  instrument and the frame-match rule.
- [ADR-0030](../adr/0030-vision-budget-sidecar.md) defines the
  projector sidecar the serve command loads.
