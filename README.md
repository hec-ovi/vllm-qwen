<h1 align="center">vllm-qwen</h1>

<p align="center">
  <strong>Qwen3.6-27B (BF16) served over OpenAI-compatible HTTP on AMD Strix Halo.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Working-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/Model-Qwen3.6--27B-0b7285" alt="Model" />
  <img src="https://img.shields.io/badge/Precision-BF16-purple" alt="Precision" />
  <img src="https://img.shields.io/badge/Context-256K_native-orange" alt="Context" />
  <img src="https://img.shields.io/badge/Decode-4.3_t%2Fs-blue" alt="Decode" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Ubuntu-26.04-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu" />
  <img src="https://img.shields.io/badge/ROCm-TheRock_7.13-ED1C24?logo=amd&logoColor=white" alt="ROCm" />
  <img src="https://img.shields.io/badge/GPU-gfx1151_(RDNA_3.5)-ED1C24?logo=amd&logoColor=white" alt="GPU" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/vLLM-0.19_(src)-4B2E83" alt="vLLM" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

---

## What this is

A thin Docker Compose wrapper that runs `Qwen/Qwen3.6-27B` (BF16) behind an
OpenAI-compatible HTTP API on an **AMD Ryzen AI Max+ 395 "Strix Halo"**
(`gfx1151`, RDNA 3.5, 128 GB UMA). Serves `/v1/completions`,
`/v1/chat/completions`, `/v1/responses`, and vision inputs through the same
endpoint. Native 256K context.

vLLM is **built from source** against a TheRock nightly ROCm SDK with a
small patch set for Strix Halo. There is no prebuilt image path — consumer
AMD GPUs aren't in AMD's mainstream ROCm support matrix yet, so source is
the only clean route.

> **Want ~75% faster decode and working tool calls instead?**
> See the sibling repo [**`llama-qwen`**](https://github.com/hec-ovi/llama-qwen) —
> same hardware, Qwen 3.6-27B Q8_0 via llama.cpp. Decode **7.5 t/s** vs
> this repo's 4.3 t/s, tool calling verified clean (vLLM currently has
> three open upstream parser bugs). Trade-off: no vision, no
> `/v1/responses`. Pick that one for agentic / coding / chat speed, this
> one for vision or structured reasoning output.

---

## Stack

| Layer | Version |
|---|---|
| Host OS | Ubuntu 26.04 (container base) |
| ROCm | **TheRock `7.13.0a20260424`** (S3 nightly; resolves to latest at build time) |
| PyTorch | `2.10.0+rocm7.12.0rc1` (AMD gfx1151 prerelease wheels) |
| Triton | `3.6.0+rocm7.12.0rc1` |
| vLLM | `0.19.2rc1` upstream HEAD, built from source, 12 local patches |
| Model | `Qwen/Qwen3.6-27B` (BF16, official) |

---

## Hardware

Tested on: **Ryzen AI Max+ 395 / 128 GB UMA** (Radeon 8060S iGPU, `gfx1151`).
Kernel ≥ 6.18. Docker with `/dev/kfd` + `/dev/dri` access.

This is **the only supported configuration today.** Other Strix Halo
variants (8050S / 8040S / lower RAM) will likely work but haven't been
tested.

### Host memory setup (required — one-time)

Strix Halo is UMA: system RAM and GPU VRAM share the same physical pool.
Out of the box the BIOS reserves a fixed chunk as "dedicated VRAM" and
the Linux TTM subsystem caps how much of the rest the GPU driver may
map as GTT. Both defaults are wrong for this workload — the 51 GiB
model won't fit unless you widen them.

**1. BIOS / UEFI:** set the dedicated GPU VRAM carve-out to its
**minimum (2 GB / 2048 MB)**. You want the GPU to take memory from the
shared pool on demand via GTT, not from a fixed-size pre-allocation.
Menu name varies by vendor — look for *UMA Frame Buffer Size*,
*UMA Buffer Size*, *iGPU Memory*, or *GPU Shared Memory*.

**2. Ubuntu GRUB:** raise the TTM page limit so the kernel will
actually let the GPU driver map ~116 GiB of GTT. Edit
`/etc/default/grub`, set:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=30408704 amdgpu.noretry=0 amdgpu.gpu_recovery=1"
```

Then:

```bash
sudo update-grub
sudo reboot
```

Verify after reboot:

```bash
cat /sys/class/drm/card1/device/mem_info_gtt_total
# expect ~124554670080  (≈ 116 GiB)
```

`ttm.pages_limit=30408704` is 30,408,704 × 4 KiB pages = **116 GiB**.
Leave 12 GiB for the OS and desktop. `amdgpu.noretry=0` +
`amdgpu.gpu_recovery=1` are stability flags — keep them on for
long-running inference.

---

## Quick start

```bash
cp .env.template .env
# edit .env:
#   - VLLM_HOST_MODELS_DIR: your HF cache directory
#   - HF_TOKEN: your HuggingFace read token (see below)

# One-time, on huggingface.co (logged in):
#   1. Open https://huggingface.co/Qwen/Qwen3.6-27B
#      and click "Agree and access repository" (the model is gated).
#   2. Mint a read token at https://huggingface.co/settings/tokens
#      and paste it into .env as HF_TOKEN.

# Fetch the model into your cache directory (uses HF_TOKEN from .env)
export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' .env | xargs)
hf download Qwen/Qwen3.6-27B --cache-dir "$VLLM_HOST_MODELS_DIR/hub"

docker compose up -d --build

# First build: ~15 min (ROCm tarball + torch wheels + vLLM source build)
# First start: ~4 min (Triton kernel JIT — one-time)
# Subsequent starts: <1 min (kernel cache persisted)

# Verify
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

---

## First boot takes ~4 minutes — don't cancel it

On a **cold start** (new container or cleared Triton cache):

| Phase | Time | Visible activity |
|---|---|---|
| Container init | ~5s | Python imports |
| Weight load | ~30s | Reading 51 GiB BF16 into UMA |
| Engine init | **~170s** | Memory profile + KV cache alloc + **Triton JIT compiling ~150 gfx1151 kernels** |
| Server bind | ~2s | Uvicorn listens on `:8000` |
| **Total** | **~4m 7s** | |

**The 170s "silent window" is Triton compiling.** One CPU core at 100%, GPU
occasionally spiking, no log output for minutes at a time. It is not stuck.
Common mistake: people see the silence, `docker compose down`, restart, and
lose the cache they were about to finish building — then it starts over.

**Subsequent boots are < 1 min** because the Triton cache persists in
`$VLLM_HOST_TRITON_CACHE` (repo-local `./.triton-cache/` by default).

**Never set `VLLM_LOGGING_LEVEL=DEBUG`.** vLLM's `ir/op.py` formats every
tensor argument into a string at every op dispatch when DEBUG is on, which
makes decode **20–100× slower** (discovered via py-spy). Default `INFO` is
fine.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat, with `messages` array. Supports vision via `image_url` content blocks. |
| `POST /v1/completions` | OpenAI text completion — raw `prompt` string, no chat template. |
| `POST /v1/responses` | OpenAI Responses API. Reasoning is separated into `output[].type == "reasoning"`. |
| `GET  /v1/models` | List the served model name (`Qwen3.6-27B`). |
| `GET  /health` | Liveness probe. |

All three generation endpoints honor the model's native **thinking mode**
(`<think>...</think>`). Chat / completions return the full stream including
thinking; Responses separates reasoning into its own output item.

---

## Benchmark

Hardware: Ryzen AI Max+ 395, 128 GB UMA, Ubuntu 26.04, TheRock 7.13 nightly.
Model: `Qwen/Qwen3.6-27B` BF16 at 256K context. 3 iterations per test.
Temperature 0. **No `max_tokens` cap** (model decides when to stop).
**Thinking mode ON** (native Qwen behavior).

| Endpoint | Prompt | prompt_tok | completion_tok | wall (s) | decode t/s |
|---|---|---|---|---|---|
| `/v1/completions` | "The capital of Argentina is" | 5 | 16 | 3.8 | **4.20** |
| `/v1/chat/completions` | "Explain what the Argentine peso is, in two short sentences." | 23 | 989 | 230.2 | **4.30** |
| `/v1/responses` | "What is the atomic number of carbon? One word answer." | (input) | 131 | 30.3 | **4.33** |
| `/v1/chat/completions + image` | "Describe this image in one sentence." + 26 KB JPEG | 164 | 457 | 107.1 | **4.27** |

### Throughput distribution (226 samples during bench)

| Stat | Value |
|---|---|
| Minimum | 3.00 t/s |
| p10 | 4.20 t/s |
| Median | **4.30 t/s** |
| Mean | 4.29 t/s |
| p90 | 4.40 t/s |
| Peak | 4.40 t/s |

Decode is **rock-steady at 4.2–4.4 t/s** across all endpoints and prompt
shapes. That's the real BF16 ceiling for a 27B model on gfx1151 — bound by
weight-streaming bandwidth through the UMA, not by compute.

### Memory footprint at idle (model loaded, KV allocated)

| Component | Size |
|---|---|
| Model weights (BF16, 27B params) | 51.2 GiB |
| KV cache capacity | 217,168 tokens |
| Total GTT used (weights + KV + compute buffers) | **104.9 GiB / 116.0 GiB** |
| Host RAM free after model load | ~23 GiB |

The whole setup uses ~105 GiB of the 128 GB UMA pool. Comfortable margin
for the OS and a desktop session alongside.

---

## Reproduce

```bash
python3 test/bench.py
# writes test/bench_results.json with full per-run detail
```

`test/bench.py` warms up once, then runs 3 iterations per endpoint with the
prompts above and records server-reported usage counters. No external deps
beyond Python 3.

---

## Which should I run: `vllm-qwen` or `llama-qwen`?

| | `vllm-qwen` (this repo) | [`llama-qwen`](https://github.com/hec-ovi/llama-qwen) |
|---|---|---|
| Weights format | BF16 safetensors (official) | Q8_0 GGUF (Unsloth re-quant) |
| Decode speed | 4.3 t/s | **~7.5 t/s** |
| Prefill speed (8K) | ~38 t/s | 200 t/s |
| Vision input | ✓ | ✘ (GGUF has no `mmproj`) |
| `/v1/responses` + separated reasoning | ✓ | ✘ (not implemented in llama.cpp) |
| Tool calling (OpenAI format) | ⚠ broken on current commit (see [vllm#40783](https://github.com/vllm-project/vllm/pull/40783) / [#40785](https://github.com/vllm-project/vllm/pull/40785) / [#40787](https://github.com/vllm-project/vllm/pull/40787)) | ✓ (via `--jinja`, verified clean) |
| Context | 256K | 256K |
| Memory footprint | ~105 GiB total | ~35 GiB total |
| Boot time cold | ~4 min | **~13s** |
| Build from source needs patches | yes (12 local patches) | no |
| Official weights | ✓ (Qwen BF16) | ✘ (Unsloth re-quant) |

**Rule of thumb:** if you need vision, reasoning separated for
structured pipelines, or weights directly from the Qwen team → this
repo. If you need raw speed, agentic loops, or a fast desktop sidekick
→ `llama-qwen`.

---

## Known non-working paths on this hardware

| Target | Status | Root cause |
|---|---|---|
| `Qwen/Qwen3.6-27B-FP8` | ✘ hangs in init | vLLM's Triton `w8a8` autotune stalls on DeltaNet's 48+48 partitions under block-128 FP8 quant. RDNA 3.5 has no hardware FP8 anyway, so even if unstuck it would emulate at BF16 speed. Upstream fix needed. |
| Unsloth `Qwen3.6-27B-GGUF` | ✘ rejected at load | HuggingFace `transformers` doesn't register the `qwen35` GGUF arch. Fixable with a small patch to `transformers`' GGUF arch map (we have a local hack in `.tests/`), but it's not a vLLM issue. |
| Qwen 3.x reasoning parser (`--reasoning-parser qwen3`) | ✘ corrupts output on raw-text `<tool_call>` | Parser only detects the special `tool_call_token_id`, so when Qwen 3.6 emits `<tool_call>` as fragmented text tokens across deltas, the reasoning→content boundary fires prematurely and the rest of the thought stream leaks into the `content` field. `usage.reasoning_tokens` also reports 0. Upstream fix: [vllm#40783](https://github.com/vllm-project/vllm/pull/40783) (open at time of writing). |
| Qwen 3.x tool-call parsers (`--tool-call-parser qwen3_coder` / `qwen3_xml`) | ✘ broken streaming, lost final `\n`, fragmented tags | Multiple streaming bugs in both parsers: tags split across deltas, content tracking drift under speculative decoding, interleaved text swallowed. Upstream fixes: [vllm#40785](https://github.com/vllm-project/vllm/pull/40785) (qwen3_coder) and [vllm#40787](https://github.com/vllm-project/vllm/pull/40787) (qwen3_xml). Both open at time of writing. |

### Tool calling caveat

The three parser bugs above affect **all** Qwen 3/3.5/3.6 versions — the
report and PRs are from upstream vLLM maintainers. They matter most for
**agentic coding loops**, where the model emits `<tool_call>` tags
repeatedly and the parser's misclassification cascades into tool-arg
corruption. Day-to-day chat and single-turn Q&A via `/v1/chat/completions`
are mostly fine.

Reproduced locally on the commit this repo ships (`51adca74e`): a
/v1/responses prompt that contains the literal string `<tool_call>`
inside the model's reasoning text causes the reasoning field to cut off
mid-sentence and the rest of the thought stream to route into `content`.

**Workarounds today:**
1. For agentic / text-only use, prefer [`llama-qwen`](https://github.com/hec-ovi/llama-qwen) — llama.cpp has an independent tool-call extractor and handles Qwen 3.6 tool calls correctly (verified: single + parallel calls with clean structured arguments). **No vision on that path**, though — the Unsloth GGUF doesn't ship the Qwen 3.6 vision tower (`mmproj`). If you need image understanding, stay on this repo and just avoid tool calling until the three PRs land.
2. Or pin `VLLM_COMMIT=<sha>` in the Dockerfile once the three PRs merge upstream.
3. Or cherry-pick the patches into `scripts/patch_strix.py` as Patches 13/14/15 and rebuild.

For Q8 GGUF serving on this hardware right now, use `llama.cpp` directly —
it accepts the Unsloth `qwen35` GGUF natively and runs at ~7.5 t/s decode.
That path is covered by [`llama-qwen`](https://github.com/hec-ovi/llama-qwen)
(text + tool calls only — **no vision** on that path).

---

## Repo layout

```
.
├── Dockerfile              multi-stage build: Ubuntu + TheRock + torch + vLLM from source
├── docker-compose.yml      one service, one model, host-mounted cache
├── .env.template           the one config file you need to edit
├── scripts/
│   ├── install_rocm_sdk.sh TheRock S3 nightly tarball → /opt/rocm
│   └── patch_strix.py      12 targeted Python patches to vLLM for gfx1151
└── test/
    └── bench.py            reproducible 4-endpoint benchmark harness
```

---

## License

[The Unlicense](LICENSE). Public domain dedication: use, modify, distribute, sell, fork, do whatever you want with this code. No attribution required, no warranty given.
