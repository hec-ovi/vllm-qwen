"""
End-to-end benchmark for vllm-qwen.

Exercises the three OpenAI-compatible endpoints and a multimodal chat:
  1. /v1/completions              — raw text continuation
  2. /v1/chat/completions         — chat completion (thinking on, no max_tokens cap)
  3. /v1/responses                — Responses API with reasoning separated
  4. /v1/chat/completions + image — vision (image from ~/Pictures)

Rules enforced everywhere:
  - no max_tokens / max_output_tokens cap (model decides when to stop)
  - thinking mode NOT disabled (native Qwen behavior)

Writes results to test/bench_results.json and prints a Markdown table to stdout.
"""
import argparse
import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

HOST = "http://127.0.0.1:8000"
MODEL = "Qwen3.6-27B"
IMAGE_PATH = "/home/hec/Pictures/profile_small.jpg"


def post(path, body, timeout=1800):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        payload = json.loads(resp.read())
        return time.time() - t0, payload, None
    except urllib.error.HTTPError as e:
        return time.time() - t0, None, f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}"
    except Exception as e:
        return time.time() - t0, None, f"{type(e).__name__}: {e}"


def wait_ready(max_wait=300):
    for i in range(max_wait):
        try:
            urllib.request.urlopen(f"{HOST}/v1/models", timeout=3).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def completion_extract(payload):
    """Extract usage + visible content from /v1/completions response."""
    usage = payload.get("usage", {})
    text = payload["choices"][0].get("text", "")
    return usage, text


def chat_extract(payload):
    usage = payload.get("usage", {})
    msg = payload["choices"][0]["message"]
    raw = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    visible = raw
    if "</think>" in raw:
        visible = raw.split("</think>", 1)[1].strip()
    return usage, visible, reasoning


def responses_extract(payload):
    usage = payload.get("usage", {})
    # /v1/responses has `output` list
    visible = ""
    reasoning = ""
    for item in payload.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    visible += c.get("text", "")
        if item.get("type") == "reasoning":
            for c in item.get("content", []):
                if c.get("type") == "reasoning_text":
                    reasoning += c.get("text", "")
    return usage, visible, reasoning


def run_test(name, path, body, extractor, runs=3):
    """Run a request `runs` times, return list of per-run dicts."""
    print(f"\n== {name} ({runs} iterations) ==")
    out = []
    for i in range(runs):
        wall, payload, err = post(path, body)
        if err:
            print(f"  run {i+1}: ERROR  {err[:200]}")
            out.append({"run": i + 1, "error": err, "wall": round(wall, 2)})
            continue
        usage, visible, *rest = extractor(payload)
        reasoning = rest[0] if rest else ""
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        decode_tps = round(completion_tokens / wall, 2) if wall > 0 else 0
        out.append({
            "run": i + 1,
            "wall": round(wall, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "decode_tps": decode_tps,
            "visible": (visible or "")[:200],
            "reasoning_len": len(reasoning or ""),
        })
        print(f"  run {i+1}: {completion_tokens} tok / {wall:.1f}s = {decode_tps} t/s  "
              f"(reasoning_chars={len(reasoning or '')})")
    return out


def capture_mem_info():
    """Pull memory / KV cache / engine init stats from the running vLLM container log."""
    import subprocess
    info = {}
    try:
        log = subprocess.run(
            ["docker", "logs", "vllm-qwen"],
            capture_output=True, text=True, timeout=10,
        ).stdout + subprocess.run(
            ["docker", "logs", "vllm-qwen"],
            capture_output=True, text=True, timeout=10,
        ).stderr
    except Exception as e:
        return {"error": str(e)}

    import re
    # Weights + model
    m = re.search(r"Loading weights took ([0-9.]+) seconds", log)
    if m: info["weights_load_seconds"] = float(m.group(1))
    m = re.search(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds", log)
    if m:
        info["model_vram_gib"] = float(m.group(1))
        info["model_load_seconds"] = float(m.group(2))
    # Engine init
    m = re.search(r"init engine .* took ([0-9.]+) s", log)
    if m: info["engine_init_seconds"] = float(m.group(1))
    # KV cache
    m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", log)
    if m: info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for [0-9]+ tokens per request: ([0-9.]+)x", log)
    if m: info["kv_max_concurrency"] = float(m.group(1))
    m = re.search(r"num_gpu_blocks[^=]*= *([0-9]+)", log)
    if m: info["num_gpu_blocks"] = int(m.group(1))

    # Host-visible GPU memory (UMA-aware)
    try:
        gtt_used = int(open("/sys/class/drm/card1/device/mem_info_gtt_used").read().strip())
        gtt_total = int(open("/sys/class/drm/card1/device/mem_info_gtt_total").read().strip())
        info["gtt_used_gib"] = round(gtt_used / 1024**3, 2)
        info["gtt_total_gib"] = round(gtt_total / 1024**3, 2)
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    if not args.skip_wait and not wait_ready():
        print("server not ready")
        return

    results = {"model": MODEL, "host": HOST, "ts": time.time(),
               "memory": capture_mem_info(), "tests": {}}
    print(f"\n== memory info ==\n{json.dumps(results['memory'], indent=2)}")

    # ---- Warmup (discarded) ----
    print("\n== warmup ==")
    post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
    }, timeout=600)

    # ---- 1. /v1/completions ----
    results["tests"]["completions"] = {
        "endpoint": "/v1/completions",
        "prompt": "The capital of Argentina is",
        "runs": run_test(
            "/v1/completions",
            "/v1/completions",
            {
                "model": MODEL,
                "prompt": "The capital of Argentina is",
                "temperature": 0,
            },
            completion_extract,
            runs=args.runs,
        ),
    }

    # ---- 2. /v1/chat/completions ----
    chat_prompt = "Explain what the Argentine peso is, in two short sentences."
    results["tests"]["chat_completions"] = {
        "endpoint": "/v1/chat/completions",
        "prompt": chat_prompt,
        "runs": run_test(
            "/v1/chat/completions",
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": chat_prompt}],
                "temperature": 0,
            },
            chat_extract,
            runs=args.runs,
        ),
    }

    # ---- 3. /v1/responses ----
    resp_prompt = "What is the atomic number of carbon? One word answer."
    results["tests"]["responses"] = {
        "endpoint": "/v1/responses",
        "prompt": resp_prompt,
        "runs": run_test(
            "/v1/responses",
            "/v1/responses",
            {
                "model": MODEL,
                "input": resp_prompt,
                "temperature": 0,
            },
            responses_extract,
            runs=args.runs,
        ),
    }

    # ---- 4. /v1/chat/completions with image ----
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        print(f"\nskipping vision test — {IMAGE_PATH} not found")
        results["tests"]["vision"] = {"endpoint": "/v1/chat/completions (image)", "error": "no image"}
    else:
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        vision_prompt = "Describe this image in one sentence."
        results["tests"]["vision"] = {
            "endpoint": "/v1/chat/completions (image)",
            "prompt": f"{vision_prompt} [image: {img_path.name}, {img_path.stat().st_size} B]",
            "runs": run_test(
                "/v1/chat/completions + image",
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    }],
                    "temperature": 0,
                },
                chat_extract,
                runs=args.runs,
            ),
        }

    out_path = Path(__file__).parent / "bench_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    # ---- Markdown table summary ----
    print("\n| Endpoint | Prompt | Runs | prompt_tok | completion_tok | wall (s) | decode t/s |")
    print("|---|---|---|---|---|---|---|")
    for key, test in results["tests"].items():
        if "runs" not in test:
            continue
        successful = [r for r in test["runs"] if "error" not in r]
        if not successful:
            continue
        avg_wall = sum(r["wall"] for r in successful) / len(successful)
        avg_prompt = sum(r["prompt_tokens"] for r in successful) / len(successful)
        avg_comp = sum(r["completion_tokens"] for r in successful) / len(successful)
        avg_tps = sum(r["decode_tps"] for r in successful) / len(successful)
        endpoint = test["endpoint"]
        prompt = (test.get("prompt") or "")[:60]
        print(f"| `{endpoint}` | {prompt} | {len(successful)} | "
              f"{int(avg_prompt)} | {int(avg_comp)} | {avg_wall:.1f} | {avg_tps:.2f} |")


if __name__ == "__main__":
    main()
