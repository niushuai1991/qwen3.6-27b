#!/usr/bin/env python3
"""Streaming latency comparison: TTFT (prefill) vs decode per-token.
Both endpoints: reasoning_effort=none (thinking disabled).
"""
import json, time, statistics, sys
import requests

SYS = open("/home/ec2-user/mof/system.txt").read()
USR = open("/home/ec2-user/mof/user.txt").read()

# load .env
ENV = {}
for line in open("/data/qwen3.6-27b/.env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        ENV[k] = v.strip().strip('"').strip("'")

MODEL = "qwen3.6-27b"
ENDPOINTS = {
    "self-hosted (vLLM L40S)": {
        "url": "http://localhost:18001/v1/chat/completions",
        "key": None,
    },
    "wangsu (edgecloud API)": {
        "url": ENV["OPENAI_BASE_URL_WANGSU"].rstrip("/") + "/chat/completions",
        "key": ENV["OPENAI_API_KEY_WANGSU"],
    },
}

def measure(label, ep, rounds=3, warmup=0):
    results = []
    for i in range(rounds + warmup):
        is_warm = i < warmup
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": USR}],
            "max_tokens": 4096,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "reasoning_effort": "none",
        }
        headers = {"Content-Type": "application/json"}
        if ep["key"]:
            headers["Authorization"] = f"Bearer {ep['key']}"
        t0 = time.monotonic()
        first_c = first_r = last_c = None
        content_chunks = 0
        reasoning_chunks = 0
        usage = None
        try:
            r = requests.post(ep["url"], headers=headers, json=payload, stream=True, timeout=300)
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "ignore")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                ch = obj.get("choices") or []
                if ch:
                    delta = ch[0].get("delta", {}) or {}
                    if delta.get("content"):
                        now = time.monotonic()
                        if first_c is None:
                            first_c = now
                        last_c = now
                        content_chunks += 1
                    if delta.get("reasoning"):
                        if first_r is None:
                            first_r = time.monotonic()
                        reasoning_chunks += 1
            tend = time.monotonic()
        except Exception as e:
            print(f"  [{label}] round {i} ERROR: {e}", file=sys.stderr)
            continue

        if is_warm:
            print(f"  [{label}] warmup done ({tend-t0:.1f}s)")
            continue

        ptoks = usage.get("prompt_tokens") if usage else None
        ctoks = usage.get("completion_tokens") if usage else content_chunks
        rtoks = 0
        cdet = (usage.get("completion_tokens_details") if usage else None) or {}
        rtoks = cdet.get("reasoning_tokens", 0) or 0
        ttft = first_c - t0 if first_c else None
        content_tokens = (ctoks - rtoks) if ctoks is not None else content_chunks
        decode_total = (last_c - first_c) if (first_c and last_c and content_tokens > 1) else 0
        dpt_ms = (decode_total / (content_tokens - 1) * 1000) if content_tokens > 1 else 0
        dtok_s = ((content_tokens - 1) / decode_total) if decode_total > 0 else 0
        rec = {
            "prompt_tokens": ptoks, "completion_tokens": ctoks,
            "reasoning_tokens": rtoks, "content_tokens": content_tokens,
            "ttft_s": ttft, "decode_total_s": decode_total,
            "decode_per_token_ms": dpt_ms, "decode_tok_s": dtok_s,
            "wall_s": tend - t0, "reasoning_chunks": reasoning_chunks,
        }
        results.append(rec)
        print(f"  [{label}] round {i}: TTFT={ttft:.2f}s decode={dpt_ms:.1f}ms/tok ({dtok_s:.1f}tok/s) "
              f"ctoks={ctoks} rtoks={rtoks} wall={tend-t0:.1f}s")
    return results

def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None

allres = {}
for label, ep in ENDPOINTS.items():
    print(f"\n=== {label} ===")
    warmup = 1 if "self-hosted" in label else 0
    allres[label] = measure(label, ep, rounds=3, warmup=warmup)

print("\n\n========== COMPARISON (median of 3 rounds) ==========")
print(f"{'metric':<26}{'self-hosted':>16}{'wangsu':>16}")
rows = allres[list(ENDPOINTS.keys())[0]]
print("-" * 58)
def pm(key, fmt="{:.2f}", unit=""):
    a = median([r[key] for r in allres["self-hosted (vLLM L40S)"]])
    b = median([r[key] for r in allres["wangsu (edgecloud API)"]])
    def f(v): return (fmt.format(v) if isinstance(v,(int,float)) else str(v)) + unit
    print(f"{key:<26}{f(a):>16}{f(b):>16}")

pm("prompt_tokens", "{:.0f}")
pm("content_tokens", "{:.0f}")
pm("reasoning_tokens", "{:.0f}")
print("-" * 58)
pm("ttft_s", "{:.2f}", "s")
pm("decode_per_token_ms", "{:.1f}", "ms")
pm("decode_tok_s", "{:.1f}", " t/s")
pm("decode_total_s", "{:.2f}", "s")
pm("wall_s", "{:.2f}", "s")
