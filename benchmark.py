"""Benchmark local vLLM/SGLang deployment with a native streaming client.

计时口径与 measure_latency.py 同源：客户端 streaming，逐 SSE chunk 记录首/末 content
token 时间戳，产出 TTFT（≈prefill）与 decode 阶段吞吐 decode_tok_s。

    TTFT            = first_content_chunk_t - start_t
    TTLT            = last_content_chunk_t  - start_t
    decode_per_token = (TTLT - TTFT) / (content_tokens - 1)
    decode_tok_s     = (content_tokens - 1) / (TTLT - TTFT)   # decode 阶段有效吞吐（不含 prefill）
    output_tps       = Σ content_tokens / total_test_time      # 整体聚合（含 TTFT），对齐 llmeter 历史

content_tokens 来自末 chunk 的 usage（stream_options.include_usage），减去 reasoning_tokens。

Acceptance 采集兼容两种引擎（--engine auto 自动探测）：
    vLLM    Prometheus Counter（累计）→ before/after delta；有逐位明细
    SGLang  Prometheus Gauge（mostrecent 瞬时）→ 直读 after 值；无逐位明细，额外给 accept_length

Usage:
    uv run python benchmark.py --label "mtp_k3_native" --concurrency 1,5,10
    uv run python benchmark.py --label "sglang_k3" --engine sglang   # 显式指定亦可

Requirements:
    pip install httpx
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DOCS_DIR = PROJECT_ROOT / "docs"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_ENDPOINT = "http://localhost:18001/v1"
DEFAULT_API_KEY = "not-needed"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 300
DEFAULT_ENGINE = "auto"
METRICS_URL = "http://localhost:18001/metrics"  # fallback if endpoint derivation fails


def parse_concurrency(raw: str) -> list[int]:
    """Parse comma-separated concurrency levels, e.g. '1,5,10' -> [1, 5, 10]."""
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_prompts(prompts_dir: Path) -> list[dict]:
    """Load prompt_*.txt files, return list of {messages, max_tokens} dicts."""
    pattern = "prompt_*.txt"
    prompt_files = sorted(prompts_dir.glob(pattern))
    if not prompt_files:
        prompt_files = sorted(prompts_dir.glob("prompt_*"))
        prompt_files = [f for f in prompt_files if f.suffix == ".txt"]

    if not prompt_files:
        print(f"  No prompt files found matching {pattern} in {prompts_dir}")
        return [{"messages": [{"role": "user", "content": "Explain what machine learning is in 3 paragraphs."}],
                 "max_tokens": 512}]

    print(f"  Loaded {len(prompt_files)} prompts:")
    payloads = []
    for p in prompt_files:
        content = p.read_text().strip()
        preview = content[:60].replace("\n", " ")
        print(f"    {p.name}: {preview}...")
        payloads.append({
            "messages": [{"role": "user", "content": content}],
            "max_tokens": None,  # set per-run below
        })
    return payloads


def metrics_url_from_endpoint(endpoint: str) -> str:
    """Derive the Prometheus metrics URL from the API endpoint.

    http://host:port/v1  ->  http://host:port/metrics
    """
    url = endpoint.rstrip("/")
    if url.endswith("/v1"):
        return url[:-3] + "/metrics"
    return url + "/metrics"


def detect_engine(text: str):
    """Detect inference engine from /metrics text. Returns 'vllm' | 'sglang' | None."""
    if not text:
        return None
    if "vllm:spec_decode_num_accepted_tokens_total" in text:
        return "vllm"
    if "sglang:spec_accept_rate" in text:
        return "sglang"
    return None


def parse_vllm_metrics(text: str) -> dict:
    """Parse vLLM spec-decode counters (cumulative; needs before/after delta)."""
    metrics = {}
    patterns = {
        "accepted_tokens": r'vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} (\S+)',
        "draft_tokens": r'vllm:spec_decode_num_draft_tokens_total\{[^}]*\} (\S+)',
        "num_drafts": r'vllm:spec_decode_num_drafts_total\{[^}]*\} (\S+)',
    }
    for name, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            metrics[name] = float(m.group(1))

    for pos in range(4):  # k=3 → positions 0,1,2
        m = re.search(
            rf'vllm:spec_decode_num_accepted_tokens_per_pos_total\{{[^}}]*position="{pos}"[^}}]*\}}\s+(\S+)',
            text,
        )
        if m:
            metrics[f"accepted_pos_{pos}"] = float(m.group(1))
    return metrics


def parse_sglang_metrics(text: str) -> dict:
    """Parse SGLang spec-decode gauges (instantaneous mostrecent; read directly).

    SGLang exposes spec_accept_rate / spec_accept_length (with bonus) /
    spec_num_steps / spec_num_draft_tokens / spec_verify_calls_total.
    No per-position breakdown.
    """
    metrics = {}
    patterns = {
        "accept_rate": r'sglang:spec_accept_rate\{[^}]*\}\s+(\S+)',
        "accept_length": r'sglang:spec_accept_length\{[^}]*\}\s+(\S+)',
        "num_steps": r'sglang:spec_num_steps\{[^}]*\}\s+(\S+)',
        "num_draft_tokens": r'sglang:spec_num_draft_tokens\{[^}]*\}\s+(\S+)',
        "verify_calls": r'sglang:spec_verify_calls_total\{[^}]*\}\s+(\S+)',
    }
    for name, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            try:
                metrics[name] = float(m.group(1))
            except ValueError:
                pass
    return metrics


def fetch_metrics(metrics_url: str, engine: str = "auto") -> dict:
    """Fetch spec-decode metrics. Returns {'engine': str|None, 'raw': dict}.

    engine='auto' probes /metrics text for the vllm vs sglang prefix.
    """
    try:
        result = subprocess.run(
            ["curl", "-s", metrics_url],
            capture_output=True, text=True, timeout=10,
        )
        text = result.stdout
    except Exception as e:
        print(f"  [WARN] Failed to fetch metrics: {e}")
        return {"engine": None, "raw": {}}

    resolved = detect_engine(text) if engine == "auto" else engine
    if resolved == "vllm":
        raw = parse_vllm_metrics(text)
    elif resolved == "sglang":
        raw = parse_sglang_metrics(text)
    else:
        raw = {}
    return {"engine": resolved, "raw": raw}


def compute_acceptance(before: dict, after: dict) -> dict:
    """Compute acceptance from before/after snapshots. Dispatches by engine.

    vLLM:    counters -> before/after delta; acceptance_rate = accepted/draft;
             accept_length = accepted/num_drafts + 1 (bonus).
    SGLang:  gauges (mostrecent) -> read after value directly; no delta, no
             per-position breakdown (SGLang does not expose one).
    """
    engine = after.get("engine") or before.get("engine")
    result = {
        "engine": engine,
        "draft_tokens": None,
        "accepted_tokens": None,
        "num_drafts": None,
        "acceptance_rate": None,
        "accept_length": None,
        "accepted_pos_0": None, "accepted_pos_1": None,
        "accepted_pos_2": None, "accepted_pos_3": None,
    }

    if engine == "vllm":
        b, a = before.get("raw", {}), after.get("raw", {})
        draft_delta = a.get("draft_tokens", 0) - b.get("draft_tokens", 0)
        accepted_delta = a.get("accepted_tokens", 0) - b.get("accepted_tokens", 0)
        drafts_delta = a.get("num_drafts", 0) - b.get("num_drafts", 0)
        result["draft_tokens"] = draft_delta
        result["accepted_tokens"] = accepted_delta
        result["num_drafts"] = drafts_delta
        result["acceptance_rate"] = accepted_delta / draft_delta if draft_delta > 0 else None
        result["accept_length"] = (accepted_delta / drafts_delta + 1) if drafts_delta > 0 else None
        for pos in range(4):
            result[f"accepted_pos_{pos}"] = a.get(f"accepted_pos_{pos}", 0) - b.get(f"accepted_pos_{pos}", 0)

    elif engine == "sglang":
        raw = after.get("raw", {})
        result["acceptance_rate"] = raw.get("accept_rate")
        result["accept_length"] = raw.get("accept_length")
        result["num_drafts"] = raw.get("verify_calls")

    return result


async def stream_one(client: httpx.AsyncClient, url: str, payload: dict, timeout: int) -> dict:
    """Single streaming request with per-token timing (measure_latency methodology).

    Records first/last content-chunk timestamps and token counts from the final
    chunk's usage. Returns a dict of per-request metrics.
    """
    start_t = time.perf_counter()
    first_t = None
    last_t = None
    chunk_content_tokens = 0  # fallback if usage missing
    usage = None
    error = None
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
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
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    if delta.get("content"):
                        now = time.perf_counter()
                        if first_t is None:
                            first_t = now
                        last_t = now
                        chunk_content_tokens += 1
    except Exception as e:
        error = str(e)
    end_t = time.perf_counter()

    ctoks = usage.get("completion_tokens") if usage else None
    cdet = (usage.get("completion_tokens_details") if usage else None) or {}
    rtoks = cdet.get("reasoning_tokens", 0) or 0
    content_tokens = (ctoks - rtoks) if ctoks is not None else chunk_content_tokens

    ttft = (first_t - start_t) if first_t is not None else None
    ttlt = (last_t - start_t) if last_t is not None else None
    has_decode = (first_t is not None and last_t is not None
                  and content_tokens is not None and content_tokens > 1)
    decode_total = (last_t - first_t) if has_decode else None
    dpt_ms = (decode_total / (content_tokens - 1) * 1000) if has_decode else None
    dtok_s = ((content_tokens - 1) / decode_total) if (has_decode and decode_total > 0) else None

    return {
        "ttft_s": ttft,
        "ttlt_s": ttlt,
        "decode_total_s": decode_total,
        "decode_per_token_ms": dpt_ms,
        "decode_tok_s": dtok_s,
        "content_tokens": content_tokens,
        "reasoning_tokens": rtoks,
        "completion_tokens": ctoks,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "wall_s": end_t - start_t,
        "error": error,
    }


def _percentile(sorted_xs: list, p: float):
    """Linear-interpolated percentile of an already-sorted list. p in [0, 100]."""
    if not sorted_xs:
        return None
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    k = (len(sorted_xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_xs) - 1)
    if f == c:
        return sorted_xs[f]
    return sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * (k - f)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


async def run_single_scenario(
    endpoint: str,
    api_key: str,
    model: str,
    payloads: list[dict],
    clients: int,
    n_requests: int,
    max_tokens: int,
    timeout: int,
    label: str,
    enable_thinking: bool = False,
    metrics_url: str = METRICS_URL,
    engine: str = DEFAULT_ENGINE,
) -> dict:
    """Run one concurrency scenario with native streaming client. Returns stats dict."""
    base = []
    for p in payloads:
        bp = {
            "model": model,
            "messages": p["messages"],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not enable_thinking:
            bp["reasoning_effort"] = "none"
        base.append(bp)
    # round-robin prompts across requests
    req_payloads = [base[i % len(base)] for i in range(n_requests)]

    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    sem = asyncio.Semaphore(clients)

    print(f"\n  [{label}] clients={clients}, n_requests={n_requests}, max_tokens={max_tokens}")
    metrics_before = fetch_metrics(metrics_url, engine)

    async with httpx.AsyncClient(headers=headers) as client:
        async def bounded(i):
            async with sem:
                return await stream_one(client, url, req_payloads[i], timeout)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[bounded(i) for i in range(n_requests)])
        test_time = time.perf_counter() - t0

    await asyncio.sleep(1)  # let server flush metrics
    metrics_after = fetch_metrics(metrics_url, engine)
    acceptance = compute_acceptance(metrics_before, metrics_after)

    ok = [r for r in results if not r["error"]]
    failed = len(results) - len(ok)

    def col(key):
        return sorted([r[key] for r in ok if r.get(key) is not None])

    ttft = col("ttft_s")
    ttlt = col("ttlt_s")
    dpt = col("decode_per_token_ms")
    dtok = col("decode_tok_s")
    walls = col("wall_s")

    total_content_tokens = sum((r["content_tokens"] or 0) for r in ok)
    total_completion_tokens = sum((r["completion_tokens"] or 0) for r in ok)
    total_prompt_tokens = sum((r["prompt_tokens"] or 0) for r in ok)
    output_tps = (total_content_tokens / test_time) if test_time > 0 else None

    return {
        "label": label,
        "clients": clients,
        "n_requests": n_requests,
        "max_tokens": max_tokens,
        "total_requests": n_requests,
        "failed_requests": failed,
        "total_test_time_s": test_time,
        "ttft_p50_s": _percentile(ttft, 50),
        "ttft_p90_s": _percentile(ttft, 90),
        "ttlt_p50_s": _percentile(ttlt, 50),
        "ttlt_p90_s": _percentile(ttlt, 90),
        "tpot_mean_ms": _mean(dpt),
        "decode_tok_s_mean": _mean(dtok),
        "output_tps": output_tps,
        "input_tokens_avg": (total_prompt_tokens / len(ok)) if ok else None,
        "output_tokens_avg": (total_completion_tokens / len(ok)) if ok else None,
        "content_tokens_total": total_content_tokens,
        "wall_p50_s": _percentile(walls, 50),
        "acceptance": acceptance,
        "acceptance_rate": acceptance.get("acceptance_rate"),
        "accept_length": acceptance.get("accept_length"),
        "spec_engine": acceptance.get("engine"),
        "per_request": results,
    }


def _fmt(val, template=".3f"):
    """Safely format a value that might be None."""
    if val is None:
        return "N/A"
    if isinstance(template, str):
        try:
            return format(val, template)
        except (ValueError, TypeError):
            return str(val)
    return str(val)


def _ms(val_ms) -> str:
    """Format milliseconds string."""
    if val_ms is None:
        return "N/A"
    return f"{val_ms:.1f}"


def generate_report(results: list[dict], label: str, model: str, endpoint: str) -> str:
    """Generate a Markdown benchmark report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Benchmark Report: {label}",
        "",
        f"**Date**: {now}",
        f"**Model**: {model}",
        f"**Endpoint**: {endpoint}",
        f"**Tool**: custom streaming client (measure_latency methodology, httpx)",
        f"**Mode**: streaming",
        f"**Thinking**: disabled (`reasoning_effort=none`)",
        "",
        "## Results",
        "",
        "| Concurrency | Requests | Time(s) | TTFT p50 | TTFT p90 | TTLT p90 | TPOT mean | decode tok/s | Output TPS | Acceptance | Accept len | Failed |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in results:
        acc = r.get("acceptance_rate")
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        al = r.get("accept_length")
        al_str = f"{al:.2f}" if al is not None else "N/A"
        lines.append(
            f"| {r['clients']} | {r['total_requests']} | "
            f"{_fmt(r['total_test_time_s'], '.1f')} | "
            f"{_fmt(r['ttft_p50_s'])}s | {_fmt(r['ttft_p90_s'])}s | "
            f"{_fmt(r['ttlt_p90_s'])}s | "
            f"{_ms(r['tpot_mean_ms'])}ms | "
            f"{_fmt(r['decode_tok_s_mean'], '.1f')} | "
            f"{_fmt(r['output_tps'], '.1f')} | "
            f"{acc_str} | "
            f"{al_str} | "
            f"{r['failed_requests']} |"
        )
    lines.append("")

    # Detail per scenario
    for r in results:
        lines.append(f"### {r['clients']} concurrent × {r['n_requests']} requests")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total requests | {r['total_requests']} |")
        lines.append(f"| Failed requests | {r['failed_requests']} |")
        lines.append(f"| Total time | {_fmt(r['total_test_time_s'], '.2f')}s |")
        lines.append(f"| Avg input tokens | {_fmt(r['input_tokens_avg'], '.0f')} |")
        lines.append(f"| Avg output tokens | {_fmt(r['output_tokens_avg'], '.0f')} |")
        lines.append(f"| TTFT p50 | {_fmt(r['ttft_p50_s'])}s |")
        lines.append(f"| TTFT p90 | {_fmt(r['ttft_p90_s'])}s |")
        lines.append(f"| TTLT p50 | {_fmt(r['ttlt_p50_s'])}s |")
        lines.append(f"| TTLT p90 | {_fmt(r['ttlt_p90_s'])}s |")
        lines.append(f"| TPOT mean (decode per-token) | {_ms(r['tpot_mean_ms'])}ms |")
        lines.append(f"| **decode tok/s (mean per-request)** | **{_fmt(r['decode_tok_s_mean'], '.1f')}** |")
        lines.append(f"| Output TPS (aggregate, incl TTFT) | {_fmt(r['output_tps'], '.1f')} tok/s |")

        acc = r.get("acceptance", {})
        lines.append(f"| Spec engine | {acc.get('engine') or 'N/A'} |")
        acc_rate = r.get("acceptance_rate")
        acc_rate_str = f"{acc_rate:.2f}" if acc_rate is not None else "N/A"
        acc_len = acc.get("accept_length")
        acc_len_str = f"{acc_len:.2f}" if acc_len is not None else "N/A"
        lines.append(f"| **Acceptance rate** | **{acc_rate_str}** |")
        lines.append(f"| **Accept length** (incl bonus) | **{acc_len_str}** |")
        if acc.get("draft_tokens") is not None:
            lines.append(f"| Draft tokens (proposed) | {acc['draft_tokens']:.0f} |")
            lines.append(f"| Accepted tokens | {acc['accepted_tokens']:.0f} |")
        for pos in range(4):
            val = acc.get(f"accepted_pos_{pos}")
            if val:  # None (sglang) or 0 (vllm k=3 pos 3) -> skip
                lines.append(f"| Accepted at position {pos} | {val:.0f} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by benchmark.py --label {label}*")
    lines.append("")
    lines.append(
        "> **计时口径**：TTFT = 首 content token 延迟（≈prefill）；"
        "decode tok/s = (content_tokens−1)/(末−首 token)，即 **decode 阶段有效吞吐（不含 prefill）**，"
        "与 `measure_latency.py` 同源。"
        "Output TPS = Σ content_tokens / 总耗时（含 TTFT，整体聚合，对齐 llmeter 历史口径）。"
    )
    lines.append("")
    lines.append(
        "> **Acceptance 口径**：acceptance_rate = accepted/proposed drafts（不含 bonus），"
        "vLLM/SGLang 跨引擎可比。accept_length 含 bonus token。"
        "vLLM 走 Counter before/after delta（含逐位明细）；"
        "SGLang 走 Gauge 瞬时直读（无逐位明细）。"
    )
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local vLLM/SGLang deployment with native streaming client",
    )
    parser.add_argument(
        "--label", required=True,
        help="Config label for this run, e.g. 'mtp_k3_native'. Used in report filename.",
    )
    parser.add_argument(
        "--concurrency", default="1,5,10",
        help="Comma-separated concurrency levels (default: 1,5,10)",
    )
    parser.add_argument(
        "--n-requests", type=int, default=10,
        help="Requests per concurrency level (default: 10)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens per request (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT,
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name for API requests (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY,
        help="API key (default: not-needed)",
    )
    parser.add_argument(
        "--prompts-dir", default=str(PROMPTS_DIR),
        help=f"Directory containing prompt_*.txt files (default: {PROMPTS_DIR})",
    )
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Enable Qwen3 thinking mode (default: disabled via reasoning_effort=none)",
    )
    parser.add_argument(
        "--engine", default=DEFAULT_ENGINE,
        choices=["auto", "vllm", "sglang"],
        help="Metrics backend for spec-decode acceptance: auto-detect (default) / vllm / sglang",
    )
    args = parser.parse_args()

    concurrency_levels = parse_concurrency(args.concurrency)
    metrics_url = metrics_url_from_endpoint(args.endpoint)

    print("=" * 60)
    print(f"  Benchmark: {args.label}")
    print(f"  Endpoint:  {args.endpoint}")
    print(f"  Metrics:   {metrics_url}  (engine={args.engine})")
    print(f"  Model:     {args.model}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Requests per level: {args.n_requests}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 60)

    # Verify metrics endpoint
    initial = fetch_metrics(metrics_url, args.engine)
    det = initial.get("engine")
    if det:
        print(f"\n  Metrics available ({metrics_url}) — engine: {det}")
    else:
        print(f"\n  [WARN] No spec-decode metrics detected at {metrics_url}"
              f"{' (engine=' + args.engine + ')' if args.engine != 'auto' else ''}.")
        print(f"  Acceptance data will be N/A. Ensure the server was launched with --enable-metrics.")

    # Load prompts
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        print(f"\n  Creating prompts directory: {prompts_dir}")
        prompts_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_prompts(prompts_dir)

    # Run scenarios sequentially
    results = []
    for clients in concurrency_levels:
        try:
            r = await run_single_scenario(
                endpoint=args.endpoint,
                api_key=args.api_key,
                model=args.model,
                payloads=payloads,
                clients=clients,
                n_requests=args.n_requests,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                label=args.label,
                enable_thinking=args.enable_thinking,
                metrics_url=metrics_url,
                engine=args.engine,
            )
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] concurrency={clients} failed: {e}")

    if not results:
        print("\n  No results collected. Aborting.")
        sys.exit(1)

    # Generate report
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, args.label, args.model, args.endpoint)

    report_path = DOCS_DIR / f"benchmark-{args.label}.md"
    report_path.write_text(report)
    print(f"\n  Report saved to {report_path}")

    # Save raw JSON (includes per-request detail)
    json_path = DOCS_DIR / f"benchmark-{args.label}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"  JSON saved to {json_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.label}")
    print(f"{'=' * 60}")
    print(f"  {'conc':>4}  {'decode/s':>8}  {'TPOT':>7}  {'out_tps':>8}  {'accept':>6}  {'acclen':>6}  {'fail':>4}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*4}")
    for r in results:
        acc = r.get("acceptance_rate")
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        al = r.get("accept_length")
        al_str = f"{al:.2f}" if al is not None else "N/A"
        print(f"  {r['clients']:>4}  "
              f"{_fmt(r['decode_tok_s_mean'], '.1f'):>8}  "
              f"{_ms(r['tpot_mean_ms']):>7}ms  "
              f"{_fmt(r['output_tps'], '.1f'):>8}  "
              f"{acc_str:>6}  "
              f"{al_str:>6}  "
              f"{r['failed_requests']:>4}")


if __name__ == "__main__":
    asyncio.run(main())
