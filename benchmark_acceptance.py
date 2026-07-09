"""Run benchmark + capture vLLM spec_decode acceptance metrics.

Usage:
    uv run python benchmark_acceptance.py --label "acceptance_check" --concurrency "1,5,10"
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

from llmeter.endpoints.openai import OpenAICompletionStreamEndpoint
from llmeter.runner import Runner

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DOCS_DIR = PROJECT_ROOT / "docs"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_API_KEY = "not-needed"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 300
METRICS_URL = "http://localhost:8000/metrics"


def fetch_metrics() -> dict:
    """Fetch spec decode metrics from vLLM Prometheus endpoint."""
    try:
        result = subprocess.run(
            ["curl", "-s", METRICS_URL],
            capture_output=True, text=True, timeout=10,
        )
        text = result.stdout
    except Exception as e:
        print(f"  [WARN] Failed to fetch metrics: {e}")
        return {}

    metrics = {}

    # Extract cumulative counters (_total)
    patterns = {
        "accepted_tokens": r'vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} (\S+)',
        "draft_tokens": r'vllm:spec_decode_num_draft_tokens_total\{[^}]*\} (\S+)',
        "num_drafts": r'vllm:spec_decode_num_drafts_total\{[^}]*\} (\S+)',
    }

    for name, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            metrics[name] = float(m.group(1))

    # Per-position accepted tokens
    for pos in range(4):  # k=3 → positions 0,1,2 (+1 for safety)
        m = re.search(
            rf'vllm:spec_decode_num_accepted_tokens_per_pos_total\{{[^}}]*position="{pos}"[^}}]*\}}\s+(\S+)',
            text,
        )
        if m:
            metrics[f"accepted_pos_{pos}"] = float(m.group(1))

    return metrics


def compute_acceptance(before: dict, after: dict) -> dict:
    """Compute acceptance rate from before/after metric snapshots."""
    draft_delta = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    accepted_delta = after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)

    result = {
        "draft_tokens": draft_delta,
        "accepted_tokens": accepted_delta,
    }

    if draft_delta > 0:
        result["acceptance_rate"] = accepted_delta / draft_delta
    else:
        result["acceptance_rate"] = None

    # Per-position acceptance
    for pos in range(4):
        pos_before = before.get(f"accepted_pos_{pos}", 0)
        pos_after = after.get(f"accepted_pos_{pos}", 0)
        result[f"accepted_pos_{pos}"] = pos_after - pos_before

    return result


def parse_concurrency(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_prompts(prompts_dir: Path) -> list[dict]:
    pattern = "prompt_*.txt"
    prompt_files = sorted(prompts_dir.glob(pattern))
    if not prompt_files:
        prompt_files = sorted(prompts_dir.glob("prompt_*"))
        prompt_files = [f for f in prompt_files if f.suffix == ".txt"]
    if not prompt_files:
        print(f"  No prompt files found matching {pattern} in {prompts_dir}")
        return [{"messages": [{"role": "user", "content": "Explain what machine learning is in 3 paragraphs."}],
                 "max_tokens": 512}]
    print(f"  Loaded {len(prompt_files)} prompts")
    payloads = []
    for p in prompt_files:
        content = p.read_text().strip()
        payloads.append({
            "messages": [{"role": "user", "content": content}],
            "max_tokens": None,
        })
    return payloads


async def run_single_scenario(
    endpoint, payloads, clients, n_requests, max_tokens, timeout, label,
):
    scenario_payloads = [{**p, "max_tokens": max_tokens} for p in payloads]
    print(f"\n  [{label}] clients={clients}, n_requests={n_requests}, max_tokens={max_tokens}")

    # Snapshot metrics BEFORE
    print(f"  Capturing pre-benchmark metrics...")
    metrics_before = fetch_metrics()

    runner = Runner(
        endpoint=endpoint,
        payload=scenario_payloads,
        clients=clients,
        n_requests=n_requests,
        timeout=timeout,
    )
    result = await runner.run()

    # Snapshot metrics AFTER
    print(f"  Capturing post-benchmark metrics...")
    time.sleep(1)  # let server flush metrics
    metrics_after = fetch_metrics()

    acceptance = compute_acceptance(metrics_before, metrics_after)

    stats = result.stats
    return {
        "label": label,
        "clients": clients,
        "n_requests": n_requests,
        "max_tokens": max_tokens,
        "total_requests": result.total_requests,
        "total_test_time_s": result.total_test_time,
        "ttft_p50_s": stats.get("time_to_first_token-p50"),
        "ttft_p90_s": stats.get("time_to_first_token-p90"),
        "ttlt_p50_s": stats.get("time_to_last_token-p50"),
        "ttlt_p90_s": stats.get("time_to_last_token-p90"),
        "tpot_p50_s": stats.get("time_per_output_token-p50"),
        "tpot_mean_s": stats.get("time_per_output_token-average"),
        "output_tps": stats.get("output_tps"),
        "rpm": stats.get("requests_per_minute"),
        "failed_requests": stats.get("failed_requests", 0),
        "input_tokens_avg": stats.get("num_tokens_input-average"),
        "output_tokens_avg": stats.get("num_tokens_output-average"),
        "acceptance": acceptance,
        # For acceptance rate, average across the run (ratio of cumulative deltas)
        "acceptance_rate": acceptance.get("acceptance_rate"),
        "accepted_tokens_delta": acceptance.get("accepted_tokens", 0),
        "draft_tokens_delta": acceptance.get("draft_tokens", 0),
    }


def _fmt(val, template=".3f"):
    if val is None:
        return "N/A"
    try:
        return format(val, template)
    except (ValueError, TypeError):
        return str(val)


def _ms(val_s) -> str:
    if val_s is None:
        return "N/A"
    return f"{val_s * 1000:.1f}"


def generate_report(results: list[dict], label: str, model: str, endpoint: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Benchmark Report: {label}",
        "",
        f"**Date**: {now}",
        f"**Model**: {model}",
        f"**Endpoint**: {endpoint}",
        f"**Tool**: [awslabs/llmeter](https://github.com/awslabs/llmeter)",
        f"**Mode**: streaming",
        "",
    ]

    # Summary table
    lines.append("## Results")
    lines.append("")
    header = ("| Concurrency | Requests | Time(s) | Output TPS | TPOT mean | "
              "Acceptance | Accepted | Draft | Failed |")
    sep = ("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    lines.append(header)
    lines.append(sep)

    for r in results:
        acc = r.get("acceptance_rate")
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        lines.append(
            f"| {r['clients']} | {r['total_requests']} | "
            f"{_fmt(r['total_test_time_s'], '.1f')} | "
            f"{_fmt(r['output_tps'], '.1f')} | "
            f"{_ms(r['tpot_mean_s'])}ms | "
            f"**{acc_str}** | "
            f"{r.get('accepted_tokens_delta', 'N/A')} | "
            f"{r.get('draft_tokens_delta', 'N/A')} | "
            f"{r['failed_requests']} |"
        )
    lines.append("")

    # Detail per scenario (with acceptance)
    for r in results:
        lines.append(f"### {r['clients']} concurrent × {r['n_requests']} requests")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total requests | {r['total_requests']} |")
        lines.append(f"| Total time | {_fmt(r['total_test_time_s'], '.2f')}s |")
        lines.append(f"| Failed requests | {r['failed_requests']} |")
        lines.append(f"| Avg input tokens | {_fmt(r['input_tokens_avg'], '.0f')} |")
        lines.append(f"| Avg output tokens | {_fmt(r['output_tokens_avg'], '.0f')} |")
        lines.append(f"| TTFT p50 | {_fmt(r['ttft_p50_s'])}s |")
        lines.append(f"| TTFT p90 | {_fmt(r['ttft_p90_s'])}s |")
        lines.append(f"| TTLT p50 | {_fmt(r['ttlt_p50_s'])}s |")
        lines.append(f"| TTLT p90 | {_fmt(r['ttlt_p90_s'])}s |")
        lines.append(f"| TPOT p50 | {_ms(r['tpot_p50_s'])}ms |")
        lines.append(f"| TPOT mean | {_ms(r['tpot_mean_s'])}ms |")
        lines.append(f"| Output throughput | {_fmt(r['output_tps'], '.1f')} tok/s |")
        lines.append(f"| Request rate | {_fmt(r['rpm'], '.1f')} rpm |")

        # Acceptance breakdown
        acc = r.get("acceptance", {})
        acc_rate = r.get("acceptance_rate")
        lines.append(f"| **Acceptance rate** | **{acc_rate:.2f}** |" if acc_rate is not None else "| **Acceptance rate** | N/A |")
        lines.append(f"| Draft tokens (total) | {acc.get('draft_tokens', 'N/A')} |")
        lines.append(f"| Accepted tokens (total) | {acc.get('accepted_tokens', 'N/A')} |")
        for pos in range(4):
            key = f"accepted_pos_{pos}"
            val = acc.get(key, "N/A")
            if val != "N/A" and val > 0:
                lines.append(f"| Accepted at position {pos} | {val:.0f} |")
            elif val == "N/A":
                lines.append(f"| Accepted at position {pos} | N/A |")
        lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by benchmark_acceptance.py --label {label}*")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM with acceptance rate capture",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", default="1,5,10")
    parser.add_argument("--n-requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--prompts-dir", default=str(PROMPTS_DIR))
    args = parser.parse_args()

    concurrency_levels = parse_concurrency(args.concurrency)

    print("=" * 60)
    print(f"  Benchmark (with acceptance): {args.label}")
    print(f"  Endpoint:  {args.endpoint}")
    print(f"  Model:     {args.model}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Requests per level: {args.n_requests}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Metrics URL: {METRICS_URL}")
    print("=" * 60)

    # Verify metrics endpoint
    initial = fetch_metrics()
    if not initial:
        print("\n  [WARN] Could not fetch vLLM metrics. Acceptance data will be unavailable.")
        print("  Ensure --enable-metrics is set in vLLM args, or check http://localhost:8000/metrics")
    else:
        print(f"\n  Initial metrics snapshot:")
        for k, v in sorted(initial.items()):
            print(f"    {k} = {v:.0f}" if isinstance(v, float) else f"    {k} = {v}")

    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        prompts_dir.mkdir(parents=True, exist_ok=True)
    payloads = load_prompts(prompts_dir)

    ep = OpenAICompletionStreamEndpoint(
        model_id=args.model,
        endpoint_name=args.label,
        provider="local-vllm",
        api_key=args.api_key,
        base_url=args.endpoint,
    )

    results = []
    for clients in concurrency_levels:
        try:
            r = await run_single_scenario(
                endpoint=ep,
                payloads=payloads,
                clients=clients,
                n_requests=args.n_requests,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                label=args.label,
            )
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] concurrency={clients} failed: {e}")

    if not results:
        print("\n  No results collected.")
        sys.exit(1)

    # Generate report
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, args.label, args.model, args.endpoint)

    report_path = DOCS_DIR / f"benchmark-{args.label}.md"
    report_path.write_text(report)
    print(f"\n  Report saved to {report_path}")

    # Save JSON
    json_path = DOCS_DIR / f"benchmark-{args.label}.json"
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items()}
        json_results.append(jr)
    json_path.write_text(json.dumps(json_results, indent=2, ensure_ascii=False, default=str))
    print(f"  JSON saved to {json_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Summary: {args.label}")
    print(f"{'=' * 70}")
    print(f"  {'conc':>4}  {'output_tps':>10}  {'TPOT':>7}  {'Acceptance':>10}  {'Accepted':>10}  {'Draft':>10}  {'failed':>6}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
    for r in results:
        acc = r.get("acceptance_rate")
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        print(
            f"  {r['clients']:>4}  "
            f"{_fmt(r['output_tps'], '.1f'):>10}  "
            f"{_ms(r['tpot_mean_s']):>7}ms  "
            f"{acc_str:>10}  "
            f"{r.get('accepted_tokens_delta', 0):>10.0f}  "
            f"{r.get('draft_tokens_delta', 0):>10.0f}  "
            f"{r['failed_requests']:>6}"
        )


if __name__ == "__main__":
    asyncio.run(main())
