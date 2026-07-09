"""Benchmark local vLLM deployment with llmeter and generate a report.

Usage:
    # Baseline (MTP)
    uv run python benchmark.py --label "mtp_k3" --concurrency 1,5,10

    # Stage A (DFlash)
    uv run python benchmark.py --label "dflash_k7" --concurrency 1,5,10

    # Compare against baseline
    uv run python benchmark.py --label "dflash_k10" --concurrency 1,5,10,20

Requirements:
    pip install llmeter
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


def parse_concurrency(raw: str) -> list[int]:
    """Parse comma-separated concurrency levels, e.g. '1,5,10' -> [1, 5, 10]."""
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_prompts(prompts_dir: Path) -> list[dict]:
    """Load prompt_*.txt files, return list of OpenAI chat payloads."""
    pattern = "prompt_*.txt"
    prompt_files = sorted(prompts_dir.glob(pattern))
    if not prompt_files:
        # Also try prompt_N.txt pattern
        prompt_files = sorted(prompts_dir.glob("prompt_*"))
        prompt_files = [f for f in prompt_files if f.suffix == ".txt"]

    if not prompt_files:
        print(f"  No prompt files found matching {pattern} in {prompts_dir}")
        # Use a default simple prompt
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


def compute_acceptance(before: dict, after: dict) -> dict:
    """Compute acceptance rate from before/after metric snapshots."""
    draft_delta = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    accepted_delta = after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)

    result = {
        "draft_tokens": draft_delta,
        "accepted_tokens": accepted_delta,
    }
    result["acceptance_rate"] = accepted_delta / draft_delta if draft_delta > 0 else None

    for pos in range(4):
        pos_before = before.get(f"accepted_pos_{pos}", 0)
        pos_after = after.get(f"accepted_pos_{pos}", 0)
        result[f"accepted_pos_{pos}"] = pos_after - pos_before

    return result


async def run_single_scenario(
    endpoint: OpenAICompletionStreamEndpoint,
    payloads: list[dict],
    clients: int,
    n_requests: int,
    max_tokens: int,
    timeout: int,
    label: str,
    disable_thinking: bool = False,
) -> dict:
    """Run one concurrency scenario. Returns stats dict."""
    # Inject max_tokens into each payload
    scenario_payloads = [{**p, "max_tokens": max_tokens} for p in payloads]
    if disable_thinking:
        # HuggingFace 官方推荐方式：chat_template_kwargs={"enable_thinking": False}
        scenario_payloads = [
            {**p, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            for p in scenario_payloads
        ]

    print(f"\n  [{label}] clients={clients}, n_requests={n_requests}, max_tokens={max_tokens}")
    metrics_before = fetch_metrics()

    runner = Runner(
        endpoint=endpoint,
        payload=scenario_payloads,
        clients=clients,
        n_requests=n_requests,
        timeout=timeout,
    )
    result = await runner.run()

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
        "prompts_used": len(payloads),
        "acceptance": acceptance,
        "acceptance_rate": acceptance.get("acceptance_rate"),
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


def _ms(val_s) -> str:
    """Format seconds as milliseconds string."""
    if val_s is None:
        return "N/A"
    return f"{val_s * 1000:.1f}"


def generate_report(results: list[dict], label: str, model: str, endpoint: str) -> str:
    """Generate a Markdown benchmark report."""
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
    header = ("| Concurrency | Requests | Time(s) | TTFT p50 | TTFT p90 | "
              "TTLT p50 | TTLT p90 | TPOT mean | Output TPS | Acceptance | RPM | Failed |")
    sep = ("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    lines.append(header)
    lines.append(sep)

    for r in results:
        acc = r.get("acceptance_rate")
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        lines.append(
            f"| {r['clients']} | {r['total_requests']} | "
            f"{_fmt(r['total_test_time_s'], '.1f')} | "
            f"{_fmt(r['ttft_p50_s'])}s | {_fmt(r['ttft_p90_s'])}s | "
            f"{_fmt(r['ttlt_p50_s'])}s | {_fmt(r['ttlt_p90_s'])}s | "
            f"{_ms(r['tpot_mean_s'])}ms | "
            f"{_fmt(r['output_tps'], '.1f')} | "
            f"{acc_str} | "
            f"{_fmt(r['rpm'], '.1f')} | "
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
        acc_rate_str = f"{acc_rate:.2f}" if acc_rate is not None else "N/A"
        lines.append(f"| **Acceptance rate** | **{acc_rate_str}** |")
        for pos in range(4):
            val = acc.get(f"accepted_pos_{pos}", 0)
            if val > 0:
                lines.append(f"| Accepted at position {pos} | {val:.0f} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by benchmark.py --label {label}*")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local vLLM deployment with llmeter",
    )
    parser.add_argument(
        "--label", required=True,
        help="Config label for this run, e.g. 'mtp_k3', 'dflash_k7'. Used in report filename.",
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
        "--disable-thinking", action="store_true",
        help="Disable Qwen3 thinking mode via chat_template_kwargs (default: thinking enabled)",
    )
    args = parser.parse_args()

    concurrency_levels = parse_concurrency(args.concurrency)

    print("=" * 60)
    print(f"  Benchmark: {args.label}")
    print(f"  Endpoint:  {args.endpoint}")
    print(f"  Model:     {args.model}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Requests per level: {args.n_requests}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 60)

    # Verify metrics endpoint
    initial = fetch_metrics()
    if initial:
        print(f"\n  vLLM metrics available ({METRICS_URL})")
    else:
        print(f"\n  [WARN] vLLM metrics not available. Acceptance data will be N/A.")
        print(f"  Ensure --enable-metrics is set, or check {METRICS_URL}")

    # Load prompts
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        print(f"\n  Creating prompts directory: {prompts_dir}")
        prompts_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_prompts(prompts_dir)

    # Setup endpoint
    ep = OpenAICompletionStreamEndpoint(
        model_id=args.model,
        endpoint_name=args.label,
        provider="local-vllm",
        api_key=args.api_key,
        base_url=args.endpoint,
    )

    # Run scenarios sequentially
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
                disable_thinking=args.disable_thinking,
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

    # Save raw JSON
    json_path = DOCS_DIR / f"benchmark-{args.label}.json"
    json_results = [{k: v for k, v in r.items()} for r in results]
    json_path.write_text(json.dumps(json_results, indent=2, ensure_ascii=False, default=str))
    print(f"  JSON saved to {json_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.label}")
    print(f"{'=' * 60}")
    print(f"  {'conc':>4}  {'output_tps':>10}  {'TPOT':>7}  {'Acceptance':>10}  {'failed':>6}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*6}")
    for r in results:
        acc = r.get("acceptance_rate")
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        print(f"  {r['clients']:>4}  "
              f"{_fmt(r['output_tps'], '.1f'):>10}  "
              f"{_ms(r['tpot_mean_s']):>7}ms  "
              f"{acc_str:>10}  "
              f"{r['failed_requests']:>6}")


if __name__ == "__main__":
    asyncio.run(main())
