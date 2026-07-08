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
import sys
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


async def run_single_scenario(
    endpoint: OpenAICompletionStreamEndpoint,
    payloads: list[dict],
    clients: int,
    n_requests: int,
    max_tokens: int,
    timeout: int,
    label: str,
) -> dict:
    """Run one concurrency scenario. Returns stats dict."""
    # Inject max_tokens into each payload
    scenario_payloads = [{**p, "max_tokens": max_tokens} for p in payloads]

    print(f"\n  [{label}] clients={clients}, n_requests={n_requests}, max_tokens={max_tokens}")
    runner = Runner(
        endpoint=endpoint,
        payload=scenario_payloads,
        clients=clients,
        n_requests=n_requests,
        timeout=timeout,
    )
    result = await runner.run()
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
              "TTLT p50 | TTLT p90 | TPOT mean | Output TPS | RPM | Failed |")
    sep = ("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    lines.append(header)
    lines.append(sep)

    for r in results:
        lines.append(
            f"| {r['clients']} | {r['total_requests']} | "
            f"{_fmt(r['total_test_time_s'], '.1f')} | "
            f"{_fmt(r['ttft_p50_s'])}s | {_fmt(r['ttft_p90_s'])}s | "
            f"{_fmt(r['ttlt_p50_s'])}s | {_fmt(r['ttlt_p90_s'])}s | "
            f"{_ms(r['tpot_mean_s'])}ms | "
            f"{_fmt(r['output_tps'], '.1f')} | {_fmt(r['rpm'], '.1f')} | "
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
    for r in results:
        print(f"  concurrency={r['clients']:>3}:  "
              f"output_tps={_fmt(r['output_tps'], '.1f'):>8}  "
              f"TTFT_p50={_fmt(r['ttft_p50_s']):>6}s  "
              f"TPOT_mean={_ms(r['tpot_mean_s']):>6}ms  "
              f"failed={r['failed_requests']}")


if __name__ == "__main__":
    asyncio.run(main())
