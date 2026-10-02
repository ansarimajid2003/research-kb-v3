"""
Benchmark KB MCP query latency and result quality.

The script exercises the same FastMCP API used by scripts/kb_query.py, keeps a
single client session open, and writes a JSON report under build/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastmcp.client import BearerAuth, Client


DEFAULT_URL = "http://localhost:8765/mcp"
MODES = ("hybrid", "semantic", "keyword")


@dataclass(frozen=True)
class BenchCase:
    name: str
    query: str
    expected_terms: tuple[str, ...]
    domain: str = "quant"
    topic: str | None = None
    strategy: str | None = None
    year_from: int | None = None
    top_k: int = 5
    min_score: float = 0.0


CASES = [
    BenchCase(
        name="pairs_cointegration",
        query="pairs trading cointegration mean reversion spread trading",
        topic="pairs_trading",
        strategy="mean-reversion",
        expected_terms=("pairs trading", "cointegration", "mean reversion", "spread"),
    ),
    BenchCase(
        name="optimal_execution",
        query="optimal execution Almgren Chriss market impact transaction costs",
        topic="execution_optimal",
        expected_terms=("optimal execution", "almgren", "market impact", "transaction cost"),
    ),
    BenchCase(
        name="crypto_pump_dump",
        query="cryptocurrency pump and dump detection thresholding social signals",
        topic="crypto_strategies",
        expected_terms=("pump", "dump", "cryptocurrency", "detection", "threshold"),
    ),
    BenchCase(
        name="market_microstructure_lob",
        query="limit order book order flow imbalance high frequency market microstructure",
        topic="market_microstructure",
        expected_terms=("limit order book", "order flow", "imbalance", "microstructure"),
    ),
    BenchCase(
        name="momentum_transformer",
        query="time series momentum transformer trend following financial time series",
        topic="momentum_trend",
        strategy="momentum",
        expected_terms=("momentum", "transformer", "trend following", "time series"),
    ),
    BenchCase(
        name="risk_var_es",
        query="value at risk expected shortfall tail risk conformal risk control",
        topic="risk_management",
        strategy="volatility",
        expected_terms=("value at risk", "expected shortfall", "tail risk", "conformal"),
    ),
    BenchCase(
        name="portfolio_optimization",
        query="portfolio optimization mean variance risk parity transaction costs",
        topic="portfolio_construction",
        expected_terms=("portfolio optimization", "mean variance", "risk parity", "transaction cost"),
    ),
    BenchCase(
        name="nlp_sentiment_oil",
        query="LLM sentiment news crude oil WTI return forecasting uncertainty",
        topic="nlp_trading",
        strategy="ml/alternative-data",
        year_from=2020,
        expected_terms=("sentiment", "news", "crude oil", "WTI", "forecasting"),
    ),
]


def auth_token() -> str:
    token = os.environ.get("MCP_API_KEY") or os.environ.get("KB_MCP_API_KEY")
    if not token:
        raise SystemExit("Set MCP_API_KEY or KB_MCP_API_KEY.")
    return token


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_ms": 0, "median_ms": 0, "p95_ms": 0, "min_ms": 0, "max_ms": 0}
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 1),
        "median_ms": round(statistics.median(values), 1),
        "p95_ms": round(percentile(values, 0.95), 1),
        "min_ms": round(min(values), 1),
        "max_ms": round(max(values), 1),
    }


def result_text(result: dict[str, Any]) -> str:
    fields = [
        result.get("title", ""),
        result.get("abstract", ""),
        result.get("text", ""),
        result.get("topic", ""),
        result.get("strategy_category", ""),
    ]
    return " ".join(str(f) for f in fields if f).lower()


def term_hit(term: str, text: str) -> bool:
    return term.lower() in text


def judge_result(case: BenchCase, result: dict[str, Any]) -> dict[str, Any]:
    text = result_text(result)
    term_hits = [term for term in case.expected_terms if term_hit(term, text)]
    topic_hit = bool(case.topic and result.get("topic") == case.topic)
    strategy_hit = bool(case.strategy and result.get("strategy_category") == case.strategy)

    # Topic/strategy metadata are strong signals; otherwise require at least one
    # query-specific term hit in the title, abstract, or payload text.
    relevant = topic_hit or strategy_hit or bool(term_hits)
    return {
        "relevant": relevant,
        "topic_hit": topic_hit,
        "strategy_hit": strategy_hit,
        "term_hits": term_hits,
    }


def quality_for(case: BenchCase, results: list[dict[str, Any]]) -> dict[str, Any]:
    judgments = [judge_result(case, r) for r in results]
    relevant_ranks = [i + 1 for i, j in enumerate(judgments) if j["relevant"]]
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
    return {
        "results": len(results),
        "hit_at_1": bool(judgments and judgments[0]["relevant"]),
        "precision_at_5": round(sum(j["relevant"] for j in judgments[:5]) / max(1, min(5, len(results))), 3),
        "mrr": round(1 / first_relevant_rank, 3) if first_relevant_rank else 0.0,
        "first_relevant_rank": first_relevant_rank,
        "topic_hits_at_5": sum(j["topic_hit"] for j in judgments[:5]),
        "strategy_hits_at_5": sum(j["strategy_hit"] for j in judgments[:5]),
        "judgments": judgments,
    }


def build_payload(case: BenchCase, mode: str, use_case_filters: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": case.query,
        "domain": case.domain,
        "top_k": case.top_k,
        "mode": mode,
        "min_score": case.min_score,
    }
    if use_case_filters and case.topic:
        payload["topic"] = case.topic
    if use_case_filters and case.strategy:
        payload["strategy"] = case.strategy
    if use_case_filters and case.year_from:
        payload["year_from"] = case.year_from
    return payload


async def timed_tool(client: Client, tool: str, payload: dict[str, Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = await client.call_tool(tool, payload)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return (result.data if hasattr(result, "data") else result), elapsed_ms


async def run_benchmark(url: str, repeats: int, modes: tuple[str, ...], use_case_filters: bool) -> dict[str, Any]:
    async with Client(url, auth=BearerAuth(auth_token()), timeout=90) as client:
        info, info_ms = await timed_tool(client, "info", {})
        runs = []
        for case in CASES:
            for mode in modes:
                payload = build_payload(case, mode, use_case_filters=use_case_filters)
                for trial in range(1, repeats + 1):
                    try:
                        data, elapsed_ms = await timed_tool(client, "search", payload)
                        results = data if isinstance(data, list) else []
                        quality = quality_for(case, results)
                        runs.append(
                            {
                                "case": case.name,
                                "query": case.query,
                                "mode": mode,
                                "trial": trial,
                                "elapsed_ms": round(elapsed_ms, 1),
                                "quality": quality,
                                "top_results": [
                                    {
                                        "rank": i + 1,
                                        "title": r.get("title"),
                                        "topic": r.get("topic"),
                                        "strategy_category": r.get("strategy_category"),
                                        "score": r.get("score"),
                                        "source_id": r.get("source_id"),
                                        "judgment": quality["judgments"][i],
                                    }
                                    for i, r in enumerate(results[:5])
                                ],
                            }
                        )
                    except Exception as exc:
                        runs.append(
                            {
                                "case": case.name,
                                "query": case.query,
                                "mode": mode,
                                "trial": trial,
                                "error": repr(exc),
                            }
                        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "repeats": repeats,
            "modes": modes,
            "use_case_filters": use_case_filters,
            "info_ms": round(info_ms, 1),
            "info": info,
            "runs": runs,
        }


def aggregate(report: dict[str, Any]) -> dict[str, Any]:
    successful = [r for r in report["runs"] if "error" not in r]
    out: dict[str, Any] = {"overall": {}, "by_mode": {}, "by_case_mode": {}}
    all_latencies = [r["elapsed_ms"] for r in successful]
    out["overall"]["latency"] = summarize(all_latencies)
    out["overall"]["errors"] = len(report["runs"]) - len(successful)
    out["overall"]["hit_at_1"] = round(
        sum(r["quality"]["hit_at_1"] for r in successful) / max(1, len(successful)), 3
    )
    out["overall"]["precision_at_5"] = round(
        statistics.fmean(r["quality"]["precision_at_5"] for r in successful), 3
    ) if successful else 0.0
    out["overall"]["mrr"] = round(statistics.fmean(r["quality"]["mrr"] for r in successful), 3) if successful else 0.0

    for mode in report["modes"]:
        rows = [r for r in successful if r["mode"] == mode]
        first_pass = [r["elapsed_ms"] for r in rows if r["trial"] == 1]
        repeat_pass = [r["elapsed_ms"] for r in rows if r["trial"] > 1]
        out["by_mode"][mode] = {
            "latency_all": summarize([r["elapsed_ms"] for r in rows]),
            "latency_first_pass": summarize(first_pass),
            "latency_repeats": summarize(repeat_pass),
            "hit_at_1": round(sum(r["quality"]["hit_at_1"] for r in rows) / max(1, len(rows)), 3),
            "precision_at_5": round(statistics.fmean(r["quality"]["precision_at_5"] for r in rows), 3) if rows else 0.0,
            "mrr": round(statistics.fmean(r["quality"]["mrr"] for r in rows), 3) if rows else 0.0,
        }

    for case in sorted({r["case"] for r in successful}):
        for mode in report["modes"]:
            rows = [r for r in successful if r["case"] == case and r["mode"] == mode]
            if not rows:
                continue
            key = f"{case}:{mode}"
            out["by_case_mode"][key] = {
                "latency": summarize([r["elapsed_ms"] for r in rows]),
                "hit_at_1": round(sum(r["quality"]["hit_at_1"] for r in rows) / len(rows), 3),
                "precision_at_5": round(statistics.fmean(r["quality"]["precision_at_5"] for r in rows), 3),
                "top_title": rows[0]["top_results"][0]["title"] if rows[0]["top_results"] else None,
            }
    return out


def print_summary(report: dict[str, Any], summary: dict[str, Any], output_path: Path) -> None:
    print(f"Report: {output_path}")
    print(f"Info call: {report['info_ms']} ms")
    print(f"Corpus: {json.dumps(report.get('info', {}).get('domains', {}), default=str)}")
    print("\nLatency by mode:")
    for mode, stats in summary["by_mode"].items():
        first = stats["latency_first_pass"]
        repeats = stats["latency_repeats"]
        print(
            f"  {mode:8} first median={first['median_ms']:>6} ms p95={first['p95_ms']:>6} ms | "
            f"repeat median={repeats['median_ms']:>6} ms p95={repeats['p95_ms']:>6} ms | "
            f"P@5={stats['precision_at_5']:.3f} H@1={stats['hit_at_1']:.3f}"
        )
    print("\nTop result by case/mode:")
    for key, stats in summary["by_case_mode"].items():
        print(f"  {key:36} {stats['latency']['median_ms']:>6} ms  P@5={stats['precision_at_5']:.3f}  {stats['top_title']}")


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("MCP_URL", DEFAULT_URL))
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="Use only the quant domain filter, ignoring case topic/strategy/year filters.",
    )
    parser.add_argument("--output-dir", default="build")
    args = parser.parse_args()

    report = await run_benchmark(args.url, args.repeats, tuple(args.modes), use_case_filters=not args.unfiltered)
    summary = aggregate(report)
    report["summary"] = summary

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "unfiltered" if args.unfiltered else "filtered"
    output_path = output_dir / f"query_benchmark_{suffix}_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print_summary(report, summary, output_path)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
