"""
Server-side KB V3 MCP tool probe.

Run inside the kb-server container after migration:
    python /tmp/verify_mcp_tools.py
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import asyncpg
from qdrant_client import QdrantClient

from src.kb.embedder import Embedder
from src.kb.ingest import COLLECTION, ingest_pdfs
from src.mcp import tools


def brief_result(row: dict) -> dict:
    return {
        "source_id": row.get("source_id"),
        "title": (row.get("title") or "")[:100],
        "year": row.get("year"),
        "topic": row.get("topic"),
        "strategy": row.get("strategy_category"),
        "score": row.get("score"),
        "mode": row.get("mode"),
    }


async def timed(label: str, coro):
    t0 = time.perf_counter()
    try:
        value = await coro
        ok = True
        error = None
    except Exception as exc:
        value = None
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return label, ok, elapsed_ms, value, error


async def main() -> None:
    qc = QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        check_compatibility=False,
        timeout=120,
    )
    pg = await asyncpg.create_pool(os.environ["PG_DSN"], min_size=1, max_size=4)
    embedder = Embedder.get()

    report: dict = {"checks": [], "samples": {}}

    # Warm the model and measure query embedding separately.
    embedder.embed_query("warmup")
    embed_times = []
    for query in ["momentum factor", "pairs trading cointegration", "portfolio optimization"]:
        t0 = time.perf_counter()
        embedder.embed_query(query)
        embed_times.append((time.perf_counter() - t0) * 1000)
    report["embed_ms"] = {
        "min": min(embed_times),
        "median": sorted(embed_times)[len(embed_times) // 2],
        "max": max(embed_times),
    }

    # Tool 7: info.
    label, ok, ms, info, err = await timed("info", tools.info(pg, qc))
    report["checks"].append({"tool": label, "ok": ok, "elapsed_ms": ms, "error": err})
    report["info"] = info

    # Tool 1: search, including the architecture checklist cases.
    search_cases = [
        (
            "search_hybrid_year_filter",
            tools.search(pg, qc, embedder, "momentum factor", domain="quant", year_from=2020, top_k=10),
        ),
        (
            "search_portfolio_topic",
            tools.search(pg, qc, embedder, "portfolio optimization", topic="portfolio_construction", top_k=10),
        ),
        (
            "search_strategy_mean_reversion",
            tools.search(pg, qc, embedder, "pairs trading", domain="quant", strategy="mean-reversion", top_k=10),
        ),
        (
            "search_semantic",
            tools.search(pg, qc, embedder, "deep learning order book", domain="quant", mode="semantic", top_k=10),
        ),
        (
            "search_keyword",
            tools.search(pg, qc, embedder, "portfolio optimization", domain="quant", mode="keyword", top_k=10),
        ),
    ]

    for label, coro in search_cases:
        name, ok, ms, value, err = await timed(label, coro)
        rows = value or []
        report["checks"].append({"tool": name, "ok": ok, "elapsed_ms": ms, "count": len(rows), "error": err})
        report["samples"][name] = [brief_result(r) for r in rows[:5]]

    year_rows = report["samples"].get("search_hybrid_year_filter", [])
    report["quality_year_filter"] = {
        "all_years_gte_2020": all((r.get("year") or 0) >= 2020 for r in year_rows),
        "years": [r.get("year") for r in year_rows],
    }

    portfolio_rows = report["samples"].get("search_portfolio_topic", [])
    expected_topics = {"mean_variance_optimization", "factor_models", "risk_parity"}
    portfolio_topics = {r.get("topic") for r in portfolio_rows}
    report["quality_portfolio_topic"] = {
        "returned_topics": sorted(t for t in portfolio_topics if t),
        "contains_expected_subtopic": bool(portfolio_topics & expected_topics),
        "expected_subtopics": sorted(expected_topics),
    }

    strategy_rows = report["samples"].get("search_strategy_mean_reversion", [])
    report["quality_strategy_filter"] = {
        "non_empty": bool(strategy_rows),
        "strategies": sorted({r.get("strategy") for r in strategy_rows if r.get("strategy")}),
    }

    hybrid_rows = report["samples"].get("search_hybrid_year_filter", [])
    semantic_rows = report["samples"].get("search_semantic", [])
    report["quality_hybrid_mode"] = {
        "hybrid_score_examples": [r.get("score") for r in hybrid_rows[:3]],
        "semantic_score_examples": [r.get("score") for r in semantic_rows[:3]],
        "hybrid_uses_rrf_shape": all((r.get("score") or 0) < 1 for r in hybrid_rows[:3]),
        "qdrant_has_sparse_vectors": "sparse" in qc.get_collection(COLLECTION).config.params.sparse_vectors,
    }

    # Tool 2: get_source.
    source_id = "2207.02458"
    label, ok, ms, source, err = await timed("get_source", tools.get_source(pg, source_id))
    if source is None:
        source_id = await pg.fetchval("SELECT source_id FROM papers WHERE is_indexed=TRUE LIMIT 1")
        label, ok, ms, source, err = await timed("get_source_fallback", tools.get_source(pg, source_id))
    report["checks"].append({"tool": label, "ok": ok, "elapsed_ms": ms, "source_id": source_id, "error": err})
    report["get_source_quality"] = {
        "found": source is not None,
        "authors_is_list": isinstance((source or {}).get("authors"), list),
        "categories_is_list": isinstance((source or {}).get("categories"), list),
        "relevance_score_is_not_null": (source or {}).get("relevance_score") is not None,
        "title": ((source or {}).get("title") or "")[:120],
    }

    # Tool 3: discover preview only.
    label, ok, ms, discovered, err = await timed(
        "discover_preview",
        tools.discover(
            pg,
            qc,
            embedder,
            "online learning market making",
            domain="quant",
            max_papers=5,
            force_download=False,
            year_from=2020,
            min_score=0.0,
        ),
    )
    report["checks"].append(
        {
            "tool": label,
            "ok": ok,
            "elapsed_ms": ms,
            "count": len((discovered or {}).get("candidates", [])),
            "search_ms": (discovered or {}).get("search_ms"),
            "error": err,
        }
    )
    report["samples"]["discover_preview"] = (discovered or {}).get("candidates", [])[:5]

    # Tool 4: add_source. Use the local health endpoint to avoid relying on internet.
    add_url = f"http://127.0.0.1:8765/health?probe={int(time.time())}"
    label, ok, ms, add_result, err = await timed(
        "add_source_local_health",
        tools.add_source(pg, qc, embedder, add_url, domain="web", topic="misc"),
    )
    report["checks"].append({"tool": label, "ok": ok, "elapsed_ms": ms, "result": add_result, "error": err})
    add_job_id = (add_result or {}).get("job_id")
    if add_job_id and add_result.get("status") == "queued":
        add_status = None
        for _ in range(60):
            add_status = await tools.job_status(pg, add_job_id)
            if add_status.get("status") in {"complete", "failed", "complete_with_warnings"}:
                break
            await asyncio.sleep(1)
        report["checks"].append({"tool": "add_source_job_status", "ok": True, "result": add_status})

    # Tool 5 and 6: index + job_status. Use web domain so this is effectively no-op.
    label, ok, ms, index_result, err = await timed("index_web_noop", tools.index(pg, qc, embedder, "web", False))
    report["checks"].append({"tool": label, "ok": ok, "elapsed_ms": ms, "result": index_result, "error": err})
    job_id = (index_result or {}).get("job_id", "missing")
    await asyncio.sleep(0.5)
    label, ok, ms, job_result, err = await timed("job_status", tools.job_status(pg, job_id))
    report["checks"].append({"tool": label, "ok": ok, "elapsed_ms": ms, "result": job_result, "error": err})

    # Checklist idempotency: re-ingest one already indexed paper if a server PDF exists.
    before_points = qc.get_collection(COLLECTION).points_count
    paper_row = None
    pdf_path = None
    papers_root = Path(os.environ.get("PAPERS_ROOT", "/data/papers"))
    rows = await pg.fetch(
        "SELECT * FROM papers WHERE domain='quant' AND is_indexed=TRUE AND chunk_count > 0 ORDER BY chunk_count ASC LIMIT 200"
    )
    for row in rows:
        row_dict = dict(row)
        local_path = Path(str(row_dict.get("local_path") or ""))
        candidates = [
            papers_root / "quant" / row["topic"] / f"{row['source_id'].replace('/', '_')}.pdf",
            papers_root / "quant" / row["topic"] / f"{row['source_id']}.pdf",
            local_path,
        ]
        if str(local_path).startswith("/DATA/"):
            candidates.append(Path("/data") / local_path.relative_to("/DATA"))
        candidate = next((p for p in candidates if p.exists()), None)
        if candidate is not None:
            paper_row = row_dict
            pdf_path = candidate
            break
    if paper_row is not None and pdf_path is not None:
        t0 = time.perf_counter()
        chunks = await ingest_pdfs(pg, qc, embedder, [(pdf_path, paper_row)], "quant", force=True)
        after_points = qc.get_collection(COLLECTION).points_count
        report["idempotency"] = {
            "tested": True,
            "source_id": paper_row["source_id"],
            "chunks_reingested": chunks,
            "before_points": before_points,
            "after_points": after_points,
            "unchanged": before_points == after_points,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }
    else:
        report["idempotency"] = {
            "tested": False,
            "before_points": before_points,
            "reason": f"no indexed PDF found under {papers_root}",
        }

    await pg.close()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
