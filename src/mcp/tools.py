"""
KB V3 MCP tool implementations — 7 tools.

All tools are async and accept pre-connected pg/qc/embedder passed from server.py.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client.models import (
    FieldCondition, Filter, FusionQuery, Fusion,
    MatchAny, MatchValue, Prefetch, Range,
)

COLLECTION = "kb"
_ARXIV_COUNT_CACHE: dict[str, Any] = {"path": None, "mtime": None, "count": None}


def _arxiv_db_path() -> Path:
    import os
    return Path(os.environ.get("ARXIV_DB", "/data/arxiv_index.db"))


def _safe_source_filename(source_id: str) -> str:
    return f"{source_id.replace('/', '_')}.pdf"


def _paper_pdf_candidates(row, papers_root: Path, domain: str) -> list[Path]:
    source_id = row["source_id"]
    topic = row["topic"]
    candidates = [
        papers_root / domain / topic / _safe_source_filename(source_id),
        papers_root / domain / topic / f"{source_id}.pdf",
    ]
    local_path = row.get("local_path") if hasattr(row, "get") else row["local_path"]
    if local_path:
        path = Path(str(local_path))
        candidates.append(path)
        if str(path).startswith("/DATA/"):
            candidates.append(Path("/data") / path.relative_to("/DATA"))
    return candidates


def _first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _sqlite_tables(con: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }


def _arxiv_paper_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    mtime = db_path.stat().st_mtime
    if (
        _ARXIV_COUNT_CACHE["path"] == str(db_path)
        and _ARXIV_COUNT_CACHE["mtime"] == mtime
        and _ARXIV_COUNT_CACHE["count"] is not None
    ):
        return int(_ARXIV_COUNT_CACHE["count"])
    with sqlite3.connect(db_path) as con:
        tables = _sqlite_tables(con)
        if "arxiv_fts" in tables:
            count = con.execute("SELECT COUNT(*) FROM arxiv_fts").fetchone()[0]
        elif "papers" in tables:
            count = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        elif "papers_fts" in tables:
            count = con.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0]
        else:
            count = 0
    _ARXIV_COUNT_CACHE.update({"path": str(db_path), "mtime": mtime, "count": count})
    return int(count)


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

async def _resolve_topic_filter(pg, topic: str | None, domain: str | None) -> list[str] | None:
    if topic is None:
        return None
    rows = await pg.fetch(
        """SELECT key FROM topics
           WHERE (domain = $2 OR $2 IS NULL)
             AND (key = $1 OR parent_key = $1)""",
        topic, domain,
    )
    return [r["key"] for r in rows] if rows else [topic]


def _build_filter(
    domain: str | None,
    topics: list[str] | None,
    year_from: int | None,
    year_to: int | None,
    source_type: str | None,
    strategy: str | None,
) -> Filter | None:
    must = []
    if domain:
        must.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    if topics:
        must.append(FieldCondition(key="topic", match=MatchAny(any=topics)))
    if year_from or year_to:
        must.append(FieldCondition(key="year", range=Range(gte=year_from, lte=year_to)))
    if source_type:
        must.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))
    if strategy:
        must.append(FieldCondition(key="strategy_category", match=MatchValue(value=strategy)))
    return Filter(must=must) if must else None


def _row_to_dict(row) -> dict:
    d = dict(row)
    # job rows always have job_type; paper/source rows never do — skip paper fields for jobs
    if "job_type" not in d:
        for field in ("authors", "categories"):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except Exception:
                    d[field] = []
            elif d.get(field) is None:
                d[field] = []
    if "relevance_score" not in d and d.get("source_type") == "paper":
        d["relevance_score"] = 0.0
    return d


async def _create_job(
    pg,
    job_type: str,
    domain: str,
    *,
    source_id: str | None = None,
    force_reindex: bool = False,
    sources_total: int = 0,
    message: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    await pg.execute(
        """INSERT INTO index_jobs (
               job_id, job_type, source_id, domain, status, force_reindex,
               sources_total, message, updated_at
           )
           VALUES ($1,$2,$3,$4,'queued',$5,$6,$7,NOW())""",
        job_id, job_type, source_id, domain, force_reindex, sources_total, message,
    )
    return job_id


async def _update_job(pg, job_id: str, **fields) -> None:
    allowed = {
        "status", "started_at", "completed_at", "sources_processed",
        "sources_total", "chunks_added", "message", "error",
    }
    updates = []
    values = []
    for key, value in fields.items():
        if key not in allowed:
            continue
        values.append(value)
        updates.append(f"{key}=${len(values) + 1}")
    if not updates:
        return
    values = [job_id] + values
    await pg.execute(
        f"UPDATE index_jobs SET {', '.join(updates)}, updated_at=NOW() WHERE job_id=$1",
        *values,
    )


# ---------------------------------------------------------------------------
# Tool 1: search
# ---------------------------------------------------------------------------

async def search(
    pg, qc, embedder,
    query: str,
    domain: str | None = None,
    topic: str | None = None,
    top_k: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    source_type: str | None = None,
    strategy: str | None = None,
    min_score: float = 0.0,
    mode: str = "hybrid",
) -> list[dict]:
    """Semantic / hybrid / keyword search across all domains."""
    topics = await _resolve_topic_filter(pg, topic, domain)
    qfilter = _build_filter(domain, topics, year_from, year_to, source_type, strategy)

    if mode == "keyword":
        tsvec = "to_tsvector('english', title || ' ' || COALESCE(abstract,''))"
        tsq   = "plainto_tsquery($1)"
        clauses = [f"{tsvec} @@ {tsq}"]
        params: list[Any] = [query]
        i = 2
        if domain:
            clauses.append(f"domain=${i}"); params.append(domain); i += 1
        if topics:
            clauses.append(f"topic=ANY(${i}::text[])"); params.append(topics); i += 1
        if year_from:
            clauses.append(f"year>=${i}"); params.append(year_from); i += 1
        if year_to:
            clauses.append(f"year<=${i}"); params.append(year_to); i += 1
        if strategy:
            clauses.append(f"strategy_category=${i}"); params.append(strategy); i += 1
        clauses.append("is_indexed=TRUE")
        sql = (
            f"SELECT *, ts_rank({tsvec}, {tsq}) AS _ts_rank "
            f"FROM papers WHERE {' AND '.join(clauses)} "
            f"ORDER BY _ts_rank DESC LIMIT ${i}"
        )
        params.append(top_k)
        rows = await pg.fetch(sql, *params)
        return [_row_to_dict(r) | {"score": float(r["_ts_rank"]), "mode": "keyword"} for r in rows]

    vecs = embedder.embed_query(query)

    if mode == "semantic":
        result = qc.query_points(
            collection_name=COLLECTION,
            query=vecs["dense"],
            using="dense",
            query_filter=qfilter,
            limit=top_k * 3,
            score_threshold=min_score if min_score > 0 else None,
            with_payload=True,
        )
        raw_hits = result.points
    else:  # hybrid — RRF fusion
        result = qc.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=vecs["dense"],  using="dense",  limit=top_k * 3),
                Prefetch(query=vecs["sparse"], using="sparse", limit=top_k * 3),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=qfilter,
            limit=top_k * 3,
            with_payload=True,
        )
        raw_hits = result.points

    seen_base, results = set(), []
    for h in raw_hits:
        sid = h.payload.get("source_id", "")
        base = h.payload.get("base_id") or (sid.split("v")[0] if sid else sid)
        if base not in seen_base:
            seen_base.add(base)
            results.append(h.payload | {"score": h.score, "mode": mode})
        if len(results) >= top_k:
            break
    return results


# ---------------------------------------------------------------------------
# Tool 2: get_source
# ---------------------------------------------------------------------------

async def get_source(pg, source_id: str) -> dict | None:
    """Full metadata for a source by ID."""
    row = await pg.fetchrow(
        """SELECT * FROM papers
           WHERE source_id=$1 OR base_id=$1
           ORDER BY source_id
           LIMIT 1""",
        source_id,
    )
    if row is None:
        row = await pg.fetchrow("SELECT * FROM sources WHERE source_id=$1", source_id)
    if row is None:
        return None
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Tool 3: discover
# ---------------------------------------------------------------------------

_FTS_STOPWORDS = {
    "a","an","the","and","or","of","in","on","for","to","with",
    "is","are","was","were","be","been","how","do","does","this",
    "that","these","those","from","by","at","as","it","its","we",
}

_FINANCIAL_PHRASES = [
    "market making", "limit order book", "order flow", "high frequency",
    "pairs trading", "statistical arbitrage", "mean reversion",
    "momentum factor", "portfolio optimization", "risk premium",
    "reinforcement learning", "deep learning", "neural network",
    "option pricing", "volatility surface", "term structure",
    "optimal execution", "price impact", "market microstructure",
]


def _build_fts_query(query: str) -> str:
    """Convert natural-language query to SQLite FTS5 MATCH expression.

    Phrases are OR-joined so that a document matching any known financial phrase
    is surfaced. Remaining individual tokens (minus stopwords) are appended with
    OR as well, giving broad recall while BM25 still ranks the best matches first.
    Falls back to the original query if nothing survives filtering.
    """
    q_lower = query.lower()
    phrase_terms: list[str] = []
    remaining = q_lower
    for phrase in _FINANCIAL_PHRASES:
        if phrase in remaining:
            phrase_terms.append(f'"{phrase}"')
            remaining = remaining.replace(phrase, " ")
    tokens = [t.strip(".,;:!?") for t in remaining.split()]
    tokens = [t for t in tokens if t not in _FTS_STOPWORDS and len(t) > 2]
    all_terms = phrase_terms + tokens
    return " OR ".join(all_terms) if all_terms else query


async def discover(
    pg, qc, embedder,
    query: str,
    domain: str = "quant",
    topic: str | None = None,
    max_papers: int = 5,
    force_download: bool = False,
    year_from: int | None = None,
    min_score: float = 0.3,
) -> dict:
    """Search arxiv snapshot for papers not yet in KB; optionally download + temp-index."""
    t0 = time.monotonic()
    db_path = _arxiv_db_path()
    fts_query = _build_fts_query(query)

    candidates = []
    warning = None
    if db_path.exists():
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            tables = _sqlite_tables(con)
            if "arxiv_fts" in tables:
                params_sql: list[Any] = [fts_query, max_papers * 8]
                year_clause = ""
                if year_from:
                    year_clause = "AND CAST(SUBSTR(update_date,1,4) AS INTEGER) >= ?"
                    params_sql.insert(-1, year_from)
                rows = con.execute(
                    f"""SELECT arxiv_id, title, abstract, update_date,
                               bm25(arxiv_fts) AS relevance
                        FROM arxiv_fts
                        WHERE arxiv_fts MATCH ?
                        {year_clause}
                        ORDER BY relevance
                        LIMIT ?""",
                    params_sql,
                ).fetchall()
                id_field = "arxiv_id"
                published_field = "update_date"
            elif "papers_fts" in tables and "papers" in tables:
                params_sql = [fts_query, max_papers * 8]
                year_clause = ""
                if year_from:
                    year_clause = "AND p.year >= ?"
                    params_sql.insert(-1, year_from)
                rows = con.execute(
                    f"""SELECT f.id AS arxiv_id, f.title, f.abstract,
                               COALESCE(p.published_date, p.update_date, '') AS update_date,
                               bm25(papers_fts) AS relevance
                        FROM papers_fts f
                        LEFT JOIN papers p ON p.id = f.id
                        WHERE papers_fts MATCH ?
                        {year_clause}
                        ORDER BY relevance
                        LIMIT ?""",
                    params_sql,
                ).fetchall()
                id_field = "arxiv_id"
                published_field = "update_date"
            else:
                rows = []
                warning = f"No supported FTS table found in {db_path}"

        # Filter already-in-KB
        known = set()
        if rows:
            ids = [r[id_field] for r in rows]
            base_ids = [i.split("v")[0] for i in ids]
            known_rows = await pg.fetch(
                """SELECT source_id, base_id FROM papers
                   WHERE source_id=ANY($1::text[]) OR base_id=ANY($2::text[])""",
                ids, base_ids,
            )
            known = {r["source_id"] for r in known_rows} | {r["base_id"] for r in known_rows}

        for r in rows:
            arxiv_id = r[id_field]
            score = abs(r["relevance"])
            if score < min_score:
                continue
            if arxiv_id not in known and arxiv_id.split("v")[0] not in known:
                candidates.append({
                    "arxiv_id":        arxiv_id,
                    "title":           r["title"],
                    "abstract":        (r["abstract"] or "")[:400],
                    "published":       r[published_field],
                    "relevance_score": score,
                    "download_status": "skipped",
                })
            if len(candidates) >= max_papers:
                break
    else:
        warning = f"arXiv FTS database not found at {db_path}"

    search_ms = int((time.monotonic() - t0) * 1000)

    download_ms = 0
    if force_download and candidates:
        from src.kb.ingest import ingest_pdfs, ensure_collection
        from src.kb.classify import classify_paper
        import aiohttp, tempfile

        ensure_collection(qc)
        t1 = time.monotonic()

        async with aiohttp.ClientSession() as session:
            for cand in candidates[:max_papers]:
                aid = cand["arxiv_id"]
                pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
                try:
                    async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            cand["download_status"] = "failed"
                            continue
                        data = await resp.read()
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(data)
                        tmp_path = Path(tmp.name)

                    cat, conf = classify_paper(cand["title"], cand.get("abstract", ""))
                    row = {
                        "source_id": aid,
                        "domain":    domain,
                        "topic":     topic or "misc",
                        "title":     cand["title"],
                        "authors":   [],
                        "abstract":  cand.get("abstract", ""),
                        "published": cand.get("published", ""),
                        "year":      int(cand.get("published", "2000")[:4]) if cand.get("published") else 0,
                        "categories": [],
                        "strategy_category":   cat,
                        "strategy_confidence": conf,
                        "citation_count": 0,
                        "pdf_url":   pdf_url,
                        "is_temp":   True,
                    }
                    await ingest_pdfs(pg, qc, embedder, [(tmp_path, row)], domain, force=True)
                    tmp_path.unlink(missing_ok=True)
                    cand["download_status"] = "downloaded"
                except asyncio.TimeoutError:
                    cand["download_status"] = "timeout"
                except Exception as e:
                    cand["download_status"] = f"failed: {e}"

        download_ms = int((time.monotonic() - t1) * 1000)

    return {
        "candidates":  candidates,
        "search_ms":   search_ms,
        "download_ms": download_ms,
        "warning":     warning,
    }


# ---------------------------------------------------------------------------
# Tool 4: add_source
# ---------------------------------------------------------------------------

async def add_source(pg, qc, embedder, url: str, domain: str = "web", topic: str | None = None) -> dict:
    """Queue a URL for async web ingest."""
    source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

    exists = await pg.fetchval(
        "SELECT is_indexed FROM sources WHERE source_id=$1", source_id
    )
    if exists is True:
        return {"job_id": source_id, "status": "already_indexed", "source_type": "web"}

    await pg.execute(
        """INSERT INTO sources (source_id, source_type, domain, topic, title, is_indexed, meta)
           VALUES ($1,'web',$2,$3,$4,FALSE,$5::jsonb)
           ON CONFLICT (source_id) DO UPDATE
             SET domain=EXCLUDED.domain, topic=EXCLUDED.topic, title=EXCLUDED.title, meta=EXCLUDED.meta""",
        source_id, domain, topic or "misc", url, json.dumps({"url": url}),
    )
    job_id = await _create_job(
        pg,
        "add_source",
        domain,
        source_id=source_id,
        sources_total=1,
        message=f"Queued web ingest for {url}",
    )
    asyncio.create_task(_run_add_source_job(pg, qc, embedder, job_id, source_id, url, domain, topic))
    return {"job_id": job_id, "status": "queued", "source_id": source_id, "source_type": "web"}


async def _run_add_source_job(
    pg,
    qc,
    embedder,
    job_id: str,
    source_id: str,
    url: str,
    domain: str,
    topic: str | None,
) -> None:
    import aiohttp
    import re
    from qdrant_client.models import PointStruct
    from src.kb.ingest import COLLECTION as KB_COLLECTION, stable_point_id
    from src.kb.parser import chunk_markdown

    await _update_job(
        pg,
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc),
        message="Fetching URL",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                html = await resp.text()
    except Exception as e:
        await _update_job(
            pg,
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc),
            error=str(e),
            message="Fetch failed",
        )
        return

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        await _update_job(
            pg,
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc),
            error="No text extracted from URL",
            message="Parse failed",
        )
        return

    meta = {
        "source_id":    source_id,
        "source_type":  "web",
        "domain":       domain,
        "topic":        topic or "misc",
        "title":        url,
        "authors":      [],
        "abstract":     text[:200],
        "published":    datetime.now(timezone.utc).date().isoformat(),
        "year":         datetime.now(timezone.utc).year,
        "categories":   [],
        "strategy_category":   "",
        "strategy_confidence": "low",
        "citation_count": 0,
        "pdf_url":      url,
        "is_temp":      False,
        "indexed_at":   datetime.now(timezone.utc).isoformat(),
    }
    chunks = chunk_markdown(text, meta)
    texts = [c["text"] for c in chunks]
    await _update_job(pg, job_id, message=f"Embedding {len(chunks)} chunks")
    try:
        vecs = embedder.embed_batch(texts)
        points = [
            PointStruct(
                id=stable_point_id(c["chunk_id"]),
                vector={"dense": v["dense"], "sparse": v["sparse"]},
                payload={k: val for k, val in c.items()} | {"text": c["text"]},
            )
            for c, v in zip(chunks, vecs)
        ]
        qc.upsert(KB_COLLECTION, points=points, wait=True)
    except Exception as e:
        await _update_job(
            pg,
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc),
            error=str(e),
            message="Indexing failed",
        )
        return

    await pg.execute(
        """INSERT INTO sources (source_id, source_type, domain, topic, title, is_indexed, chunk_count, meta)
           VALUES ($1,$2,$3,$4,$5,TRUE,$6,$7::jsonb)
           ON CONFLICT (source_id) DO UPDATE SET is_indexed=TRUE, chunk_count=$6""",
        source_id, "web", domain, topic or "misc", url, len(chunks),
        json.dumps({"url": url}),
    )
    await _update_job(
        pg,
        job_id,
        status="complete",
        completed_at=datetime.now(timezone.utc),
        sources_processed=1,
        chunks_added=len(chunks),
        message="Indexed",
    )


# ---------------------------------------------------------------------------
# Tool 5: index
# ---------------------------------------------------------------------------

async def index(pg, qc, embedder, domain: str = "quant", force_reindex: bool = False) -> dict:
    """Start a background batch indexing job for PDFs on server disk."""
    where = "TRUE" if force_reindex else "is_indexed=FALSE"
    total = await pg.fetchval(f"SELECT COUNT(*) FROM papers WHERE domain=$1 AND {where}", domain)
    job_id = await _create_job(
        pg,
        "index",
        domain,
        force_reindex=force_reindex,
        sources_total=total or 0,
        message="Queued batch index",
    )
    asyncio.create_task(_run_index_job(pg, qc, embedder, job_id, domain, force_reindex))
    return {"job_id": job_id, "status": "queued", "domain": domain, "sources_total": total or 0}


async def _run_index_job(pg, qc, embedder, job_id: str, domain: str, force_reindex: bool) -> None:
    from src.kb.ingest import ensure_collection, ingest_pdfs

    await _update_job(
        pg,
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc),
        message="Scanning PDFs",
    )
    try:
        ensure_collection(qc)
        papers_root = Path(os.environ.get("PAPERS_ROOT", "/data/papers"))
        where = "TRUE" if force_reindex else "is_indexed=FALSE"
        rows = await pg.fetch(
            f"SELECT * FROM papers WHERE domain=$1 AND {where} ORDER BY source_id",
            domain,
        )
        await _update_job(pg, job_id, sources_total=len(rows))

        missing = 0
        processed = 0
        chunks_added = 0
        for row in rows:
            source_id = row["source_id"]
            pdf = _first_existing_path(_paper_pdf_candidates(row, papers_root, domain))
            if pdf is None:
                missing += 1
                await _update_job(
                    pg,
                    job_id,
                    sources_processed=processed,
                    chunks_added=chunks_added,
                    message=f"Missing {missing} PDFs; latest missing {source_id}",
                )
                continue
            added = await ingest_pdfs(pg, qc, embedder, [(pdf, dict(row))], domain, force_reindex)
            processed += 1
            chunks_added += added
            await _update_job(
                pg,
                job_id,
                sources_processed=processed,
                chunks_added=chunks_added,
                message=f"Indexed {source_id}",
            )

        status = "complete" if missing == 0 else "complete_with_warnings"
        message = "Index complete" if missing == 0 else f"Index complete; {missing} PDFs missing"
        await _update_job(
            pg,
            job_id,
            status=status,
            completed_at=datetime.now(timezone.utc),
            sources_processed=processed,
            chunks_added=chunks_added,
            message=message,
        )
    except Exception as e:
        await _update_job(
            pg,
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc),
            error=str(e),
            message="Index failed",
        )


# ---------------------------------------------------------------------------
# Tool 6: job_status (lightweight — actual async tracking is future work)
# ---------------------------------------------------------------------------

async def job_status(pg, job_id: str) -> dict:
    """Check status of an async job."""
    row = await pg.fetchrow("SELECT * FROM index_jobs WHERE job_id=$1", job_id)
    if row is None:
        return {"job_id": job_id, "status": "not_found"}
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Tool 7: info
# ---------------------------------------------------------------------------

async def info(pg, qc) -> dict:
    import os
    col = qc.get_collection(COLLECTION)
    q_points = col.points_count or 0

    domain_rows = await pg.fetch(
        """SELECT domain,
                  COUNT(*) AS sources,
                  SUM(CASE WHEN is_indexed THEN 1 ELSE 0 END) AS indexed,
                  COALESCE(SUM(chunk_count),0) AS chunks
           FROM papers
           GROUP BY domain"""
    )
    domains = {
        r["domain"]: {
            "sources": r["sources"],
            "indexed": r["indexed"],
            "chunks":  r["chunks"],
        }
        for r in domain_rows
    }

    pg_size = await pg.fetchval(
        "SELECT pg_database_size(current_database()) / 1024 / 1024"
    )
    arxiv_papers = 0
    db_path = _arxiv_db_path()
    arxiv_papers = _arxiv_paper_count(db_path)
    papers_root = Path(os.environ.get("PAPERS_ROOT", "/data/papers"))
    recent_jobs = await pg.fetch(
        """SELECT job_id, job_type, domain, status, sources_processed,
                  sources_total, chunks_added, message, error, updated_at
           FROM index_jobs
           WHERE updated_at >= NOW() - INTERVAL '7 days'
              OR status IN ('running', 'queued')
           ORDER BY updated_at DESC
           LIMIT 5"""
    )

    return {
        "kb_version":       "3.0",
        "embedding_model":  "BAAI/bge-m3",
        "qdrant_vectors":   q_points,
        "qdrant_points":    q_points,
        "domains":          domains,
        "postgres_size_mb": pg_size,
        "arxiv_fts_papers": arxiv_papers,
        "arxiv_db_path":    str(db_path),
        "arxiv_db_exists":  db_path.exists(),
        "papers_root":      str(papers_root),
        "papers_root_exists": papers_root.exists(),
        "recent_jobs":      [_row_to_dict(r) for r in recent_jobs],
        "qdrant_status":    str(col.status),
    }
