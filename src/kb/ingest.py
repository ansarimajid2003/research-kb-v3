"""
Ingest pipeline: PDF → parse → embed → upsert to Qdrant + update Postgres.

Usage (from project root with venv active):
    python -m src.kb.ingest --domain quant --all-topics
    python -m src.kb.ingest --domain quant --topic momentum_trend
    python -m src.kb.ingest --domain quant --file 2301.12345v2.pdf
    python -m src.kb.ingest --source-dir /path/to/pdfs --domain quant --topic misc
    python -m src.kb.ingest --domain quant --all-topics --force   # re-embed already indexed
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, HnswConfigDiff, OptimizersConfigDiff,
    PayloadSchemaType, PointStruct, SparseVectorParams,
    VectorParams, NamedVector, NamedSparseVector,
)
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")

COLLECTION = "kb"
BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "64"))
EMBED_BATCH = int(os.environ.get("INGEST_EMBED_BATCH", "64"))
UPSERT_RETRIES = int(os.environ.get("QDRANT_UPSERT_RETRIES", "3"))
UPSERT_RETRY_BASE_SECONDS = int(os.environ.get("QDRANT_UPSERT_RETRY_BASE_SECONDS", "10"))


def stable_point_id(chunk_id: str) -> int:
    """SHA-256 → uint64. Deterministic: same chunk_id → same point id across reruns."""
    digest = hashlib.sha256(chunk_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def ensure_collection(qc: QdrantClient):
    existing = [c.name for c in qc.get_collections().collections]
    if COLLECTION in existing:
        return
    print(f"[qdrant] creating collection '{COLLECTION}'")
    qc.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE, on_disk=True),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100, on_disk=True),
        optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
    )
    for field, schema in [
        ("domain",            PayloadSchemaType.KEYWORD),
        ("topic",             PayloadSchemaType.KEYWORD),
        ("year",              PayloadSchemaType.INTEGER),
        ("strategy_category", PayloadSchemaType.KEYWORD),
        ("source_type",       PayloadSchemaType.KEYWORD),
        ("language",          PayloadSchemaType.KEYWORD),
        ("is_temp",           PayloadSchemaType.BOOL),
    ]:
        qc.create_payload_index(COLLECTION, field, schema)
    print(f"[qdrant] collection '{COLLECTION}' ready")


def qdrant_health_summary(qc: QdrantClient) -> str:
    """Small diagnostic string for abort messages after write failures."""
    try:
        info = qc.get_collection(COLLECTION)
    except Exception as exc:
        return f"unreachable: {exc}"
    return (
        f"status={info.status}, optimizer={info.optimizer_status}, "
        f"points={info.points_count}, segments={info.segments_count}"
    )


async def get_pdfs_for_topic(pg, domain: str, topic: str, source_dir: Path | None,
                              force: bool) -> list[tuple[Path, dict]]:
    """Return list of (pdf_path, paper_row) for PDFs that need indexing."""
    if source_dir:
        pdfs = list(source_dir.rglob("*.pdf"))
    else:
        # Default: look in ingest/<domain>/<topic>/
        pdfs = list((ROOT / "ingest" / domain / topic).glob("*.pdf"))

    rows = []
    for pdf in pdfs:
        source_id = pdf.stem
        row = await pg.fetchrow(
            "SELECT * FROM papers WHERE source_id=$1", source_id
        )
        if row is None:
            print(f"  [skip] {source_id} — not in DB (run seed_db.py first)")
            continue
        if not force and row["is_indexed"]:
            continue
        rows.append((pdf, dict(row)))
    return rows


async def ingest_pdfs(pg, qc: QdrantClient, embedder, pdfs: list[tuple[Path, dict]],
                      domain: str, force: bool):
    from .parser import parse_pdf, chunk_markdown
    from .classify import classify_paper
    from datetime import datetime, timezone

    total_chunks = 0
    for pdf, row in tqdm(pdfs, unit="pdf", desc="ingesting"):
        source_id = row["source_id"]

        # Build meta dict for chunker
        meta = {
            "source_id":    source_id,
            "source_type":  "paper",
            "domain":       domain,
            "topic":        row["topic"],
            "title":        row["title"],
            "authors":      json.loads(row["authors"]) if isinstance(row["authors"], str) else list(row["authors"]),
            "abstract":     row["abstract"],
            "published":    str(row["published"]) if row["published"] else "",
            "year":         row["year"],
            "categories":   json.loads(row["categories"]) if isinstance(row["categories"], str) else list(row["categories"]),
            "strategy_category":   row["strategy_category"],
            "strategy_confidence": row["strategy_confidence"],
            "citation_count":      row["citation_count"],
            "pdf_url":      row["pdf_url"],
            "is_temp":      False,
            "indexed_at":   datetime.now(timezone.utc).isoformat(),
        }

        # Classify if not already done
        if not meta["strategy_category"]:
            cat, conf = classify_paper(meta["title"], meta["abstract"])
            meta["strategy_category"] = cat
            meta["strategy_confidence"] = conf

        # Parse → chunk
        try:
            text = parse_pdf(pdf)
        except Exception as e:
            print(f"  [error] parse {source_id}: {e}")
            continue

        chunks = chunk_markdown(text, meta)
        texts = [c["text"] for c in chunks]

        # Embed
        vecs = embedder.embed_batch(texts, batch_size=EMBED_BATCH)

        # Build Qdrant points
        points = []
        for c, v in zip(chunks, vecs):
            payload = {k: val for k, val in c.items() if k != "text"}
            payload["text"] = c["text"]
            points.append(PointStruct(
                id=stable_point_id(c["chunk_id"]),
                vector={
                    "dense":  v["dense"],
                    "sparse": v["sparse"],
                },
                payload=payload,
            ))

        # Repeated Qdrant write failure is a server health problem. Abort and
        # let the next run resume safely from Postgres' is_indexed flags.
        import time as _time
        total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            batch_no = i // BATCH_SIZE + 1
            for attempt in range(1, UPSERT_RETRIES + 1):
                try:
                    qc.upsert(COLLECTION, points=batch, wait=True)
                    break
                except Exception as exc:
                    if attempt == UPSERT_RETRIES:
                        health = qdrant_health_summary(qc)
                        raise RuntimeError(
                            f"Qdrant upsert failed after {UPSERT_RETRIES} attempts "
                            f"for {source_id} batch {batch_no}/{total_batches}; "
                            f"health={health}. Aborting migration so the server can "
                            f"be inspected/restarted and this run can resume later."
                        ) from exc
                    print(
                        f"  [retry] upsert {source_id} batch {batch_no}/{total_batches} "
                        f"attempt {attempt} failed: {exc}"
                    )
                    _time.sleep(UPSERT_RETRY_BASE_SECONDS * attempt)

        # Update Postgres
        await pg.execute(
            """UPDATE papers SET is_indexed=TRUE, chunk_count=$1, last_parsed=NOW(),
               strategy_category=$2, strategy_confidence=$3
               WHERE source_id=$4""",
            len(chunks), meta["strategy_category"], meta["strategy_confidence"], source_id,
        )
        await pg.execute(
            "UPDATE sources SET is_indexed=TRUE, chunk_count=$1 WHERE source_id=$2",
            len(chunks), source_id,
        )
        total_chunks += len(chunks)

    return total_chunks


async def main(args):
    from .embedder import Embedder

    qc = QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        check_compatibility=False,
        timeout=120,
    )
    pg = await asyncpg.connect(os.environ["PG_DSN"])

    ensure_collection(qc)
    embedder = Embedder.get()

    source_dir = Path(args.source_dir) if args.source_dir else None

    # Collect (pdf, row) pairs to process
    to_ingest: list[tuple[Path, dict]] = []

    if args.file:
        pdf = Path(args.file)
        row = await pg.fetchrow("SELECT * FROM papers WHERE source_id=$1", pdf.stem)
        if row is None:
            print(f"[error] {pdf.stem} not found in DB")
            sys.exit(1)
        to_ingest = [(pdf, dict(row))]

    elif args.all_topics:
        topics = await pg.fetch(
            "SELECT DISTINCT topic FROM papers WHERE domain=$1 AND topic != ''", args.domain
        )
        for t in topics:
            topic = t["topic"]
            # For all-topics mode use source_dir as the papers root
            effective_dir = (source_dir / topic if source_dir else
                             ROOT / "ingest" / args.domain / topic)
            pairs = await get_pdfs_for_topic(pg, args.domain, topic, effective_dir, args.force)
            to_ingest.extend(pairs)

    else:
        effective_dir = (source_dir / args.topic if source_dir else None)
        to_ingest = await get_pdfs_for_topic(
            pg, args.domain, args.topic, effective_dir, args.force
        )

    print(f"[ingest] {len(to_ingest)} PDFs to process")
    if not to_ingest:
        print("[ingest] nothing to do")
        await pg.close()
        return

    total_chunks = await ingest_pdfs(pg, qc, embedder, to_ingest, args.domain, args.force)

    # Final stats
    count = qc.get_collection(COLLECTION).points_count
    print(f"\n[done] {len(to_ingest)} PDFs → {total_chunks} chunks | "
          f"Qdrant total: {count} points")
    await pg.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KB V3 ingest pipeline")
    parser.add_argument("--domain",     default="quant")
    parser.add_argument("--topic",      default=None)
    parser.add_argument("--all-topics", action="store_true")
    parser.add_argument("--file",       default=None, help="Single PDF path")
    parser.add_argument("--source-dir", default=None, help="Override PDF source directory")
    parser.add_argument("--force",      action="store_true", help="Re-index already indexed papers")
    args = parser.parse_args()

    if not args.file and not args.all_topics and not args.topic and not args.source_dir:
        parser.error("specify --topic, --all-topics, --file, or --source-dir")

    asyncio.run(main(args))
