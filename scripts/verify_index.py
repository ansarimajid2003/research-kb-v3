"""
Post-migration verification: confirms Qdrant index and Postgres are consistent.

Usage:
    python scripts/verify_index.py
    python scripts/verify_index.py --quick   # skip search tests
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(ROOT / ".env")

COLLECTION = "kb"


async def main(args):
    qc = QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        check_compatibility=False,
    )
    pg = await asyncpg.connect(os.environ["PG_DSN"])

    # --- Qdrant stats ---
    info = qc.get_collection(COLLECTION)
    q_points = info.points_count
    q_indexed = info.indexed_vectors_count
    print(f"[qdrant] points={q_points}, indexed_vectors={q_indexed}")
    print(f"[qdrant] status={info.status}, optimizer={info.optimizer_status}")

    # --- Postgres stats ---
    total    = await pg.fetchval("SELECT COUNT(*) FROM papers")
    indexed  = await pg.fetchval("SELECT COUNT(*) FROM papers WHERE is_indexed=TRUE")
    chunks   = await pg.fetchval("SELECT COALESCE(SUM(chunk_count),0) FROM papers WHERE is_indexed=TRUE")
    failed   = await pg.fetchval("SELECT COUNT(*) FROM papers WHERE is_indexed=FALSE AND domain='quant'")
    print(f"[postgres] total={total}, indexed={indexed}, failed={failed}, chunks_sum={chunks}")

    # Consistency check
    if abs(q_points - chunks) > 500:
        print(f"[WARN] Qdrant points ({q_points}) vs Postgres chunk_count sum ({chunks}) mismatch > 500")
    else:
        print(f"[OK] Qdrant/Postgres chunk counts consistent (delta={abs(q_points - chunks)})")

    if args.quick:
        await pg.close()
        return

    # --- Per-topic breakdown ---
    rows = await pg.fetch(
        "SELECT topic, COUNT(*) n, SUM(chunk_count) chunks "
        "FROM papers WHERE is_indexed=TRUE GROUP BY topic ORDER BY n DESC LIMIT 10"
    )
    print("\nTop topics:")
    for r in rows:
        print(f"  {r['topic']:<30} papers={r['n']:4d}  chunks={r['chunks'] or 0:6d}")

    # --- Semantic search test ---
    from src.kb.embedder import Embedder
    embedder = Embedder.get()

    queries = [
        ("momentum factor cross-sectional", "quant"),
        ("pairs trading cointegration", "quant"),
        ("deep learning order book", "quant"),
    ]
    print()
    for q, domain in queries:
        v = embedder.embed_query(q)
        result = qc.query_points(
            COLLECTION,
            query=v["dense"],
            using="dense",
            query_filter=None,
            limit=3,
            with_payload=True,
        )
        hits = result.points
        top = hits[0] if hits else None
        if top:
            print(f"[search] '{q[:40]}' -> score={top.score:.3f} | {top.payload.get('title','')[:50]}")
        else:
            print(f"[search] '{q[:40]}' -> NO RESULTS")

    # --- Idempotency test ---
    source_id = await pg.fetchval("SELECT source_id FROM papers WHERE is_indexed=TRUE LIMIT 1")
    pts_before = qc.get_collection(COLLECTION).points_count
    print(f"\n[idempotency] {source_id}: Qdrant has {pts_before} points before re-ingest")
    print("[idempotency] (skipping re-ingest in verify — run with --reindex to test)")

    await pg.close()
    print("\n[done] verification complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip search tests")
    args = parser.parse_args()
    asyncio.run(main(args))
