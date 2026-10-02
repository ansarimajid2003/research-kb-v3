"""
One-shot migration: all 1,865 V1 papers → V3 Qdrant + Postgres.
Reads original local_path from papers_meta.json so filenames match exactly.

Usage:
    python scripts/migrate_quant.py                # migrate all not-yet-indexed
    python scripts/migrate_quant.py --force        # re-index everything
    python scripts/migrate_quant.py --topic momentum_trend
    python scripts/migrate_quant.py --dry-run      # show counts, no changes
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm import tqdm

load_dotenv(ROOT / ".env")

PAPERS_META = Path(r"C:\Users\ansar\Documents\Research Knowledge Base\research\papers_meta.json")


async def main(args):
    from src.kb.embedder import Embedder
    from src.kb.ingest import ensure_collection, ingest_pdfs, COLLECTION

    qc = QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        check_compatibility=False,
        timeout=120,
    )
    pg = await asyncpg.connect(os.environ["PG_DSN"])

    ensure_collection(qc)

    # Load original paths from papers_meta.json
    with open(PAPERS_META, encoding="utf-8") as f:
        meta_map = json.load(f)  # {arxiv_id: {local_path, ...}}

    # Query DB for papers to ingest
    where = "is_indexed = FALSE" if not args.force else "TRUE"
    if args.topic:
        rows = await pg.fetch(
            f"SELECT * FROM papers WHERE domain='quant' AND topic=$1 AND {where}",
            args.topic,
        )
    else:
        rows = await pg.fetch(
            f"SELECT * FROM papers WHERE domain='quant' AND {where}"
        )

    print(f"[migrate] {len(rows)} papers to process")

    # Build (pdf_path, row) pairs using original laptop paths from papers_meta.json
    to_ingest = []
    missing = []
    for row in rows:
        arxiv_id = row["source_id"]
        m = meta_map.get(arxiv_id, {})
        lp = m.get("local_path", "")
        if not lp:
            missing.append(arxiv_id)
            continue
        pdf = Path(lp)
        if not pdf.exists():
            missing.append(arxiv_id)
            continue
        to_ingest.append((pdf, dict(row)))

    if missing:
        print(f"[migrate] {len(missing)} PDFs not found on disk — skipping")

    if args.dry_run:
        print(f"[migrate] dry-run: would ingest {len(to_ingest)} PDFs")
        await pg.close()
        return

    if not to_ingest:
        print("[migrate] nothing to ingest")
        await pg.close()
        return

    embedder = Embedder.get()
    total_chunks = await ingest_pdfs(pg, qc, embedder, to_ingest, "quant", args.force)

    count = qc.get_collection(COLLECTION).points_count
    print(f"\n[done] {len(to_ingest)} PDFs -> {total_chunks} chunks | "
          f"Qdrant total: {count} points")
    await pg.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--topic",     default=None)
    args = parser.parse_args()
    asyncio.run(main(args))
