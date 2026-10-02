"""
Seed the KB V3 PostgreSQL database.

Usage:
    python scripts/seed_db.py              # create schema + load all data
    python scripts/seed_db.py --schema-only
    python scripts/seed_db.py --data-only

Run from project root with venv active.
"""
import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import date

import asyncpg
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

PAPERS_META = Path(r"C:\Users\ansar\Documents\Research Knowledge Base\research\papers_meta.json")
DOMAINS_YAML = ROOT / "domains.yaml"

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS domains (
    key         TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS topics (
    key         TEXT NOT NULL,
    domain      TEXT NOT NULL REFERENCES domains(key),
    label       TEXT NOT NULL,
    parent_key  TEXT,
    description TEXT NOT NULL DEFAULT '',
    paper_count INT NOT NULL DEFAULT 0,
    PRIMARY KEY (key, domain)
);

CREATE TABLE IF NOT EXISTS papers (
    source_id             TEXT PRIMARY KEY,
    base_id               TEXT NOT NULL,
    domain                TEXT NOT NULL DEFAULT 'quant',
    topic                 TEXT NOT NULL DEFAULT '',
    title                 TEXT NOT NULL DEFAULT '',
    abstract              TEXT NOT NULL DEFAULT '',
    authors               JSONB NOT NULL DEFAULT '[]',
    categories            JSONB NOT NULL DEFAULT '[]',
    published             DATE,
    year                  SMALLINT,
    strategy_category     TEXT NOT NULL DEFAULT '',
    strategy_confidence   TEXT NOT NULL DEFAULT '',
    citation_count        INT NOT NULL DEFAULT 0,
    influential_citations INT NOT NULL DEFAULT 0,
    relevance_score       REAL NOT NULL DEFAULT 0,
    pdf_url               TEXT NOT NULL DEFAULT '',
    local_path            TEXT NOT NULL DEFAULT '',
    is_indexed            BOOLEAN NOT NULL DEFAULT FALSE,
    chunk_count           INT NOT NULL DEFAULT 0,
    last_parsed           TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    domain      TEXT NOT NULL,
    topic       TEXT,
    title       TEXT NOT NULL DEFAULT '',
    is_indexed  BOOLEAN NOT NULL DEFAULT FALSE,
    chunk_count INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta        JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS strategy_rules (
    id         SERIAL PRIMARY KEY,
    category   TEXT NOT NULL,
    rule_type  TEXT NOT NULL,
    value      TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS index_jobs (
    job_id             TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    job_type           TEXT NOT NULL DEFAULT 'index',
    source_id          TEXT,
    domain             TEXT NOT NULL DEFAULT 'quant',
    status             TEXT NOT NULL DEFAULT 'pending',
    force_reindex      BOOLEAN NOT NULL DEFAULT FALSE,
    sources_total      INT NOT NULL DEFAULT 0,
    started_at         TIMESTAMPTZ,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at       TIMESTAMPTZ,
    sources_processed  INT NOT NULL DEFAULT 0,
    chunks_added       INT NOT NULL DEFAULT 0,
    message            TEXT,
    error              TEXT
);

CREATE INDEX IF NOT EXISTS idx_papers_fts
    ON papers USING GIN (to_tsvector('english', title || ' ' || abstract));
CREATE INDEX IF NOT EXISTS idx_papers_year       ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_domain_topic ON papers(domain, topic);
CREATE INDEX IF NOT EXISTS idx_papers_strategy   ON papers(strategy_category, strategy_confidence);
CREATE INDEX IF NOT EXISTS idx_papers_strategy_year ON papers(strategy_category, year);
CREATE INDEX IF NOT EXISTS idx_sources_domain    ON sources(domain, source_type);
CREATE INDEX IF NOT EXISTS idx_topics_domain     ON topics(domain);
CREATE INDEX IF NOT EXISTS idx_topics_parent     ON topics(parent_key);

ALTER TABLE papers ADD COLUMN IF NOT EXISTS relevance_score REAL NOT NULL DEFAULT 0;
ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS job_type TEXT NOT NULL DEFAULT 'index';
ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS source_id TEXT;
ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS sources_total INT NOT NULL DEFAULT 0;
ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE index_jobs ADD COLUMN IF NOT EXISTS message TEXT;
"""


def safe_source_filename(source_id: str, local_path: str = "") -> str:
    if local_path:
        return Path(local_path).name
    return f"{source_id.replace('/', '_')}.pdf"


async def create_schema(pg):
    print("Creating schema...")
    await pg.execute(SCHEMA_SQL)
    print("  Schema OK")


async def seed_domains(pg, domains_cfg: dict):
    print("Seeding domains and topics...")
    for domain_key, domain_cfg in domains_cfg.items():
        await pg.execute(
            """INSERT INTO domains (key, label)
               VALUES ($1, $2)
               ON CONFLICT (key) DO UPDATE SET label = EXCLUDED.label""",
            domain_key, domain_cfg["label"],
        )
        for topic_key, topic_cfg in (domain_cfg.get("topics") or {}).items():
            await pg.execute(
                """INSERT INTO topics (key, domain, label, parent_key)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (key, domain) DO UPDATE
                     SET label=EXCLUDED.label, parent_key=EXCLUDED.parent_key""",
                topic_key, domain_key, topic_cfg["label"], topic_cfg.get("parent"),
            )
    print(f"  {len(domains_cfg)} domains seeded")


async def seed_papers(pg, papers_meta: dict):
    print(f"Seeding {len(papers_meta)} papers...")
    inserted = skipped = 0

    for arxiv_id, p in tqdm(papers_meta.items(), unit="paper"):
        # Derive fields
        base_id = arxiv_id.split("v")[0]
        topic = p.get("topic_group", "")
        pub_str = p.get("published")
        pub_date = date.fromisoformat(pub_str) if pub_str else None
        year = pub_date.year if pub_date else None

        # Reconstruct server-side local_path from laptop path filename
        lp = p.get("local_path", "")
        filename = safe_source_filename(arxiv_id, lp)
        server_path = f"/DATA/papers/quant/{topic}/{filename}" if filename else ""

        try:
            await pg.execute(
                """INSERT INTO papers (
                       source_id, base_id, domain, topic, title, abstract,
                       authors, categories, published, year,
                       citation_count, influential_citations, pdf_url, local_path
                   ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                   ON CONFLICT (source_id) DO UPDATE SET
                       relevance_score = COALESCE(papers.relevance_score, 0)""",
                arxiv_id, base_id, "quant", topic,
                p.get("title", ""), p.get("abstract", ""),
                json.dumps(p.get("authors", [])),
                json.dumps(p.get("categories", [])),
                pub_date, year,
                p.get("citation_count") or 0,
                p.get("influential_citations") or 0,
                p.get("pdf_url", ""), server_path,
            )
            await pg.execute(
                """INSERT INTO sources (source_id, source_type, domain, topic, title)
                   VALUES ($1, 'paper', $2, $3, $4)
                   ON CONFLICT (source_id) DO NOTHING""",
                arxiv_id, "quant", topic, p.get("title", ""),
            )
            inserted += 1
        except Exception as e:
            print(f"  WARN {arxiv_id}: {e}")
            skipped += 1

    print(f"  {inserted} inserted, {skipped} skipped")


async def update_topic_counts(pg):
    print("Updating topic paper counts...")
    await pg.execute("""
        UPDATE topics t SET paper_count = (
            SELECT COUNT(*) FROM papers p
            WHERE p.domain = t.domain AND p.topic = t.key
        )
    """)
    print("  Done")


async def main(schema_only=False, data_only=False):
    dsn = os.environ["PG_DSN"]
    print(f"Connecting to {dsn.split('@')[1]}...")
    pg = await asyncpg.connect(dsn)

    try:
        if not data_only:
            await create_schema(pg)

        if not schema_only:
            with open(DOMAINS_YAML) as f:
                cfg = yaml.safe_load(f)
            await seed_domains(pg, cfg["domains"])

            with open(PAPERS_META, encoding="utf-8") as f:
                papers_meta = json.load(f)
            await seed_papers(pg, papers_meta)
            await update_topic_counts(pg)

        # Summary
        counts = await pg.fetchrow(
            "SELECT (SELECT COUNT(*) FROM papers) AS papers, "
            "(SELECT COUNT(*) FROM sources) AS sources, "
            "(SELECT COUNT(*) FROM topics) AS topics, "
            "(SELECT COUNT(*) FROM domains) AS domains"
        )
        print(f"\nDB state: {dict(counts)}")

    finally:
        await pg.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(schema_only=args.schema_only, data_only=args.data_only))
