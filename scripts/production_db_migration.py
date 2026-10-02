"""
Production DB fixups after the V3 migration.

Applies additive schema changes and refreshes paths/topic counts without
destructively re-seeding indexed state.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.seed_db import DOMAINS_YAML, PAPERS_META, SCHEMA_SQL, safe_source_filename


async def main() -> None:
    load_dotenv()
    pg = await asyncpg.connect(os.environ["PG_DSN"])
    try:
        await pg.execute(SCHEMA_SQL)

        with open(DOMAINS_YAML, encoding="utf-8") as f:
            domains = yaml.safe_load(f)["domains"]
        for domain_key, domain_cfg in domains.items():
            await pg.execute(
                """INSERT INTO domains (key, label)
                   VALUES ($1, $2)
                   ON CONFLICT (key) DO UPDATE SET label=EXCLUDED.label""",
                domain_key,
                domain_cfg["label"],
            )
            for topic_key, topic_cfg in (domain_cfg.get("topics") or {}).items():
                await pg.execute(
                    """INSERT INTO topics (key, domain, label, parent_key)
                       VALUES ($1,$2,$3,$4)
                       ON CONFLICT (key, domain) DO UPDATE
                         SET label=EXCLUDED.label, parent_key=EXCLUDED.parent_key""",
                    topic_key,
                    domain_key,
                    topic_cfg["label"],
                    topic_cfg.get("parent"),
                )

        with open(PAPERS_META, encoding="utf-8") as f:
            papers_meta = json.load(f)
        updated = 0
        for source_id, meta in papers_meta.items():
            topic = meta.get("topic_group") or "quant_finance"
            filename = safe_source_filename(source_id, meta.get("local_path", ""))
            server_path = f"/DATA/papers/quant/{topic}/{filename}"
            result = await pg.execute(
                """UPDATE papers
                   SET local_path=$2, relevance_score=COALESCE(relevance_score, 0)
                   WHERE source_id=$1""",
                source_id,
                server_path,
            )
            if result.endswith(" 1"):
                updated += 1

        await pg.execute(
            """
            UPDATE topics t SET paper_count = (
                SELECT COUNT(*) FROM papers p
                WHERE p.domain = t.domain AND p.topic = t.key
            )
            """
        )
        counts = await pg.fetchrow(
            """SELECT
                   (SELECT COUNT(*) FROM papers) AS papers,
                   (SELECT COUNT(*) FROM papers WHERE is_indexed) AS indexed,
                   (SELECT COUNT(*) FROM topics) AS topics,
                   (SELECT COUNT(*) FROM index_jobs) AS jobs
            """
        )
        print({"updated_local_paths": updated, "counts": dict(counts)})
    finally:
        await pg.close()


if __name__ == "__main__":
    asyncio.run(main())
