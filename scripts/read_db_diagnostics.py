from __future__ import annotations

import asyncio
import os

import asyncpg
from dotenv import load_dotenv


async def main() -> None:
    load_dotenv()
    pg = await asyncpg.connect(os.environ["PG_DSN"])
    rows = await pg.fetch(
        """
        SELECT key, parent_key
        FROM topics
        WHERE key = 'portfolio_construction'
           OR parent_key = 'portfolio_construction'
        ORDER BY key
        """
    )
    print("portfolio_topics=", [dict(r) for r in rows])
    rows = await pg.fetch(
        """
        SELECT topic,
               COUNT(*) AS sources,
               SUM(CASE WHEN is_indexed THEN 1 ELSE 0 END) AS indexed,
               COALESCE(SUM(chunk_count), 0) AS chunks
        FROM papers
        WHERE topic = ANY($1::text[])
        GROUP BY topic
        ORDER BY topic
        """,
        ["portfolio_construction", "factor_models", "mean_variance_optimization", "risk_parity"],
    )
    print("portfolio_topic_counts=", [dict(r) for r in rows])

    row = await pg.fetchrow(
        """
        SELECT source_id, title, is_indexed
        FROM papers
        WHERE source_id = '2207.02458'
        """
    )
    print("source_2207_02458=", dict(row) if row else None)

    rows = await pg.fetch(
        """
        SELECT source_id, title, is_indexed
        FROM papers
        WHERE source_id LIKE '2207.02458%'
        """
    )
    print("source_2207_02458_like=", [dict(r) for r in rows])

    rows = await pg.fetch(
        """
        SELECT source_id, topic, title
        FROM papers
        WHERE is_indexed = FALSE
        ORDER BY source_id
        LIMIT 20
        """
    )
    print("unindexed_sample=", [dict(r) for r in rows])
    await pg.close()


if __name__ == "__main__":
    asyncio.run(main())
