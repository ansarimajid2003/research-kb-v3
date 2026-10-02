"""Remove is_temp=True Qdrant points older than --max-age-hours."""
import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchValue


async def cleanup(max_age_hours: int):
    qc = QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        check_compatibility=False,
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    result = qc.delete(
        collection_name="kb",
        points_selector=Filter(must=[
            FieldCondition(key="is_temp",    match=MatchValue(value=True)),
            FieldCondition(key="indexed_at", range=DatetimeRange(lt=cutoff)),
        ]),
    )
    print(f"[cleanup] purged temp points older than {max_age_hours}h (op={result.operation_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-age-hours", type=int, default=24)
    args = parser.parse_args()
    asyncio.run(cleanup(args.max_age_hours))
