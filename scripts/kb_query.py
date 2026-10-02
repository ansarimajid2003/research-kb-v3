"""
Small CLI for querying the production KB MCP server.

Examples:
  python scripts/kb_query.py info
  python scripts/kb_query.py search "pairs trading cointegration" --domain quant --top-k 5
  python scripts/kb_query.py discover "online learning market making" --year-from 2020
  python scripts/kb_query.py get-source 2207.02458
  python scripts/kb_query.py job-status <job_id>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from fastmcp.client import BearerAuth, Client


DEFAULT_URL = "http://localhost:8765/mcp"


def auth_token() -> str:
    token = os.environ.get("MCP_API_KEY") or os.environ.get("KB_MCP_API_KEY")
    if not token:
        raise SystemExit("Set MCP_API_KEY or KB_MCP_API_KEY.")
    return token


async def call_tool(url: str, tool: str, args: dict[str, Any]) -> Any:
    async with Client(url, auth=BearerAuth(auth_token()), timeout=60) as client:
        result = await client.call_tool(tool, args)
    return result.data if hasattr(result, "data") else result


def add_common_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query")
    parser.add_argument("--domain")
    parser.add_argument("--topic")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--source-type")
    parser.add_argument("--strategy")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("MCP_URL", DEFAULT_URL))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info")

    search_parser = sub.add_parser("search")
    add_common_search_args(search_parser)

    discover_parser = sub.add_parser("discover")
    discover_parser.add_argument("query")
    discover_parser.add_argument("--domain", default="quant")
    discover_parser.add_argument("--topic")
    discover_parser.add_argument("--max-papers", type=int, default=5)
    discover_parser.add_argument("--force-download", action="store_true")
    discover_parser.add_argument("--year-from", type=int)
    discover_parser.add_argument("--min-score", type=float, default=0.3)

    source_parser = sub.add_parser("get-source")
    source_parser.add_argument("source_id")

    add_parser = sub.add_parser("add-source")
    add_parser.add_argument("url")
    add_parser.add_argument("--domain", default="web")
    add_parser.add_argument("--topic")

    index_parser = sub.add_parser("index")
    index_parser.add_argument("--domain", default="quant")
    index_parser.add_argument("--force-reindex", action="store_true")

    job_parser = sub.add_parser("job-status")
    job_parser.add_argument("job_id")

    args = parser.parse_args()
    command = args.command.replace("-", "_")
    tool_name = args.command.replace("-", "_")

    if command == "info":
        payload = {}
    elif command == "get_source":
        payload = {"source_id": args.source_id}
    elif command == "job_status":
        payload = {"job_id": args.job_id}
    else:
        payload = {
            k: v
            for k, v in vars(args).items()
            if k not in {"command", "url"} and v is not None
        }
    if command == "index":
        payload["force_reindex"] = payload.pop("force_reindex", False)
    if command == "discover":
        payload["force_download"] = payload.pop("force_download", False)
        payload["max_papers"] = payload.pop("max_papers")
        if "year_from" in payload:
            payload["year_from"] = payload.pop("year_from")
        if "min_score" in payload:
            payload["min_score"] = payload.pop("min_score")
    if command == "search":
        payload["top_k"] = payload.pop("top_k")
        if "year_from" in payload:
            payload["year_from"] = payload.pop("year_from")
        if "year_to" in payload:
            payload["year_to"] = payload.pop("year_to")
        if "source_type" in payload:
            payload["source_type"] = payload.pop("source_type")
        if "min_score" in payload:
            payload["min_score"] = payload.pop("min_score")

    print(json.dumps(await call_tool(args.url, tool_name, payload), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
