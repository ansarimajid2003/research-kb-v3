"""
Probe the deployed FastMCP HTTP endpoint without printing secrets.
"""
from __future__ import annotations

import asyncio
import os
import time

from fastmcp.client import BearerAuth, Client


def get_server_key() -> str:
    token = os.environ.get("MCP_API_KEY") or os.environ.get("KB_MCP_API_KEY")
    if not token:
        raise SystemExit("Set MCP_API_KEY or KB_MCP_API_KEY.")
    return token


async def probe(url: str, token: str) -> dict:
    t0 = time.perf_counter()
    async with Client(url, auth=BearerAuth(token), timeout=30) as client:
        tools = await client.list_tools()
        info = await client.call_tool("info", {})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    info_payload = info.data if hasattr(info, "data") else info
    return {
        "url": url,
        "ok": True,
        "elapsed_ms": elapsed_ms,
        "tool_names": [t.name for t in tools],
        "info": info_payload,
    }


async def main() -> None:
    token = get_server_key()
    url = os.environ.get("MCP_URL", "http://localhost:8765/mcp")
    for url in [url]:
        try:
            result = await probe(url, token)
        except Exception as exc:
            result = {
                "url": url,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
