"""
KB V3 FastMCP server — runs inside Docker on the HP server.

Start:
    uvicorn src.mcp.server:app --host 0.0.0.0 --port 8765
Or via docker compose:
    docker compose up -d kb-server
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")

from src.kb.embedder import Embedder
from src.mcp.auth import BearerAuth
import src.mcp.tools as _tools

# ---------------------------------------------------------------------------
# Shared state (initialised once at startup)
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app):
    _state["qc"] = QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        check_compatibility=False,
    )
    _state["pg"] = await asyncpg.create_pool(
        os.environ["PG_DSN"],
        min_size=int(os.environ.get("PG_POOL_MIN_SIZE", "1")),
        max_size=int(os.environ.get("PG_POOL_MAX_SIZE", "8")),
    )
    _state["embedder"] = Embedder.get()
    print("[kb-server] ready")
    yield
    await _state["pg"].close()


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------
mcp = FastMCP("kb-v3", lifespan=lifespan)

async def _health(request: Request):
    return JSONResponse({"status": "ok", "service": "kb-v3"})

_starlette_app = mcp.http_app(stateless_http=True)
_starlette_app.add_middleware(BearerAuth, api_key=os.environ.get("MCP_API_KEY", ""))
_starlette_app.add_route("/health", _health, methods=["GET"])
app = _starlette_app


def _ctx():
    return _state["pg"], _state["qc"], _state["embedder"]


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
async def search(
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
    """Search the KB. mode: hybrid (default) | semantic | keyword."""
    pg, qc, emb = _ctx()
    return await _tools.search(pg, qc, emb, query, domain, topic, top_k,
                                year_from, year_to, source_type, strategy,
                                min_score, mode)


@mcp.tool()
async def get_source(source_id: str) -> dict | None:
    """Full metadata for a source by ID (arxiv ID, URL hash, etc.)."""
    pg, _, _ = _ctx()
    return await _tools.get_source(pg, source_id)


@mcp.tool()
async def discover(
    query: str,
    domain: str = "quant",
    topic: str | None = None,
    max_papers: int = 5,
    force_download: bool = False,
    year_from: int | None = None,
    min_score: float = 0.3,
) -> dict:
    """Search arxiv snapshot for papers not yet in KB. force_download=True downloads + temp-indexes them."""
    pg, qc, emb = _ctx()
    return await _tools.discover(pg, qc, emb, query, domain, topic,
                                  max_papers, force_download, year_from, min_score)


@mcp.tool()
async def add_source(url: str, domain: str = "web", topic: str | None = None) -> dict:
    """Fetch a URL, chunk it, and index it immediately."""
    pg, qc, emb = _ctx()
    return await _tools.add_source(pg, qc, emb, url, domain, topic)


@mcp.tool()
async def index(domain: str = "quant", force_reindex: bool = False) -> dict:
    """Start background batch indexing for PDFs on server disk."""
    pg, qc, emb = _ctx()
    return await _tools.index(pg, qc, emb, domain, force_reindex)


@mcp.tool()
async def job_status(job_id: str) -> dict:
    """Check status of a background job."""
    pg, _, _ = _ctx()
    return await _tools.job_status(pg, job_id)


@mcp.tool()
async def info() -> dict:
    """KB health snapshot: vector counts, domain stats, server metrics."""
    pg, qc, _ = _ctx()
    return await _tools.info(pg, qc)
