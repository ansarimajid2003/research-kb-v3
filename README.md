# Research KB V3

Research KB V3 is a self-hosted MCP service for hybrid retrieval over a research corpus. It separates a GPU-capable laptop build/indexing workflow from a low-power server that provides authenticated search, metadata lookup, corpus discovery, and controlled indexing.

The repository contains source code, deployment templates, taxonomy, and operational scripts. It deliberately excludes private credentials, model caches, PDFs, the arXiv SQLite index, Postgres/Qdrant volumes, and benchmark output. Those assets stay on the operator-controlled host.

## Architecture

```text
Laptop build environment                 Server deployment
PDFs + metadata -> parse/chunk -> embed -> Qdrant dense+sparse vectors
                                      \-> Postgres metadata/taxonomy/jobs
MCP client -> FastMCP HTTP API -> hybrid/semantic/keyword retrieval
                                  -> SQLite FTS discovery snapshot
```

- **MCP API:** FastMCP/Starlette with bearer-token middleware and `/health`.
- **Retrieval:** bge-m3 dense (1,024-d cosine) and sparse vectors fused by Qdrant reciprocal-rank fusion; metadata filters include domain, topic, year, source type, language, temporary-state, and strategy category.
- **Storage:** Qdrant stores chunk vectors and payload; PostgreSQL stores sources, taxonomy, jobs, and metadata; SQLite FTS supports discovery beyond the indexed corpus.
- **Operations:** source IDs yield stable chunk IDs; a SHA-256-derived Qdrant point ID makes re-indexing an upsert rather than a duplicate insert. Cleanup removes expired temporary discovery material.

## Components

| Location | Responsibility |
|---|---|
| Laptop | Parse PDFs, classify, chunk, embed, stage data, seed/migrate metadata, and run indexing/verification scripts. |
| Server | Run `kb-server`, Qdrant, PostgreSQL, and cleanup in Docker; retain raw corpus, SQLite discovery snapshot, model cache, and database volumes. |

## MCP tools

`search`, `get_source`, `discover`, `add_source`, `index`, `job_status`, and `info`.

`search` supports `hybrid` (default), `semantic`, and `keyword` modes. `discover` searches an external-to-the-index SQLite snapshot; optional download/index work and `add_source` run as tracked jobs.

## Run locally or deploy

1. Create a private data directory containing your corpus and discovery database; do not commit it.
2. Copy `.env.example` to `.env`, replace all placeholder secrets, and set the data paths for your host.
3. Copy `docker-compose.example.yml` to `docker-compose.yml`, review volume mounts and port exposure, then run `docker compose up -d --build`.
4. Check `http://localhost:8765/health`. Supply `Authorization: Bearer <MCP_API_KEY>` to `/mcp`.

The compose template keeps Qdrant and PostgreSQL loopback-bound. Publish the MCP port only through an authenticated network boundary you control.

## Build environment

The laptop workflow uses `requirements_build.txt`; the server image uses `requirements_serve.txt`. `scripts/seed_db.py`, `src/kb/ingest.py`, `scripts/verify_index.py`, and `scripts/benchmark_queries.py` provide the main data lifecycle. Query an endpoint with:

```powershell
$env:MCP_API_KEY = 'your-secret'
python scripts/kb_query.py info --url http://localhost:8765/mcp
```

## Evidence and limitations

The last captured benchmark and corpus counts are intentionally documented in the accompanying project summary, with their collection date. They are historical measurements, not a guarantee of current server availability, capacity, or retrieval quality. Benchmark relevance is a small hand-curated test set and should be extended before making quality claims.

## Security and data policy

Never commit `.env`, Qdrant/Postgres storage, model caches, PDFs, SQLite databases, staged corpus data, or generated reports. Confirm that you have redistribution rights before publishing any corpus artifacts.
