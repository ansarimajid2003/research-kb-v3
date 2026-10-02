"""
PDF parser + tiktoken chunker.
chunk_id = "{source_id}:{chunk_index}" — deterministic, stable across reruns.
"""
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import pymupdf4llm
import tiktoken

_PARSE_TIMEOUT = 120  # seconds; some PDFs hang indefinitely in pymupdf4llm

TOKENIZER = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE = 512     # tokens per chunk (including header)
CHUNK_OVERLAP = 64   # token overlap between consecutive chunks

# Persistent pool with one worker — spawned once, reused across all papers.
# Eliminates the ~60s Windows spawn+import overhead per paper.
_ctx = mp.get_context("spawn")
_parser_pool: Optional[mp.pool.Pool] = None


def _get_pool() -> mp.pool.Pool:
    global _parser_pool
    if _parser_pool is None:
        _parser_pool = _ctx.Pool(processes=1)
    return _parser_pool


def _reset_pool() -> None:
    global _parser_pool
    if _parser_pool is not None:
        try:
            _parser_pool.terminate()
            _parser_pool.join()
        except Exception:
            pass
    _parser_pool = None


def parse_pdf(path: str | Path) -> str:
    """Return full document as Markdown string. Kills worker if it hangs."""
    pool = _get_pool()
    try:
        result = pool.apply_async(pymupdf4llm.to_markdown, (str(path),), {"show_progress": False})
        return result.get(timeout=_PARSE_TIMEOUT)
    except mp.TimeoutError:
        _reset_pool()
        raise TimeoutError(f"PDF parsing timed out after {_PARSE_TIMEOUT}s")
    except Exception:
        _reset_pool()
        raise


def chunk_markdown(text: str, meta: dict) -> list[dict]:
    """
    Split Markdown text into overlapping token-bounded chunks.
    Each chunk carries a copy of meta plus text, chunk_id, chunk_index, total_chunks.
    """
    source_id = meta["source_id"]
    header = (
        f"Title: {meta.get('title', '')}\n"
        f"Authors: {', '.join(meta.get('authors') or [])}\n"
        f"Published: {meta.get('published', '')}\n"
        f"Domain: {meta.get('domain', '')} / {meta.get('topic', '')}\n"
        f"Source: {source_id}\n\n"
    )
    h_tok = TOKENIZER.encode(header)
    b_tok = TOKENIZER.encode(text)
    eff = CHUNK_SIZE - len(h_tok)          # body tokens available per chunk
    if eff <= 0:
        eff = CHUNK_SIZE // 2              # header unusually long — halve chunk

    chunks: list[dict] = []
    start = 0
    idx = 0
    while start < len(b_tok):
        end = min(start + eff, len(b_tok))
        chunk_text = TOKENIZER.decode(h_tok + b_tok[start:end])
        chunks.append({
            **meta,
            "text":        chunk_text,
            "chunk_index": idx,
            "chunk_id":    f"{source_id}:{idx}",
            "page_start":  None,           # pymupdf4llm doesn't give page per token
            "page_end":    None,
        })
        start += eff - CHUNK_OVERLAP
        idx += 1

    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total
    return chunks
