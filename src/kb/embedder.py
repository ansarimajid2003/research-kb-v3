"""
bge-m3 embedder — dense + sparse vectors simultaneously.
Use FlagEmbedding (NOT sentence-transformers) for sparse output.
"""
import os
from functools import lru_cache
from pathlib import Path

import torch
from FlagEmbedding import BGEM3FlagModel
from qdrant_client.models import SparseVector

_MODELS_DIR = Path(os.environ.get("MODELS_DIR",
    str(Path(__file__).parent.parent.parent / "models" / "bge-m3")))


class Embedder:
    _instance: "Embedder | None" = None

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # FP16 on CUDA: faster. FP16 on CPU: half the memory (2.2GB→1.1GB), slightly slower.
        use_fp16 = self.device == "cuda" or os.environ.get("FORCE_FP16", "").lower() in ("1", "true")
        self.model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=use_fp16,
            cache_dir=str(_MODELS_DIR),
        )
        self.dim = 1024
        cache_size = int(os.environ.get("QUERY_EMBED_CACHE_SIZE", "512"))
        self._cached_embed_query = lru_cache(maxsize=cache_size)(self._embed_query_uncached)
        print(f"[Embedder] device={self.device}")

    def embed_batch(self, texts: list[str], batch_size: int = 64, max_length: int = 512) -> list[dict]:
        """Return list of {"dense": list[float], "sparse": SparseVector} per text."""
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
        )
        results = []
        for i in range(len(texts)):
            weights = out["lexical_weights"][i]
            results.append({
                "dense": out["dense_vecs"][i].tolist(),
                "sparse": SparseVector(
                    indices=list(weights.keys()),
                    values=[float(v) for v in weights.values()],
                ),
            })
        return results

    def embed_query(self, text: str) -> dict:
        """Single query vector — same format as embed_batch output."""
        return self._cached_embed_query(text)

    def _embed_query_uncached(self, text: str) -> dict:
        max_length = int(os.environ.get("QUERY_MAX_LENGTH", "128"))
        return self.embed_batch([text], batch_size=1, max_length=max_length)[0]
