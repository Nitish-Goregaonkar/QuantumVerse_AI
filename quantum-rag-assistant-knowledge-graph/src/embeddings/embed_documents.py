"""
Generates embeddings for text chunks using sentence-transformers.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class EmbeddingEngine:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    def _load_model(self):
        if self._model is None:
            log.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                log.info(f"Embedding model loaded (dim={self.get_embedding_dim()})")
            except ImportError:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._load_model()

        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        log.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  
            convert_to_numpy=True,
        )

        log.info(f"Generated embeddings: shape={embeddings.shape}")
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)

    def get_embedding_dim(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()