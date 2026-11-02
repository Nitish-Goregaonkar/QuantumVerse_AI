"""
Text chunking strategies for RAG pipelines.

Strategies:
1. Fixed-size chunking with overlap
2. Sentence-aware chunking
3. Paragraph-aware chunking (default — best for quantum docs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.ingestion.load_documents import Document
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:

    text: str
    source: str
    chunk_id: int
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            text=data["text"],
            source=data["source"],
            chunk_id=data["chunk_id"],
            page=data.get("page"),
            metadata=data.get("metadata", {}),
        )


class TextChunker:

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = "paragraph",
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.strategy = strategy

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        chunk_id = 0

        for doc in documents:
            if not doc.content.strip():
                continue

            raw_chunks = self._split(doc.content)

            for text in raw_chunks:
                if len(text.strip()) < 30:  
                    continue
                all_chunks.append(
                    Chunk(
                        text=text.strip(),
                        source=doc.source,
                        chunk_id=chunk_id,
                        page=doc.page,
                        metadata=doc.metadata,
                    )
                )
                chunk_id += 1

        log.info(
            f"Created {len(all_chunks)} chunks from {len(documents)} documents "
            f"(strategy={self.strategy}, size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def _split(self, text: str) -> List[str]:
        """Route to the appropriate splitting strategy."""
        if self.strategy == "fixed":
            return self._fixed_split(text)
        elif self.strategy == "sentence":
            return self._sentence_split(text)
        else:
            return self._paragraph_split(text)

    def _fixed_split(self, text: str) -> List[str]:
        """Simple fixed-size character split with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sentence_split(self, text: str) -> List[str]:
        """Split on sentence boundaries, group into chunk_size windows."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += " " + sentence
            else:
                if current:
                    chunks.append(current.strip())
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = overlap_text + " " + sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _paragraph_split(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n{2,}", text)
        chunks = []
        current_parts = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            if current_len + para_len > self.chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))
                if self.chunk_overlap > 0 and current_parts:
                    current_parts = [current_parts[-1]]
                    current_len = len(current_parts[0])
                else:
                    current_parts = []
                    current_len = 0

            current_parts.append(para)
            current_len += para_len

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks