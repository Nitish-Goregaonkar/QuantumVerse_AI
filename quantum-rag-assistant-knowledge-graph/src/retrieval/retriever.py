"""Knowledge-graph retriever: embedding-ranked entry chunks + graph expansion."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re
import numpy as np

from src.embeddings.embed_documents import EmbeddingEngine
from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.utils.config import settings

@dataclass
class RetrievalResult:
    text: str
    source: str
    score: float
    page: int | None = None
    chunk_id: int | None = None
    document_id: str | None = None
    metadata: dict | None = None

class Retriever:
    def __init__(self, graph_store: KnowledgeGraphStore | None = None):
        self.graph_store = graph_store or KnowledgeGraphStore()
        self.embedding_engine: EmbeddingEngine = self.graph_store.embedding_engine

    def load(self):
        self.graph_store.load()

    def retrieve(self, query: str, top_k: int = None, threshold: float = None, use_graph: bool = True,
                 document_ids: list[str] | None = None, debug: bool = False) -> List[RetrievalResult]:
        if not self.graph_store.loaded:
            self.graph_store.load()
        top_k = top_k or settings.top_k_results
        threshold = settings.similarity_threshold if threshold is None else threshold
        query_vector = self.embedding_engine.embed_query(query)
        query_terms = set(re.findall(r"[a-zA-Z0-9'-]+", query.casefold()))
        scored = []
        for node_id, data in self.graph_store.graph.nodes(data=True):
            if data.get("node_type") != "chunk" or not data.get("embedding"):
                continue
            source = data.get("source", "unknown")
            stable_id = f"doc::{source}"
            if document_ids is not None and stable_id not in document_ids and source not in document_ids:
                continue
            vector = np.asarray(data["embedding"], dtype=float)
            denom = (np.linalg.norm(query_vector) * np.linalg.norm(vector)) or 1.0
            similarity = float(np.dot(query_vector, vector) / denom)
            # Small lexical boost for exact terms and concepts directly linked to the chunk.
            chunk_terms = set(re.findall(r"[a-zA-Z0-9'-]+", data.get("text", "").casefold()))
            lexical = len(query_terms & chunk_terms) / max(1, len(query_terms))
            concept_names = [self.graph_store.graph.nodes[n].get("label", "").casefold() for n in self.graph_store.graph.neighbors(node_id)
                             if self.graph_store.graph.nodes[n].get("node_type") == "concept"]
            concept_boost = 0.08 if any(c and c in query.casefold() for c in concept_names) else 0.0
            score = max(0.0, min(1.0, 0.82 * similarity + 0.18 * lexical + concept_boost))
            scored.append((score, node_id, data))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        seen = set()
        # Broad document-summary requests should sample across the corpus rather
        # than require one chunk to exceed a semantic similarity threshold.
        summary_intent = bool(re.search(
            r"\b(summar(y|ize|ise)|overview|uploaded documents|all documents|key concepts|knowledge base)\b",
            query.casefold(),
        ))
        if summary_intent:
            # Pick best-scoring chunks per source first, then fill remaining slots.
            per_source = {}
            for score, node_id, data in scored:
                source = data.get("source", "unknown")
                if source not in per_source:
                    per_source[source] = (score, node_id, data)
            candidates = sorted(per_source.values(), key=lambda item: item[0], reverse=True)
            candidates += [item for item in scored if item not in candidates]
            for score, node_id, data in candidates:
                if len(results) >= top_k:
                    break
                key = (data.get("source"), data.get("text"))
                if key not in seen:
                    results.append(RetrievalResult(data.get("text", ""), data.get("source", "unknown"), score, data.get("page"),
                        data.get("chunk_id"), f"doc::{data.get('source', 'unknown')}", data.get("metadata", {})))
                    seen.add(key)
        # Normal retrieval: if threshold rejects every chunk, return the best
        # candidates rather than falsely claiming the knowledge base is empty.
        if not results:
            scored = [item for item in scored if item[0] >= max(0.05, threshold * 0.5)][:top_k]
        # Expand through graph links from the best entry nodes to related chunks.
        for score, node_id, data in scored:
            if (not summary_intent and score < threshold) or len(results) >= top_k:
                continue
            key = (data.get("source"), data.get("text"))
            if key not in seen:
                results.append(RetrievalResult(data.get("text", ""), data.get("source", "unknown"), score, data.get("page"),
                        data.get("chunk_id"), f"doc::{data.get('source', 'unknown')}", data.get("metadata", {})))
                seen.add(key)
            if not use_graph:
                continue
            for concept_id in self.graph_store.graph.neighbors(node_id):
                if self.graph_store.graph.nodes[concept_id].get("node_type") != "concept":
                    continue
                for related_id in self.graph_store.graph.neighbors(concept_id):
                    related = self.graph_store.graph.nodes[related_id]
                    if related.get("node_type") != "chunk" or related_id == node_id:
                        continue
                    related_doc_id = f"doc::{related.get('source', 'unknown')}"
                    if document_ids is not None and related_doc_id not in document_ids and related.get("source") not in document_ids:
                        continue
                    related_key = (related.get("source"), related.get("text"))
                    if related_key in seen or len(results) >= top_k:
                        continue
                    results.append(RetrievalResult(related.get("text", ""), related.get("source", "unknown"), max(0.0, score * 0.85),
                        related.get("page"), related.get("chunk_id"), f"doc::{related.get('source', 'unknown')}",
                        related.get("metadata", {})))
                    seen.add(related_key)
        if use_graph and document_ids is None:
            # User-defined graph nodes (added in the UI) contribute their description as context.
            q = query.casefold()
            for _, data in self.graph_store.graph.nodes(data=True):
                if data.get("custom") and data.get("label", "").casefold() in q and data.get("description"):
                    results.append(RetrievalResult(f"{data['label']}: {data['description']}", "Custom graph node", 0.9))
        return results

    @staticmethod
    def format_context(results: List[RetrievalResult]) -> str:
        return "\n\n".join(f"[Source: {r.source}" + (f", page {r.page}" if r.page else "") + f"]\n{r.text}" for r in results)
