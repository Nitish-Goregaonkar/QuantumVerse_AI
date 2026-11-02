"""A lightweight local knowledge graph with embeddings stored on chunk nodes.

This deliberately does not use a vector database. NetworkX stores document,
chunk, and concept nodes; embeddings are node attributes used to rank graph
entry points, after which retrieval expands through graph relationships.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

from src.embeddings.embed_documents import EmbeddingEngine
from src.ingestion.chunker import Chunk
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

CONCEPT_STOPWORDS = {
    "about", "after", "also", "among", "because", "between", "could", "does",
    "each", "from", "have", "into", "more", "most", "other", "should", "some",
    "such", "than", "that", "their", "there", "these", "they", "this", "through",
    "under", "using", "very", "what", "when", "where", "which", "while", "with",
    "would", "your", "quantum computing", "computing", "system", "systems",
    # Capitalised sentence starters that are not real concepts
    "the", "for", "then", "since", "let", "here", "note", "thus", "hence", "now",
    "figure", "example", "exercise", "chapter", "section", "table", "and", "but",
    "its", "our", "any", "all", "one", "two", "can", "may", "was", "are", "not",
    "how", "why", "who", "say", "next", "you", "finally", "similarly", "however",
    "therefore", "suppose", "recall", "again", "so", "in", "if", "it", "we", "this", "once", "both", "first", "second", "given", "consider",
}


def extract_concepts(text: str, max_concepts: int = 18) -> list[str]:
    """Extract useful named/technical phrases without requiring an LLM call."""
    candidates: list[str] = []
    # Technical phrases / capitalized names, e.g. Quantum Entanglement, QAOA.
    patterns = [
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9-]+){0,3}\b", 0),
        (r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b", 0),
        (r"\b(?:qubit|qubits|superposition|entanglement|decoherence|quantum gate|quantum gates|Bloch sphere|Hadamard gate|CNOT gate|Shor's algorithm|Grover's algorithm|variational quantum algorithm|VQE|QAOA|embedding|measurement|interference)\b", re.IGNORECASE),
    ]
    for pattern, flags in patterns:
        candidates.extend(re.findall(pattern, text, flags=flags))
    seen = set()
    output = []
    for item in candidates:
        concept = re.sub(r"\s+", " ", item).strip(" .,:;()[]{}")
        key = concept.casefold()
        if len(concept) < 3 or key in CONCEPT_STOPWORDS or key in seen:
            continue
        if len(concept.split()) == 1 and concept.islower() and len(concept) < 5:
            continue
        seen.add(key)
        output.append(concept)
        if len(output) >= max_concepts:
            break
    return output


QUANTUM_TERMS = re.compile(
    r"qubit|superposition|entangle|decoheren|quantum|bloch|hadamard|cnot|shor|grover|"
    r"vqe|qaoa|interference|unitary|hilbert|eigen|photon|spin|bell state|teleport",
    re.IGNORECASE,
)


def categorize_node(node_type: str, label: str) -> str:
    """Map internal graph node types onto the categories shown in the web UI."""
    if node_type == "document":
        return "Document"
    if QUANTUM_TERMS.search(label or ""):
        return "Quantum Entity"
    return "Concept"


def short_id(node_id: str) -> str:
    return "node_" + hashlib.md5(node_id.encode("utf-8")).hexdigest()[:8]


def display_name(source: str) -> str:
    """Strip the random 8-hex upload prefix added by the upload endpoint."""
    return re.sub(r"^[0-9a-f]{8}_", "", source)


class KnowledgeGraphStore:
    def __init__(self, path: Path | str | None = None, embedding_engine: EmbeddingEngine | None = None):
        self.path = Path(path or settings.knowledge_graph_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.graph = nx.Graph()
        self.loaded = False

    def load(self) -> None:
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.graph = nx.node_link_graph(payload)
            self.loaded = True
            log.info("Loaded knowledge graph: %s nodes, %s edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        else:
            self.graph = nx.Graph()
            self.loaded = True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = nx.node_link_data(self.graph)
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def rebuild_from_chunks(self, chunks: Iterable[Chunk], replace: bool = True) -> dict:
        chunks = list(chunks)
        if not chunks:
            raise ValueError("No readable text chunks were produced from the uploaded documents.")
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_engine.embed_texts(texts)
        custom_nodes: list[tuple[str, dict]] = []
        custom_edges: list[tuple[str, str, dict]] = []
        if replace:
            # Keep user-created nodes (added from the UI) when the graph is rebuilt.
            custom_nodes = [(n, dict(d)) for n, d in self.graph.nodes(data=True) if d.get("custom")]
            keep = {n for n, _ in custom_nodes}
            custom_edges = [(u, v, dict(d)) for u, v, d in self.graph.edges(data=True) if u in keep and v in keep]
            self.graph = nx.Graph()
        # Stable document nodes and chunk nodes.
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            source = str(chunk.source)
            doc_id = f"doc::{source}"
            chunk_id = f"chunk::{source}::{chunk.chunk_id}::{index}"
            self.graph.add_node(doc_id, node_type="document", label=source, source=source)
            self.graph.add_node(
                chunk_id, node_type="chunk", label=f"{source} chunk {chunk.chunk_id}",
                source=source, text=chunk.text, page=chunk.page, chunk_id=chunk.chunk_id, metadata=chunk.metadata,
                embedding=np.asarray(vector, dtype=float).tolist(),
            )
            self.graph.add_edge(doc_id, chunk_id, relation="CONTAINS")
            concepts = extract_concepts(chunk.text)
            for concept in concepts:
                concept_id = f"concept::{concept.casefold()}"
                self.graph.add_node(concept_id, node_type="concept", label=concept)
                self.graph.add_edge(chunk_id, concept_id, relation="MENTIONS")
        # Connect concepts that occur in the same chunk, making multi-hop graph expansion useful.
        for node_id, data in list(self.graph.nodes(data=True)):
            if data.get("node_type") != "chunk":
                continue
            concept_neighbors = [n for n in self.graph.neighbors(node_id) if self.graph.nodes[n].get("node_type") == "concept"]
            for i, left in enumerate(concept_neighbors):
                for right in concept_neighbors[i + 1:]:
                    if left != right:
                        self.graph.add_edge(left, right, relation="CO_OCCURS")
        for node_id, data in custom_nodes:
            self.graph.add_node(node_id, **data)
        for u, v, data in custom_edges:
            self.graph.add_edge(u, v, **data)
        self._relink_custom_nodes()
        self.save()
        self.loaded = True
        return {
            "documents": len({c.source for c in chunks}),
            "chunks": len(chunks),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "concepts": sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "concept"),
        }

    def stats(self) -> dict:
        if not self.loaded:
            self.load()
        return {
            "documents": sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "document"),
            "chunks": sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "chunk"),
            "concepts": sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "concept"),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }


    # ------------------------------------------------------------------
    # Helpers used by the web UI (graph canvas, document vault)
    # ------------------------------------------------------------------
    def _typed(self, node_type: str):
        return [(n, d) for n, d in self.graph.nodes(data=True) if d.get("node_type") == node_type]

    def _relink_custom_nodes(self) -> None:
        """Re-attach custom nodes to a concept whose label matches, if one exists."""
        by_label = {d.get("label", "").casefold(): n for n, d in self._typed("concept") if not d.get("custom")}
        for node_id, data in self._typed("concept"):
            if not data.get("custom"):
                continue
            target = by_label.get(data.get("label", "").casefold())
            if target and target != node_id:
                self.graph.add_edge(node_id, target, relation="RELATED_TO")

    def _docs_for_concept(self, concept_id: str) -> set[str]:
        docs: set[str] = set()
        for chunk_id in self.graph.neighbors(concept_id):
            cdata = self.graph.nodes[chunk_id]
            if cdata.get("node_type") == "chunk":
                docs.add(f"doc::{cdata.get('source')}")
        return docs

    def overview(self, limit: int = 60) -> dict:
        """Return a compact graph view while keeping every document connected
        to concepts extracted from its own chunks.

        The previous implementation selected only globally popular concepts.
        A document containing useful but less frequent concepts could therefore
        appear as an isolated document node even though its chunks and concept
        edges existed in the full graph.
        """
        if not self.loaded:
            self.load()
        g = self.graph
        documents = self._typed("document")
        concepts = self._typed("concept")
        custom = [(n, d) for n, d in concepts if d.get("custom")]
        regular = [
            (n, d) for n, d in concepts
            if not d.get("custom") and d.get("label", "").casefold() not in CONCEPT_STOPWORDS
        ]
        regular.sort(key=lambda item: g.degree(item[0]), reverse=True)

        # Keep the globally important concepts, but also select a few concepts
        # local to EACH document. This prevents low-frequency documents from
        # becoming disconnected merely because their concepts fell outside
        # the global top-N cutoff.
        selected_ids = {n for n, _ in regular[:max(0, limit)]}
        document_concepts: dict[str, list[tuple[int, str]]] = {}
        for doc_id, _doc_data in documents:
            counts: dict[str, int] = {}
            for chunk_id in g.neighbors(doc_id):
                if g.nodes[chunk_id].get("node_type") != "chunk":
                    continue
                for concept_id in g.neighbors(chunk_id):
                    cdata = g.nodes[concept_id]
                    if cdata.get("node_type") != "concept":
                        continue
                    if cdata.get("label", "").casefold() in CONCEPT_STOPWORDS:
                        continue
                    counts[concept_id] = counts.get(concept_id, 0) + 1
            ranked = sorted(counts.items(), key=lambda item: (-item[1], g.nodes[item[0]].get("label", "").casefold()))
            document_concepts[doc_id] = ranked[:3]
            selected_ids.update(concept_id for concept_id, _ in ranked[:3])

        selected = [(n, g.nodes[n]) for n in selected_ids if n in g]
        # Preserve deterministic ordering for stable layouts between refreshes.
        selected.sort(key=lambda item: (
            0 if item[1].get("custom") else 1,
            item[1].get("label", "").casefold(),
            item[0],
        ))
        picked = selected + custom
        # Avoid duplicate custom concepts if they were already selected.
        deduped = {}
        for node_id, data in picked:
            deduped[node_id] = data
        picked = list(deduped.items())
        included = {n for n, _ in documents} | {n for n, _ in picked}

        nodes = []
        for node_id, data in documents + picked:
            nodes.append({
                "id": node_id,
                "label": display_name(data.get("label", node_id)),
                "type": categorize_node(data.get("node_type"), data.get("label", "")),
                "degree": g.degree(node_id),
                "custom": bool(data.get("custom")),
            })

        links, seen = [], set()

        def add_link(a, b, relation):
            if a == b or a not in included or b not in included:
                return
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                links.append({"source": key[0], "target": key[1], "relation": relation})

        # Explicit document-to-concept links are derived from the document's
        # own chunks. These are presentation edges; the underlying graph remains
        # document -> chunk -> concept.
        for doc_id, ranked in document_concepts.items():
            for concept_id, _count in ranked:
                add_link(doc_id, concept_id, "CONTAINS_CONCEPT")

        # Retain a small number of concept co-occurrence links to avoid a hairball.
        picked_ids = [n for n, _ in picked]
        chunk_sets = {
            n: {c for c in g.neighbors(n) if g.nodes[c].get("node_type") == "chunk"}
            for n in picked_ids
        }
        max_per_node = 2
        for node_id in picked_ids:
            candidates = []
            for other in picked_ids:
                if other == node_id or not g.has_edge(node_id, other):
                    continue
                weight = len(chunk_sets[node_id] & chunk_sets[other]) + 1
                candidates.append((weight, other))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _weight, other in candidates[:max_per_node]:
                add_link(node_id, other, g.edges[node_id, other].get("relation", "CO_OCCURS"))

        return {
            "nodes": nodes,
            "links": links,
            "stats": self.stats(),
            "shown_concepts": len(picked),
            "total_concepts": len(concepts),
        }

    def node_detail(self, node_id: str) -> dict | None:
        if not self.loaded:
            self.load()
        if node_id not in self.graph:
            return None
        g = self.graph
        data = g.nodes[node_id]
        node_type = data.get("node_type")
        label = display_name(data.get("label", node_id))
        neighbors = list(g.neighbors(node_id))
        detail = {
            "id": node_id,
            "label": label,
            "type": categorize_node(node_type, label),
            "node_id": short_id(node_id),
            "degree": len(neighbors),
            "custom": bool(data.get("custom")),
        }
        if data.get("custom"):
            detail["description"] = data.get("description", "")
            detail["chunks"] = 0
        elif node_type == "document":
            chunks = [n for n in neighbors if g.nodes[n].get("node_type") == "chunk"]
            detail["chunks"] = len(chunks)
            detail["description"] = f"Source document split into {len(chunks)} embedded chunks in the knowledge graph."
        else:
            chunks = [n for n in neighbors if g.nodes[n].get("node_type") == "chunk"]
            related = sorted(
                (n for n in neighbors if g.nodes[n].get("node_type") == "concept"),
                key=lambda n: g.degree(n), reverse=True,
            )[:5]
            docs = {display_name(g.nodes[c].get("source", "")) for c in chunks}
            text = f"Mentioned in {len(chunks)} chunk(s) across {len(docs)} document(s)."
            if related:
                text += " Frequently co-occurs with: " + ", ".join(g.nodes[n].get("label", "") for n in related) + "."
            if chunks:
                snippet = (g.nodes[chunks[0]].get("text") or "").strip().replace("\n", " ")
                if snippet:
                    text += f' Example context: "{snippet[:220]}..."'
            detail["chunks"] = len(chunks)
            detail["description"] = text
        return detail

    def add_custom_node(self, label: str, category: str, description: str) -> dict:
        if not self.loaded:
            self.load()
        label = re.sub(r"\s+", " ", label).strip()
        if not label:
            raise ValueError("Label is required.")
        node_id = f"concept::custom::{label.casefold()}"
        if node_id in self.graph:
            raise ValueError(f"A custom node named '{label}' already exists.")
        self.graph.add_node(node_id, node_type="concept", label=label, custom=True,
                            category=category, description=description)
        existing = [n for n, d in self._typed("concept") if not d.get("custom") and d.get("label", "").casefold() == label.casefold()]
        if existing:
            self.graph.add_edge(node_id, existing[0], relation="RELATED_TO")
        else:
            # Attach to the most connected concept so the node is visible in the graph.
            hubs = sorted((n for n, d in self._typed("concept") if not d.get("custom")), key=lambda n: self.graph.degree(n), reverse=True)
            if hubs:
                self.graph.add_edge(node_id, hubs[0], relation="RELATED_TO")
        self.save()
        return self.node_detail(node_id)

    def list_documents(self) -> list[dict]:
        """Documents currently in the graph, with file size from data/raw when available."""
        if not self.loaded:
            self.load()
        files = {f.name: f for f in Path(settings.data_raw_dir).rglob("*") if f.is_file()}
        result = []
        for doc_id, data in self._typed("document"):
            source = data.get("source", "")
            chunks = sum(1 for n in self.graph.neighbors(doc_id) if self.graph.nodes[n].get("node_type") == "chunk")
            path = files.get(source)
            size = path.stat().st_size if path else 0
            result.append({
                "id": f"doc::{source}",
                "source": source,
                "name": display_name(source),
                "size_bytes": size,
                "chunks": chunks,
                "date": __import__("datetime").datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d") if path else "",
                "status": "Indexed",
            })
        return sorted(result, key=lambda d: d["name"].casefold())

    def delete_document(self, source: str) -> dict:
        """Remove a document, its chunks, orphaned concepts, and its source file.

        Accept either the stored source filename or UI node ID (``doc::<source>``).
        """
        if not self.loaded:
            self.load()
        source = str(source)
        if source.startswith("doc::"):
            source = source[5:]
        doc_id = f"doc::{source}"
        if doc_id not in self.graph or self.graph.nodes[doc_id].get("node_type") != "document":
            raise KeyError(source)

        chunk_ids = [n for n, d in self._typed("chunk") if d.get("source") == source]
        self.graph.remove_nodes_from(chunk_ids + [doc_id])
        orphans = [
            n for n, d in self._typed("concept")
            if not d.get("custom") and not any(
                self.graph.nodes[x].get("node_type") == "chunk"
                for x in list(self.graph.neighbors(n))
            )
        ]
        self.graph.remove_nodes_from(orphans)

        removed_file = False
        base = Path(settings.data_raw_dir).resolve()
        for candidate in base.rglob("*"):
            if candidate.is_file() and candidate.name == Path(source).name:
                resolved = candidate.resolve()
                if base in resolved.parents:
                    candidate.unlink()
                    removed_file = True
                    break
        self.save()
        return {"source": source, "removed_chunks": len(chunk_ids),
                "removed_concepts": len(orphans), "file_deleted": removed_file}
