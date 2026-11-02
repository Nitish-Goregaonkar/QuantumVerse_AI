from pathlib import Path

import networkx as nx

from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.utils.config import settings


def test_delete_document_removes_document_chunks_and_orphan_concepts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_raw_dir", tmp_path)
    source = "sample.pdf"
    (tmp_path / source).write_bytes(b"source")

    store = object.__new__(KnowledgeGraphStore)
    store.loaded = True
    store.graph = nx.Graph()
    store.graph.add_node(f"doc::{source}", node_type="document", source=source)
    store.graph.add_node("chunk::1", node_type="chunk", source=source)
    store.graph.add_node("concept::orphan", node_type="concept", label="Orphan")
    store.graph.add_node("concept::shared", node_type="concept", label="Shared")
    store.graph.add_node("chunk::2", node_type="chunk", source="other.pdf")
    store.graph.add_node("concept::custom", node_type="concept", label="Custom", custom=True)
    store.graph.add_edges_from([
        (f"doc::{source}", "chunk::1"),
        ("chunk::1", "concept::orphan"),
        ("chunk::2", "concept::shared"),
        ("chunk::1", "concept::shared"),
    ])
    store.save = lambda: None

    result = store.delete_document(f"doc::{source}")

    assert result["removed_chunks"] == 1
    assert result["removed_concepts"] == 1
    assert result["file_deleted"] is True
    assert f"doc::{source}" not in store.graph
    assert "chunk::1" not in store.graph
    assert "concept::orphan" not in store.graph
    assert "concept::shared" in store.graph
    assert "concept::custom" in store.graph
    assert not (tmp_path / source).exists()


def test_delete_document_unknown_source_raises_key_error():
    store = object.__new__(KnowledgeGraphStore)
    store.loaded = True
    store.graph = nx.Graph()
    try:
        store.delete_document("missing.pdf")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for an unknown document")
