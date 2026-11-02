import networkx as nx

from src.knowledge_graph.graph_store import KnowledgeGraphStore


def test_overview_connects_each_document_to_its_own_concepts():
    store = object.__new__(KnowledgeGraphStore)
    store.loaded = True
    store.graph = nx.Graph()

    # A very popular concept belongs to document A, while document B has
    # low-frequency concepts that would not make the global top-N cutoff.
    store.graph.add_node("doc::a.pdf", node_type="document", label="a.pdf", source="a.pdf")
    store.graph.add_node("doc::b.pdf", node_type="document", label="b.pdf", source="b.pdf")
    store.graph.add_node("chunk::a", node_type="chunk", source="a.pdf", text="quantum")
    store.graph.add_node("chunk::b", node_type="chunk", source="b.pdf", text="teleportation")
    store.graph.add_node("concept::quantum", node_type="concept", label="Quantum")
    store.graph.add_node("concept::teleportation", node_type="concept", label="Teleportation")
    store.graph.add_edges_from([
        ("doc::a.pdf", "chunk::a"),
        ("chunk::a", "concept::quantum"),
        ("doc::b.pdf", "chunk::b"),
        ("chunk::b", "concept::teleportation"),
    ])

    result = store.overview(limit=1)
    links = {(link["source"], link["target"]) for link in result["links"]}

    assert ("doc::a.pdf", "concept::quantum") in links or ("concept::quantum", "doc::a.pdf") in links
    assert ("doc::b.pdf", "concept::teleportation") in links or ("concept::teleportation", "doc::b.pdf") in links
    assert len([n for n in result["nodes"] if n["type"] == "Document"]) == 2
