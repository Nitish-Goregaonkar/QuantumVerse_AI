
import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

path = Path(__file__).parent / "data" / "knowledge_graph" / "knowledge_graph.json"

if not path.exists():
    print("Graph not found. Upload and process a document first.")
    raise SystemExit(1)

with open(path, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

G = nx.node_link_graph(graph_data)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Display a manageable number of nodes
nodes = list(G.nodes())[:100]
H = G.subgraph(nodes).copy()

colors = []
for node in H.nodes:
    kind = H.nodes[node].get("node_type")
    colors.append({
        "document": "skyblue",
        "chunk": "lightgreen",
        "concept": "orange"
    }.get(kind, "lightgray"))

labels = {
    node: str(H.nodes[node].get("label", node))[:30]
    for node in H.nodes
}

plt.figure(figsize=(16, 10))
pos = nx.spring_layout(H, seed=42)

nx.draw_networkx(
    H,
    pos,
    labels=labels,
    node_color=colors,
    node_size=900,
    font_size=7,
    edge_color="gray",
    with_labels=True
)

plt.title("Quantum RAG Assistant - Knowledge Graph")
plt.axis("off")
plt.tight_layout()
plt.savefig("knowledge_graph.png", dpi=200)
print("Saved visual graph to knowledge_graph.png")
plt.show()
