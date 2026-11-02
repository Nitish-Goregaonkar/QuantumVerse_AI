"""Build the local knowledge graph from previously processed chunks."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.chunker import Chunk
from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.utils.config import settings

def main():
    chunks_path = settings.chunks_file
    if not chunks_path.exists():
        raise SystemExit(f"Chunks file not found: {chunks_path}. Run python scripts/ingest.py first, or upload documents in the UI.")
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(item) for item in data]
    store = KnowledgeGraphStore()
    stats = store.rebuild_from_chunks(chunks, replace=True)
    print("Knowledge graph built successfully:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"Saved to: {store.path}")

if __name__ == "__main__":
    main()
