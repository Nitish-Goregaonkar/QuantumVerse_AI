"""
CLI script to load, chunk, and save documents for RAG.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --source data/raw/
    python scripts/ingest.py --source data/raw/ --strategy paragraph --chunk-size 512
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.chunker import TextChunker
from src.ingestion.load_documents import DocumentLoader
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents for the Quantum RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py
  python scripts/ingest.py --source data/raw/
  python scripts/ingest.py --strategy sentence --chunk-size 256
        """,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(settings.data_raw_dir),
        help=f"Directory containing source documents (default: {settings.data_raw_dir})",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["fixed", "sentence", "paragraph"],
        default="paragraph",
        help="Text chunking strategy (default: paragraph)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Target chunk size in characters (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Chunk overlap in characters (default: {settings.chunk_overlap})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.chunks_file),
        help=f"Output JSON file for chunks (default: {settings.chunks_file})",
    )

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Quantum RAG Assistant — Document Ingestion")
    log.info("=" * 60)
    log.info(f"Source directory : {args.source}")
    log.info(f"Chunking strategy: {args.strategy}")
    log.info(f"Chunk size       : {args.chunk_size} chars")
    log.info(f"Chunk overlap    : {args.chunk_overlap} chars")
    log.info(f"Output file      : {args.output}")
    log.info("=" * 60)

    # Step 1: Load documents
    loader = DocumentLoader()
    documents = loader.load_directory(args.source)

    if not documents:
        log.error(f"No documents found in {args.source}")
        log.info("Add PDF, TXT, MD, or DOCX files to the source directory.")
        sys.exit(1)

    # Step 2: Chunk documents
    chunker = TextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        strategy=args.strategy,
    )
    chunks = chunker.chunk_documents(documents)

    if not chunks:
        log.error("No chunks created. Check your documents.")
        sys.exit(1)

    # Step 3: Save chunks to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_data = [c.to_dict() for c in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info("✅ Ingestion complete!")
    log.info(f"   Documents loaded : {len(documents)}")
    log.info(f"   Chunks created   : {len(chunks)}")
    log.info(f"   Saved to         : {output_path}")
    log.info("")
    log.info("Next step: python scripts/build_index.py (build the knowledge graph)")


if __name__ == "__main__":
    main()