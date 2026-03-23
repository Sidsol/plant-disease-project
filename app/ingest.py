"""
Knowledge base ingestion CLI.

Usage:
    python -m app.ingest              # Ingest (upsert) knowledge docs
    python -m app.ingest --reset      # Delete collection and re-ingest
    python -m app.ingest --stats      # Show collection stats only
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import ingest, get_collection, KNOWLEDGE_DIR, CHROMA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Ingest plant disease knowledge base into ChromaDB."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and re-ingest from scratch.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics without ingesting.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL (default: http://127.0.0.1:11434).",
    )
    args = parser.parse_args()

    if args.stats:
        collection = get_collection(args.ollama_url)
        count = collection.count()
        print(f"Collection: {collection.name}")
        print(f"Documents (chunks): {count}")
        print(f"ChromaDB path: {CHROMA_DIR}")
        return

    print(f"Knowledge directory: {KNOWLEDGE_DIR}")
    print(f"ChromaDB storage:    {CHROMA_DIR}")
    if args.reset:
        print("Mode: RESET + re-ingest")
    else:
        print("Mode: Upsert (update/insert)")
    print()

    result = ingest(
        knowledge_dir=KNOWLEDGE_DIR,
        ollama_url=args.ollama_url,
        reset=args.reset,
    )

    print(f"Documents processed: {result['documents']}")
    print(f"Total chunks stored: {result['chunks']}")
    print(f"Collection:          {result['collection']}")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
