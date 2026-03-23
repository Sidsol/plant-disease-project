"""
RAG (Retrieval-Augmented Generation) service.

Uses ChromaDB for vector storage and Ollama for embeddings.
Loads markdown documents from the knowledge base, chunks them,
and provides similarity search for the chat service.
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
CHROMA_DIR = PROJECT_ROOT / "data" / "chromadb"

COLLECTION_NAME = "plant_disease_knowledge"

# ---------------------------------------------------------------------------
# Text splitter configuration
# ---------------------------------------------------------------------------
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    length_function=len,
)

# ---------------------------------------------------------------------------
# Singleton ChromaDB client & collection
# ---------------------------------------------------------------------------
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_embedding_fn(ollama_url: str = "http://127.0.0.1:11434") -> OllamaEmbeddingFunction:
    return OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url=ollama_url,
    )


def get_collection(
    ollama_url: str = "http://127.0.0.1:11434",
) -> chromadb.Collection:
    """Return (and lazily create) the ChromaDB collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_fn(ollama_url),
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# ---------------------------------------------------------------------------
# Document loading & chunking
# ---------------------------------------------------------------------------

def _detect_category(path: Path) -> str:
    """Infer document category from its parent folder name."""
    parent = path.parent.name
    if parent == "diseases":
        return "disease"
    if parent == "plants":
        return "plant"
    if parent == "general_care":
        return "general_care"
    return "other"


def _extract_title(text: str) -> str:
    """Extract the first markdown heading as a document title."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.lstrip("# ").strip()
    return "Untitled"


def load_documents(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[dict]:
    """
    Read all .md files from the knowledge directory tree.

    Walks the knowledge directory recursively, extracts a title from
    each document's first ``#`` heading, infers the category from the
    parent folder name, and splits the text into overlapping chunks
    using ``RecursiveCharacterTextSplitter``.

    Args:
        knowledge_dir: Root path of the markdown knowledge base.

    Returns:
        List of dicts, each containing:
          - ``path``:     relative path within the knowledge dir
          - ``title``:    first markdown heading (or 'Untitled')
          - ``category``: 'disease', 'plant', 'general_care', or 'other'
          - ``chunks``:   list of {id, text, metadata} dicts
    """
    docs: list[dict] = []
    md_files = sorted(knowledge_dir.rglob("*.md"))

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        title = _extract_title(text)
        category = _detect_category(md_path)
        rel_path = md_path.relative_to(knowledge_dir).as_posix()

        chunks = _splitter.split_text(text)
        chunk_records = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{rel_path}::chunk_{i}"
            chunk_records.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": rel_path,
                        "title": title,
                        "category": category,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                }
            )
        docs.append(
            {
                "path": rel_path,
                "title": title,
                "category": category,
                "chunks": chunk_records,
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(
    knowledge_dir: Path = KNOWLEDGE_DIR,
    ollama_url: str = "http://127.0.0.1:11434",
    reset: bool = False,
) -> dict:
    """
    Ingest all knowledge-base documents into ChromaDB.

    Args:
        knowledge_dir: Root of the markdown knowledge tree.
        ollama_url: Base URL of the Ollama server.
        reset: If True, delete the existing collection before ingesting.

    Returns:
        Summary dict with document/chunk counts.
    """
    global _client, _collection
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _collection = None

    collection = get_collection(ollama_url)

    docs = load_documents(knowledge_dir)
    total_chunks = 0

    for doc in docs:
        ids = [c["id"] for c in doc["chunks"]]
        texts = [c["text"] for c in doc["chunks"]]
        metadatas = [c["metadata"] for c in doc["chunks"]]
        if ids:
            collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            total_chunks += len(ids)

    return {
        "documents": len(docs),
        "chunks": total_chunks,
        "collection": COLLECTION_NAME,
        "chroma_dir": str(CHROMA_DIR),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    n_results: int = 5,
    category_filter: Optional[str] = None,
    ollama_url: str = "http://127.0.0.1:11434",
) -> list[dict]:
    """
    Retrieve the most relevant knowledge chunks for a user query.

    Embeds the query with nomic-embed-text, performs cosine similarity
    search against the ChromaDB collection, and returns ranked results
    with similarity scores converted from distance to [0, 1] similarity.

    Args:
        query: Natural-language question from the user.
        n_results: Maximum number of chunks to return (default 5).
        category_filter: Optional category ('disease', 'plant',
            'general_care') to restrict retrieval scope.
        ollama_url: Ollama server URL for embedding the query.

    Returns:
        List of dicts, each containing:
          - ``text``:     the chunk content
          - ``source``:   relative path to the source document
          - ``title``:    document title
          - ``category``: document category
          - ``score``:    cosine similarity (1.0 = identical)
    """
    collection = get_collection(ollama_url)

    where_filter = {"category": category_filter} if category_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    retrieved = []
    if results and results["documents"]:
        for i, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            retrieved.append(
                {
                    "text": doc_text,
                    "source": meta.get("source", ""),
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "score": round(1 - distance, 4) if distance is not None else None,
                }
            )
    return retrieved
