"""
rag_service.py

RAG layer for the voice AI system:
- Embeds text using Gemini's embedding API
- Stores/retrieves chunks from a local ChromaDB collection
- Provides vector_search() for use inside execute_tool()
- Provides ingest_documents() for one-time/offline indexing

Requires:
    pip install chromadb google-genai
"""

import os
import asyncio
import chromadb
from google import genai
from google.genai import types

# ── Config ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "company_knowledge")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

_embed_client = None
_chroma_client = None
_collection = None


def get_embed_client():
    global _embed_client
    if _embed_client is None:
        _embed_client = genai.Client(api_key=GEMINI_API_KEY)
    return _embed_client


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # No embedding_function passed — we embed manually via Gemini
        # and pass vectors directly, so Chroma doesn't need its own embedder.
        _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def _embed_sync(text: str, task_type: str = "RETRIEVAL_QUERY"):
    """Blocking call to Gemini embeddings API. Run inside executor when called from async code."""
    client = get_embed_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values


async def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY"):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _embed_sync(text, task_type))


async def vector_search(query: str, top_k: int = TOP_K) -> str:
    """
    Embeds the query and searches ChromaDB for the closest matching chunks.
    Returns a single string of concatenated chunk text (what the tool result becomes).
    """
    if not query or not query.strip():
        return "No query provided."

    try:
        query_vector = await embed_text(query, task_type="RETRIEVAL_QUERY")
    except Exception as e:
        print(f"[RAG] embedding error: {type(e).__name__}: {e}")
        return "Sorry, I couldn't search the knowledge base right now."

    collection = get_collection()
    loop = asyncio.get_event_loop()

    def _query():
        return collection.query(query_embeddings=[query_vector], n_results=top_k)

    try:
        results = await loop.run_in_executor(None, _query)
    except Exception as e:
        print(f"[RAG] chroma query error: {type(e).__name__}: {e}")
        return "Sorry, I couldn't search the knowledge base right now."

    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No relevant information found in the knowledge base."

    return "\n\n---\n\n".join(docs)


# ── Chunking + Ingestion (run offline, not during live calls) ──

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """Simple sliding-window chunker by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_documents(documents: list[dict]):
    """
    documents: list of {"id": str, "text": str, "metadata": dict (optional)}
    Chunks each doc, embeds each chunk with RETRIEVAL_DOCUMENT task type,
    and upserts into ChromaDB.
    """
    collection = get_collection()

    all_ids, all_embeddings, all_docs, all_metas = [], [], [], []

    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]
        metadata = doc.get("metadata", {})

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            vector = _embed_sync(chunk, task_type="RETRIEVAL_DOCUMENT")

            all_ids.append(chunk_id)
            all_embeddings.append(vector)
            all_docs.append(chunk)
            all_metas.append({**metadata, "source_doc": doc_id, "chunk_index": i})

    if all_ids:
        collection.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_docs,
            metadatas=all_metas
        )
        print(f"[RAG] Ingested {len(all_ids)} chunks from {len(documents)} document(s)")
    else:
        print("[RAG] No chunks to ingest")


if __name__ == "__main__":
    # Example one-time ingestion run:
    #   python rag_service.py
    from iae_knowledge import IEI_KNOWLEDGE

    ingest_documents([
        {"id": "iei_knowledge", "text": IEI_KNOWLEDGE, "metadata": {"topic": "iei"}}
    ])