"""rag/retriever.py — RAG pipeline: load -> split -> embed -> retrieve.

Indexing is IDEMPOTENT: every chunk gets a stable id (source + position + content
hash). On startup we only embed chunks that aren't already stored and delete ones
whose source text changed — so restarting the app never duplicates passages, and
editing a sample doc re-indexes just that doc.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "sample_docs"
DB_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION = "qa_docs"
_retriever = None


def _load() -> list[Document]:
    return [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
            for p in sorted(DOCS_DIR.glob("*.md"))]


def _chunk_id(doc: Document, i: int) -> str:
    h = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:10]
    return f"{doc.metadata.get('source', '?')}:{i}:{h}"


def _embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))


def build_store(embeddings=None, persist_dir: Path | str = DB_DIR) -> Chroma:
    """Create/open the store and sync it with sample_docs (adds new, removes stale)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(_load())
    ids = [_chunk_id(c, i) for i, c in enumerate(chunks)]

    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings or _embeddings(),
                   persist_directory=str(persist_dir))
    existing = set(store.get(include=[])["ids"])
    wanted = dict(zip(ids, chunks))

    stale = list(existing - wanted.keys())
    if stale:
        store.delete(ids=stale)
    new_ids = [i for i in ids if i not in existing]
    if new_ids:
        store.add_documents([wanted[i] for i in new_ids], ids=new_ids)
    log.info("RAG index: %d chunks (%d added, %d removed)", len(ids), len(new_ids), len(stale))
    return store


def get_retriever(k: int = 3):
    global _retriever
    if _retriever is None:
        _retriever = build_store().as_retriever(search_kwargs={"k": k})
    return _retriever