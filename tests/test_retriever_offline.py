"""Bug 6 regression: re-indexing must not duplicate chunks (no API key needed)."""
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag import retriever


def test_reindex_is_idempotent(tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    first = len(retriever.build_store(emb, tmp_path).get(include=[])["ids"])
    second = len(retriever.build_store(emb, tmp_path).get(include=[])["ids"])
    third = len(retriever.build_store(emb, tmp_path).get(include=[])["ids"])
    assert first > 0 and first == second == third


def test_changed_doc_replaces_stale_chunks(tmp_path, monkeypatch):
    docs = tmp_path / "docs"; docs.mkdir()
    (docs / "a.md").write_text("original content about login", encoding="utf-8")
    monkeypatch.setattr(retriever, "DOCS_DIR", docs)
    emb = DeterministicFakeEmbedding(size=32)
    retriever.build_store(emb, tmp_path / "db")
    (docs / "a.md").write_text("updated content about sessions", encoding="utf-8")
    store = retriever.build_store(emb, tmp_path / "db")
    texts = store.get()["documents"]
    assert texts == ["updated content about sessions"]
