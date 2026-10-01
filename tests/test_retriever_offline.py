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


def test_top_k_defaults_to_5_and_is_configurable(monkeypatch):
    """Live finding: with top-3, 'export' matched the release notes so strongly that the
    other docs were pushed out (BUG-112 and the vendor doc were never seen)."""
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    assert retriever.top_k() == 5
    monkeypatch.setenv("RAG_TOP_K", "8")
    assert retriever.top_k() == 8
    monkeypatch.setenv("RAG_TOP_K", "oops")
    assert retriever.top_k() == 5


def test_more_results_reach_the_second_strongest_doc(tmp_path, monkeypatch):
    docs = tmp_path / "docs"; docs.mkdir()
    for i in range(4):
        (docs / f"exports_{i}.md").write_text(f"Export feature notes part {i}. Exports exports.", encoding="utf-8")
    (docs / "vendor.md").write_text("Export vendor support line and sales contact.", encoding="utf-8")
    monkeypatch.setattr(retriever, "DOCS_DIR", docs)
    from langchain_core.embeddings import DeterministicFakeEmbedding
    store = retriever.build_store(DeterministicFakeEmbedding(size=32), tmp_path / "db")
    assert len(store.similarity_search("export vendor", k=5)) == 5