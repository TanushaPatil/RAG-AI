"""
Tests for the RAG pipeline components (chunker, embedder, vector store, retriever).
Run from project root: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processor.validator import DisasterDocument
from rag.chunker import Chunk, chunk_document, chunk_documents, chunk_text


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_doc(content: str = None) -> DisasterDocument:
    if content is None:
        content = (
            "Sentinel-1 is a C-band SAR satellite used for flood detection. "
            "It provides all-weather, day-and-night imaging. "
            "Flood pixels are identified by low backscatter values. "
            "The Copernicus Emergency Management Service uses Sentinel-1 data. "
            "Change detection compares pre- and post-event images. "
        ) * 20  # ~1000 words
    return DisasterDocument(
        title="Test SAR Document",
        source_url="https://example.com/sar-test",
        content=content,
        hazard_types=["flood"],
        region="Europe",
        tags=["SAR", "Sentinel-1"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chunker tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_text_basic(self):
        text = " ".join(f"word{i}" for i in range(1000))
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            words = chunk.split()
            assert len(words) <= 120  # some tolerance for sentence boundaries

    def test_chunk_text_empty(self):
        assert chunk_text("") == []

    def test_chunk_text_short(self):
        text = "Short text that fits in one chunk."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_chunk_overlap(self):
        """Chunks should share some words at boundaries."""
        text = " ".join(f"word{i}." for i in range(200))
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        if len(chunks) >= 2:
            # Last words of chunk[0] should appear at start of chunk[1]
            end_words = set(chunks[0].split()[-10:])
            start_words = set(chunks[1].split()[:10])
            assert len(end_words & start_words) > 0

    def test_chunk_document_returns_chunks(self):
        doc = make_doc()
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_are_unique(self):
        doc = make_doc()
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_preserved(self):
        doc = make_doc()
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        for chunk in chunks:
            assert chunk.source_url == doc.source_url
            assert chunk.title == doc.title
            assert chunk.document_id == doc.document_id
            assert "flood" in chunk.hazard_types
            assert chunk.region == "Europe"

    def test_total_chunks_consistent(self):
        doc = make_doc()
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total

    def test_chunk_to_metadata_types(self):
        doc = make_doc()
        chunk = chunk_document(doc, chunk_size=100, overlap=10)[0]
        meta = chunk.to_metadata()
        # ChromaDB requires str/int/float/bool — no lists
        for key, val in meta.items():
            assert isinstance(val, (str, int, float, bool)), \
                f"Metadata key '{key}' has non-primitive type {type(val)}"

    def test_chunk_documents_multiple(self):
        docs = [make_doc() for _ in range(3)]
        all_chunks = chunk_documents(docs, chunk_size=100, overlap=10)
        assert len(all_chunks) > 3  # more chunks than docs


# ─────────────────────────────────────────────────────────────────────────────
# Embedder tests (uses actual model — skipped if not installed)
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedder:
    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

    def test_embed_returns_vectors(self):
        from rag.embedder import Embedder
        embedder = Embedder()
        vecs = embedder.embed(["Flood in Bangladesh", "Earthquake in Turkey"])
        assert len(vecs) == 2
        assert len(vecs[0]) > 0  # non-empty vectors

    def test_embed_one(self):
        from rag.embedder import Embedder
        embedder = Embedder()
        vec = embedder.embed_one("Sentinel-1 SAR flood detection")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_embed_normalised(self):
        """Vectors should be approximately unit-length (normalised=True)."""
        import math
        from rag.embedder import Embedder
        embedder = Embedder()
        vec = embedder.embed_one("Test sentence")
        magnitude = math.sqrt(sum(v ** 2 for v in vec))
        assert 0.99 < magnitude < 1.01

    def test_empty_input(self):
        from rag.embedder import Embedder
        embedder = Embedder()
        result = embedder.embed([])
        assert result == []

    def test_dimension(self):
        from rag.embedder import Embedder
        embedder = Embedder()
        assert embedder.dimension > 0


# ─────────────────────────────────────────────────────────────────────────────
# Vector store + retriever integration test (in-memory chroma)
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStoreRetriever:
    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("chromadb", reason="chromadb not installed")
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

    @pytest.fixture
    def in_memory_store(self, tmp_path):
        from rag.embedder import Embedder
        from rag.vector_store import VectorStore
        embedder = Embedder()
        store = VectorStore(
            embedder=embedder,
            persist_dir=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )
        return store

    def test_add_and_count(self, in_memory_store):
        doc = make_doc(content="Sentinel-1 SAR flood detection method.")
        chunks = chunk_document(doc, chunk_size=50, overlap=5)
        added = in_memory_store.add_chunks(chunks)
        assert added > 0
        assert in_memory_store.count() == added

    def test_query_returns_results(self, in_memory_store):
        doc = make_doc(content="Sentinel-1 SAR detects floods using backscatter analysis.")
        chunks = chunk_document(doc, chunk_size=50, overlap=5)
        in_memory_store.add_chunks(chunks)
        results = in_memory_store.query("flood detection SAR", n_results=3)
        assert len(results) > 0
        assert "text" in results[0]
        assert "relevance_score" in results[0]

    def test_retriever_reranks(self, in_memory_store):
        from rag.retriever import Retriever
        doc = make_doc(content="Sentinel-1 detects floods. Earthquake damage assessment via SAR.")
        chunks = chunk_document(doc, chunk_size=50, overlap=5)
        in_memory_store.add_chunks(chunks)
        retriever = Retriever(vector_store=in_memory_store, top_k=5, final_k=3)
        results = retriever.retrieve("SAR flood detection")
        assert len(results) <= 3
        # Results should be sorted by relevance descending
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retriever_empty_store(self, in_memory_store):
        from rag.retriever import Retriever
        retriever = Retriever(vector_store=in_memory_store)
        results = retriever.retrieve("any question")
        assert results == []

    def test_format_context(self, in_memory_store):
        from rag.retriever import Retriever
        doc = make_doc(content="Bangladesh flood 2024 affected millions.")
        chunks = chunk_document(doc, chunk_size=50, overlap=5)
        in_memory_store.add_chunks(chunks)
        retriever = Retriever(vector_store=in_memory_store)
        results = retriever.retrieve("Bangladesh flood")
        context = retriever.format_context(results)
        assert "[SOURCE 1]" in context
        assert "URL:" in context
