"""
RAG Retriever — finds the most relevant chunks for a question.

Adds a simple keyword re-scoring step on top of vector similarity
to boost chunks that contain exact query terms (important for
disaster types, place names, and technical terms).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _keyword_boost(text: str, query: str, boost: float = 0.05) -> float:
    """
    Add a small boost to the relevance score for each query keyword
    that appears in the chunk text.
    """
    words = re.findall(r"\w+", query.lower())
    text_lower = text.lower()
    matched = sum(1 for w in words if w in text_lower and len(w) > 3)
    return min(boost * matched, 0.20)  # cap at +0.20


class Retriever:
    """
    Retrieves and re-ranks chunks from the vector store.

    Parameters
    ----------
    vector_store : VectorStore
        Initialised and populated vector store.
    top_k : int
        Number of candidate chunks to retrieve before re-ranking.
    final_k : int
        Number of chunks to return after re-ranking.
    min_relevance : float
        Discard chunks below this cosine-based relevance score.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 10,
        final_k: int = 5,
        min_relevance: float = 0.20,
    ):
        self.store = vector_store
        self.top_k = top_k
        self.final_k = final_k
        self.min_relevance = min_relevance

    def retrieve(
        self,
        query: str,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve and re-rank the best chunks for a query.

        Returns
        -------
        List[Dict]
            Re-ranked list of chunk dicts with boosted relevance_score.
        """
        if self.store.count() == 0:
            logger.warning("Vector store is empty. Run build_index.py first.")
            return []

        # Step 1: vector similarity search
        candidates = self.store.query(
            query_text=query,
            n_results=self.top_k,
            where=where,
        )

        # Step 2: keyword re-scoring
        for c in candidates:
            boost = _keyword_boost(c["text"], query)
            c["relevance_score"] = min(1.0, c["relevance_score"] + boost)

        # Step 3: filter low-relevance chunks
        candidates = [c for c in candidates if c["relevance_score"] >= self.min_relevance]

        # Step 4: sort by final score
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Step 5: return top final_k
        result = candidates[: self.final_k]
        logger.info(
            "Retrieved %d chunks for query '%s...'",
            len(result),
            query[:60],
        )
        return result

    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a context block for the LLM prompt.
        Each chunk is labelled with its source and relevance.
        """
        if not chunks:
            return "No relevant documents found in the knowledge base."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            title = meta.get("title", "Unknown")
            url = meta.get("source_url", "")
            score = chunk["relevance_score"]
            text = chunk["text"]
            parts.append(
                f"[SOURCE {i}] {title}\n"
                f"URL: {url}\n"
                f"Relevance: {score:.2%}\n"
                f"---\n{text}"
            )

        return "\n\n".join(parts)
