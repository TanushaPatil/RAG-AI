"""
RAG Answer Generator.

Takes retrieved chunks and a user question, builds a grounded prompt,
and calls an LLM to produce an evidence-based answer with source citations.

Supports:
  - Google Gemini   (GOOGLE_API_KEY)
  - OpenAI GPT      (OPENAI_API_KEY)
  - Offline fallback (no API key — returns formatted context only)

The system works without any API key in offline/demo mode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a disaster intelligence analyst specialising in satellite imagery, \
SAR (Synthetic Aperture Radar), flood mapping, and humanitarian response. \
Answer the user's question using ONLY the provided source documents. \
If the documents do not contain enough information to answer confidently, \
say so clearly — do NOT invent facts or extrapolate beyond provided data. \
When source documents present conflicting or varying reports (e.g. differing fatality counts, \
discrepant affected population figures, conflicting dates, or differing flood extents), \
you must explicitly highlight the disagreement, report the differing perspectives with their respective \
citations, and NOT arbitrarily pick or guess which one is correct. \
Always cite the source number (e.g. [SOURCE 1]) when you use information from it. \
Be concise, precise, and technically accurate."""

USER_PROMPT_TEMPLATE = """\
## Retrieved Source Documents

{context}

## Question

{question}

## Instructions

Answer the question using the source documents above. \
Reference specific sources with [SOURCE N]. \
If the answer spans multiple sources, cite all relevant ones. \
If the sources provide conflicting facts, figures, or dates, explicitly state the conflict \
and attribute each figure to its source rather than choosing one. \
If the evidence is insufficient to answer completely, acknowledge the limitations clearly. \
End your answer with a "## Sources" section listing each cited source's title and URL.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[dict]          # [{title, url, relevance_score, chunk_id}]
    model_used: str
    chunk_count: int
    offline_mode: bool = False

    def display(self) -> str:
        """Formatted string for terminal display."""
        divider = "═" * 70
        source_lines = "\n".join(
            f"  [{i+1}] {s['title']}\n      {s['url']}"
            for i, s in enumerate(self.sources)
        )
        return (
            f"\n{divider}\n"
            f"❓ QUESTION\n{self.question}\n\n"
            f"{'📴 OFFLINE MODE — No LLM called' if self.offline_mode else f'🤖 MODEL: {self.model_used}'}\n"
            f"{divider}\n"
            f"💡 ANSWER\n\n{self.answer}\n\n"
            f"📚 SOURCES ({self.chunk_count} chunks retrieved)\n{source_lines}\n"
            f"{divider}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class Generator:
    """
    Wraps LLM calls for the RAG answer generation step.

    Auto-selects backend based on available API keys:
      1. Google Gemini  (GOOGLE_API_KEY)
      2. OpenAI GPT-4o  (OPENAI_API_KEY)
      3. Offline mode   (no key — returns context without LLM)
    """

    def __init__(self, model: Optional[str] = None):
        self._model_override = model
        self._backend, self._model_name = self._detect_backend()

    # ── Backend detection ─────────────────────────────────────────────────────

    def _detect_backend(self):
        if os.environ.get("GOOGLE_API_KEY"):
            model = self._model_override or "gemini-1.5-flash"
            logger.info("Generator: using Google Gemini (%s)", model)
            return "gemini", model

        if os.environ.get("OPENAI_API_KEY"):
            model = self._model_override or "gpt-4o-mini"
            logger.info("Generator: using OpenAI (%s)", model)
            return "openai", model

        logger.info(
            "Generator: no API key found — running in OFFLINE mode. "
            "Set GOOGLE_API_KEY or OPENAI_API_KEY in .env to enable LLM answers."
        )
        return "offline", "offline"

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, question: str, chunks: List[dict], context: str) -> RAGResponse:
        """
        Generate a grounded answer.

        Parameters
        ----------
        question : str
            The user's question.
        chunks : List[dict]
            Retrieved chunks (from Retriever.retrieve).
        context : str
            Pre-formatted context string (from Retriever.format_context).

        Returns
        -------
        RAGResponse
            Structured response with answer, sources, and metadata.
        """
        sources = self._extract_sources(chunks)

        if self._backend == "offline":
            answer = self._offline_answer(context)
            return RAGResponse(
                question=question,
                answer=answer,
                sources=sources,
                model_used="offline",
                chunk_count=len(chunks),
                offline_mode=True,
            )

        full_prompt = USER_PROMPT_TEMPLATE.format(
            context=context, question=question
        )

        if self._backend == "gemini":
            answer = self._call_gemini(full_prompt)
        elif self._backend == "openai":
            answer = self._call_openai(full_prompt)
        else:
            answer = context

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            model_used=self._model_name,
            chunk_count=len(chunks),
        )

    # ── LLM backends ──────────────────────────────────────────────────────────

    def _call_gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=SYSTEM_PROMPT,
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return f"[LLM ERROR] {e}\n\nFalling back to raw context:\n\n{prompt}"

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            return f"[LLM ERROR] {e}\n\nFalling back to raw context:\n\n{prompt}"

    def _offline_answer(self, context: str) -> str:
        return (
            "⚠️  OFFLINE MODE — No LLM API key configured.\n\n"
            "The following source documents were retrieved from the knowledge base "
            "and are relevant to your question. Set GOOGLE_API_KEY or OPENAI_API_KEY "
            "in your .env file to get an AI-generated answer.\n\n"
            "── RETRIEVED CONTEXT ──\n\n"
            + context
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_sources(chunks: List[dict]) -> List[dict]:
        """Deduplicate sources from retrieved chunks."""
        seen_urls = set()
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            url = meta.get("source_url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append(
                    {
                        "title": meta.get("title", "Unknown"),
                        "url": url,
                        "relevance_score": chunk.get("relevance_score", 0),
                        "chunk_id": chunk.get("chunk_id", ""),
                    }
                )
        return sources
