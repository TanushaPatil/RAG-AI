#!/usr/bin/env python3
"""
CLI Script: Query the RAG system with a question.

Usage:
    python scripts/query_rag.py "How does Sentinel-1 detect floods?"
    python scripts/query_rag.py "What happened during the Turkey earthquake?" --top-k 8
    python scripts/query_rag.py "Cyclone impact on Bangladesh" --region Bangladesh
    python scripts/query_rag.py --interactive    # continuous Q&A session

This script must be run from the project root directory.
Requires the vector index to be built first: python scripts/build_index.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,  # Keep output clean for users
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("query_rag")


def build_rag(persist_dir: str, top_k: int, final_k: int):
    from rag.embedder import Embedder
    from rag.generator import Generator
    from rag.retriever import Retriever
    from rag.vector_store import VectorStore

    embedder = Embedder()
    store = VectorStore(embedder=embedder, persist_dir=persist_dir)

    if store.count() == 0:
        print(
            "\n❌ Vector index is empty!\n"
            "   Build the index first:\n"
            "   python scripts/build_index.py --create-fixtures\n"
        )
        sys.exit(1)

    retriever = Retriever(vector_store=store, top_k=top_k, final_k=final_k)
    generator = Generator()
    return retriever, generator


def ask(question: str, retriever, generator, region_filter: str | None = None):
    where = None
    if region_filter:
        where = {"region": region_filter}

    chunks = retriever.retrieve(question, where=where)
    if not chunks:
        print("\n⚠️  No relevant documents found for your question.")
        print("Try running the scraper or building the index with fixtures first.")
        return

    context = retriever.format_context(chunks)
    response = generator.generate(question=question, chunks=chunks, context=context)
    print(response.display())


def main():
    parser = argparse.ArgumentParser(
        description="Query the Disaster Intelligence RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/query_rag.py "How is Sentinel-1 used for flood detection?"
  python scripts/query_rag.py "What is the impact of cyclone Mocha?" --top-k 8
  python scripts/query_rag.py --interactive
        """,
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to answer (omit for --interactive mode)",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/chroma_db",
        help="ChromaDB directory (default: data/chroma_db)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidate chunks to retrieve (default: 10)",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
        help="Number of chunks after re-ranking (default: 5)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Optional region/country filter (e.g. 'Bangladesh')",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive Q&A session",
    )
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.print_help()
        sys.exit(0)

    print("⏳ Loading RAG system (first run downloads embedding model ~90 MB) ...")
    retriever, generator = build_rag(args.persist_dir, args.top_k, args.final_k)
    print(f"✅ Index loaded ({retriever.store.count()} chunks)\n")

    if args.interactive:
        print("🛰️  Disaster Intelligence RAG — Interactive Mode")
        print("   Type your question and press Enter. Type 'exit' to quit.\n")
        while True:
            try:
                question = input("❓ Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue
            ask(question, retriever, generator, region_filter=args.region)
    else:
        ask(args.question, retriever, generator, region_filter=args.region)


if __name__ == "__main__":
    main()
