"""
ingest_nasm.py — NASM Knowledge Base Ingestion for Neuro-Fit
=============================================================

Loads the NASM Essentials of Personal Fitness Training PDF, chunks it
with ``RecursiveCharacterTextSplitter``, generates dense embeddings
via ``OllamaEmbeddings`` (``nomic-embed-text``), and persists the
resulting FAISS index to ``vector_store/`` so the LangGraph Refuter
Agent can load it at runtime.

Prerequisites:
    1. Ollama running locally with ``nomic-embed-text`` pulled:
           ollama pull nomic-embed-text
    2. The NASM PDF placed at:
           datasets/NASM_Essentials_of_Personal_Fitness_Training.pdf

Run:
    python scripts/ingest_nasm.py

Output:
    vector_store/index.faiss
    vector_store/index.pkl
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "datasets" / "NASM_Essentials_of_Personal_Fitness_Training.pdf"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text"


def main() -> None:
    # ── Validate PDF existence ────────────────────────────────────────
    if not PDF_PATH.is_file():
        print(f"[ERROR] NASM PDF not found: {PDF_PATH}", file=sys.stderr)
        print(
            "  Place the file at:\n"
            f"    {PDF_PATH}\n"
            "  and re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading PDF: {PDF_PATH}")

    # ── Load PDF with LangChain PyPDFLoader ───────────────────────────
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(PDF_PATH))
    raw_docs = loader.load()
    print(f"  Loaded {len(raw_docs)} pages from PDF.")

    # ── Chunk with RecursiveCharacterTextSplitter ─────────────────────
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  Split into {len(chunks):,} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # ── Generate embeddings with Ollama nomic-embed-text ──────────────
    from langchain_ollama import OllamaEmbeddings

    print(f"Generating embeddings with Ollama '{EMBEDDING_MODEL}'...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    t0 = time.perf_counter()

    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.perf_counter() - t0
    print(f"  Embeddings generated in {elapsed:.1f}s.")

    # ── Persist FAISS index to disk ───────────────────────────────────
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_DIR))
    print(f"\nFAISS index saved to: {VECTOR_STORE_DIR}/")
    print(f"  Total chunks indexed: {len(chunks):,}")
    print(
        "\nThe LangGraph Refuter Agent can now load this index:\n"
        "    from langchain_community.vectorstores import FAISS\n"
        "    from langchain_ollama import OllamaEmbeddings\n"
        "    embeddings = OllamaEmbeddings(model='nomic-embed-text')\n"
        f"    vs = FAISS.load_local('{VECTOR_STORE_DIR}', embeddings, "
        "allow_dangerous_deserialization=True)\n"
        "    retriever = vs.as_retriever(search_kwargs={'k': 3})"
    )


if __name__ == "__main__":
    main()
