"""Reasoning layer (SR-RAG).  Lazy import avoids runpy warnings when using
``python -m reasoning.rag_orchestrator``."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["SRRAGOrchestrator"]


def __getattr__(name: str):
    if name == "SRRAGOrchestrator":
        from reasoning.rag_orchestrator import SRRAGOrchestrator

        return SRRAGOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from reasoning.rag_orchestrator import SRRAGOrchestrator
