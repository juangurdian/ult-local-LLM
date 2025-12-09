"""Chroma-based vector store (stub for now)."""

from __future__ import annotations

from typing import List, Dict


class VectorStore:
    """Placeholder vector store wrapper."""

    def __init__(self):
        # TODO: initialize ChromaDB client when ready
        self.ready = False

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        # Stubbed search
        return [{"text": f"(rag stub result) {query}", "score": 0.0}]

    async def add_documents(self, docs: List[Dict]) -> int:
        # Stubbed ingestion
        return len(docs)

