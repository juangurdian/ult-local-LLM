"""Deep research pipeline (stub)."""

from __future__ import annotations

from typing import Dict, List

from .vector_store import VectorStore


class DeepResearch:
    def __init__(self, store: VectorStore):
        self.store = store

    async def search_and_summarize(self, query: str) -> Dict[str, List[str]]:
        results = await self.store.search(query, top_k=5)
        summaries = [r["text"] for r in results]
        return {"query": query, "summaries": summaries}

