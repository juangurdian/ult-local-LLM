"""Document ingestion pipeline (stub)."""

from __future__ import annotations

from typing import List, Dict

from .vector_store import VectorStore


class DocumentIngestion:
    def __init__(self, store: VectorStore):
        self.store = store

    async def ingest(self, documents: List[Dict]) -> Dict[str, int]:
        count = await self.store.add_documents(documents)
        return {"ingested": count}

