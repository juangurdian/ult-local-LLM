from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from ..rag import VectorStore

router = APIRouter(prefix="/rag", tags=["rag"])
vector_store = VectorStore()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    results: List[str]


@router.post("/search", response_model=SearchResult)
async def rag_search(body: SearchRequest):
    """Search local vector store (stubbed)."""
    hits = await vector_store.search(body.query, top_k=body.top_k)
    return SearchResult(results=[hit["text"] for hit in hits])

