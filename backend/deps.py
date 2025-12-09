"""Shared dependencies for the FastAPI app."""

from functools import lru_cache
from .router.router import ModelRouter


@lru_cache()
def get_model_router() -> ModelRouter:
    """Singleton ModelRouter instance."""
    return ModelRouter()

