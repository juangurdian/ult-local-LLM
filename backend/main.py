from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from .config import get_settings
from .deps import get_model_router
from .api import chat, agents, rag, images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Local AI Beast",
    description="Unified backend with intelligent routing and agents",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(agents.router, prefix=settings.api_prefix)
app.include_router(rag.router, prefix=settings.api_prefix)
app.include_router(images.router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": "1.1.0",
        "endpoints": [
            f"{settings.api_prefix}/chat",
            f"{settings.api_prefix}/agents/research",
            f"{settings.api_prefix}/agents/code",
            f"{settings.api_prefix}/rag/search",
            f"{settings.api_prefix}/images/generate",
        ],
    }


@app.get("/health")
async def health():
    router = get_model_router()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": list(router.model_configs.keys()),
    }


@app.get(f"{settings.api_prefix}/status")
async def status():
    router = get_model_router()
    return {
        "service": settings.app_name,
        "online": True,
        "models": list(router.model_configs.keys()),
        "timestamp": datetime.now().isoformat(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc), "timestamp": datetime.now().isoformat()},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True, log_level="info")
