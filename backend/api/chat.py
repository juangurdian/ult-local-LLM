from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import json
import logging

from ..deps import get_model_router
from ..router.router import ModelRouter

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "auto"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    context: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    model_used: str
    execution_time_ms: int
    routing_info: Dict[str, Any]


def _format_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_text(text: str, chunk_size: int = 32):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, model_router: ModelRouter = Depends(get_model_router)):
    """Chat endpoint with intelligent routing."""
    start = time.perf_counter()

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Use last user message for routing
    last_user_msg = next((m for m in reversed(request.messages) if m.role == "user"), None)
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    routing_result = await model_router.route_query(
        query=last_user_msg.content,
        images=last_user_msg.images,
        context=request.context,
        force_model=request.model,
        conversation_history=[msg.dict() for msg in request.messages[:-1]],
    )
    logger.info("Routing decision (non-stream): %s", routing_result)

    execution_result = await model_router.execute_query(
        routing_result=routing_result,
        messages=[msg.dict() for msg in request.messages],
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    logger.info("Execution result (non-stream) success=%s model=%s", execution_result.get("success"), execution_result.get("model_used"))

    packing = execution_result.get("packing") or {}

    # Attach packing info to routing
    routing_result_with_packing = {**routing_result, "packing": packing}

    return ChatResponse(
        success=execution_result["success"],
        response=execution_result.get("response"),
        error=execution_result.get("error"),
        model_used=execution_result["model_used"],
        execution_time_ms=int((time.perf_counter() - start) * 1000),
        routing_info=routing_result_with_packing,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest, model_router: ModelRouter = Depends(get_model_router)):
    """Stream responses via server-sent events."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_user_msg = next((m for m in reversed(request.messages) if m.role == "user"), None)
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    routing_result = await model_router.route_query(
        query=last_user_msg.content,
        images=last_user_msg.images,
        context=request.context,
        force_model=request.model,
        conversation_history=[msg.dict() for msg in request.messages[:-1]],
    )
    logger.info("Routing decision (stream): %s", routing_result)

    execution_result = await model_router.execute_query(
        routing_result=routing_result,
        messages=[msg.dict() for msg in request.messages],
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    response_text = execution_result.get("response", "") or ""
    success = execution_result.get("success", False)
    packing = execution_result.get("packing") or {}
    model_meta = model_router.model_configs.get(routing_result.get("model"), {})
    logger.info(
        "Execution result (stream) success=%s model=%s tokens_kept=%s tokens_dropped=%s",
        success,
        execution_result.get("model_used"),
        packing.get("tokens_kept"),
        packing.get("tokens_dropped"),
    )

    async def event_generator():
        # Emit routing information first
        yield _format_event(
            {
                "type": "routing",
                "payload": {
                    **routing_result,
                    "packing": packing,
                    "model_meta": {
                        "context_window": model_meta.get("context_window"),
                        "estimated_vram_gb": model_meta.get("estimated_vram_gb"),
                        "estimated_tokens_per_sec": model_meta.get("estimated_tokens_per_sec"),
                    },
                },
            }
        )

        # Stream assistant content in chunks
        for chunk in _chunk_text(response_text):
            yield _format_event({"type": "delta", "text": chunk})

        # Emit completion metadata
        yield _format_event(
            {
                "type": "done",
                "payload": {
                    "success": success,
                    "model_used": execution_result.get("model_used"),
                    "error": execution_result.get("error"),
                },
            }
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")