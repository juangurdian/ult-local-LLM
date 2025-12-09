from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from ..deps import get_model_router
from ..router.router import ModelRouter
from ..agents import AgentManager, AgentContext

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentRequest(BaseModel):
    query: str
    context: Optional[str] = None
    web: Optional[bool] = False
    code: Optional[str] = None


class AgentResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    model_used: Optional[str] = None
    routing_info: Optional[Dict[str, Any]] = None


def get_agent_manager(model_router: ModelRouter = Depends(get_model_router)) -> AgentManager:
    return AgentManager(model_router)


@router.post("/research", response_model=AgentResponse)
async def research_agent(
    body: AgentRequest,
    manager: AgentManager = Depends(get_agent_manager),
):
    """Research agent endpoint."""
    ctx = AgentContext(query=body.query, context=body.context, web=body.web)
    result = await manager.run("research", ctx)
    return AgentResponse(
        success=result.success,
        result=result.result,
        model_used=result.model_used,
        routing_info=result.routing_info,
    )


@router.post("/code", response_model=AgentResponse)
async def coding_agent(
    body: AgentRequest,
    manager: AgentManager = Depends(get_agent_manager),
):
    """Coding agent endpoint."""
    ctx = AgentContext(query=body.query, context=body.context, code=body.code)
    result = await manager.run("code", ctx)
    return AgentResponse(
        success=result.success,
        result=result.result,
        model_used=result.model_used,
        routing_info=result.routing_info,
    )

