"""Research agent (stub) that can expand later to web + RAG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..router.router import ModelRouter
from .base import AgentBase, AgentContext, AgentResult


class ResearchAgent(AgentBase):
    name = "research"

    def __init__(self, router: ModelRouter):
        self.router = router

    async def run(self, ctx: AgentContext) -> AgentResult:
        # For now, just route the query and return a stubbed response.
        routing = await self.router.route_query(ctx.query)
        return AgentResult(
            success=True,
            result="(research stub) " + ctx.query,
            model_used=routing["model"],
            routing_info=routing,
        )

