"""Coding agent (stub) that can expand into code execution sandbox."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..router.router import ModelRouter
from .base import AgentBase, AgentContext, AgentResult


class CodingAgent(AgentBase):
    name = "coding"

    def __init__(self, router: ModelRouter):
        self.router = router

    async def run(self, ctx: AgentContext) -> AgentResult:
        # Prefer the code snippet if provided
        target_text = ctx.code or ctx.query
        routing = await self.router.route_query(target_text)
        return AgentResult(
            success=True,
            result="(code stub) " + target_text,
            model_used=routing["model"],
            routing_info=routing,
        )

