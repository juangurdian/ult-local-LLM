"""Agent package for Local AI Beast."""

from .manager import AgentManager
from .research_agent import ResearchAgent
from .coding_agent import CodingAgent
from .base import AgentResult, AgentContext, AgentBase

__all__ = [
    "AgentManager",
    "ResearchAgent",
    "CodingAgent",
    "AgentResult",
    "AgentContext",
    "AgentBase",
]

