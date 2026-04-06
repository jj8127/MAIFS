"""
Debate System Module
Multi-Agent 토론 시스템
"""
from .debate_protocol import (
    DebateProtocol,
    DebateTurn,
    DebateResult,
    DebateTerminationReason
)

__all__ = [
    "DebateProtocol",
    "DebateTurn",
    "DebateResult",
    "DebateTerminationReason"
]
