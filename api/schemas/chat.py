from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str | None = None
    session_id: str | None = None
    message: str
    location: dict[str, Any] | None = None
    debug: bool = False


class Citation(BaseModel):
    evidence_id: str
    domain: str | None = None
    title: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    issue_id: str | None = None
    issue_status: str
    trace_id: str
    confidence: float | None = None
    citations: list[Citation] = Field(default_factory=list)
    need_human_escalation: bool = False
