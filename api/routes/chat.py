from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter

from api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or f"s_{uuid.uuid4().hex[:12]}"
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"

    return ChatResponse(
        answer=f"[Demo] 收到您的消息：{request.message[:50]}...",
        session_id=session_id,
        issue_id=None,
        issue_status="NEW",
        trace_id=trace_id,
        confidence=None,
        citations=[],
        need_human_escalation=False,
    )
