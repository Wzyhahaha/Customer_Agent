from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class IssueEvent(BaseModel):
    event_id: str
    issue_id: str
    from_status: str
    to_status: str
    trigger: str
    created_at: datetime


class IssueResponse(BaseModel):
    issue_id: str
    customer_id: str | None = None
    session_id: str | None = None
    title: str
    status: str
    category: str | None = None
    summary: str | None = None
    created_at: datetime
    updated_at: datetime
    events: list[IssueEvent] = []
