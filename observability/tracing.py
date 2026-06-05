from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def generate_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:12]}"


@dataclass
class RetrievalTraceData:
    trace_id: str = field(default_factory=generate_trace_id)
    session_id: str | None = None
    message_id: str | None = None
    query: str = ""
    pipeline: str = "baseline"
    query_analysis: dict[str, Any] | None = None
    rewritten_queries: list[str] = field(default_factory=list)
    recall_counts: dict[str, int] = field(default_factory=dict)
    selected_evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    flags: dict[str, bool] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "query": self.query,
            "pipeline": self.pipeline,
            "query_analysis": self.query_analysis,
            "rewritten_queries": self.rewritten_queries,
            "recall_counts": self.recall_counts,
            "selected_evidence": self.selected_evidence,
            "confidence": self.confidence,
            "flags": self.flags,
            "tool_calls": self.tool_calls,
            "created_at": self.created_at,
        }
