from __future__ import annotations

import json

from sqlalchemy.orm import Session

from storage.models import RetrievalTrace, ToolCall


class TraceRepo:
    def __init__(self, db: Session):
        self.db = db

    def save_retrieval_trace(
        self,
        trace_id: str,
        session_id: str | None,
        message_id: str | None,
        query: str,
        pipeline: str = "baseline",
        query_analysis: dict | None = None,
        rewritten_queries: list[str] | None = None,
        recall_counts: dict | None = None,
        selected_evidence: list[dict] | None = None,
        confidence: float | None = None,
        flags: dict | None = None,
    ) -> RetrievalTrace:
        rt = RetrievalTrace(
            trace_id=trace_id,
            session_id=session_id,
            message_id=message_id,
            query=query,
            pipeline=pipeline,
            query_analysis_json=json.dumps(query_analysis, ensure_ascii=False) if query_analysis else None,
            rewritten_queries_json=json.dumps(rewritten_queries, ensure_ascii=False) if rewritten_queries else None,
            recall_counts_json=json.dumps(recall_counts, ensure_ascii=False) if recall_counts else None,
            selected_evidence_json=json.dumps(selected_evidence, ensure_ascii=False) if selected_evidence else None,
            confidence=confidence,
            flags_json=json.dumps(flags, ensure_ascii=False) if flags else None,
        )
        self.db.add(rt)
        self.db.commit()
        return rt

    def save_tool_call(
        self,
        trace_id: str,
        session_id: str | None,
        tool_name: str,
        input_summary: str | None = None,
        output_summary: str | None = None,
        status: str = "success",
        latency_ms: int | None = None,
        error_code: str | None = None,
    ) -> ToolCall:
        tc = ToolCall(
            trace_id=trace_id,
            session_id=session_id,
            tool_name=tool_name,
            input_summary=input_summary,
            output_summary=output_summary,
            status=status,
            latency_ms=latency_ms,
            error_code=error_code,
        )
        self.db.add(tc)
        self.db.commit()
        return tc

    def get_trace(self, trace_id: str) -> RetrievalTrace | None:
        return self.db.query(RetrievalTrace).filter(RetrievalTrace.trace_id == trace_id).first()

    def get_tool_calls(self, trace_id: str) -> list[ToolCall]:
        return self.db.query(ToolCall).filter(ToolCall.trace_id == trace_id).all()
