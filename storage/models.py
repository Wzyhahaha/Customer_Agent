from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow():
    return datetime.now(timezone.utc)


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class Customer(Base):
    __tablename__ = "customers"

    id = Column(String, primary_key=True, default=lambda: _new_id("c_"))
    name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: _new_id("s_"))
    customer_id = Column(String, ForeignKey("customers.id"), nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    messages = relationship("Message", back_populates="session", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: _new_id("m_"))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    trace_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    session = relationship("Session", back_populates="messages")


class Issue(Base):
    __tablename__ = "issues"

    id = Column(String, primary_key=True, default=lambda: _new_id("i_"))
    customer_id = Column(String, ForeignKey("customers.id"), nullable=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=True)
    title = Column(String, nullable=True)
    status = Column(String, nullable=False, default="NEW")
    category = Column(String, nullable=True)
    priority = Column(String, nullable=True, default="normal")
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    resolved_at = Column(DateTime, nullable=True)
    escalated_at = Column(DateTime, nullable=True)

    events = relationship("IssueEvent", back_populates="issue", order_by="IssueEvent.created_at")


class IssueEvent(Base):
    __tablename__ = "issue_events"

    id = Column(String, primary_key=True, default=lambda: _new_id("ie_"))
    issue_id = Column(String, ForeignKey("issues.id"), nullable=False)
    from_status = Column(String, nullable=False)
    to_status = Column(String, nullable=False)
    trigger = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    issue = relationship("Issue", back_populates="events")


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"

    id = Column(String, primary_key=True, default=lambda: _new_id("rt_"))
    trace_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=True)
    message_id = Column(String, nullable=True)
    query = Column(Text, nullable=False)
    pipeline = Column(String, nullable=False, default="baseline")
    query_analysis_json = Column(Text, nullable=True)
    rewritten_queries_json = Column(Text, nullable=True)
    recall_counts_json = Column(Text, nullable=True)
    selected_evidence_json = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    flags_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id = Column(String, primary_key=True, default=lambda: _new_id("tc_"))
    trace_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=True)
    tool_name = Column(String, nullable=False)
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="success")
    latency_ms = Column(Integer, nullable=True)
    error_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
