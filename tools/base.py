from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


class ToolSpec(BaseModel):
    name: str
    description: str
    timeout_seconds: int = 10
    retryable: bool = False
    sensitive: bool = False


class ToolContext(BaseModel):
    user_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None


class BaseTool(Protocol):
    spec: ToolSpec

    def run(self, input_data: BaseModel, context: ToolContext) -> BaseModel:
        ...


@dataclass
class ToolCallRecord:
    tool_name: str
    input_summary: str
    output_summary: str
    status: str
    latency_ms: int
    error_code: str | None = None
