from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tools.base import BaseTool, ToolCallRecord, ToolContext, ToolSpec

if TYPE_CHECKING:
    from pydantic import BaseModel


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._call_history: list[ToolCallRecord] = []

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def run(self, name: str, input_data: BaseModel, context: ToolContext) -> tuple[BaseModel, ToolCallRecord]:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not registered")

        start = time.perf_counter()
        try:
            result = tool.run(input_data, context)
            latency_ms = int((time.perf_counter() - start) * 1000)
            record = ToolCallRecord(
                tool_name=name,
                input_summary=str(input_data)[:200],
                output_summary=str(result)[:200],
                status="success",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            record = ToolCallRecord(
                tool_name=name,
                input_summary=str(input_data)[:200],
                output_summary=str(exc)[:200],
                status="error",
                latency_ms=latency_ms,
                error_code="TOOL_EXECUTION_FAILED",
            )
            raise

        self._call_history.append(record)
        return result, record

    def get_call_history(self) -> list[ToolCallRecord]:
        return list(self._call_history)


# Global registry
tool_registry = ToolRegistry()
