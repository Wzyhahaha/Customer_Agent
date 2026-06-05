from __future__ import annotations

from tools.base import ToolContext, ToolSpec


spec = ToolSpec(
    name="report",
    description="Read external CSV report data for customer monthly records",
    timeout_seconds=10,
    retryable=False,
    sensitive=True,
)


class ReportTool:
    spec = spec

    def run(self, input_data, context: ToolContext):
        from pydantic import BaseModel

        class ReportOutput(BaseModel):
            data: list[dict] = []
            summary: str = ""

        return ReportOutput(data=[], summary="No report data available")
