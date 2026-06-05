from __future__ import annotations

from tools.base import ToolContext, ToolSpec


spec = ToolSpec(
    name="escalation",
    description="Escalate the current issue to human after-sales support",
    timeout_seconds=10,
    retryable=False,
    sensitive=True,
)


class EscalationTool:
    spec = spec

    def run(self, input_data, context: ToolContext):
        from pydantic import BaseModel

        class EscalationOutput(BaseModel):
            escalated: bool = True
            reason: str = ""
            ticket_id: str = ""

        reason = getattr(input_data, "reason", "Customer requested human support")
        return EscalationOutput(escalated=True, reason=str(reason), ticket_id="")
