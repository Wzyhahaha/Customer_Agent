from __future__ import annotations

from tools.base import ToolContext, ToolSpec


spec = ToolSpec(
    name="rag_search",
    description="Search the RAG knowledge base for after-sales support information",
    timeout_seconds=15,
    retryable=True,
    sensitive=False,
)


class RagTool:
    spec = spec

    def run(self, input_data, context: ToolContext):
        # Wraps the RAG retrieval pipeline
        from pydantic import BaseModel

        class RagOutput(BaseModel):
            answer: str
            evidence: list[dict] = []
            confidence: float = 0.0

        return RagOutput(answer="[RAG response placeholder]", evidence=[], confidence=0.0)
