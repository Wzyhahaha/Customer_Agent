from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryAnalysis:
    original_query: str
    domains: list[str] = field(default_factory=list)
    intent: str = ""
    keywords: list[str] = field(default_factory=list)
    rewritten_queries: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    risk_flags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.rewritten_queries:
            self.rewritten_queries = [self.original_query]


@dataclass
class RetrievedEvidence:
    evidence_id: str
    content: str
    domain: str
    source_store: str
    source_file: str = ""
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    matched_by: list[str] = field(default_factory=list)

    def add_score(self, name: str, value: float):
        self.scores[name] = float(value)

    def add_match(self, source: str):
        if source and source not in self.matched_by:
            self.matched_by.append(source)


@dataclass
class RetrievalTrace:
    query_analysis: QueryAnalysis
    recall_counts: dict[str, int] = field(default_factory=dict)
    fusion_strategy: str = ""
    rerank_strategy: str = ""
    selected_evidence_ids: list[str] = field(default_factory=list)
    dropped_evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    flags: dict[str, bool] = field(default_factory=dict)
