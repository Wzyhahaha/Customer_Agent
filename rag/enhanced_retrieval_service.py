from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.evidence import evidence_from_document
from rag.fusion import FusionRanker
from rag.pipeline_types import RetrievedEvidence, RetrievalTrace
from rag.query_analysis import QueryAnalyzer, QueryRewriter
from rag.reranker import EvidenceReranker
from rag.structured_retriever import StructuredFieldRetriever


_UNSET = object()


@dataclass
class EnhancedRetrievalResult:
    query: str
    evidence: list[RetrievedEvidence] = field(default_factory=list)
    trace: RetrievalTrace | None = None


class EnhancedRetrievalService:
    def __init__(
        self,
        question_retriever: Any = None,
        policy_retriever: Any = None,
        troubleshooting_retriever: Any = None,
        maintenance_retriever: Any = None,
        structured_retriever: StructuredFieldRetriever | None | object = _UNSET,
        analyzer: QueryAnalyzer | None = None,
        rewriter: QueryRewriter | None = None,
        fusion_ranker: FusionRanker | None = None,
        reranker: EvidenceReranker | None = None,
    ):
        self.question_retriever = question_retriever
        self.policy_retriever = policy_retriever
        self.troubleshooting_retriever = troubleshooting_retriever
        self.maintenance_retriever = maintenance_retriever
        self.structured_retriever = structured_retriever
        self.analyzer = analyzer or QueryAnalyzer()
        self.rewriter = rewriter or QueryRewriter()
        self.fusion_ranker = fusion_ranker or FusionRanker()
        self.reranker = reranker or EvidenceReranker()

    def retrieve(self, query: str) -> EnhancedRetrievalResult:
        analysis = self.rewriter.rewrite(self.analyzer.analyze(query))
        trace = RetrievalTrace(query_analysis=analysis)
        trace.flags["mixed_query"] = len(analysis.domains) > 1

        groups: dict[str, list[RetrievedEvidence]] = {
            "question_recall": self._retrieve_vector(
                self._question_retriever(),
                analysis.original_query,
                "question_recall",
                "question",
            )
        }

        if "policy" in analysis.domains:
            groups["policy_rules"] = self._retrieve_vector(
                self._policy_retriever(),
                analysis.original_query,
                "policy_rules",
                "policy",
            )
        if "troubleshooting" in analysis.domains:
            groups["troubleshooting_cases"] = self._retrieve_vector(
                self._troubleshooting_retriever(),
                analysis.original_query,
                "troubleshooting_cases",
                "troubleshooting",
            )
        if "maintenance" in analysis.domains:
            groups["maintenance_guides"] = self._retrieve_vector(
                self._maintenance_retriever(),
                analysis.original_query,
                "maintenance_guides",
                "maintenance",
            )

        structured = self._structured_retriever()
        if structured is not None:
            groups["structured_field"] = structured.retrieve(analysis)

        for source_name, items in groups.items():
            trace.recall_counts[source_name] = len(items)

        fused = self.fusion_ranker.fuse(
            {source_name: items for source_name, items in groups.items() if items}
        )
        reranked, confidence = self.reranker.rerank(analysis, fused)
        selected = reranked[:8]

        trace.fusion_strategy = "rrf"
        trace.rerank_strategy = "rule"
        trace.confidence = confidence
        trace.selected_evidence_ids = [item.evidence_id for item in selected]
        trace.dropped_evidence_ids = [item.evidence_id for item in reranked[8:]]
        trace.flags["no_domain_evidence"] = not any(
            item.domain != "question" for item in selected
        )
        trace.flags["low_confidence"] = confidence < 0.35

        return EnhancedRetrievalResult(query=query, evidence=selected, trace=trace)

    @staticmethod
    def _retrieve_vector(
        retriever: Any,
        query: str,
        source_store: str,
        domain: str,
    ) -> list[RetrievedEvidence]:
        if retriever is None:
            return []
        return [
            evidence_from_document(
                doc,
                source_store=source_store,
                domain=domain,
                matched_by="vector",
            )
            for doc in retriever.invoke(query)
        ]

    def _question_retriever(self):
        if self.question_retriever is None:
            from rag.vector_store import VectorStoreService

            self.question_retriever = VectorStoreService("question_recall").get_retriever()
        return self.question_retriever

    def _policy_retriever(self):
        if self.policy_retriever is None:
            from rag.vector_store import VectorStoreService

            self.policy_retriever = VectorStoreService("policy_rules").get_retriever()
        return self.policy_retriever

    def _troubleshooting_retriever(self):
        if self.troubleshooting_retriever is None:
            from rag.vector_store import VectorStoreService

            self.troubleshooting_retriever = VectorStoreService(
                "troubleshooting_cases"
            ).get_retriever()
        return self.troubleshooting_retriever

    def _maintenance_retriever(self):
        if self.maintenance_retriever is None:
            from rag.vector_store import VectorStoreService

            self.maintenance_retriever = VectorStoreService(
                "maintenance_guides"
            ).get_retriever()
        return self.maintenance_retriever

    def _structured_retriever(self):
        if self.structured_retriever is _UNSET:
            self.structured_retriever = StructuredFieldRetriever()
        return self.structured_retriever
