import pytest

from rag.pipeline_types import QueryAnalysis, RetrievedEvidence
from rag.reranker import EvidenceReranker


def evidence(evidence_id: str, content: str, domain: str):
    item = RetrievedEvidence(
        evidence_id=evidence_id,
        content=content,
        domain=domain,
        source_store=evidence_id.split(":")[0],
    )
    item.add_score("rrf_score", 0.01)
    return item


def test_reranker_prefers_domain_and_keyword_match():
    analysis = QueryAnalysis(
        original_query="滤网堵了怎么清理",
        domains=["maintenance"],
        keywords=["滤网", "清理"],
    )
    weak = evidence("policy_rules:policy_001", "保修判断需要购买凭证", "policy")
    strong = evidence("maintenance_guides:maint_001", "滤网堵塞后需要清理并定期更换", "maintenance")

    reranked, confidence = EvidenceReranker().rerank(analysis, [weak, strong])

    assert reranked[0].evidence_id == "maintenance_guides:maint_001"
    assert reranked[0].scores["rerank_score"] > reranked[1].scores["rerank_score"]
    assert confidence > 0.5


def test_reranker_low_confidence_for_unmatched_evidence():
    analysis = QueryAnalysis(original_query="有没有儿童模式", domains=[], keywords=["儿童模式"])
    item = evidence("policy_rules:policy_001", "保修判断需要购买凭证", "policy")

    reranked, confidence = EvidenceReranker().rerank(analysis, [item])

    assert reranked[0].scores["rerank_score"] < 0.3
    assert confidence < 0.3


def test_reranker_deduplicates_and_ignores_empty_keywords():
    analysis = QueryAnalysis(
        original_query="滤网",
        domains=[],
        keywords=["滤网", "滤网", ""],
    )
    item = evidence("maintenance_guides:maint_001", "需要清理滤网", "maintenance")

    reranked, confidence = EvidenceReranker().rerank(analysis, [item])

    assert reranked[0].scores["rerank_score"] == pytest.approx(0.48)
    assert confidence == pytest.approx(0.48)


def test_reranker_empty_evidence_returns_empty_result_and_zero_confidence():
    analysis = QueryAnalysis(original_query="滤网", domains=[], keywords=["滤网"])

    reranked, confidence = EvidenceReranker().rerank(analysis, [])

    assert reranked == []
    assert confidence == 0.0


def test_reranker_writes_rerank_score_to_input_evidence():
    analysis = QueryAnalysis(original_query="滤网", domains=[], keywords=["滤网"])
    item = evidence("maintenance_guides:maint_001", "需要清理滤网", "maintenance")

    reranked, _ = EvidenceReranker().rerank(analysis, [item])

    assert "rerank_score" in item.scores
    assert item.scores["rerank_score"] == reranked[0].scores["rerank_score"]


def test_reranker_preserves_input_order_for_exact_score_ties():
    analysis = QueryAnalysis(original_query="无匹配", domains=[], keywords=[])
    first = evidence("policy_rules:policy_001", "保修判断需要购买凭证", "policy")
    second = evidence("maintenance_guides:maint_001", "滤网清理说明", "maintenance")

    reranked, _ = EvidenceReranker().rerank(analysis, [first, second])

    assert [item.evidence_id for item in reranked] == [
        "policy_rules:policy_001",
        "maintenance_guides:maint_001",
    ]
