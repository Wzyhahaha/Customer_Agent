from rag.fusion import FusionRanker
from rag.pipeline_types import RetrievedEvidence


def evidence(evidence_id: str, content: str = "content"):
    return RetrievedEvidence(
        evidence_id=evidence_id,
        content=content,
        domain="policy",
        source_store=evidence_id.split(":")[0],
    )


def test_rrf_fusion_combines_scores_from_multiple_sources():
    shared_a = evidence("policy_rules:policy_001")
    shared_b = evidence("policy_rules:policy_001")
    unique = evidence("question_recall:q_001")

    fused = FusionRanker(k=60).fuse(
        {
            "vector": [shared_a, unique],
            "structured": [shared_b],
        }
    )

    assert [item.evidence_id for item in fused] == ["policy_rules:policy_001", "question_recall:q_001"]
    assert fused[0].scores["rrf_score"] > fused[1].scores["rrf_score"]
    assert fused[0].matched_by == ["vector", "structured"]


def test_rrf_fusion_keeps_highest_existing_scores():
    first = evidence("policy_rules:policy_001")
    first.add_score("field_score", 0.4)
    second = evidence("policy_rules:policy_001")
    second.add_score("field_score", 1.2)

    fused = FusionRanker().fuse({"a": [first], "b": [second]})

    assert fused[0].scores["field_score"] == 1.2


def test_rrf_fusion_ignores_input_rrf_scores():
    first = evidence("policy_rules:policy_001")
    first.add_score("rrf_score", 999.0)
    second = evidence("policy_rules:policy_001")
    second.add_score("rrf_score", 500.0)

    fused = FusionRanker(k=60).fuse({"a": [first], "b": [second]})

    assert fused[0].scores["rrf_score"] == 1 / (60 + 1) + 1 / (60 + 1)
