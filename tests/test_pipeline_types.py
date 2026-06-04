from rag.pipeline_types import QueryAnalysis, RetrievedEvidence, RetrievalTrace


def test_query_analysis_defaults_to_original_rewrite():
    analysis = QueryAnalysis(original_query="APP搜不到设备怎么办", domains=["troubleshooting"])

    assert analysis.rewritten_queries == ["APP搜不到设备怎么办"]
    assert analysis.intent == ""
    assert analysis.needs_clarification is False
    assert analysis.risk_flags == []


def test_retrieved_evidence_updates_scores_and_matched_by():
    evidence = RetrievedEvidence(
        evidence_id="policy_rules:policy_001",
        content="保内且非人为故障通常可申请保修",
        domain="policy",
        source_store="policy_rules",
        source_file="policy_rules.jsonl",
        doc_id="policy_001",
        metadata={"scene": "保修判断"},
    )

    evidence.add_score("field_score", 0.8)
    evidence.add_match("structured_field")
    evidence.add_match("structured_field")

    assert evidence.scores["field_score"] == 0.8
    assert evidence.matched_by == ["structured_field"]


def test_retrieval_trace_records_selected_and_flags():
    analysis = QueryAnalysis(original_query="滤网堵了坏了算保修吗", domains=["maintenance", "policy"])
    trace = RetrievalTrace(query_analysis=analysis)

    trace.recall_counts["question_recall"] = 3
    trace.selected_evidence_ids.append("maintenance_guides:maint_001")
    trace.flags["mixed_query"] = True
    trace.confidence = 0.72

    assert trace.recall_counts == {"question_recall": 3}
    assert trace.selected_evidence_ids == ["maintenance_guides:maint_001"]
    assert trace.flags["mixed_query"] is True
    assert trace.confidence == 0.72
