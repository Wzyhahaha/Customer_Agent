from rag.pipeline_types import QueryAnalysis
from rag.structured_retriever import StructuredFieldRetriever


def test_structured_field_retriever_matches_aliases_and_scores():
    records = [
        {
            "id": "trouble_001",
            "knowledge_type": "troubleshooting",
            "scene": "故障排查",
            "symptom": "无法连接 WiFi",
            "aliases": ["连不上 WiFi", "APP 搜不到设备", "配网失败"],
            "content": "症状：无法连接 WiFi",
        }
    ]
    retriever = StructuredFieldRetriever(records_by_domain={"troubleshooting": records})
    analysis = QueryAnalysis(
        original_query="APP搜不到设备怎么办",
        domains=["troubleshooting"],
        keywords=["APP搜不到", "搜不到设备"],
    )

    evidence = retriever.retrieve(analysis)

    assert len(evidence) == 1
    assert evidence[0].evidence_id == "structured_troubleshooting:trouble_001"
    assert evidence[0].scores["field_score"] > 0
    assert evidence[0].matched_by == ["structured_field"]


def test_structured_field_retriever_respects_domains():
    retriever = StructuredFieldRetriever(
        records_by_domain={
            "policy": [{"id": "policy_001", "knowledge_type": "policy", "topic": "保修", "content": "保修规则"}],
            "maintenance": [{"id": "maint_001", "knowledge_type": "maintenance", "topic": "滤网", "content": "滤网维护"}],
        }
    )
    analysis = QueryAnalysis(original_query="滤网怎么清理", domains=["maintenance"], keywords=["滤网", "清理"])

    evidence = retriever.retrieve(analysis)

    assert [item.evidence_id for item in evidence] == ["structured_maintenance:maint_001"]


def test_structured_field_retriever_preserves_empty_records_mapping():
    retriever = StructuredFieldRetriever(records_by_domain={})
    analysis = QueryAnalysis(original_query="anything", domains=["policy"], keywords=["anything"])

    evidence = retriever.retrieve(analysis)

    assert retriever.records_by_domain == {}
    assert evidence == []


def test_structured_field_retriever_uses_record_source_file():
    records = [
        {
            "id": "maint_custom",
            "knowledge_type": "maintenance",
            "topic": "filter",
            "content": "filter maintenance",
            "source_file": "custom.jsonl",
        }
    ]
    retriever = StructuredFieldRetriever(records_by_domain={"maintenance": records})
    analysis = QueryAnalysis(original_query="filter", domains=["maintenance"], keywords=["filter"])

    evidence = retriever.retrieve(analysis)

    assert evidence[0].source_file == "custom.jsonl"
