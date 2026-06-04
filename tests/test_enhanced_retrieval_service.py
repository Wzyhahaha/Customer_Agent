from langchain_core.documents import Document

from rag.enhanced_retrieval_service import EnhancedRetrievalService


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


def test_enhanced_retrieval_service_returns_evidence_and_trace():
    question_doc = Document(
        page_content="问题：APP 搜不到设备",
        metadata={"id": 4, "source": "机器人FAQ知识库_合并版.jsonl"},
    )
    trouble_doc = Document(
        page_content="症状：无法连接 WiFi\n别名：APP 搜不到设备；配网失败",
        metadata={"id": "trouble_001", "source": "troubleshooting_cases.jsonl"},
    )
    service = EnhancedRetrievalService(
        question_retriever=FakeRetriever([question_doc]),
        policy_retriever=FakeRetriever([]),
        troubleshooting_retriever=FakeRetriever([trouble_doc]),
        maintenance_retriever=FakeRetriever([]),
        structured_retriever=None,
    )

    result = service.retrieve("APP搜不到设备怎么办")

    assert result.query == "APP搜不到设备怎么办"
    assert result.trace.query_analysis.domains == ["troubleshooting"]
    assert result.trace.recall_counts["question_recall"] == 1
    assert result.trace.recall_counts["troubleshooting_cases"] == 1
    assert result.evidence[0].domain == "troubleshooting"
    assert result.trace.confidence > 0


def test_enhanced_retrieval_service_flags_no_domain_evidence():
    service = EnhancedRetrievalService(
        question_retriever=FakeRetriever([]),
        policy_retriever=FakeRetriever([]),
        troubleshooting_retriever=FakeRetriever([]),
        maintenance_retriever=FakeRetriever([]),
        structured_retriever=None,
    )

    result = service.retrieve("有没有儿童模式")

    assert result.evidence == []
    assert result.trace.flags["no_domain_evidence"] is True
    assert result.trace.confidence == 0.0
