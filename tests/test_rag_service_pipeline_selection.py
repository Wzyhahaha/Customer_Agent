from rag.enhanced_retrieval_service import EnhancedRetrievalResult
from rag.pipeline_types import QueryAnalysis, RetrievalTrace
from rag.rag_service import RagSummarizeService
import pytest


class FakeBaselineRetrieval:
    def retrieve(self, query):
        class Route:
            route = "policy"
            reason = "policy keywords"

        class Bundle:
            route = Route()
            question_docs = []
            policy_docs = []
            troubleshooting_docs = []
            maintenance_docs = []

        return Bundle()


class FakeEnhancedRetrieval:
    def retrieve(self, query):
        analysis = QueryAnalysis(original_query=query, domains=["policy"])
        trace = RetrievalTrace(query_analysis=analysis, confidence=0.4)
        return EnhancedRetrievalResult(query=query, evidence=[], trace=trace)


def test_rag_service_builds_baseline_inputs_by_default():
    service = RagSummarizeService(init_chain=False)
    service.retrieval_service = FakeBaselineRetrieval()

    inputs = service.build_chain_inputs("还保修吗")

    assert inputs["route"] == "policy"
    assert "## 路由结果" in inputs["context"]
    assert service.enhanced_retrieval_service is None


def test_rag_service_builds_enhanced_inputs_when_selected():
    service = RagSummarizeService(pipeline="enhanced", init_chain=False)
    service.enhanced_retrieval_service = FakeEnhancedRetrieval()

    inputs = service.build_chain_inputs("还保修吗")

    assert inputs["route"] == "enhanced"
    assert "## 检索 Trace" in inputs["context"]


def test_rag_service_rejects_unknown_pipeline():
    with pytest.raises(ValueError, match="unsupported rag pipeline"):
        RagSummarizeService(pipeline="unknown", init_chain=False)


def test_rag_service_init_chain_false_defers_model_initialization():
    service = RagSummarizeService(init_chain=False)

    assert service.chain is None
    assert service.model is None


def test_rag_summarize_initializes_chain_lazily():
    class FakeChain:
        def invoke(self, inputs):
            return f"answer:{inputs['input']}"

    service = RagSummarizeService(init_chain=False)
    service.retrieval_service = FakeBaselineRetrieval()
    service._init_chain = lambda: FakeChain()

    assert service.rag_summarize("还保修吗") == "answer:还保修吗"
