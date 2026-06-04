import unittest
from langchain_core.documents import Document

from rag.context_formatter import format_retrieval_bundle
from rag.context_formatter import format_enhanced_context
from rag.enhanced_retrieval_service import EnhancedRetrievalResult
from rag.pipeline_types import QueryAnalysis, RetrievedEvidence, RetrievalTrace
from rag.retrieval_service import RetrievalBundle
from rag.query_router import QueryRoute


class TestContextFormatter(unittest.TestCase):
    def test_format_bundle_contains_route_and_two_knowledge_sections(self):
        bundle = RetrievalBundle(
            query="机器不出水，如果是正常使用坏的还能保修吗",
            route=QueryRoute(route="mixed", reason="policy+troubleshooting keywords"),
            question_docs=[Document(page_content="相似问法", metadata={"question": "不出水还能保修吗"})],
            policy_docs=[Document(page_content="规则正文", metadata={"topic": "免费维修判断"})],
            troubleshooting_docs=[Document(page_content="排障正文", metadata={"symptom": "机器人不出水"})],
        )

        context = format_retrieval_bundle(bundle)

        self.assertIn("## 路由结果", context)
        self.assertIn("## 规则依据", context)
        self.assertIn("## 排障依据", context)
        self.assertIn("mixed", context)

    def test_format_bundle_contains_maintenance_section(self):
        bundle = RetrievalBundle(
            query="拖布怎么清洗",
            route=QueryRoute(route="maintenance", reason="maintenance keywords"),
            maintenance_docs=[Document(page_content="维护正文", metadata={"section": "耗材专项维护与更换"})],
        )

        context = format_retrieval_bundle(bundle)

        self.assertIn("## 维护依据", context)
        self.assertIn("维护正文", context)


if __name__ == "__main__":
    unittest.main()


def test_format_enhanced_context_groups_evidence_and_citations():
    analysis = QueryAnalysis(original_query="进水还能保修吗", domains=["policy"], keywords=["进水", "保修"])
    trace = RetrievalTrace(query_analysis=analysis, confidence=0.82)
    evidence = RetrievedEvidence(
        evidence_id="policy_rules:policy_002",
        content="通常非保修：进水、跌落、私拆、第三方维修导致的损坏",
        domain="policy",
        source_store="policy_rules",
        doc_id="policy_002",
    )
    evidence.add_score("rerank_score", 0.82)
    result = EnhancedRetrievalResult(query="进水还能保修吗", evidence=[evidence], trace=trace)

    context = format_enhanced_context(result)

    assert "## 检索 Trace" in context
    assert "confidence=0.82" in context
    assert "## 政策依据" in context
    assert "[policy_rules:policy_002]" in context


def test_format_enhanced_context_includes_low_confidence_guidance():
    analysis = QueryAnalysis(original_query="有没有儿童模式", domains=[], keywords=["儿童模式"])
    trace = RetrievalTrace(
        query_analysis=analysis,
        confidence=0.0,
        flags={"no_domain_evidence": True, "low_confidence": True},
    )
    result = EnhancedRetrievalResult(query="有没有儿童模式", evidence=[], trace=trace)

    context = format_enhanced_context(result)

    assert "知识库没有召回到直接领域依据" in context
    assert "需要补充信息或转人工核验" in context
