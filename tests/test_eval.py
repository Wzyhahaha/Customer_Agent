import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import rag.eval as eval_module
from rag.enhanced_retrieval_service import EnhancedRetrievalResult
from rag.pipeline_types import QueryAnalysis, RetrievedEvidence, RetrievalTrace


class FakeRoute:
    def __init__(self, route):
        self.route = route
        self.reason = "fake"


class FakeBundle:
    def __init__(self, route):
        self.route = FakeRoute(route)
        self.question_docs = []
        self.policy_docs = []
        self.troubleshooting_docs = []
        self.maintenance_docs = []


class FakeRetriever:
    search_kwargs = {"k": 1}


class FakeRetrievalService:
    def __init__(self, route="policy"):
        self.route = route
        self.question_retriever = FakeRetriever()
        self.policy_retriever = FakeRetriever()
        self.troubleshooting_retriever = FakeRetriever()
        self.maintenance_retriever = FakeRetriever()

    def retrieve(self, query):
        return FakeBundle(self.route)


class TestEvalModule(unittest.TestCase):
    def test_eval_module_can_import_without_chat_model(self):
        self.assertTrue(hasattr(eval_module, "load_test_queries"))
        self.assertTrue(hasattr(eval_module, "PolicySectionResolver"))

    def test_policy_section_resolver_uses_scene_metadata(self):
        resolver = eval_module.PolicySectionResolver()
        doc = Document(
            page_content="主题：是否在保修期内",
            metadata={"scene": "保修判断", "source": "data/structured_policies/policy_rules.jsonl"},
        )

        self.assertEqual(resolver.resolve(doc), "保修判断")

    def test_evaluate_reports_route_accuracy(self):
        fake_cases = [
            {
                "name": "case",
                "query": "这个机器还在保修期吗",
                "expected_route": "policy",
                "expected_question_refs": [],
                "expected_domain_sections": [],
            }
        ]

        with patch.object(
            eval_module,
            "TypedRetrievalService",
            return_value=FakeRetrievalService(),
        ):
            with patch.object(eval_module, "load_test_queries", return_value=fake_cases):
                result = eval_module.evaluate()

        self.assertIn("route_stage", result)
        self.assertEqual(result["route_stage"]["accuracy"], 1.0)

    def test_policy_section_resolver_uses_section_metadata(self):
        resolver = eval_module.PolicySectionResolver()
        doc = Document(
            page_content="维护正文",
            metadata={"section": "耗材专项维护与更换", "source": "data/structured_policies/maintenance_guides.jsonl"},
        )

        self.assertEqual(resolver.resolve(doc), "耗材专项维护与更换")

    def test_evaluate_reports_maintenance_domain_metrics(self):
        fake_cases = [
            {
                "name": "maintenance-case",
                "query": "拖布怎么清洗",
                "expected_route": "maintenance",
                "expected_question_refs": [],
                "expected_domain_sections": ["maintenance:耗材专项维护与更换"],
            }
        ]

        class MaintenanceFakeRetriever:
            search_kwargs = {"k": 1}

        class MaintenanceFakeRetrievalService:
            def __init__(self):
                self.route = "maintenance"
                self.question_retriever = MaintenanceFakeRetriever()
                self.policy_retriever = MaintenanceFakeRetriever()
                self.troubleshooting_retriever = MaintenanceFakeRetriever()
                self.maintenance_retriever = MaintenanceFakeRetriever()

            def retrieve(self, query):
                bundle = FakeBundle("maintenance")
                bundle.maintenance_docs = [
                    Document(
                        page_content="维护正文",
                        metadata={"section": "耗材专项维护与更换", "source": "data/structured_policies/maintenance_guides.jsonl"},
                    )
                ]
                return bundle

        with patch.object(
            eval_module,
            "TypedRetrievalService",
            return_value=MaintenanceFakeRetrievalService(),
        ):
            with patch.object(eval_module, "load_test_queries", return_value=fake_cases):
                result = eval_module.evaluate()

        self.assertIn("domain_stage", result)
        self.assertEqual(result["route_stage"]["total_queries"], 1)
        self.assertEqual(result["domain_stage"]["relevant_hit_total"], 1)

    def test_evaluate_supports_enhanced_pipeline(self):
        fake_cases = [
            {
                "name": "enhanced-case",
                "query": "进水还能保修吗",
                "expected_route": "policy",
                "expected_question_refs": [],
                "expected_domain_sections": ["policy:免费维修判断"],
            }
        ]

        class EnhancedFakeRetrievalService:
            def retrieve(self, query):
                analysis = QueryAnalysis(
                    original_query=query,
                    domains=["policy"],
                    keywords=["进水", "保修"],
                )
                trace = RetrievalTrace(query_analysis=analysis, confidence=0.8)
                evidence = RetrievedEvidence(
                    evidence_id="policy_rules:policy_002",
                    content="通常非保修：进水、跌落、私拆",
                    domain="policy",
                    source_store="policy_rules",
                    doc_id="policy_002",
                    metadata={"scene": "免费维修判断"},
                )
                return EnhancedRetrievalResult(query=query, evidence=[evidence], trace=trace)

        with patch(
            "rag.enhanced_retrieval_service.EnhancedRetrievalService",
            return_value=EnhancedFakeRetrievalService(),
        ):
            with patch.object(eval_module, "load_test_queries", return_value=fake_cases):
                result = eval_module.evaluate(pipeline="enhanced")

        self.assertEqual(result["route_stage"]["accuracy"], 1.0)
        self.assertEqual(result["domain_stage"]["relevant_hit_total"], 1)
        self.assertEqual(result["details"][0]["domain_bad_case"], "ok")


if __name__ == "__main__":
    unittest.main()


def test_reciprocal_rank_returns_first_relevant_position():
    assert eval_module.reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert eval_module.reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_at_k_scores_relevant_items_higher_when_early():
    early = eval_module.ndcg_at_k(["good", "bad"], {"good"}, 2)
    late = eval_module.ndcg_at_k(["bad", "good"], {"good"}, 2)

    assert early == 1.0
    assert 0 < late < early


def test_classify_bad_case_route_recall_rank_and_context():
    assert (
        eval_module.classify_bad_case(
            route_correct=False,
            expected=set(),
            retrieved=[],
            matched=[],
        )
        == "route_error"
    )
    assert (
        eval_module.classify_bad_case(
            route_correct=True,
            expected={"a"},
            retrieved=["b"],
            matched=[],
        )
        == "recall_miss"
    )
    assert (
        eval_module.classify_bad_case(
            route_correct=True,
            expected={"a"},
            retrieved=["b", "a"],
            matched=["a"],
            top_k=1,
        )
        == "rank_error"
    )
    assert (
        eval_module.classify_bad_case(
            route_correct=True,
            expected={"a"},
            retrieved=["a", "x", "y", "z"],
            matched=["a"],
            top_k=4,
        )
        == "context_noise"
    )
