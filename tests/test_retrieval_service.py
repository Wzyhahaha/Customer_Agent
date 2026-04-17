import unittest
from langchain_core.documents import Document

from rag.retrieval_service import RetrievalBundle, TypedRetrievalService


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.docs


class FakeRoute:
    def __init__(self, route, reason):
        self.route = route
        self.reason = reason


class FakeRouter:
    def __init__(self, route, reason):
        self._route = FakeRoute(route, reason)

    def route(self, query):
        return self._route


class TestTypedRetrievalService(unittest.TestCase):
    def test_policy_query_only_hits_policy_store(self):
        question = FakeRetriever([Document(page_content="相似问法", metadata={})])
        policy = FakeRetriever([Document(page_content="规则", metadata={"topic": "保修期"})])
        troubleshooting = FakeRetriever([Document(page_content="排障", metadata={"symptom": "不出水"})])

        service = TypedRetrievalService(question_retriever=question, policy_retriever=policy, troubleshooting_retriever=troubleshooting)
        bundle = service.retrieve("这个机器还在保修期吗")

        self.assertEqual(bundle.route.route, "policy")
        self.assertEqual(len(bundle.policy_docs), 1)
        self.assertEqual(len(bundle.troubleshooting_docs), 0)

    def test_mixed_query_hits_both_structured_stores(self):
        question = FakeRetriever([])
        policy = FakeRetriever([Document(page_content="规则", metadata={"topic": "保修期"})])
        troubleshooting = FakeRetriever([Document(page_content="排障", metadata={"symptom": "不出水"})])
        maintenance = FakeRetriever([Document(page_content="维护", metadata={"section": "耗材专项维护与更换"})])

        service = TypedRetrievalService(
            question_retriever=question,
            policy_retriever=policy,
            troubleshooting_retriever=troubleshooting,
            maintenance_retriever=maintenance,
            router=FakeRouter("mixed", "policy+troubleshooting keywords"),
        )
        bundle = service.retrieve("机器不出水，如果是正常使用坏的还能保修吗")

        self.assertEqual(bundle.route.route, "mixed")
        self.assertEqual(len(bundle.policy_docs), 1)
        self.assertEqual(len(bundle.troubleshooting_docs), 1)
        self.assertEqual(maintenance.queries, [])

    def test_other_query_skips_structured_stores(self):
        question = FakeRetriever([Document(page_content="相似问法", metadata={})])
        policy = FakeRetriever([])
        troubleshooting = FakeRetriever([])

        service = TypedRetrievalService(question_retriever=question, policy_retriever=policy, troubleshooting_retriever=troubleshooting)
        bundle = service.retrieve("谢谢，已经解决了")

        self.assertEqual(bundle.route.route, "other")
        self.assertEqual(policy.queries, [])
        self.assertEqual(troubleshooting.queries, [])

    def test_maintenance_query_only_hits_maintenance_store(self):
        question = FakeRetriever([])
        policy = FakeRetriever([])
        troubleshooting = FakeRetriever([])
        maintenance = FakeRetriever([Document(page_content="维护正文", metadata={"section": "耗材专项维护与更换"})])

        service = TypedRetrievalService(
            question_retriever=question,
            policy_retriever=policy,
            troubleshooting_retriever=troubleshooting,
            maintenance_retriever=maintenance,
        )
        bundle = service.retrieve("拖布怎么清洗，多久换一次")

        self.assertEqual(bundle.route.route, "maintenance")
        self.assertEqual(len(bundle.maintenance_docs), 1)
        self.assertEqual(policy.queries, [])
        self.assertEqual(troubleshooting.queries, [])

    def test_mixed_query_hits_maintenance_and_policy(self):
        question = FakeRetriever([])
        policy = FakeRetriever([Document(page_content="规则正文", metadata={"scene": "免费维修判断"})])
        troubleshooting = FakeRetriever([])
        maintenance = FakeRetriever([Document(page_content="维护正文", metadata={"section": "耗材专项维护与更换"})])

        service = TypedRetrievalService(
            question_retriever=question,
            policy_retriever=policy,
            troubleshooting_retriever=troubleshooting,
            maintenance_retriever=maintenance,
        )
        bundle = service.retrieve("滤网堵了怎么清理，坏了算保修吗")

        self.assertEqual(bundle.route.route, "mixed")
        self.assertEqual(len(bundle.policy_docs), 1)
        self.assertEqual(len(bundle.maintenance_docs), 1)
        self.assertEqual(troubleshooting.queries, [])

    def test_three_domain_mixed_query_hits_all_three_stores(self):
        question = FakeRetriever([])
        policy = FakeRetriever([Document(page_content="规则正文", metadata={"scene": "免费维修判断"})])
        troubleshooting = FakeRetriever([Document(page_content="排障正文", metadata={"scene": "故障排查"})])
        maintenance = FakeRetriever([Document(page_content="维护正文", metadata={"section": "耗材专项维护与更换"})])

        service = TypedRetrievalService(
            question_retriever=question,
            policy_retriever=policy,
            troubleshooting_retriever=troubleshooting,
            maintenance_retriever=maintenance,
            router=FakeRouter("mixed", "policy+troubleshooting+maintenance keywords"),
        )
        bundle = service.retrieve("拖布不出水，怎么清洗，坏了还能保修吗")

        self.assertEqual(bundle.route.route, "mixed")
        self.assertEqual(len(bundle.policy_docs), 1)
        self.assertEqual(len(bundle.troubleshooting_docs), 1)
        self.assertEqual(len(bundle.maintenance_docs), 1)


if __name__ == "__main__":
    unittest.main()
