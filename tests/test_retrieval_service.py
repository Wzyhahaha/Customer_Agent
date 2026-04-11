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

        service = TypedRetrievalService(question_retriever=question, policy_retriever=policy, troubleshooting_retriever=troubleshooting)
        bundle = service.retrieve("机器不出水，如果是正常使用坏的还能保修吗")

        self.assertEqual(bundle.route.route, "mixed")
        self.assertEqual(len(bundle.policy_docs), 1)
        self.assertEqual(len(bundle.troubleshooting_docs), 1)

    def test_other_query_skips_structured_stores(self):
        question = FakeRetriever([Document(page_content="相似问法", metadata={})])
        policy = FakeRetriever([])
        troubleshooting = FakeRetriever([])

        service = TypedRetrievalService(question_retriever=question, policy_retriever=policy, troubleshooting_retriever=troubleshooting)
        bundle = service.retrieve("谢谢，已经解决了")

        self.assertEqual(bundle.route.route, "other")
        self.assertEqual(policy.queries, [])
        self.assertEqual(troubleshooting.queries, [])


if __name__ == "__main__":
    unittest.main()