import unittest
from langchain_core.documents import Document

from rag.vector_store import VectorStoreService


class TestVectorStoreService(unittest.TestCase):
    def test_structured_store_skips_splitter(self):
        service = VectorStoreService("policy_rules")
        docs = [Document(page_content="规则正文", metadata={"topic": "保修期判断"})]

        prepared = service._prepare_documents_for_store(docs)

        self.assertEqual(prepared, docs)

    def test_legacy_policy_store_keeps_splitter_path(self):
        service = VectorStoreService("policy_answer")
        docs = [Document(page_content="这是一段很长的正文。" * 50, metadata={})]

        prepared = service._prepare_documents_for_store(docs)

        self.assertGreaterEqual(len(prepared), 1)
        self.assertNotEqual(prepared, [])


if __name__ == "__main__":
    unittest.main()