import unittest
from langchain_core.documents import Document

from rag.context_formatter import format_retrieval_bundle
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
