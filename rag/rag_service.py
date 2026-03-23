import os
import sys

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    logger.debug(f"[rag_prompt]{prompt}")
    return prompt


class RagSummarizeService:
    def __init__(self):
        self.question_store = VectorStoreService("question_recall")
        self.policy_store = VectorStoreService("policy_answer")
        self.question_retriever = self.question_store.get_retriever()
        self.policy_retriever = self.policy_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | print_prompt | self.model | StrOutputParser()

    def recall_question_docs(self, query: str) -> list[Document]:
        return self.question_retriever.invoke(query)

    def _build_policy_query(self, query: str, recalled_docs: list[Document]) -> str:
        query_parts = [query]
        for doc in recalled_docs[:3]:
            metadata = doc.metadata or {}
            matched_question = str(metadata.get("question") or doc.page_content).strip()
            intent = str(metadata.get("intent") or "").strip()
            category = str(metadata.get("category") or "").strip()
            answer_hint = str(metadata.get("answer_hint") or "").strip()

            if matched_question:
                query_parts.append(f"相似问法：{matched_question}")
            if intent:
                query_parts.append(f"意图：{intent}")
            if category:
                query_parts.append(f"分类：{category}")
            if answer_hint:
                query_parts.append(f"处理提示：{answer_hint}")

        # 去重后再检索政策库，避免 query 被重复短语污染。
        return "\n".join(dict.fromkeys(part for part in query_parts if part))

    def retrieve_policy_docs(self, query: str, recalled_docs: list[Document]) -> list[Document]:
        policy_query = self._build_policy_query(query, recalled_docs)
        return self.policy_retriever.invoke(policy_query)

    @staticmethod
    def _format_question_context(recalled_docs: list[Document]) -> str:
        if not recalled_docs:
            return "无匹配到相似问法。"

        context_parts = []
        for index, doc in enumerate(recalled_docs, start=1):
            metadata = doc.metadata or {}
            matched_question = str(metadata.get("question") or doc.page_content).strip()
            intent = str(metadata.get("intent") or "").strip()
            category = str(metadata.get("category") or "").strip()
            answer_hint = str(metadata.get("answer_hint") or "").strip()
            linked_policy_ids = metadata.get("linked_policy_ids") or []

            context_parts.append(
                "\n".join(
                    [
                        f"【相似问法{index}】{matched_question}",
                        f"分类：{category or '未知'}",
                        f"意图：{intent or '未知'}",
                        f"处理提示：{answer_hint or '无'}",
                        f"关联政策ID：{linked_policy_ids or '无'}",
                    ]
                )
            )
        return "\n\n".join(context_parts)

    @staticmethod
    def _format_policy_context(policy_docs: list[Document]) -> str:
        if not policy_docs:
            return "无匹配到政策依据。"

        context_parts = []
        for index, doc in enumerate(policy_docs, start=1):
            source = str((doc.metadata or {}).get("source") or "未知来源").strip()
            context_parts.append(f"【政策依据{index}】来源：{source}\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def rag_summarize(self, query: str):
        recalled_question_docs = self.recall_question_docs(query)
        policy_docs = self.retrieve_policy_docs(query, recalled_question_docs)

        context = "\n\n".join(
            [
                "## 相似问法召回",
                self._format_question_context(recalled_question_docs),
                "## 政策与处理依据",
                self._format_policy_context(policy_docs),
            ]
        )

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("我的扫地机器人坏了，还在保修期吗"))
