import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.factory import chat_model
from rag.context_formatter import format_retrieval_bundle
from rag.retrieval_service import TypedRetrievalService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    logger.debug(f"[rag_prompt]{prompt}")
    return prompt


class RagSummarizeService:
    def __init__(self):
        self.retrieval_service = TypedRetrievalService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | print_prompt | self.model | StrOutputParser()

    def build_chain_inputs(self, query: str) -> dict[str, str]:
        bundle = self.retrieval_service.retrieve(query)
        return {
            "input": query,
            "route": bundle.route.route,
            "context": format_retrieval_bundle(bundle),
        }

    def rag_summarize(self, query: str):
        return self.chain.invoke(self.build_chain_inputs(query))


if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("我的扫地机器人坏了，还在保修期吗"))
