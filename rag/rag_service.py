import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag.context_formatter import format_retrieval_bundle
from rag.retrieval_service import TypedRetrievalService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    logger.debug(f"[rag_prompt]{prompt}")
    return prompt


class RagSummarizeService:
    def __init__(self, pipeline: str = "baseline", init_chain: bool = True):
        if pipeline not in {"baseline", "enhanced"}:
            raise ValueError(f"unsupported rag pipeline: {pipeline}")
        self.pipeline = pipeline
        self.retrieval_service = TypedRetrievalService()
        self.enhanced_retrieval_service = None
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = None
        self.chain = self._init_chain() if init_chain else None

    def _init_chain(self):
        if self.model is None:
            from model.factory import chat_model

            self.model = chat_model
        return self.prompt_template | print_prompt | self.model | StrOutputParser()

    def build_chain_inputs(self, query: str) -> dict[str, str]:
        if self.pipeline == "enhanced":
            from rag.context_formatter import format_enhanced_context
            from rag.enhanced_retrieval_service import EnhancedRetrievalService

            if self.enhanced_retrieval_service is None:
                self.enhanced_retrieval_service = EnhancedRetrievalService()
            result = self.enhanced_retrieval_service.retrieve(query)
            return {
                "input": query,
                "route": "enhanced",
                "context": format_enhanced_context(result),
            }

        bundle = self.retrieval_service.retrieve(query)
        return {
            "input": query,
            "route": bundle.route.route,
            "context": format_retrieval_bundle(bundle),
        }

    def rag_summarize(self, query: str):
        if self.chain is None:
            self.chain = self._init_chain()
        return self.chain.invoke(self.build_chain_inputs(query))


if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("我的扫地机器人坏了，还在保修期吗"))
