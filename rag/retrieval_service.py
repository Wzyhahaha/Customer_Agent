from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

if TYPE_CHECKING:
    from rag.query_router import QueryRoute, QueryRouter
    from rag.vector_store import VectorStoreService


@dataclass
class RetrievalBundle:
    query: str
    route: "QueryRoute"
    question_docs: list[Document] = field(default_factory=list)
    policy_docs: list[Document] = field(default_factory=list)
    troubleshooting_docs: list[Document] = field(default_factory=list)


class TypedRetrievalService:
    def __init__(
        self,
        question_retriever: Any = None,
        policy_retriever: Any = None,
        troubleshooting_retriever: Any = None,
        router: "QueryRouter | None" = None,
    ):
        self._router = router
        self._question_retriever = question_retriever
        self._policy_retriever = policy_retriever
        self._troubleshooting_retriever = troubleshooting_retriever

    @property
    def router(self) -> "QueryRouter":
        if self._router is None:
            from rag.query_router import QueryRouter

            self._router = QueryRouter()
        return self._router

    @property
    def question_retriever(self) -> BaseRetriever:
        if self._question_retriever is None:
            from rag.vector_store import VectorStoreService

            self._question_retriever = VectorStoreService("question_recall").get_retriever()
        return self._question_retriever

    @property
    def policy_retriever(self) -> BaseRetriever:
        if self._policy_retriever is None:
            from rag.vector_store import VectorStoreService

            self._policy_retriever = VectorStoreService("policy_rules").get_retriever()
        return self._policy_retriever

    @property
    def troubleshooting_retriever(self) -> BaseRetriever:
        if self._troubleshooting_retriever is None:
            from rag.vector_store import VectorStoreService

            self._troubleshooting_retriever = VectorStoreService("troubleshooting_cases").get_retriever()
        return self._troubleshooting_retriever

    def retrieve(self, query: str) -> RetrievalBundle:
        route = self.router.route(query)
        question_docs: list[Document] = self.question_retriever.invoke(query)
        policy_docs: list[Document] = []
        troubleshooting_docs: list[Document] = []

        # other 路由只检索相似问法，不检索结构化知识库
        if route.route == "other":
            return RetrievalBundle(
                query=query,
                route=route,
                question_docs=question_docs,
                policy_docs=[],
                troubleshooting_docs=[],
            )

        if route.route in {"policy", "mixed"}:
            policy_docs = self.policy_retriever.invoke(query)
        if route.route in {"troubleshooting", "mixed"}:
            troubleshooting_docs = self.troubleshooting_retriever.invoke(query)

        return RetrievalBundle(
            query=query,
            route=route,
            question_docs=question_docs,
            policy_docs=policy_docs,
            troubleshooting_docs=troubleshooting_docs,
        )