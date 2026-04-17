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
    maintenance_docs: list[Document] = field(default_factory=list)


class TypedRetrievalService:
    def __init__(
        self,
        question_retriever: Any = None,
        policy_retriever: Any = None,
        troubleshooting_retriever: Any = None,
        maintenance_retriever: Any = None,
        router: "QueryRouter | None" = None,
    ):
        self._router = router
        self._question_retriever = question_retriever
        self._policy_retriever = policy_retriever
        self._troubleshooting_retriever = troubleshooting_retriever
        self._maintenance_retriever = maintenance_retriever

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

    @property
    def maintenance_retriever(self) -> BaseRetriever:
        if self._maintenance_retriever is None:
            from rag.vector_store import VectorStoreService

            self._maintenance_retriever = VectorStoreService("maintenance_guides").get_retriever()
        return self._maintenance_retriever

    @staticmethod
    def _domains_for_route(route: "QueryRoute") -> set[str]:
        if route.route == "other":
            return set()
        if route.route != "mixed":
            return {route.route}

        reason = str(route.reason or "")
        if reason.endswith(" keywords"):
            reason = reason[:-9]

        domains = {
            domain
            for domain in reason.split("+")
            if domain in {"policy", "troubleshooting", "maintenance"}
        }
        return domains

    def retrieve(self, query: str) -> RetrievalBundle:
        route = self.router.route(query)
        question_docs: list[Document] = self.question_retriever.invoke(query)
        policy_docs: list[Document] = []
        troubleshooting_docs: list[Document] = []
        maintenance_docs: list[Document] = []

        # other 路由只检索相似问法，不检索结构化知识库
        if route.route == "other":
            return RetrievalBundle(
                query=query,
                route=route,
                question_docs=question_docs,
                policy_docs=[],
                troubleshooting_docs=[],
                maintenance_docs=[],
            )

        routed_domains = self._domains_for_route(route)

        if "policy" in routed_domains:
            policy_docs = self.policy_retriever.invoke(query)
        if "troubleshooting" in routed_domains:
            troubleshooting_docs = self.troubleshooting_retriever.invoke(query)
        if "maintenance" in routed_domains:
            maintenance_docs = self.maintenance_retriever.invoke(query)

        return RetrievalBundle(
            query=query,
            route=route,
            question_docs=question_docs,
            policy_docs=policy_docs,
            troubleshooting_docs=troubleshooting_docs,
            maintenance_docs=maintenance_docs,
        )
