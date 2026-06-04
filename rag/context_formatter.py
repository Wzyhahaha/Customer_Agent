from typing import TYPE_CHECKING

from rag.retrieval_service import RetrievalBundle
from rag.pipeline_types import RetrievedEvidence

if TYPE_CHECKING:
    from rag.enhanced_retrieval_service import EnhancedRetrievalResult


def _format_docs(title: str, docs: list) -> str:
    if not docs:
        return f"## {title}\n无"
    rows = []
    for index, doc in enumerate(docs, start=1):
        rows.append(f"【{title}{index}】{doc.page_content}")
    return f"## {title}\n" + "\n\n".join(rows)


def format_retrieval_bundle(bundle: RetrievalBundle) -> str:
    return "\n\n".join(
        [
            "## 路由结果",
            f"route={bundle.route.route}",
            f"reason={bundle.route.reason}",
            _format_docs("相似问法", bundle.question_docs),
            _format_docs("规则依据", bundle.policy_docs),
            _format_docs("排障依据", bundle.troubleshooting_docs),
            _format_docs("维护依据", bundle.maintenance_docs),
        ]
    )


DOMAIN_TITLES = {
    "policy": "政策依据",
    "troubleshooting": "排障依据",
    "maintenance": "维护依据",
    "question": "相似问法参考",
}


def _format_evidence_group(title: str, evidence: list[RetrievedEvidence]) -> str:
    if not evidence:
        return f"## {title}\n无"
    rows = []
    for item in evidence:
        score = item.scores.get("rerank_score", item.scores.get("rrf_score", 0.0))
        rows.append(f"[{item.evidence_id}] score={score:.2f}\n{item.content}")
    return f"## {title}\n" + "\n\n".join(rows)


def format_enhanced_context(result: "EnhancedRetrievalResult") -> str:
    trace = result.trace
    if trace is None:
        return "## 检索 Trace\n无"

    sections = [
        "## 检索 Trace",
        f"query={result.query}",
        f"domains={','.join(trace.query_analysis.domains) or 'none'}",
        f"intent={trace.query_analysis.intent}",
        f"confidence={trace.confidence:.2f}",
        f"fusion={trace.fusion_strategy or 'none'}",
        f"rerank={trace.rerank_strategy or 'none'}",
    ]

    if trace.flags.get("no_domain_evidence"):
        sections.append("知识库没有召回到直接领域依据。")
    if trace.flags.get("low_confidence"):
        sections.append("需要补充信息或转人工核验。")

    grouped: dict[str, list[RetrievedEvidence]] = {}
    for item in result.evidence:
        grouped.setdefault(item.domain, []).append(item)

    for domain in ("policy", "troubleshooting", "maintenance", "question"):
        sections.append(_format_evidence_group(DOMAIN_TITLES[domain], grouped.get(domain, [])))

    return "\n\n".join(sections)
