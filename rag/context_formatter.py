from rag.retrieval_service import RetrievalBundle


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
