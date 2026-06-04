from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document

from rag.pipeline_types import RetrievedEvidence


def _source_file_name(source: str) -> str:
    normalized_source = source.replace("\\", "/")
    return Path(normalized_source).name if normalized_source else ""


def stable_doc_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    for key in ("id", "doc_id", "source_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"


def evidence_from_document(
    doc: Document,
    source_store: str,
    domain: str,
    matched_by: str,
    score_name: str = "",
    score_value: float | None = None,
) -> RetrievedEvidence:
    metadata = dict(doc.metadata or {})
    doc_id = stable_doc_id(doc)
    source = str(metadata.get("source") or "")
    source_file = _source_file_name(source)
    evidence = RetrievedEvidence(
        evidence_id=f"{source_store}:{doc_id}",
        content=doc.page_content,
        domain=domain,
        source_store=source_store,
        source_file=source_file,
        doc_id=doc_id,
        metadata=metadata,
    )
    evidence.add_match(matched_by)
    if score_name and score_value is not None:
        evidence.add_score(score_name, score_value)
    return evidence
