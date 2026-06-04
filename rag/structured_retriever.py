from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.pipeline_types import QueryAnalysis, RetrievedEvidence
from utils.path_tool import get_abs_path


DOMAIN_FILES = {
    "policy": "data/structured_policies/policy_rules.jsonl",
    "troubleshooting": "data/structured_policies/troubleshooting_cases.jsonl",
    "maintenance": "data/structured_policies/maintenance_guides.jsonl",
}

FIELD_NAMES = (
    "scene",
    "topic",
    "symptom",
    "aliases",
    "user_intents",
    "judgement_factors",
    "required_info",
    "related_parts",
    "content",
)


class StructuredFieldRetriever:
    def __init__(self, records_by_domain: dict[str, list[dict[str, Any]]] | None = None):
        self.records_by_domain = self._load_default_records() if records_by_domain is None else records_by_domain

    def retrieve(self, analysis: QueryAnalysis, limit_per_domain: int = 4) -> list[RetrievedEvidence]:
        results: list[RetrievedEvidence] = []
        domains = analysis.domains or list(self.records_by_domain)
        for domain in domains:
            scored: list[tuple[float, dict[str, Any]]] = []
            for record in self.records_by_domain.get(domain, []):
                score = self._score_record(analysis, record)
                if score > 0:
                    scored.append((score, record))

            scored.sort(key=lambda item: item[0], reverse=True)
            for score, record in scored[:limit_per_domain]:
                doc_id = str(record.get("id") or f"{domain}_{len(results) + 1}")
                evidence = RetrievedEvidence(
                    evidence_id=f"structured_{domain}:{doc_id}",
                    content=str(record.get("content") or self._record_text(record)),
                    domain=domain,
                    source_store=f"structured_{domain}",
                    source_file=str(record.get("source_file") or Path(DOMAIN_FILES.get(domain, "")).name),
                    doc_id=doc_id,
                    metadata=dict(record),
                )
                evidence.add_score("field_score", score)
                evidence.add_match("structured_field")
                results.append(evidence)
        return results

    @staticmethod
    def _load_default_records() -> dict[str, list[dict[str, Any]]]:
        records: dict[str, list[dict[str, Any]]] = {}
        for domain, relative_path in DOMAIN_FILES.items():
            path = Path(get_abs_path(relative_path))
            domain_records: list[dict[str, Any]] = []
            if path.exists():
                with path.open("r", encoding="utf-8") as file:
                    for line in file:
                        raw = line.strip()
                        if raw:
                            domain_records.append(json.loads(raw))
            records[domain] = domain_records
        return records

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        values: list[str] = []
        for field in FIELD_NAMES:
            value = record.get(field)
            if isinstance(value, list):
                values.extend(str(item) for item in value)
            elif value is not None:
                values.append(str(value))
        return " ".join(values)

    def _score_record(self, analysis: QueryAnalysis, record: dict[str, Any]) -> float:
        text = self._record_text(record).lower().replace(" ", "")
        query = analysis.original_query.lower().replace(" ", "")
        terms = [term for term in analysis.keywords if term]
        score = 0.0

        for term in terms:
            normalized_term = term.lower().replace(" ", "")
            if normalized_term and normalized_term in text:
                score += 2.0

        for rewritten in analysis.rewritten_queries:
            for token in rewritten.replace("，", " ").replace("。", " ").split():
                normalized_token = token.lower().replace(" ", "")
                if len(normalized_token) >= 2 and normalized_token in text:
                    score += 0.5

        compact_query = query
        if compact_query and compact_query in text:
            score += 3.0

        return score
