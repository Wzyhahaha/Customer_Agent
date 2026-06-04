from __future__ import annotations

from rag.pipeline_types import QueryAnalysis, RetrievedEvidence


class EvidenceReranker:
    def rerank(
        self,
        analysis: QueryAnalysis,
        evidence: list[RetrievedEvidence],
    ) -> tuple[list[RetrievedEvidence], float]:
        for item in evidence:
            score = self._score_item(analysis, item)
            item.add_score("rerank_score", score)

        reranked = sorted(
            evidence,
            key=lambda item: (
                item.scores.get("rerank_score", 0.0),
                item.scores.get("rrf_score", 0.0),
            ),
            reverse=True,
        )
        confidence = reranked[0].scores.get("rerank_score", 0.0) if reranked else 0.0
        return reranked, min(confidence, 1.0)

    def _score_item(self, analysis: QueryAnalysis, evidence: RetrievedEvidence) -> float:
        score = 0.0
        content = evidence.content.lower()

        if evidence.domain in analysis.domains:
            score += 0.35

        normalized_keywords = {
            keyword.lower() for keyword in analysis.keywords if keyword and keyword.strip()
        }
        keyword_hits = 0
        for keyword in normalized_keywords:
            if keyword in content:
                keyword_hits += 1
        if normalized_keywords:
            score += 0.45 * (keyword_hits / len(normalized_keywords))

        if evidence.scores.get("field_score", 0.0) > 0:
            score += 0.15

        if evidence.scores.get("rrf_score", 0.0) > 0:
            score += min(evidence.scores["rrf_score"] * 3, 0.05)

        return score
