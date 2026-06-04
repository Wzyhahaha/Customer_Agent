from __future__ import annotations

from copy import deepcopy

from rag.pipeline_types import RetrievedEvidence


class FusionRanker:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, groups: dict[str, list[RetrievedEvidence]]) -> list[RetrievedEvidence]:
        merged: dict[str, RetrievedEvidence] = {}

        for source_name, evidence_list in groups.items():
            for rank, evidence in enumerate(evidence_list, start=1):
                item = merged.get(evidence.evidence_id)
                if item is None:
                    item = deepcopy(evidence)
                    item.scores.pop("rrf_score", None)
                    merged[evidence.evidence_id] = item

                item.add_match(source_name)
                item.scores["rrf_score"] = item.scores.get("rrf_score", 0.0) + 1.0 / (self.k + rank)

                for score_name, score_value in evidence.scores.items():
                    if score_name == "rrf_score":
                        continue
                    existing = item.scores.get(score_name)
                    if existing is None or score_value > existing:
                        item.scores[score_name] = score_value

        return sorted(
            merged.values(),
            key=lambda item: (
                item.scores.get("rrf_score", 0.0),
                item.scores.get("field_score", 0.0),
            ),
            reverse=True,
        )
