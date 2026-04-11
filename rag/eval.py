"""
RAG 检索评估模块。

测试集中的每条样本显式标注：
1. expected_question_refs: 期望命中的问题库文档
2. expected_policy_sections: 期望命中的政策分区

输出指标：
1. Recall@K: top-k 中命中的相关目标数 / 标注相关目标总数
2. Precision@K: top-k 中命中的相关目标数 / top-k 返回总数
3. Hit@K: 至少命中 1 个相关目标的查询占比
4. Complete@K: 命中全部相关目标的查询占比
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag.retrieval_service import TypedRetrievalService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = PROJECT_ROOT / "data" / "test_queries.jsonl"
POLICY_HEADING_RE = re.compile(r"^##\s+(.+?)(?:（\d+条）)?\s*$", re.MULTILINE)


def load_test_queries() -> list[dict[str, Any]]:
    """加载测试集并校验字段。"""
    test_queries: list[dict[str, Any]] = []
    with TEST_FILE.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            raw = line.strip()
            if not raw:
                continue

            case = json.loads(raw)
            required_fields = ("query", "expected_question_refs", "expected_policy_sections")
            missing_fields = [field for field in required_fields if field not in case]
            if missing_fields:
                raise ValueError(
                    f"{TEST_FILE}:{line_no} 缺少字段: {', '.join(missing_fields)}"
                )

            test_queries.append(case)

    return test_queries


def normalize_question_ref(ref: dict[str, Any]) -> str:
    """把测试集里的 question ref 归一化成稳定字符串。"""
    source = str(ref["source"]).strip()
    doc_id = int(ref["id"])
    return f"{source}#{doc_id}"


def extract_question_ref(doc) -> str:
    """从召回文档中提取问题库文档引用。"""
    metadata = doc.metadata or {}
    source = metadata.get("source")
    doc_id = metadata.get("id")
    if source is None or doc_id is None:
        return ""

    source_name = Path(str(source)).name
    return f"{source_name}#{doc_id}"


class PolicySectionResolver:
    """通过原始 policy 文件内容反查 chunk 所属 section。"""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}

    def _load_source(self, source_path: str) -> dict[str, Any]:
        cached = self._cache.get(source_path)
        if cached is not None:
            return cached

        path = Path(source_path)
        text = path.read_text(encoding="utf-8")
        matches = list(POLICY_HEADING_RE.finditer(text))

        sections: list[dict[str, Any]] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append(
                {
                    "title": match.group(1).strip(),
                    "start": start,
                    "end": end,
                }
            )

        cached = {"text": text, "sections": sections}
        self._cache[source_path] = cached
        return cached

    @staticmethod
    def _extract_heading_from_chunk(content: str) -> str:
        match = POLICY_HEADING_RE.search(content)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _locate_chunk(text: str, chunk: str) -> int:
        if not chunk:
            return -1

        index = text.find(chunk)
        if index != -1:
            return index

        normalized_chunk = chunk.strip()
        if normalized_chunk:
            index = text.find(normalized_chunk)
            if index != -1:
                return index

        for line in (line.strip() for line in chunk.splitlines()):
            if len(line) < 12:
                continue
            index = text.find(line)
            if index != -1:
                return index

        return -1

    def resolve(self, doc) -> str:
        metadata = doc.metadata or {}
        explicit_category = str(metadata.get("category") or "").strip()
        if explicit_category:
            return explicit_category

        chunk_heading = self._extract_heading_from_chunk(doc.page_content)
        if chunk_heading:
            return chunk_heading

        source = str(metadata.get("source") or "").strip()
        if not source:
            return ""

        source_path = Path(source)
        if not source_path.exists():
            return ""

        bundle = self._load_source(str(source_path))
        index = self._locate_chunk(bundle["text"], doc.page_content)
        if index == -1:
            return ""

        for section in bundle["sections"]:
            if section["start"] <= index < section["end"]:
                return str(section["title"])

        return ""


@dataclass
class StageStats:
    name: str
    expected_total: int = 0
    retrieved_total: int = 0
    relevant_hit_total: int = 0
    relevant_retrieved_total: int = 0
    hit_queries: int = 0
    complete_queries: int = 0
    total_queries: int = 0

    @property
    def recall_at_k(self) -> float:
        return self.relevant_hit_total / self.expected_total if self.expected_total else 0.0

    @property
    def precision_at_k(self) -> float:
        return (
            self.relevant_retrieved_total / self.retrieved_total
            if self.retrieved_total
            else 0.0
        )

    @property
    def hit_at_k(self) -> float:
        return self.hit_queries / self.total_queries if self.total_queries else 0.0

    @property
    def complete_at_k(self) -> float:
        return self.complete_queries / self.total_queries if self.total_queries else 0.0

    @property
    def f1_at_k(self) -> float:
        recall = self.recall_at_k
        precision = self.precision_at_k
        return (
            2 * recall * precision / (recall + precision)
            if (recall + precision) > 0
            else 0.0
        )


def _format_ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "0/0"


def _print_stage_summary(stats: StageStats, top_k: int):
    print("\n" + "=" * 70)
    print(f"{stats.name} 评估结果 (K={top_k})")
    print("=" * 70)
    print(
        f"Recall@{top_k}:    {_format_ratio(stats.relevant_hit_total, stats.expected_total)}"
        f" = {stats.recall_at_k:.1%}"
    )
    print(
        f"Precision@{top_k}: {_format_ratio(stats.relevant_retrieved_total, stats.retrieved_total)}"
        f" = {stats.precision_at_k:.1%}"
    )
    print(
        f"Hit@{top_k}:       {_format_ratio(stats.hit_queries, stats.total_queries)}"
        f" = {stats.hit_at_k:.1%}"
    )
    print(
        f"Complete@{top_k}:  {_format_ratio(stats.complete_queries, stats.total_queries)}"
        f" = {stats.complete_at_k:.1%}"
    )
    print(f"F1@{top_k}:        {stats.f1_at_k:.1%}")


def evaluate():
    """评估两阶段检索效果。"""
    retrieval_service = TypedRetrievalService()
    resolver = PolicySectionResolver()
    test_queries = load_test_queries()

    question_stats = StageStats(name="问题库")
    policy_stats = StageStats(name="政策库")
    joint_hit_queries = 0
    details: list[dict[str, Any]] = []

    print("=" * 70)
    print("RAG 两阶段检索评估")
    print("=" * 70)
    print(f"测试集文件: {TEST_FILE}")
    print(f"样本数: {len(test_queries)}")
    print(f"问题库 K: {retrieval_service.question_retriever.search_kwargs.get('k', 'N/A')}")
    print(f"政策库 K: {retrieval_service.policy_retriever.search_kwargs.get('k', 'N/A')}")

    for index, case in enumerate(test_queries, start=1):
        query = str(case["query"]).strip()
        case_name = str(case.get("name") or f"case-{index}")

        expected_question_refs = {
            normalize_question_ref(ref) for ref in case["expected_question_refs"]
        }
        expected_policy_sections = {
            str(section).strip()
            for section in case["expected_policy_sections"]
            if str(section).strip()
        }

        # 使用 TypedRetrievalService 进行检索
        bundle = retrieval_service.retrieve(query)

        question_docs = bundle.question_docs
        retrieved_question_refs = [
            ref for ref in (extract_question_ref(doc) for doc in question_docs) if ref
        ]
        retrieved_question_ref_set = set(retrieved_question_refs)
        matched_question_refs = sorted(expected_question_refs & retrieved_question_ref_set)

        routed_policy_docs = bundle.policy_docs
        routed_troubleshooting_docs = bundle.troubleshooting_docs
        merged_domain_docs = routed_policy_docs + routed_troubleshooting_docs

        retrieved_policy_sections = [
            resolver.resolve(doc) or "(未识别分区)" for doc in merged_domain_docs
        ]
        retrieved_policy_section_set = {
            section for section in retrieved_policy_sections if section != "(未识别分区)"
        }
        matched_policy_sections = sorted(
            expected_policy_sections & retrieved_policy_section_set
        )

        question_relevant_retrieved = sum(
            1 for ref in retrieved_question_refs if ref in expected_question_refs
        )
        policy_relevant_retrieved = sum(
            1 for section in retrieved_policy_sections if section in expected_policy_sections
        )

        question_hit = bool(matched_question_refs)
        policy_hit = bool(matched_policy_sections)
        question_complete = matched_question_refs == sorted(expected_question_refs)
        policy_complete = matched_policy_sections == sorted(expected_policy_sections)

        question_stats.expected_total += len(expected_question_refs)
        question_stats.retrieved_total += len(retrieved_question_refs)
        question_stats.relevant_hit_total += len(matched_question_refs)
        question_stats.relevant_retrieved_total += question_relevant_retrieved
        question_stats.hit_queries += int(question_hit)
        question_stats.complete_queries += int(question_complete)
        question_stats.total_queries += 1

        policy_stats.expected_total += len(expected_policy_sections)
        policy_stats.retrieved_total += len(retrieved_policy_sections)
        policy_stats.relevant_hit_total += len(matched_policy_sections)
        policy_stats.relevant_retrieved_total += policy_relevant_retrieved
        policy_stats.hit_queries += int(policy_hit)
        policy_stats.complete_queries += int(policy_complete)
        policy_stats.total_queries += 1

        joint_hit_queries += int(question_hit and policy_hit)

        question_recall = (
            len(matched_question_refs) / len(expected_question_refs)
            if expected_question_refs
            else 0.0
        )
        question_precision = (
            question_relevant_retrieved / len(retrieved_question_refs)
            if retrieved_question_refs
            else 0.0
        )
        policy_recall = (
            len(matched_policy_sections) / len(expected_policy_sections)
            if expected_policy_sections
            else 0.0
        )
        policy_precision = (
            policy_relevant_retrieved / len(retrieved_policy_sections)
            if retrieved_policy_sections
            else 0.0
        )

        print("\n" + "-" * 70)
        print(f"[{index}] {case_name}")
        print(f"Query: {query}")
        print(f"  问题库期望: {sorted(expected_question_refs)}")
        print(f"  问题库召回: {retrieved_question_refs}")
        question_k = retrieval_service.question_retriever.search_kwargs.get('k', 'N/A')
        print(
            f"  问题库命中: {matched_question_refs} | "
            f"Recall@{question_k}={question_recall:.1%}, "
            f"Precision@{question_k}={question_precision:.1%}"
        )
        print(f"  政策库期望: {sorted(expected_policy_sections)}")
        print(f"  政策库召回: {retrieved_policy_sections}")
        policy_k = retrieval_service.policy_retriever.search_kwargs.get('k', 'N/A')
        print(
            f"  政策库命中: {matched_policy_sections} | "
            f"Recall@{policy_k}={policy_recall:.1%}, "
            f"Precision@{policy_k}={policy_precision:.1%}"
        )

        details.append(
            {
                "name": case_name,
                "query": query,
                "expected_question_refs": sorted(expected_question_refs),
                "retrieved_question_refs": retrieved_question_refs,
                "matched_question_refs": matched_question_refs,
                "question_recall_at_k": question_recall,
                "question_precision_at_k": question_precision,
                "expected_policy_sections": sorted(expected_policy_sections),
                "retrieved_policy_sections": retrieved_policy_sections,
                "matched_policy_sections": matched_policy_sections,
                "policy_recall_at_k": policy_recall,
                "policy_precision_at_k": policy_precision,
                "joint_hit": question_hit and policy_hit,
            }
        )

    _print_stage_summary(question_stats, retrieval_service.question_retriever.search_kwargs.get('k', 0))
    _print_stage_summary(policy_stats, retrieval_service.policy_retriever.search_kwargs.get('k', 0))

    joint_hit_at_k = joint_hit_queries / len(test_queries) if test_queries else 0.0
    print("\n" + "=" * 70)
    print("联合指标")
    print("=" * 70)
    print(
        f"Joint Hit@K: {_format_ratio(joint_hit_queries, len(test_queries))}"
        f" = {joint_hit_at_k:.1%}"
    )

    return {
        "question_stage": {
            "recall_at_k": question_stats.recall_at_k,
            "precision_at_k": question_stats.precision_at_k,
            "hit_at_k": question_stats.hit_at_k,
            "complete_at_k": question_stats.complete_at_k,
            "f1_at_k": question_stats.f1_at_k,
            "expected_total": question_stats.expected_total,
            "retrieved_total": question_stats.retrieved_total,
            "relevant_hit_total": question_stats.relevant_hit_total,
            "relevant_retrieved_total": question_stats.relevant_retrieved_total,
        },
        "policy_stage": {
            "recall_at_k": policy_stats.recall_at_k,
            "precision_at_k": policy_stats.precision_at_k,
            "hit_at_k": policy_stats.hit_at_k,
            "complete_at_k": policy_stats.complete_at_k,
            "f1_at_k": policy_stats.f1_at_k,
            "expected_total": policy_stats.expected_total,
            "retrieved_total": policy_stats.retrieved_total,
            "relevant_hit_total": policy_stats.relevant_hit_total,
            "relevant_retrieved_total": policy_stats.relevant_retrieved_total,
        },
        "joint_hit_at_k": joint_hit_at_k,
        "details": details,
    }


if __name__ == "__main__":
    evaluate()
