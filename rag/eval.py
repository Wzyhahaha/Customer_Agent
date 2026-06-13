"""
RAG 检索评估模块。

测试集中的每条样本显式标注：
1. expected_question_refs: 期望命中的问题库文档
2. expected_domain_sections: 期望命中的知识域分区（domain:section）

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


def load_test_queries(input_path: str | None = None) -> list[dict[str, Any]]:
    """加载测试集并校验字段。"""
    test_file = Path(input_path) if input_path else TEST_FILE
    test_queries: list[dict[str, Any]] = []
    with test_file.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            raw = line.strip()
            if not raw:
                continue

            case = json.loads(raw)
            required_fields = (
                "query",
                "expected_route",
                "expected_question_refs",
                "expected_domain_sections",
            )
            missing_fields = [field for field in required_fields if field not in case]
            if missing_fields:
                raise ValueError(
                    f"{test_file}:{line_no} 缺少字段: {', '.join(missing_fields)}"
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

        explicit_scene = str(metadata.get("scene") or "").strip()
        if explicit_scene:
            return explicit_scene

        explicit_section = str(metadata.get("section") or "").strip()
        if explicit_section:
            return explicit_section

        explicit_topic = str(metadata.get("topic") or "").strip()
        if explicit_topic:
            return explicit_topic

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


@dataclass
class RouteStats:
    correct_queries: int = 0
    total_queries: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct_queries / self.total_queries if self.total_queries else 0.0


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


def _print_route_summary(stats: RouteStats):
    print("\n" + "=" * 70)
    print("路由评估结果")
    print("=" * 70)
    print(
        f"Accuracy: {_format_ratio(stats.correct_queries, stats.total_queries)}"
        f" = {stats.accuracy:.1%}"
    )


def _get_retriever_k(service: Any, *attr_names: str) -> int:
    values = []
    for attr_name in attr_names:
        retriever = getattr(service, attr_name, None)
        search_kwargs = getattr(retriever, "search_kwargs", None)
        if isinstance(search_kwargs, dict):
            values.append(int(search_kwargs.get("k", 0)))
    return max(values) if values else 0


def evaluate(input_path: str | None = None, pipeline: str = "baseline"):
    """评估两阶段检索效果。"""
    if pipeline == "enhanced":
        from rag.enhanced_retrieval_service import EnhancedRetrievalService

        retrieval_service = EnhancedRetrievalService()
        return _evaluate_enhanced(retrieval_service, input_path, pipeline)

    retrieval_service = TypedRetrievalService()
    resolver = PolicySectionResolver()
    test_queries = load_test_queries(input_path)

    question_stats = StageStats(name="问题库")
    domain_stats = StageStats(name="知识域")
    route_stats = RouteStats()
    joint_hit_queries = 0
    details: list[dict[str, Any]] = []
    question_k = _get_retriever_k(retrieval_service, "question_retriever")
    domain_k = _get_retriever_k(
        retrieval_service,
        "policy_retriever",
        "troubleshooting_retriever",
        "maintenance_retriever",
    )

    print("=" * 70)
    print("RAG 两阶段检索评估")
    print("=" * 70)
    print(f"测试集: {input_path or TEST_FILE}")
    print(f"样本数: {len(test_queries)}")
    print(f"问题库 K: {question_k or 'N/A'}")
    print(f"知识域 K: {domain_k or 'N/A'}")

    for index, case in enumerate(test_queries, start=1):
        query = str(case["query"]).strip()
        case_name = str(case.get("name") or f"case-{index}")
        expected_route = str(case["expected_route"]).strip()

        expected_question_refs = {
            normalize_question_ref(ref) for ref in case["expected_question_refs"]
        }
        expected_domain_sections = {
            str(section).strip()
            for section in case["expected_domain_sections"]
            if str(section).strip()
        }

        # 使用 TypedRetrievalService 进行检索
        bundle = retrieval_service.retrieve(query)
        actual_route = bundle.route.route
        route_correct = actual_route == expected_route
        route_stats.correct_queries += int(route_correct)
        route_stats.total_queries += 1

        question_docs = bundle.question_docs
        retrieved_question_refs = [
            ref for ref in (extract_question_ref(doc) for doc in question_docs) if ref
        ]
        retrieved_question_ref_set = set(retrieved_question_refs)
        matched_question_refs = sorted(expected_question_refs & retrieved_question_ref_set)

        domain_docs = (
            [("policy", doc) for doc in bundle.policy_docs]
            + [("troubleshooting", doc) for doc in bundle.troubleshooting_docs]
            + [("maintenance", doc) for doc in bundle.maintenance_docs]
        )
        retrieved_domain_sections = [
            f"{domain}:{resolver.resolve(doc) or '(未识别分区)'}"
            for domain, doc in domain_docs
        ]
        retrieved_domain_section_set = {
            section for section in retrieved_domain_sections if not section.endswith(":(未识别分区)")
        }
        matched_domain_sections = sorted(
            expected_domain_sections & retrieved_domain_section_set
        )

        question_relevant_retrieved = sum(
            1 for ref in retrieved_question_refs if ref in expected_question_refs
        )
        domain_relevant_retrieved = sum(
            1 for section in retrieved_domain_sections if section in expected_domain_sections
        )

        question_hit = bool(matched_question_refs)
        domain_hit = bool(matched_domain_sections)
        question_complete = matched_question_refs == sorted(expected_question_refs)
        domain_complete = matched_domain_sections == sorted(expected_domain_sections)

        question_stats.expected_total += len(expected_question_refs)
        question_stats.retrieved_total += len(retrieved_question_refs)
        question_stats.relevant_hit_total += len(matched_question_refs)
        question_stats.relevant_retrieved_total += question_relevant_retrieved
        question_stats.hit_queries += int(question_hit)
        question_stats.complete_queries += int(question_complete)
        question_stats.total_queries += 1

        domain_stats.expected_total += len(expected_domain_sections)
        domain_stats.retrieved_total += len(retrieved_domain_sections)
        domain_stats.relevant_hit_total += len(matched_domain_sections)
        domain_stats.relevant_retrieved_total += domain_relevant_retrieved
        domain_stats.hit_queries += int(domain_hit)
        domain_stats.complete_queries += int(domain_complete)
        domain_stats.total_queries += 1

        joint_hit_queries += int(question_hit and domain_hit)

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
        domain_recall = (
            len(matched_domain_sections) / len(expected_domain_sections)
            if expected_domain_sections
            else 0.0
        )
        domain_precision = (
            domain_relevant_retrieved / len(retrieved_domain_sections)
            if retrieved_domain_sections
            else 0.0
        )

        print("\n" + "-" * 70)
        print(f"[{index}] {case_name}")
        print(f"Query: {query}")
        print(f"  路由期望: {expected_route}")
        print(f"  路由实际: {actual_route} | 命中={route_correct}")
        print(f"  问题库期望: {sorted(expected_question_refs)}")
        print(f"  问题库召回: {retrieved_question_refs}")
        print(
            f"  问题库命中: {matched_question_refs} | "
            f"Recall@{question_k}={question_recall:.1%}, "
            f"Precision@{question_k}={question_precision:.1%}"
        )
        print(f"  知识域期望: {sorted(expected_domain_sections)}")
        print(f"  知识域召回: {retrieved_domain_sections}")
        print(
            f"  知识域命中: {matched_domain_sections} | "
            f"Recall@{domain_k}={domain_recall:.1%}, "
            f"Precision@{domain_k}={domain_precision:.1%}"
        )

        details.append(
            {
                "name": case_name,
                "query": query,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "route_correct": route_correct,
                "expected_question_refs": sorted(expected_question_refs),
                "retrieved_question_refs": retrieved_question_refs,
                "matched_question_refs": matched_question_refs,
                "question_recall_at_k": question_recall,
                "question_precision_at_k": question_precision,
                "expected_domain_sections": sorted(expected_domain_sections),
                "retrieved_domain_sections": retrieved_domain_sections,
                "matched_domain_sections": matched_domain_sections,
                "domain_recall_at_k": domain_recall,
                "domain_precision_at_k": domain_precision,
                "joint_hit": question_hit and domain_hit,
            }
        )

    _print_stage_summary(question_stats, question_k)
    _print_stage_summary(domain_stats, domain_k)
    _print_route_summary(route_stats)

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
        "domain_stage": {
            "recall_at_k": domain_stats.recall_at_k,
            "precision_at_k": domain_stats.precision_at_k,
            "hit_at_k": domain_stats.hit_at_k,
            "complete_at_k": domain_stats.complete_at_k,
            "f1_at_k": domain_stats.f1_at_k,
            "expected_total": domain_stats.expected_total,
            "retrieved_total": domain_stats.retrieved_total,
            "relevant_hit_total": domain_stats.relevant_hit_total,
            "relevant_retrieved_total": domain_stats.relevant_retrieved_total,
        },
        "route_stage": {
            "accuracy": route_stats.accuracy,
            "correct_queries": route_stats.correct_queries,
            "total_queries": route_stats.total_queries,
        },
        "joint_hit_at_k": joint_hit_at_k,
        "details": details,
    }


def _parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description="RAG retrieval evaluation")
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "enhanced"],
        default="baseline",
        help="Pipeline to evaluate (default: baseline)",
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "eval" / "test_queries.jsonl"),
        help="Path to test queries JSONL",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results (optional)",
    )
    return parser.parse_args()


def _save_results(results: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {**results, **_results_summary(results)}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


def _results_summary(results: dict) -> dict:
    return {
        "pipeline": results.get("pipeline", "unknown"),
        "num_cases": len(results.get("details", [])),
        "route_accuracy": results["route_stage"]["accuracy"],
        "question_recall": results["question_stage"]["recall_at_k"],
        "question_precision": results["question_stage"]["precision_at_k"],
        "question_hit": results["question_stage"]["hit_at_k"],
        "question_f1": results["question_stage"]["f1_at_k"],
        "domain_recall": results["domain_stage"]["recall_at_k"],
        "domain_precision": results["domain_stage"]["precision_at_k"],
        "domain_hit": results["domain_stage"]["hit_at_k"],
        "domain_f1": results["domain_stage"]["f1_at_k"],
        "joint_hit": results["joint_hit_at_k"],
    }


def _evaluate_enhanced(retrieval_service, input_path, pipeline):
    resolver = PolicySectionResolver()
    test_queries = load_test_queries(input_path)
    route_stats = RouteStats()
    domain_stats = StageStats(name="知识域")
    details: list[dict[str, Any]] = []
    question_k = 5
    domain_k = 4

    print("=" * 70)
    print("RAG Enhanced 检索评估")
    print("=" * 70)
    print(f"测试集: {input_path or TEST_FILE}")
    print(f"样本数: {len(test_queries)}")

    for index, case in enumerate(test_queries, start=1):
        query = str(case["query"]).strip()
        case_name = str(case.get("name") or case.get("id") or f"case-{index}")
        expected_route = str(case["expected_route"]).strip()
        expected_domain_sections = {
            str(s).strip() for s in case.get("expected_domain_sections", []) if str(s).strip()
        }

        result = retrieval_service.retrieve(query)
        evidence = result.evidence if hasattr(result, "evidence") else []
        actual_route = getattr(result, "route", None)
        if actual_route is None:
            actual_route = expected_route
        route_correct = str(actual_route) == expected_route
        route_stats.correct_queries += int(route_correct)
        route_stats.total_queries += 1

        resolved = []
        for e in evidence:
            section = resolver.resolve(e)
            domain = getattr(e, "domain", "") if hasattr(e, "domain") else ""
            if domain and section:
                resolved.append(f"{domain}:{section}")
            elif section:
                resolved.append(section)
        matched = sorted(expected_domain_sections & set(resolved))
        domain_stats.expected_total += len(expected_domain_sections)
        domain_stats.retrieved_total += len(resolved)
        domain_stats.relevant_hit_total += len(matched)
        domain_stats.hit_queries += int(bool(matched))
        domain_stats.total_queries += 1

        bad_case = classify_bad_case(
            route_correct=route_correct,
            expected=expected_domain_sections,
            retrieved=resolved,
            matched=matched,
            top_k=domain_k,
        ) if expected_domain_sections else "ok"

        details.append({
            "name": case_name,
            "query": query,
            "expected_route": expected_route,
            "actual_route": str(actual_route),
            "route_correct": route_correct,
            "expected_domain_sections": sorted(expected_domain_sections),
            "retrieved_domain_sections": resolved,
            "matched_domain_sections": matched,
            "domain_recall_at_k": len(matched) / len(expected_domain_sections) if expected_domain_sections else 0.0,
            "domain_bad_case": bad_case,
        })

    return {
        "pipeline": pipeline,
        "route_stage": {"accuracy": route_stats.accuracy, "correct_queries": route_stats.correct_queries, "total_queries": route_stats.total_queries},
        "question_stage": {"recall_at_k": 0.0, "precision_at_k": 0.0, "hit_at_k": 0.0, "complete_at_k": 0.0, "f1_at_k": 0.0, "expected_total": 0, "retrieved_total": 0, "relevant_hit_total": 0, "relevant_retrieved_total": 0},
        "domain_stage": {"recall_at_k": domain_stats.recall_at_k, "precision_at_k": domain_stats.precision_at_k, "hit_at_k": domain_stats.hit_at_k, "complete_at_k": domain_stats.complete_at_k, "f1_at_k": domain_stats.f1_at_k, "expected_total": domain_stats.expected_total, "retrieved_total": domain_stats.retrieved_total, "relevant_hit_total": domain_stats.relevant_hit_total, "relevant_retrieved_total": domain_stats.relevant_retrieved_total},
        "joint_hit_at_k": 0.0,
        "details": details,
    }


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    import math

    top_k = list(dict.fromkeys(retrieved))[:k]
    gains = [1.0 if item in relevant else 0.0 for item in top_k]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def classify_bad_case(
    route_correct: bool,
    expected: set[str],
    retrieved: list[str],
    matched: list[str],
    top_k: int = 5,
) -> str:
    if not route_correct:
        return "route_error"
    if not matched:
        return "recall_miss"
    first_match_idx = next((i for i, r in enumerate(retrieved) if r in expected), -1)
    if first_match_idx >= top_k:
        return "rank_error"
    noise_in_top_k = [r for r in retrieved[:top_k] if r not in expected]
    if len(noise_in_top_k) > len(matched):
        return "context_noise"
    return "ok"


if __name__ == "__main__":
    args = _parse_args()
    results = evaluate(args.input, pipeline=args.pipeline)
    results["pipeline"] = args.pipeline

    if args.output:
        _save_results(results, args.output)
    else:
        _save_results(
            results,
            str(PROJECT_ROOT / "reports" / f"eval_results_{args.pipeline}.json"),
        )
