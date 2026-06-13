from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.eval_metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

HF_HOME = Path.home() / ".cache" / "huggingface" / "hub"
WIX_SNAPSHOT = (
    HF_HOME
    / "datasets--Wix--WixQA"
    / "snapshots"
    / "d662dc42479c14e202eccd832f8c4b66a035c4cc"
)
TECHQA_SNAPSHOT = (
    HF_HOME
    / "datasets--nvidia--TechQA-RAG-Eval"
    / "snapshots"
    / "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a"
)
DEFAULT_MODEL_PATH = Path.home() / "Desktop" / "my-mini-swe" / "models" / "bge-m3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db" / "external_eval"


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    text: str
    metadata: dict


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    relevant_ids: set[str]
    answer: str = ""


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    score: float


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if raw:
                yield json.loads(raw)


def load_wix_documents(corpus_path: Path) -> Iterable[CorpusDocument]:
    for row in iter_jsonl(corpus_path):
        doc_id = str(row["id"])
        yield CorpusDocument(
            doc_id=doc_id,
            text=str(row.get("contents") or row.get("title") or "").strip(),
            metadata={
                "source_dataset": "wixqa",
                "doc_id": doc_id,
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "article_type": row.get("article_type", ""),
            },
        )


def load_wix_queries(query_path: Path) -> Iterable[EvalCase]:
    for index, row in enumerate(iter_jsonl(query_path), start=1):
        relevant_ids = {str(item) for item in row.get("article_ids", []) if str(item)}
        yield EvalCase(
            case_id=f"wixqa_{index:05d}",
            query=str(row["question"]).strip(),
            relevant_ids=relevant_ids,
            answer=str(row.get("answer") or ""),
        )


def load_techqa(path: Path, include_impossible: bool = False) -> tuple[list[CorpusDocument], list[EvalCase]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    documents_by_id: dict[str, CorpusDocument] = {}
    queries: list[EvalCase] = []

    for row in rows:
        contexts = row.get("contexts") or []
        relevant_ids: set[str] = set()
        for context in contexts:
            doc_id = str(context.get("filename") or "").strip()
            text = str(context.get("text") or "").strip()
            if not doc_id or not text:
                continue
            relevant_ids.add(doc_id)
            documents_by_id.setdefault(
                doc_id,
                CorpusDocument(
                    doc_id=doc_id,
                    text=text,
                    metadata={
                        "source_dataset": "techqa_rag_eval",
                        "doc_id": doc_id,
                    },
                ),
            )

        if row.get("is_impossible") and not include_impossible:
            continue
        if not relevant_ids:
            continue

        queries.append(
            EvalCase(
                case_id=str(row.get("id") or f"techqa_{len(queries) + 1:05d}"),
                query=str(row["question"]).strip(),
                relevant_ids=relevant_ids,
                answer=str(row.get("answer") or ""),
            )
        )

    return list(documents_by_id.values()), queries


def chunk_document(
    document: CorpusDocument,
    *,
    chunk_chars: int,
    chunk_overlap: int,
    max_doc_chars: int | None = None,
) -> list[IndexedChunk]:
    text = document.text[:max_doc_chars] if max_doc_chars else document.text
    text = text.strip()
    if not text:
        return []

    title = str(document.metadata.get("title") or "").strip()
    article_type = str(document.metadata.get("article_type") or "").strip()
    prefix_parts = []
    if title:
        prefix_parts.append(f"Title: {title}")
    if article_type:
        prefix_parts.append(f"Article type: {article_type}")
    prefix = "\n".join(prefix_parts)

    step = max(1, chunk_chars - chunk_overlap)
    chunks: list[IndexedChunk] = []
    for chunk_index, start in enumerate(range(0, len(text), step)):
        raw_chunk = text[start : start + chunk_chars].strip()
        if not raw_chunk:
            continue
        chunk_text = f"{prefix}\nContent:\n{raw_chunk}" if prefix else raw_chunk
        metadata = dict(document.metadata)
        metadata["doc_id"] = document.doc_id
        metadata["chunk_id"] = f"{document.doc_id}::chunk-{chunk_index}"
        metadata["chunk_index"] = chunk_index
        chunks.append(
            IndexedChunk(
                chunk_id=metadata["chunk_id"],
                doc_id=document.doc_id,
                text=chunk_text,
                metadata=metadata,
            )
        )
        if start + chunk_chars >= len(text):
            break
    return chunks


def aggregate_doc_ids(chunks: list[RetrievedChunk], top_k: int) -> list[str]:
    article_ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if not chunk.doc_id or chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        article_ids.append(chunk.doc_id)
        if len(article_ids) >= top_k:
            break
    return article_ids


def is_chroma_index_error(error: Exception) -> bool:
    message = str(error).lower()
    return "hnsw" in message and ("loading" in message or "segment reader" in message)


class ChromaIndexCorruptedError(RuntimeError):
    pass


def close_vector_store(vector_store) -> None:
    if vector_store is None:
        return
    client = getattr(vector_store, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def compute_metrics(
    cases: list[EvalCase],
    retrieved_by_case_id: dict[str, list[str]],
    top_k: int,
) -> dict:
    if not cases:
        return {
            "num_cases": 0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    recall_values = []
    precision_values = []
    hit_values = []
    mrr_values = []
    ndcg_values = []
    for case in cases:
        retrieved = retrieved_by_case_id.get(case.case_id, [])
        recall_values.append(recall_at_k(case.relevant_ids, retrieved, top_k))
        precision_values.append(precision_at_k(case.relevant_ids, retrieved, top_k))
        hit_values.append(1.0 if set(retrieved[:top_k]) & case.relevant_ids else 0.0)
        mrr_values.append(mrr_at_k(case.relevant_ids, retrieved, top_k))
        ndcg_values.append(ndcg_at_k(case.relevant_ids, retrieved, top_k))

    count = len(cases)
    return {
        "num_cases": count,
        "recall_at_k": sum(recall_values) / count,
        "precision_at_k": sum(precision_values) / count,
        "hit_at_k": sum(hit_values) / count,
        "mrr_at_k": sum(mrr_values) / count,
        "ndcg_at_k": sum(ndcg_values) / count,
    }


def resolve_dataset(dataset: str) -> tuple[list[CorpusDocument], list[EvalCase], str]:
    if dataset in {"wixqa_expertwritten", "wixqa_simulated"}:
        corpus_path = WIX_SNAPSHOT / "wix_kb_corpus" / "wix_kb_corpus.jsonl"
        query_path = WIX_SNAPSHOT / dataset / "test.jsonl"
        documents = list(load_wix_documents(corpus_path))
        queries = list(load_wix_queries(query_path))
        return documents, queries, str(query_path)

    if dataset == "techqa":
        dataset_path = TECHQA_SNAPSHOT / "train.json"
        documents, queries = load_techqa(dataset_path)
        return documents, queries, str(dataset_path)

    raise ValueError(f"未知外部评测集：{dataset}")


def build_or_load_store(
    *,
    dataset: str,
    documents: list[CorpusDocument],
    model_path: Path,
    chroma_dir: Path,
    rebuild: bool,
    max_doc_chars: int,
    chunk_chars: int,
    chunk_overlap: int,
):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    from model.factory import LocalBgeM3Embeddings

    persist_directory = chroma_dir / dataset

    def create_vector_store():
        return Chroma(
            collection_name=f"external_{dataset}",
            embedding_function=LocalBgeM3Embeddings(model_path),
            persist_directory=str(persist_directory),
        )

    if rebuild and persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = create_vector_store()

    try:
        collection_count = vector_store._collection.count()
    except Exception as exc:
        close_vector_store(vector_store)
        if not is_chroma_index_error(exc):
            raise
        raise ChromaIndexCorruptedError(
            f"检测到 Chroma HNSW 索引损坏：{persist_directory}\n"
            "请重新执行本命令并加上 --rebuild。不要在已打开坏库的同一进程里删除该目录。"
        ) from exc

    if rebuild or collection_count == 0:
        chunks = [
            chunk
            for doc in documents
            for chunk in chunk_document(
                doc,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
                max_doc_chars=max_doc_chars,
            )
        ]
        langchain_docs = [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        batch_size = 128
        for start in range(0, len(langchain_docs), batch_size):
            end = start + batch_size
            vector_store.add_documents(langchain_docs[start:end], ids=ids[start:end])
            print(f"已写入 {min(end, len(langchain_docs))}/{len(langchain_docs)} 个 chunk")

        # Chroma can serve queries from the current process even when the persisted
        # HNSW files are not yet readable by a fresh process. Reopen before eval so
        # reported metrics only use a verified on-disk index.
        close_vector_store(vector_store)
        vector_store = create_vector_store()
        try:
            reopened_count = vector_store._collection.count()
        except Exception as exc:
            close_vector_store(vector_store)
            if not is_chroma_index_error(exc):
                raise
            raise ChromaIndexCorruptedError(
                f"重建后仍无法重新打开 Chroma HNSW 索引：{persist_directory}"
            ) from exc
        if reopened_count == 0 and chunks:
            close_vector_store(vector_store)
            raise ChromaIndexCorruptedError(
                f"重建后 Chroma 集合为空，索引未正确落盘：{persist_directory}"
            )

    return vector_store


def retrieve_cases(
    vector_store,
    cases: list[EvalCase],
    top_k: int,
    candidate_k: int,
) -> dict[str, list[str]]:
    retrieved: dict[str, list[str]] = {}
    for index, case in enumerate(cases, start=1):
        docs_with_scores = vector_store.similarity_search_with_score(case.query, k=candidate_k)
        chunks = [
            RetrievedChunk(
                doc_id=str(doc.metadata.get("doc_id") or ""),
                score=float(score),
            )
            for doc, score in docs_with_scores
        ]
        retrieved[case.case_id] = aggregate_doc_ids(chunks, top_k)
        if index % 50 == 0 or index == len(cases):
            print(f"已检索 {index}/{len(cases)} 条查询")
    return retrieved


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存：{path}")


def save_markdown_report(path: Path, payload: dict) -> None:
    metrics = payload["metrics"]
    lines = [
        f"# External RAG Eval: {payload['dataset']}",
        "",
        f"- Source: `{payload['source']}`",
        f"- Cases: {metrics['num_cases']}",
        f"- K: {payload['top_k']}",
        f"- Candidate chunks: {payload['candidate_k']}",
        f"- Chunk chars: {payload['chunk_chars']}",
        f"- Chunk overlap: {payload['chunk_overlap']}",
        f"- Max doc chars: {payload['max_doc_chars'] or 'unlimited'}",
        f"- Doc limit: {payload['doc_limit'] or 'none'}",
        f"- Query limit: {payload['limit'] or 'none'}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Recall@K | {metrics['recall_at_k']:.1%} |",
        f"| Precision@K | {metrics['precision_at_k']:.1%} |",
        f"| Hit@K | {metrics['hit_at_k']:.1%} |",
        f"| MRR@K | {metrics['mrr_at_k']:.1%} |",
        f"| nDCG@K | {metrics['ndcg_at_k']:.1%} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已保存：{path}")


def output_stem(dataset: str, *, limit: int, doc_limit: int) -> str:
    suffix = "_sample" if limit or doc_limit else ""
    return f"external_eval_{dataset}{suffix}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external RAG eval on cached WixQA/TechQA.")
    parser.add_argument(
        "--dataset",
        choices=["wixqa_expertwritten", "wixqa_simulated", "techqa"],
        default="wixqa_expertwritten",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30, help="先检索多少个 chunk，再聚合为 top-k 文档。")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条，0 表示全量。")
    parser.add_argument("--doc-limit", type=int, default=0, help="只索引前 N 篇文档，0 表示全量。")
    parser.add_argument("--max-doc-chars", type=int, default=0, help="每篇文档最多索引多少字符，0 表示不截断。")
    parser.add_argument("--chunk-chars", type=int, default=1600, help="每个检索 chunk 的字符数。")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="相邻 chunk 的重叠字符数。")
    parser.add_argument("--rebuild", action="store_true", help="重建外部数据集 Chroma 库。")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    documents, queries, source = resolve_dataset(args.dataset)
    if args.doc_limit:
        documents = documents[: args.doc_limit]
    if args.limit:
        queries = queries[: args.limit]

    print(f"外部评测集: {args.dataset}")
    print(f"知识库文档数: {len(documents)}")
    print(f"评测问题数: {len(queries)}")
    print(f"来源: {source}")

    store_dataset = f"{args.dataset}_sample" if args.doc_limit else args.dataset
    vector_store = None
    try:
        vector_store = build_or_load_store(
            dataset=store_dataset,
            documents=documents,
            model_path=args.model_path,
            chroma_dir=args.chroma_dir,
            rebuild=args.rebuild,
            max_doc_chars=args.max_doc_chars,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
        )
        retrieved = retrieve_cases(vector_store, queries, args.top_k, args.candidate_k)
    finally:
        close_vector_store(vector_store)

    metrics = compute_metrics(queries, retrieved, args.top_k)
    details = [
        {
            "case_id": case.case_id,
            "query": case.query,
            "relevant_ids": sorted(case.relevant_ids),
            "retrieved_ids": retrieved.get(case.case_id, []),
        }
        for case in queries
    ]
    payload = {
        "dataset": args.dataset,
        "source": source,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "chunk_chars": args.chunk_chars,
        "chunk_overlap": args.chunk_overlap,
        "max_doc_chars": args.max_doc_chars,
        "doc_limit": args.doc_limit,
        "limit": args.limit,
        "metrics": metrics,
        "details": details,
    }

    stem = output_stem(args.dataset, limit=args.limit, doc_limit=args.doc_limit)
    json_path = args.output_dir / f"{stem}.json"
    report_path = args.output_dir / f"{stem}.md"
    save_results(json_path, payload)
    save_markdown_report(report_path, payload)

    print("\n外部评测完成")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
