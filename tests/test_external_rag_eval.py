import json
import shutil
from pathlib import Path

from scripts import run_external_rag_eval as external_eval


FIXTURE_DIR = Path(__file__).resolve().parent / "_external_eval_tmp"


def make_fixture_dir(name: str) -> Path:
    path = FIXTURE_DIR / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_wixqa_reads_corpus_and_queries():
    fixture_dir = make_fixture_dir("wixqa")
    corpus_path = fixture_dir / "corpus.jsonl"
    query_path = fixture_dir / "test.jsonl"
    write_jsonl(
        corpus_path,
        [
            {
                "id": "article-1",
                "title": "Article One",
                "contents": "Article body",
                "url": "https://example.test/1",
            }
        ],
    )
    write_jsonl(
        query_path,
        [
            {
                "question": "How do I do this?",
                "answer": "Use article one.",
                "article_ids": ["article-1"],
            }
        ],
    )

    documents = list(external_eval.load_wix_documents(corpus_path))
    queries = list(external_eval.load_wix_queries(query_path))

    assert documents[0].doc_id == "article-1"
    assert documents[0].text == "Article body"
    assert queries[0].query == "How do I do this?"
    assert queries[0].relevant_ids == {"article-1"}


def test_load_techqa_builds_context_corpus_and_skips_impossible_by_default():
    fixture_dir = make_fixture_dir("techqa")
    dataset_path = fixture_dir / "train.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Why did it fail?",
                    "answer": "Because config is missing.",
                    "is_impossible": False,
                    "contexts": [{"filename": "ctx1.txt", "text": "Config docs"}],
                },
                {
                    "id": "q2",
                    "question": "Unknown?",
                    "answer": "",
                    "is_impossible": True,
                    "contexts": [],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    documents, queries = external_eval.load_techqa(dataset_path)

    assert documents[0].doc_id == "ctx1.txt"
    assert queries[0].case_id == "q1"
    assert queries[0].relevant_ids == {"ctx1.txt"}
    assert len(queries) == 1


def test_compute_metrics_scores_retrieval_ranking():
    cases = [
        external_eval.EvalCase("q1", "first", {"a"}),
        external_eval.EvalCase("q2", "second", {"c"}),
    ]
    retrieved = {
        "q1": ["b", "a"],
        "q2": ["c", "d"],
    }

    results = external_eval.compute_metrics(cases, retrieved, top_k=2)

    assert results["num_cases"] == 2
    assert results["recall_at_k"] == 1.0
    assert results["precision_at_k"] == 0.5
    assert results["hit_at_k"] == 1.0
    assert results["mrr_at_k"] == 0.75


def test_chunk_document_prefixes_title_and_preserves_article_id():
    document = external_eval.CorpusDocument(
        doc_id="article-1",
        text="abcdefghij",
        metadata={"title": "Useful Title", "article_type": "article", "doc_id": "article-1"},
    )

    chunks = external_eval.chunk_document(document, chunk_chars=4, chunk_overlap=1)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["doc_id"] == "article-1" for chunk in chunks)
    assert chunks[0].text.startswith("Title: Useful Title\nArticle type: article\nContent:\n")
    assert chunks[1].text.endswith("defg")


def test_aggregate_doc_ids_deduplicates_chunk_results_in_rank_order():
    retrieved_chunks = [
        external_eval.RetrievedChunk("article-a", 0.1),
        external_eval.RetrievedChunk("article-b", 0.2),
        external_eval.RetrievedChunk("article-a", 0.3),
        external_eval.RetrievedChunk("article-c", 0.4),
    ]

    article_ids = external_eval.aggregate_doc_ids(retrieved_chunks, top_k=2)

    assert article_ids == ["article-a", "article-b"]


def test_output_stem_marks_sample_runs():
    assert external_eval.output_stem("wixqa_expertwritten", limit=0, doc_limit=0) == (
        "external_eval_wixqa_expertwritten"
    )
    assert external_eval.output_stem("wixqa_expertwritten", limit=5, doc_limit=0) == (
        "external_eval_wixqa_expertwritten_sample"
    )


def test_chroma_index_error_detection():
    error = RuntimeError("Error loading hnsw index")

    assert external_eval.is_chroma_index_error(error)
    assert not external_eval.is_chroma_index_error(RuntimeError("other failure"))


def test_close_vector_store_calls_underlying_client_close():
    class FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeVectorStore:
        def __init__(self):
            self._client = FakeClient()

    vector_store = FakeVectorStore()

    external_eval.close_vector_store(vector_store)

    assert vector_store._client.closed
