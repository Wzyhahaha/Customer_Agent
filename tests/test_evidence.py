from langchain_core.documents import Document

from rag.evidence import evidence_from_document, stable_doc_id


def test_stable_doc_id_prefers_metadata_id():
    doc = Document(page_content="content", metadata={"id": "policy_001"})

    assert stable_doc_id(doc) == "policy_001"


def test_evidence_from_document_uses_source_name_and_domain():
    doc = Document(
        page_content="主题：是否在保修期内",
        metadata={"id": "policy_001", "source": "data/structured_policies/policy_rules.jsonl", "scene": "保修判断"},
    )

    evidence = evidence_from_document(doc, source_store="policy_rules", domain="policy", matched_by="vector")

    assert evidence.evidence_id == "policy_rules:policy_001"
    assert evidence.source_file == "policy_rules.jsonl"
    assert evidence.domain == "policy"
    assert evidence.doc_id == "policy_001"
    assert evidence.matched_by == ["vector"]
    assert evidence.metadata["scene"] == "保修判断"


def test_evidence_from_document_uses_source_name_from_windows_path():
    doc = Document(
        page_content="content",
        metadata={"id": "policy_001", "source": r"C:\data\structured_policies\policy_rules.jsonl"},
    )

    evidence = evidence_from_document(doc, source_store="policy_rules", domain="policy", matched_by="vector")

    assert evidence.source_file == "policy_rules.jsonl"


def test_evidence_from_document_copies_metadata_top_level():
    doc = Document(page_content="content", metadata={"id": "policy_001", "scene": "warranty"})

    evidence = evidence_from_document(doc, source_store="policy_rules", domain="policy", matched_by="vector")
    evidence.metadata["scene"] = "changed"

    assert doc.metadata["scene"] == "warranty"


def test_evidence_from_document_records_float_score_when_provided():
    doc = Document(page_content="content", metadata={"id": "policy_001"})

    evidence = evidence_from_document(
        doc,
        source_store="policy_rules",
        domain="policy",
        matched_by="vector",
        score_name="vector_score",
        score_value="0.7",
    )

    assert evidence.scores == {"vector_score": 0.7}
    assert isinstance(evidence.scores["vector_score"], float)


def test_evidence_from_document_leaves_scores_empty_when_not_provided():
    doc = Document(page_content="content", metadata={"id": "policy_001"})

    evidence = evidence_from_document(doc, source_store="policy_rules", domain="policy", matched_by="vector")

    assert evidence.scores == {}
