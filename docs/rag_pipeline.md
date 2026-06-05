# RAG Pipeline

## Overview

本项目的 RAG pipeline 支持两种模式：**baseline**（基础 typed retrieval）和 **enhanced**（多阶段可追踪检索）。

## Baseline Pipeline

```
User Query → QueryRouter (rule-based) → Vector Search → Context Format → LLM Answer
```

特点：
- 基于关键词的简单路由
- 单次向量检索
- 直接拼接上下文

## Enhanced Pipeline

```
User Query
  → QueryAnalyzer (LLM route + intent)
  → QueryRewriter (multi-perspective rewrite)
  → Multi-source Retrieval
      ├── Vector Search
      ├── Keyword Search
      └── Structured Field Retrieval
  → RRF Fusion
  → EvidenceReranker (rule-based)
  → ContextBuilder + Citation
  → LLM Answer + Confidence
```

## Pipeline Components

### QueryAnalyzer
Analyses user query to determine: route (policy/troubleshooting/maintenance), intent, and required info fields.

### QueryRewriter
Generates multiple rewritten versions of the query to improve recall for paraphrased questions.

### Multi-source Retrieval
- **Vector Search**: Dense retrieval via Chroma embeddings
- **Keyword Search**: BM25-style sparse retrieval
- **Structured Field Retrieval**: Direct lookup of structured policy fields (warranty, conditions, etc.)

### RRF Fusion
Reciprocal Rank Fusion combines results from multiple retrieval sources into a single ranked list.

### EvidenceReranker
Rule-based reranking considering:
- Source authority (policy > faq)
- Recency
- Query-answer semantic overlap
- Risk tags (warranty claims need authoritative sources)

### Citation
Each evidence piece gets a unique `citation_id` for traceability.

### Confidence
Calculated from retrieval scores, source consistency, and answer-evidence alignment.

## Retrieval Trace

Each retrieval produces a `RetrievalTrace`:

```python
class RetrievalTrace:
    trace_id: str
    query: str
    pipeline: str
    query_analysis: dict
    rewritten_queries: list[str]
    recall_counts: dict
    selected_evidence: list[RetrievedEvidence]
    confidence: float
    flags: dict[str, bool]
```

## Fallback Strategy

When confidence is below threshold:
1. Trigger conservative response template
2. Request clarification from user
3. Flag for human review
4. Do NOT fabricate unsupported claims
