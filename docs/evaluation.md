# Evaluation

## Dataset

Evaluation uses a structured JSONL benchmark at `data/eval/test_queries.jsonl`.

### Format

```json
{
  "id": "case_policy_001",
  "query": "机器进水了还能保修吗？",
  "difficulty": "medium",
  "expected_route": "policy",
  "expected_evidence_ids": ["policy_water_damage"],
  "expected_answer_points": [
    "不能直接承诺免费保修",
    "需要确认进水原因和检测结果",
    "建议联系售后检测"
  ],
  "risk_tags": ["warranty_commitment", "needs_evidence"],
  "requires_clarification": true
}
```

### Distribution

| Category | Count | Description |
|----------|------:|-------------|
| Policy (保修政策) | 10 | Warranty, water damage, out-of-warranty |
| Troubleshooting (故障排查) | 12 | APP pairing, charging, noise, map issues |
| Maintenance (维护建议) | 10 | Filter, brush, dust bin, sensor cleaning |
| Mixed (混合问题) | 8 | Maintenance + warranty, troubleshooting + escalation |
| Low Confidence (低置信度) | 5 | Vague descriptions, insufficient evidence |
| Escalation (人工升级) | 5 | Multi-round failure, complaints, complex repairs |

## Metrics

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| Route Accuracy | Correct routing to policy/troubleshooting/maintenance |
| Recall@K | Relevant evidence is in top K results |
| Precision@K | Fraction of top K results that are relevant |
| MRR@K | Mean reciprocal rank of first relevant result |
| nDCG@K | Normalized discounted cumulative gain |
| Joint Hit@K | Both question bank and knowledge domain results hit |

### Answer Metrics

| Metric | Description |
|--------|-------------|
| Answer Point Coverage | Expected answer points are covered |
| Citation Accuracy | Citations support the answer |
| Faithfulness | Answer is faithful to evidence |
| Safe Fallback Rate | Low-confidence cases correctly trigger fallback |

### Process Metrics

| Metric | Description |
|--------|-------------|
| Issue State Accuracy | State transitions are correct |
| Escalation Precision | Human escalation decisions are reasonable |

## Experiment Groups

| Group | Description |
|-------|-------------|
| `baseline` | Current typed retrieval |
| `enhanced` | Full multi-stage pipeline |
| `no_rewrite` | Enhanced without query rewriting |
| `no_structured` | Enhanced without structured field retrieval |
| `no_rerank` | Enhanced without evidence reranker |
| `vector_only` | Vector retrieval only |
| `question_only` | Similar question only |
| `domain_only` | Knowledge domain only |

## Running Evaluation

```bash
make eval-baseline     # Run baseline pipeline
make eval-enhanced     # Run enhanced pipeline
make eval-report       # Generate Markdown report
```

## Output

Reports are generated to `reports/`:
- `eval_results_baseline.json`
- `eval_results_enhanced.json`
- `ablation_results.json`
- `eval_report.md`
