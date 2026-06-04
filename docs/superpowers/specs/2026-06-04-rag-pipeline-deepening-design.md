# RAG Pipeline 深化设计

日期：2026-06-04

## 背景

当前项目已经具备 typed RAG 基础：`rag/query_router.py` 负责领域路由，`rag/retrieval_service.py` 按问题库、政策、排障、维护知识库检索，`rag/context_formatter.py` 格式化上下文，`rag/eval.py` 提供离线评估。

现有实现已经比简单 demo 更进一步，但主要瓶颈是检索链路仍偏规则化和单阶段化：

- 路由主要依赖关键词命中。
- 多个知识源的结果缺少统一 evidence 对象。
- 多路召回结果没有融合排序和显式重排阶段。
- 上下文构建缺少置信度、引用、证据分组和降级策略。
- 评估可以衡量召回和路由，但还不能完整解释 bad case 来自路由、召回、排序还是上下文噪声。

本设计目标是把现有 typed RAG 深化为可解释、可评估、可对比的多阶段检索系统，主服务于面试和答辩中的技术纵深展示，次要支持论文/课程项目式的实验对比。

## 目标

1. 将 RAG 链路从“关键词路由 + 向量召回”升级为多阶段 pipeline。
2. 让每个检索阶段都有明确输入、输出、分数和 trace，便于解释和调试。
3. 支持 baseline 与 enhanced pipeline 的离线指标对比。
4. 在客服场景中加入低置信度、检索为空、混合问题和证据冲突的降级策略。
5. 控制实现范围，第一版优先完成稳定、可测试、可展示的工程闭环。

## 非目标

1. 第一版不强依赖大模型 Query Rewrite、HyDE 或 cross-encoder rerank。
2. 第一版不引入复杂在线学习或用户反馈训练。
3. 第一版不重做 Streamlit UI，也不重构 Agent 主流程之外的无关模块。
4. 第一版不把 Chroma 替换成其他向量数据库。

## 推荐方案

采用“多阶段检索链路升级”为主线，吸收评估实验体系和少量领域元数据检索能力。

核心链路：

```text
QueryAnalyzer
  -> QueryRewriter
  -> MultiRetriever
  -> FusionRanker
  -> EvidenceReranker
  -> ContextBuilder
  -> RagSummarizer
```

这条路线适合当前项目，因为已有 `query_router.py`、`retrieval_service.py`、`context_formatter.py` 和 `eval.py` 可以作为 baseline 和改造入口，不需要推倒重来。

## 模块设计

### QueryAnalyzer

职责：分析用户问题，输出领域、意图、关键词、风险标记和是否需要澄清。

第一版保留现有关键词路由作为 baseline，同时新增结构化分析结果。分析可以先基于规则和领域词典实现，避免引入不稳定的大模型依赖。

输出字段：

- `original_query`
- `domains`
- `intent`
- `keywords`
- `needs_clarification`
- `risk_flags`

### QueryRewriter

职责：生成更适合检索的查询变体。

第一版输出：

- 原始 query。
- 标准化 query，例如补全领域词、统一 WiFi/配网等表达。
- 扩展 query，例如把“APP 搜不到设备”扩展为“配网失败、无法连接 WiFi、设备搜索不到”。

第一版不做 LLM rewrite 和 HyDE，但保留接口，后续可作为第二阶段增强。

### MultiRetriever

职责：从多个召回源取回候选证据，并统一转换为 `RetrievedEvidence`。

第一版召回源：

- 现有 Chroma 问题库召回。
- 现有政策、排障、维护 Chroma 召回。
- 结构化字段召回：对 JSONL 中的 `aliases`、`user_intents`、`symptom`、`topic`、`required_info` 等字段做轻量打分。

可选召回源：

- BM25 或本地 token overlap 召回。若新增依赖成本较高，第一版优先实现无依赖的轻量关键词打分。

### FusionRanker

职责：将多路召回结果融合、去重并排序。

第一版使用 RRF（Reciprocal Rank Fusion）。它适合答辩展示，因为可以解释为“不同召回器的排名投票”，比简单拼接结果更合理。

每条 evidence 保留：

- `vector_score`
- `field_score`
- `rrf_score`
- `matched_by`
- `source_store`
- `domain`

### EvidenceReranker

职责：对融合后的 top N evidence 进行二次排序。

第一版实现规则 rerank：

- 领域是否匹配 QueryAnalysis。
- 用户 query 与 evidence 的关键词覆盖率。
- 结构化字段是否命中。
- 问题库意图与领域证据是否一致。

接口预留后续接入 DashScope rerank、LLM pairwise rerank 或 cross-encoder rerank。

### ContextBuilder

职责：把最终 evidence 构造成给 LLM 的上下文。

第一版要求：

- 按领域分组：政策、排障、维护、相似问法。
- 对重复 evidence 去重。
- 控制上下文长度。
- 为每条最终证据生成引用编号，例如 `[policy_rules:policy_001]`。
- 区分“答案依据”和“问法参考”，相似问法不直接作为政策结论依据。

## 数据结构

### QueryAnalysis

```python
QueryAnalysis
- original_query: str
- domains: list[str]
- intent: str
- keywords: list[str]
- rewritten_queries: list[str]
- needs_clarification: bool
- risk_flags: list[str]
```

### RetrievedEvidence

```python
RetrievedEvidence
- evidence_id: str
- content: str
- domain: str
- source_store: str
- source_file: str
- doc_id: str
- metadata: dict
- scores: dict[str, float]
- matched_by: list[str]
```

### RetrievalTrace

```python
RetrievalTrace
- query_analysis: QueryAnalysis
- recall_counts: dict[str, int]
- fusion_strategy: str
- rerank_strategy: str
- selected_evidence_ids: list[str]
- dropped_evidence_ids: list[str]
- confidence: float
- flags: dict[str, bool]
```

`Document` 只作为 LangChain/Chroma 的底层输入输出。进入业务检索链路后，统一转换为 `RetrievedEvidence`，方便融合、重排、评估和日志追踪。

## 可信回答与降级策略

### 检索为空

如果领域证据为空，只能使用相似问法理解用户意图，不直接生成确定结论。回答应提示当前知识库没有直接依据，并要求用户补充型号、故障现象、购买时间或凭证等信息。

Trace 标记：`no_domain_evidence=True`。

### 低置信度

根据 top evidence 的 rerank 分数、关键词覆盖率、领域一致性计算 confidence。低于阈值时，系统不直接给确定结论，而是输出可能方向和需要确认的信息。

例如保修问题必须要求购买时间、凭证和故障原因，不能直接承诺免费维修。

### 混合问题

混合问题按领域拆分回答。比如“滤网堵了怎么清理，坏了算保修吗”应拆成维护建议和保修判断两个部分，分别引用维护和政策证据。

### 证据冲突

如果 evidence 存在冲突，优先级为：

```text
policy_rules > troubleshooting_cases > maintenance_guides > question_recall
```

`question_recall` 只作为意图参考，不作为最终政策依据。冲突时回答应提示以检测或人工核验为准。

## 评估设计

扩展现有 `rag/eval.py`，支持 baseline 与 enhanced pipeline 对比：

```powershell
python rag/eval.py --pipeline baseline
python rag/eval.py --pipeline enhanced
```

指标：

- Route Accuracy
- Recall@K
- Precision@K
- MRR@K
- nDCG@K
- Joint Hit@K

Bad case 分类：

- `route_error`：领域分析错误。
- `recall_miss`：相关证据没有被召回。
- `rank_error`：相关证据召回了但排序过低。
- `context_noise`：上下文中无关证据过多。

评估输出应包含总体指标和逐条 case 明细，便于展示一次检索为什么成功或失败。

## 测试策略

单元测试覆盖：

- `QueryAnalyzer`：保修、排障、维护、混合问题识别。
- `QueryRewriter`：标准化和扩展 query 是否稳定。
- `MultiRetriever`：不同召回源是否统一输出 `RetrievedEvidence`。
- `FusionRanker`：RRF 融合、去重、排序是否正确。
- `EvidenceReranker`：领域匹配、关键词覆盖、结构化字段命中是否影响排序。
- `ContextBuilder`：领域分组、引用编号、长度控制和降级提示。

离线评估覆盖：

- 单领域政策问题。
- 单领域排障问题。
- 单领域维护问题。
- 混合领域问题。
- 检索为空或低置信度问题。

## 展示方式

答辩或面试中建议展示三类材料：

1. 多阶段 RAG pipeline 架构图。
2. 一次检索 trace：输入问题、改写 query、多路召回、RRF 分数、rerank 分数、最终引用证据。
3. baseline vs enhanced 指标对比表。

推荐展示样例：

- “进水还能保修吗？”
- “APP 搜不到设备怎么办？”
- “滤网堵了怎么清理，坏了算保修吗？”

## 分阶段实施建议

第一阶段：

- 新增核心数据结构。
- 新增 QueryAnalyzer 和 QueryRewriter。
- 将现有 retriever 输出转换为 `RetrievedEvidence`。

第二阶段：

- 新增结构化字段召回。
- 新增 RRF FusionRanker。
- 新增规则 EvidenceReranker。

第三阶段：

- 改造 ContextBuilder。
- 增加置信度、引用和降级策略。
- 扩展 eval 支持 baseline/enhanced 对比。

第四阶段：

- 增加可选 LLM rewrite、HyDE 或模型 rerank 接口。
- 基于 bad case 做小规模实验对比。

## 风险与取舍

- 模块增多会提升复杂度，因此第一版必须通过清晰数据结构和单元测试控制边界。
- LLM rewrite 和模型 rerank 展示价值高，但不确定性和外部依赖更强，第一版只预留接口。
- 结构化字段召回能增强垂直场景表现，但不应替代向量召回，而应作为多路召回的一部分。
- 评估指标需要测试集质量支撑，后续应继续扩充 `data/test_queries.jsonl` 的场景覆盖。

## 验收标准

1. enhanced pipeline 能与 baseline pipeline 并存，并通过配置或命令行选择。
2. 检索链路输出 `RetrievalTrace`，包含分析、召回、融合、重排和最终证据信息。
3. 最终上下文包含引用编号和领域分组。
4. 低置信度、检索为空、混合问题和证据冲突有明确降级行为。
5. `rag/eval.py` 能输出 baseline/enhanced 对比指标和 bad case 分类。
6. 核心模块有单元测试覆盖。
