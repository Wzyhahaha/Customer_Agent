# RAG Eval 失真修复设计

## 背景

当前 `rag.eval` 分数很差，不是单一召回能力下降，而是评估目标、路由策略和结构化知识装载同时错位。

已经确认的直接问题有三类：

- 评估集仍包含大量旧维护保养样本，但新的结构化知识只覆盖保修规则和故障排查。
- `QueryRouter` 关键词范围过窄，很多真实售后问法会落到 `other`，进而跳过结构化检索。
- `policy_rules` 与 `troubleshooting_cases` 共用同一个数据目录，但向量库装载逻辑没有按文件名或知识类型过滤，导致双库串库。

这三个问题叠加后，`rag.eval` 的低分主要反映“评估口径失真”和“数据污染”，并不能准确反映当前结构化检索链路的真实质量。

## 目标

- 修复结构化双库串库，保证 `policy_rules` 和 `troubleshooting_cases` 只加载各自数据。
- 扩大路由覆盖，减少明显售后问题被误判为 `other`。
- 将 `data/test_queries.jsonl` 改成与当前结构化知识覆盖一致的评估集。
- 让 `rag.eval` 显式校验样本期望路由，避免“路由错了但评估静默通过”。

## 非目标

- 不在这次修复中扩充大规模维护保养知识。
- 不在这次修复中重构生成层 Prompt 或新增 rerank。
- 不在这次修复中解决 DashScope 网络失败问题，只把离线评估目标校正到当前知识覆盖。

## 方案

### 1. 结构化双库隔离

在 `VectorStoreService` 增加按 store 过滤知识文件的能力：

- `policy_rules` 只加载 `policy_rules.jsonl`
- `troubleshooting_cases` 只加载 `troubleshooting_cases.jsonl`
- 其他 store 继续保持原有行为

这样即使两个文件仍位于同一目录，也不会互相写入错误集合。

### 2. 路由覆盖修正

保留轻量关键词路由，但补充当前结构化知识真实覆盖到的保修和排障问法，重点覆盖：

- 非保修因素：进水、跌落、摔坏、私拆、外力损坏
- WiFi 故障别名：配网失败、搜不到设备、无法联网
- 排障词：处理、排查、清理后仍异常等

不把这次修复扩展成复杂分类器，避免超出最小闭环。

### 3. 评估集重写

`data/test_queries.jsonl` 改为只包含当前结构化知识能支撑的样本，覆盖：

- `policy`
- `troubleshooting`
- `mixed`

每条样本新增 `expected_route`，并继续保留：

- `expected_question_refs`
- `expected_policy_sections`

这样评估分数才和当前系统目标一致。

### 4. 评估校验增强

`rag.eval` 增加对 `expected_route` 的加载和统计：

- 若样本声明了 `expected_route`，则计算路由命中情况
- 评估输出增加路由准确率
- 详情中记录 `expected_route` 与 `actual_route`

这样可以把“路由错导致召回空”的问题单独暴露出来。

## 测试策略

采用最小 TDD 闭环：

- `tests/test_vector_store_service.py`
  验证结构化双库只保留各自目标文件。
- `tests/test_query_router.py`
  验证新增保修/排障问法不会被误判为 `other`。
- `tests/test_eval.py`
  验证评估样本支持 `expected_route`，并返回路由统计。

最后运行针对性单测和完整相关测试集。

## 风险与取舍

- 这次把评估集收缩到当前知识覆盖，会让分数更能解释当前系统状态，但不能代表维护保养场景已被解决。
- 路由仍是规则式方案，后续若结构化知识继续扩张，仍需要重新审视分类边界。
- 串库修复后需要重建 `policy_rules` 与 `troubleshooting_cases` 库，否则历史污染仍在本地 Chroma 中残留。
