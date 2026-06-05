# Interview Talking Points

## 推荐讲述主线

1. **业务问题**：售后客服不是简单聊天，而是要识别用户、理解问题、查知识、跟踪状态、必要时升级人工。
2. **系统设计**：把系统拆成 Agent Runtime、RAG Engine、Tool Registry、Persistence、Evaluation。
3. **RAG 深度**：不是单次向量检索，而是 query analysis → rewrite → 多源召回 → RRF → rerank → citation → confidence。
4. **工程化**：不是只做页面，而是补 API、数据库、CI、Docker、日志、测试。
5. **研究潜力**：构造 benchmark，对 baseline/enhanced/ablation 做指标对比和 bad case 分析。
6. **结果展示**：给出架构图、trace 示例、评估表、典型 case 演示。
7. **反思**：真实业务还需要权限、工单系统、反馈学习、监控告警和更大规模数据。

## 常见追问

### Q1：为什么不用简单向量检索？

售后问题有 policy、troubleshooting、maintenance 等不同知识域。相似问法不一定等于可靠政策依据。多阶段 RAG 可以先判断 query 类型，再做多源召回、融合、重排和引用。评估中通过 baseline/enhanced 对比验证改动价值。

### Q2：如何减少幻觉？

- 回答必须基于 retrieved evidence
- 高风险保修类问题走 response policy
- 低置信度触发 fallback，不直接给确定结论
- 引用 citation，trace 中可追溯证据

### Q3：为什么要做状态机？

客服不是单轮问答，真实流程包括识别用户、补充信息、解决、确认、升级人工。状态机让流程可控、可测试、可持久化。issue_events 可以回放问题处理过程。

### Q4：如何评价 RAG 效果？

- 构建售后场景 JSONL benchmark
- 检索层用 Route Accuracy、Recall@K、MRR、nDCG、Joint Hit
- 回答层看 Answer Point Coverage、Citation Accuracy、Safe Fallback
- 额外做 ablation 和 bad case taxonomy

### Q5：如果线上化还缺什么？

- 权限和用户认证
- 真实工单系统集成
- 反馈学习闭环
- 更严格监控告警
- 大规模并发和成本控制
- 数据脱敏和合规
