# Customer Agent: Traceable RAG Customer Support Agent Platform

面向扫地机器人售后场景的可追踪 RAG 智能客服 Agent 平台，集成 **ReAct Agent**、**多阶段 RAG**、**问题状态机**、**工具调用**、**人工升级**、**检索 trace** 与 **离线评估体系**，支持从用户咨询、证据检索、回答生成到问题流转的完整闭环。

> 本项目不是简单的聊天机器人 Demo，而是一个围绕售后服务流程设计的、可评估、可追踪、可部署的 Agent 系统工程。

## Highlights

| 维度 | 能力 |
|------|------|
| **Agent** | ReAct 决策、客户识别、问题状态机、补充信息请求、人工升级 |
| **RAG** | Query Analysis → Query Rewrite → 多源召回 → RRF Fusion → Rerank → Citation + Confidence |
| **Engineering** | `pyproject.toml` + `Makefile` + CI + Docker + Structured Logging + Tests |
| **Trace** | `trace_id` 串联 Agent/RAG/Tool 全链路，可查询、可回放 |
| **Evaluation** | Baseline / Enhanced / Ablation 对比，Recall@K, MRR@K, nDCG@K, Bad Case Taxonomy |

## Architecture

```mermaid
graph TD
    UI[Streamlit Web UI] --> API[FastAPI Backend]
    API --> Agent[Agent Runtime / State Machine]
    API --> RAG[RAG Engine: baseline | enhanced]
    API --> Tools[Tool Registry]
    Agent --> DB[(SQLite: sessions, issues, traces)]
    RAG --> DB
    Tools --> DB
    RAG --> Vector[(Chroma Vector Store)]
    DB --> Obs[Observability: trace, logs, metrics]
    Obs --> Eval[Offline Evaluation]
```

## Quick Start

```bash
# 1. Install
git clone <repo-url> && cd <repo>
cp .env.example .env   # 填入 DASHSCOPE_API_KEY 和 AMAP_API_KEY
make install

# 2. Run UI (standalone)
make run-ui

# 3. Or start API + UI
make run-api          # http://localhost:8000
make run-ui           # http://localhost:8501

# 4. Run tests
make test

# 5. Run evaluation
make eval-baseline
make eval-enhanced
make eval-report
```

## Demo Scenarios

| # | Scenario | Key Points |
|---|----------|------------|
| 1 | 保修政策咨询 | Policy 路由、证据引用、谨慎承诺 |
| 2 | 故障排查 | Troubleshooting 路由、相似问法召回、步骤生成 |
| 3 | 维护建议 | Maintenance 路由、结构化步骤 |
| 4 | 混合问题 | 多域拆分、分别引用证据 |
| 5 | 低置信度处理 | Fallback 策略、请求补充信息 |
| 6 | 人工升级 | 多轮失败累计 → Escalation |

## Project Structure

```text
Customer_Agent/
├── app.py                  # Streamlit UI 入口
├── api/                    # FastAPI 后端
│   ├── main.py
│   ├── routes/             # chat, issues, health, traces
│   └── schemas/            # Pydantic 请求/响应模型
├── agent/                  # ReAct Agent, 状态机, 工具, 中间件
│   ├── react_agent.py
│   ├── state.py
│   ├── response_policy.py
│   └── tools/
├── rag/                    # RAG Engine
│   ├── query_router.py
│   ├── retrieval_service.py
│   ├── vector_store.py
│   ├── eval.py
│   ├── eval_metrics.py
│   └── eval_report.py
├── tools/                  # Tool Registry (RAG, Weather, Report, Escalation)
├── storage/                # SQLite + SQLAlchemy persistence layer
│   ├── database.py
│   ├── models.py
│   └── repositories/
├── observability/          # Structured logging, tracing, error codes
├── data/
│   └── eval/               # Benchmark test queries
├── reports/                # Evaluation reports (generated)
├── docs/                   # Architecture, RAG pipeline, evaluation, deployment docs
├── tests/                  # Unit & integration tests
├── pyproject.toml          # Dependency & tool config
├── Makefile                # Unified command entry
├── Dockerfile
└── docker-compose.yml
```

## RAG Pipeline

| Stage | Baseline | Enhanced |
|-------|----------|----------|
| Query Analysis | Simple keyword | LLM-based route + intent |
| Query Rewrite | — | Multi-perspective rewriting |
| Retrieval | Single vector search | Multi-source: vector + keyword + structured |
| Fusion | — | RRF (Reciprocal Rank Fusion) |
| Rerank | — | Rule-based Evidence Reranker |
| Output | Raw chunks | Citations + confidence + flags |

## Evaluation

| Metric | Baseline | Enhanced |
|--------|----------|----------|
| Route Accuracy | TBD | TBD |
| Recall@5 | TBD | TBD |
| MRR@5 | TBD | TBD |
| nDCG@5 | TBD | TBD |
| Joint Hit@5 | TBD | TBD |

> 指标由 `make eval-report` 生成。运行前请确保已安装依赖并配置 API Key。

See [docs/evaluation.md](docs/evaluation.md) for full methodology, ablation study and bad case analysis.

## Roadmap

- [x] P0: Project foundation (`pyproject.toml`, `Makefile`, CI, README)
- [ ] P3-A: Evaluation MVP (benchmark dataset, baseline vs enhanced report)
- [ ] P1: FastAPI backend + SQLite persistence + issue state machine
- [ ] P2: Agent trace, tool registry, trace viewer UI
- [ ] P3-B: Full ablation study + bad case taxonomy
- [ ] P4: Docker Compose, demo screenshots, release

## Resume Highlights

**中文：**

> 面向扫地机器人售后场景，设计并实现集成 ReAct Agent、多阶段 RAG、问题状态机、工具调用、人工升级、检索 trace 与离线评估体系的智能客服系统。

- 设计多阶段 RAG pipeline：Query Analysis → Query Rewrite → 多源召回 → RRF Fusion → Rule Rerank → Citation + Confidence
- 构建客服 Issue Lifecycle State Machine，支持客户识别、问题跟踪、补充信息请求、人工升级
- 将 Streamlit Demo 重构为 FastAPI + SQLite 服务化架构，持久化会话、问题单、工具调用和检索 trace
- 建立离线评估体系，对 Baseline/Enhanced/Ablation Pipeline 进行 Route Accuracy、Recall@K、MRR@K、nDCG@K、Bad Case 分析

**English:**

> Built an engineering-oriented customer support Agent platform for robotic vacuum after-sales scenarios, integrating ReAct-style orchestration, multi-stage RAG, issue state tracking, tool calls, human escalation, retrieval trace logging, and reproducible offline evaluation.

- Designed a traceable multi-stage RAG pipeline with query analysis, query rewriting, multi-source retrieval, RRF fusion, rule-based reranking, citation generation, and confidence-aware fallback
- Implemented an issue lifecycle state machine for customer identification, troubleshooting progress tracking, clarification requests, resolution confirmation, and human escalation
- Refactored the demo-oriented Streamlit app into a service-oriented architecture with FastAPI, SQLite persistence, structured logs, and traceable Agent/tool/RAG execution records
- Built an offline evaluation framework comparing baseline, enhanced, and ablation pipelines using Route Accuracy, Recall@K, MRR@K, nDCG@K, Joint Hit@K, and bad-case taxonomy

## CI Status

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
