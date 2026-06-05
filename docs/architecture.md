# Architecture

## System Overview

Customer Agent 采用分层架构：UI 层 → API 层 → 核心服务层 → 持久化层，每层职责清晰。

## Module Diagram

```
┌─────────────────────────────────┐
│     Streamlit Web UI            │  演示与交互层
│     (app.py)                    │
└─────────────┬───────────────────┘
              │ HTTP
┌─────────────▼───────────────────┐
│     FastAPI Backend             │  API 网关
│     /chat /issues /health       │
│     /traces /eval               │
└──────┬──────┬───────┬───────────┘
       │      │       │
┌──────▼──┐ ┌─▼────┐ ┌▼──────────┐
│ Agent   │ │ RAG  │ │ Tool      │  核心服务
│ Runtime │ │Engine│ │ Registry  │
└────┬────┘ └──┬───┘ └─────┬─────┘
     │         │            │
┌────▼─────────▼────────────▼─────┐
│     Persistence Layer           │  数据层
│     SQLite + Chroma             │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│     Observability & Eval        │  可观测与评估
│     trace / logs / metrics      │
└─────────────────────────────────┘
```

## Request Flow

1. User sends message via Streamlit UI
2. Streamlit calls `POST /chat` with `user_id`, `message`, `location`
3. FastAPI checks/create session, then dispatches to Agent Runtime
4. Agent Runtime evaluates current issue state, decides action
5. If RAG needed: RAG Engine runs query analysis → retrieval → fusion → rerank
6. If tool needed: Tool Registry executes and logs
7. Agent generates response via Response Policy
8. Results persisted: message, issue event, retrieval trace, tool call log
9. Response returned with `trace_id`, `issue_status`, `confidence`, `citations`

## Directory Structure

```
api/           FastAPI routes and Pydantic schemas
agent/         ReAct Agent, state machine, response policy
rag/           RAG pipeline: routing, retrieval, fusion, eval
tools/         Unified tool interface and registry
storage/       SQLAlchemy models and repositories
observability/ Structured logging, tracing, error codes
```

## Technology Choices

| Layer | Choice | Reason |
|-------|--------|--------|
| Web Framework | FastAPI | Type-safe, async, auto OpenAPI docs |
| Frontend | Streamlit | Rapid UI, focus on backend engineering |
| Database | SQLite + SQLAlchemy | Zero-config local dev, easy migration to Postgres |
| Vector Store | Chroma | Persistent, local, already integrated |
| LLM | DashScope (Tongyi) | Chinese-optimized, existing integration |
| CI | GitHub Actions | Zero-config for public repos |
