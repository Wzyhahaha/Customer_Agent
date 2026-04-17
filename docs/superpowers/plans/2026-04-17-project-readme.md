# Project README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为当前仓库补一份可直接指导新同学在本地跑通项目的 `README.md`，覆盖环境准备、依赖安装、密钥配置、启动命令、RAG 评估与常见问题。

**Architecture:** 先从现有代码中核实依赖、环境变量、启动入口和知识库同步行为，再把这些信息压缩成“双入口型 README”：前部解决快速上手，后部补足项目结构和排障。文档以 `Windows + PowerShell` 为主示例，其他系统只保留简短提示，避免 README 变成泛化模板。

**Tech Stack:** Markdown, Streamlit, LangChain, LangGraph, DashScope, Chroma, PyYAML, PowerShell

---

### Task 1: 核对启动前置条件与运行命令

**Files:**
- Modify: `README.md`
- Reference: `app.py`
- Reference: `model/factory.py`
- Reference: `utils/amap_service.py`
- Reference: `rag/vector_store.py`
- Reference: `rag/eval.py`

- [ ] **Step 1: 确认模型与地图服务的环境变量来源**

Run:
```powershell
rg -n "DASHSCOPE|AMAP|API_KEY|getenv|os\.getenv" -S model utils app.py rag
```

Expected:
```text
utils\amap_service.py 中出现 AMAP_API_KEY
model\factory.py 显示使用 DashScope 的聊天模型和嵌入模型
```

- [ ] **Step 2: 确认应用入口和首次知识库同步行为**

Run:
```powershell
Get-Content app.py -Encoding utf8 -Raw | Out-String -Width 4096
Get-Content rag/vector_store.py -Encoding utf8 -Raw | Out-String -Width 4096
```

Expected:
```text
app.py 在 Agent 初始化前调用 initialize_vector_store_once()
VectorStoreService.ensure_all_vector_stores_synced() 会在应用启动时同步全部向量库
```

- [ ] **Step 3: 确认评估入口和测试集来源**

Run:
```powershell
Get-Content rag/eval.py -Encoding utf8 -Raw | Out-String -Width 4096
```

Expected:
```text
评估入口是 python rag/eval.py
测试集文件路径是 data/test_queries.jsonl
输出包含问题库、知识域、路由与联合指标
```

- [ ] **Step 4: 列出 README 中必须明确声明的前置依赖**

Write this checklist into the README draft:
```markdown
- Python 3.11+
- Windows + PowerShell（主示例）
- DashScope 访问能力与 `DASHSCOPE_API_KEY`
- 高德地图访问能力与 `AMAP_API_KEY`
- `data/external/records.csv` 外部报表数据
- 首次运行可联网访问模型与地图服务
```

- [ ] **Step 5: 提交前置条件核对结果**

```bash
git add README.md
git commit -m "docs: document project prerequisites"
```

### Task 2: 编写快速开始与配置检查清单

**Files:**
- Create: `README.md`
- Reference: `config/rag.yml`
- Reference: `config/chroma.yml`
- Reference: `config/agent.yml`

- [ ] **Step 1: 写出 README 标题、项目简介和功能概览**

Add this content near the top of `README.md`:
```markdown
# 智扫通机器人智能客服项目

这是一个基于 `Streamlit + LangChain + RAG + DashScope + Chroma` 的智能客服演示项目，面向扫地机器人售后与问答场景。系统支持对话式客服、知识库检索、客户编号识别、问题状态流转、定位天气上下文和 RAG 检索评估。

## 功能概览

- ReAct Agent 对话式问答
- 问题库与结构化知识库的分类型 RAG 检索
- 客户编号识别与聊天历史持久化
- 问题单状态管理与人工升级提示
- 城市定位、天气上下文与外部报表查询
- `rag/eval.py` 检索评估脚本
```

- [ ] **Step 2: 写出 Windows 主路径的快速开始**

Add this section to `README.md`:
```markdown
## 快速开始

### 1. 环境要求

- Python 3.11 或更高版本
- Windows + PowerShell（以下命令以此环境为主）
- 可访问 DashScope 与高德接口的网络环境

### 2. 创建虚拟环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. 安装依赖

当前仓库未提供统一的 `requirements.txt` 或 `pyproject.toml`。按现有代码，至少需要安装以下依赖：

```powershell
pip install streamlit langchain langchain-community langchain-core langgraph langchain-chroma langchain-text-splitters chromadb dashscope pyyaml requests pypdf pytest
```
```

- [ ] **Step 3: 写出密钥与配置检查清单**

Add this section to `README.md`:
```markdown
## 启动前检查清单

### 1. `DASHSCOPE_API_KEY`

- 用途：用于 `ChatTongyi` 聊天模型和 `DashScopeEmbeddings` 向量模型。
- 配置方式：在启动 `streamlit` 前写入终端环境变量。
- 缺失现象：Agent 初始化失败，或向量库同步失败。

### 2. `AMAP_API_KEY`

- 用途：用于逆地理编码和实时天气查询。
- 配置方式：在启动 `streamlit` 前写入终端环境变量。
- 缺失现象：页面可打开，但会提示定位/天气获取失败。

### 3. 浏览器定位权限

- 用途：让前端组件拿到用户经纬度。
- 缺失现象：无法自动识别当前城市，只能看到定位失败提示。

### 4. `data/external/records.csv`

- 用途：报告相关工具从外部数据表中读取用户月度记录。
- 缺失现象：报告类工具调用失败，日志中会出现文件不存在错误。
```

- [ ] **Step 4: 写出启动命令与最小验证步骤**

Add this section to `README.md`:
```markdown
### 4. 启动项目

```powershell
$env:DASHSCOPE_API_KEY="你的 DashScope Key"
$env:AMAP_API_KEY="你的高德 Key"
streamlit run app.py
```

首次启动时，应用会自动检查并同步 `config/chroma.yml` 中定义的所有知识库到本地 `chroma_db/`，因此速度通常比后续启动慢。

### 5. 最小运行验证

启动后，按以下顺序验证：

1. 浏览器能正常打开 Streamlit 页面。
2. 首条消息先输入有效客户编号，例如 `1001`。
3. 再输入一个售后问题，确认系统能返回回答。
4. 若已允许定位权限，页面应显示城市和天气信息。
5. 检查 `data/chat_history.json` 与 `data/issue_status.json` 是否有更新记录。
```

- [ ] **Step 5: 提交快速开始文档**

```bash
git add README.md
git commit -m "docs: add readme quickstart"
```

### Task 3: 补充项目结构、评估说明与常见问题

**Files:**
- Modify: `README.md`
- Reference: `agent/react_agent.py`
- Reference: `agent/tools/agent_tools.py`
- Reference: `rag/retrieval_service.py`
- Reference: `utils/city_locator_component.py`

- [ ] **Step 1: 写出核心目录结构**

Append this section to `README.md`:
```markdown
## 项目结构

```text
.
├── app.py                     # Streamlit 入口
├── agent/                     # ReAct Agent 与工具、中间件
├── rag/                       # 路由、检索、向量库与评估
├── data/                      # 知识库、测试集、聊天历史、问题状态、外部数据
├── config/                    # 模型、向量库、提示词和外部数据配置
├── utils/                     # 定位、天气、日志、运行时上下文、问题单处理
├── chroma_db/                 # 本地持久化向量库
└── docs/                      # 设计与计划文档
```
```

- [ ] **Step 2: 写出核心运行流程与 RAG 评估说明**

Append this content to `README.md`:
```markdown
## 核心运行流程

1. 启动 `app.py` 后，系统会先同步全部向量库。
2. 页面收到用户消息后，先确认客户编号并装载历史对话。
3. Agent 调用 `rag_summarize`、天气、位置、外部数据等工具完成回答。
4. RAG 会先从问题库召回相似问法，再按路由结果检索结构化知识域。
5. 对话与问题状态会持久化到本地 JSON 文件中。

## RAG 评估

运行命令：

```powershell
python rag/eval.py
```

评估脚本会读取 `data/test_queries.jsonl`，输出：

- 问题库 Recall / Precision / Hit / Complete / F1
- 知识域 Recall / Precision / Hit / Complete / F1
- 路由准确率
- 联合命中率
```

- [ ] **Step 3: 写出常见问题与排障说明**

Append this content to `README.md`:
```markdown
## 常见问题

### 1. `ModuleNotFoundError`

- 原因：依赖未安装完整。
- 处理：重新检查虚拟环境是否激活，并补装缺失的包。

### 2. Agent 或向量模型初始化失败

- 原因：通常是 `DASHSCOPE_API_KEY` 未配置，或当前网络无法访问 DashScope。
- 处理：重新设置环境变量后重启终端和 Streamlit。

### 3. 页面提示定位失败

- 原因：浏览器未授权定位，或 `AMAP_API_KEY` 未配置，或高德接口调用失败。
- 处理：先确认浏览器定位权限，再检查高德 Key。

### 4. 首次启动耗时较长

- 原因：系统会自动检查并同步知识库到 `chroma_db/`。
- 处理：这是预期行为，后续启动通常会更快。

### 5. 报告相关功能报错

- 原因：`data/external/records.csv` 缺失或内容格式不符合预期。
- 处理：确认文件存在，且包含用户 ID、时间等必要字段。
```

- [ ] **Step 4: 运行文档一致性检查**

Run:
```powershell
Test-Path README.md
rg -n "DASHSCOPE_API_KEY|AMAP_API_KEY|streamlit run app.py|python rag/eval.py" README.md
```

Expected:
```text
README.md 存在
README 中能找到环境变量、启动命令和评估命令
```

- [ ] **Step 5: 提交完整 README**

```bash
git add README.md
git commit -m "docs: add project readme"
```
