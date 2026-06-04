# 智扫通机器人智能客服项目

这是一个基于 `Streamlit + LangChain + RAG + DashScope + Chroma` 的智能客服演示项目，场景聚焦在扫地机器人售后咨询。系统把 ReAct Agent、知识库检索、客户编号识别、问题状态流转、定位天气上下文和外部报表查询组合在一起，既能跑页面演示，也能单独评估 RAG 检索链路。

这份 README 以“新同学拿到仓库后能直接跑起来”为目标编写。主示例环境是 `Windows + PowerShell`，`macOS/Linux` 只补充必要差异说明。

## 功能概览

- ReAct Agent 对话式客服
- 问题库与结构化知识库的分类型 RAG 检索
- 客户编号识别与聊天历史持久化
- 问题状态跟踪、已解决标记、升级人工提示
- 浏览器定位、城市天气上下文
- 外部 CSV 数据读取，用于报告类能力
- `rag/eval.py` 检索评估脚本

## 快速开始

### 1. 环境要求

- Python `3.11+`
- Windows + PowerShell
- 可访问 DashScope 和高德接口的网络环境
- 浏览器允许页面获取定位权限

如果你使用 `macOS/Linux`，主要差异只有虚拟环境激活命令和环境变量写法，后文给了简短示例。

### 2. 创建虚拟环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

`macOS/Linux` 可改为：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. 安装依赖

当前仓库里没有现成的 `requirements.txt` 或 `pyproject.toml`。按代码中的实际导入，至少需要安装这些包：

```powershell
pip install streamlit langchain langchain-core langchain-community langgraph langchain-chroma langchain-text-splitters chromadb dashscope pyyaml requests pypdf pytest
```

说明：

- `streamlit`：页面入口
- `langchain` / `langgraph`：Agent 与工具链
- `langchain-chroma` / `chromadb`：本地向量库
- `dashscope`：DashScope 模型 SDK
- `requests`：高德接口调用
- `pypdf`：PDF 知识文件加载
- `pytest`：测试与回归验证

## 启动前检查清单

### 1. `DASHSCOPE_API_KEY`

- 用途：`model/factory.py` 中的 `ChatTongyi` 聊天模型和 `DashScopeEmbeddings` 向量模型都依赖它。
- 配置方式：在启动 `streamlit` 前写入当前终端环境变量。
- 缺失现象：Agent 初始化失败，或向量库同步 / RAG 检索时模型调用失败。

### 2. `AMAP_API_KEY`

- 用途：`utils/amap_service.py` 会用它访问高德逆地理编码和实时天气接口。
- 配置方式：在启动 `streamlit` 前写入当前终端环境变量。
- 缺失现象：页面能打开，但会提示定位失败，城市和天气信息无法展示。

### 3. 浏览器定位权限

- 用途：`components/city_locator` 前端组件先拿到经纬度，再由后端查询城市与天气。
- 缺失现象：系统无法自动识别当前城市，只会显示定位失败或未获取城市信息。

### 4. `data/external/records.csv`

- 用途：Agent 的报表类工具会从这个 CSV 中读取用户月度记录。
- 缺失现象：普通售后问答通常还能运行，但涉及报告/外部数据的能力会报文件不存在错误。

### 5. 首次运行需要联网

- 用途：首次启动时会同步向量库，并调用 DashScope 模型与高德服务。
- 缺失现象：首次建库、聊天生成、定位天气查询都可能失败。

## 启动项目

先在当前终端设置环境变量：

```powershell
$env:DASHSCOPE_API_KEY="你的 DashScope Key"
$env:AMAP_API_KEY="你的高德 Key"
```

然后启动 Streamlit：

```powershell
streamlit run app.py
```

`macOS/Linux` 可改成：

```bash
export DASHSCOPE_API_KEY="你的 DashScope Key"
export AMAP_API_KEY="你的高德 Key"
streamlit run app.py
```

首次启动时，`app.py` 会先调用 `VectorStoreService.ensure_all_vector_stores_synced()`，自动检查并同步 `config/chroma.yml` 中定义的所有知识库到本地 `chroma_db/`。所以第一次通常会比后续启动慢，这是预期行为。

## 最小运行验证

页面打开后，建议按下面顺序验证：

1. 浏览器能正常打开 Streamlit 页面。
2. 首条消息先输入有效客户编号，例如 `1001`。
3. 再输入一个售后问题，例如“我的机器回不去充电座怎么办？”
4. 确认系统能返回回答，而不是初始化失败或报错。
5. 如果浏览器已允许定位，页面应显示当前城市和天气。
6. 检查 `data/chat_history.json` 与 `data/issue_status.json` 是否出现新的记录。

## 项目结构

```text
.
├── app.py                     # Streamlit 页面入口
├── agent/                     # ReAct Agent、工具函数、中间件
├── rag/                       # 路由、检索、向量库、上下文格式化、评估脚本
├── data/                      # 知识库、测试集、聊天历史、问题状态、外部 CSV
├── config/                    # 模型、向量库、提示词、外部数据路径配置
├── utils/                     # 定位、天气、日志、运行时上下文、问题单处理
├── prompts/                   # 主提示词、RAG 提示词、报告提示词
├── components/                # Streamlit 前端组件（如定位组件）
├── chroma_db/                 # 本地持久化向量库
├── tests/                     # 单元测试
└── docs/                      # 设计与计划文档
```

比较关键的目录再补充一句：

- `data/questions/`：相似问法问题库
- `data/policies/`：原始文本知识
- `data/structured_policies/`：结构化政策、故障排查、维护保养知识
- `config/chroma.yml`：定义每个向量库对应的数据目录、集合名和 `k` 值
- `config/rag.yml`：定义聊天模型和向量模型名称

## 核心运行流程

1. 启动 `app.py` 后，系统先同步全部向量库。
2. 页面收到用户消息后，先要求用户确认客户编号。
3. Agent 再结合聊天上下文调用 `rag_summarize`、天气、位置、外部数据等工具。
4. RAG 会先从问题库召回相似问法，再根据 `QueryRouter` 的路由结果去检索结构化知识域。
5. 聊天记录写入 `data/chat_history.json`，问题状态写入 `data/issue_status.json`。

## RAG 评估

如果你想单独评估检索效果，而不是只启动页面，可以运行：

```powershell
python rag/eval.py
```

也可以显式对比两条 RAG 管线：

```powershell
python rag/eval.py --pipeline baseline
python rag/eval.py --pipeline enhanced
```

`baseline` 使用原有分类型检索流程。`enhanced` 会启用查询分析、确定性查询改写、多源证据检索、RRF 融合、规则重排、证据引用和检索 Trace。两种模式都会输出 `Recall@K`、`Precision@K`、`MRR@K`、`nDCG@K`、路由准确率、`Joint Hit@K` 和坏例分类，便于横向比较检索质量。

这个脚本会读取 `data/test_queries.jsonl`，并输出：

- 问题库的 `Recall@K`、`Precision@K`、`Hit@K`、`Complete@K`、`F1@K`
- 知识域的同类指标
- 路由准确率
- 联合命中指标 `Joint Hit@K`

注意：`rag/eval.py` 依赖检索链路和模型能力已可用。如果 DashScope 模型或向量能力没有配置好，评估也无法正常运行。

## 常见问题

### 1. `ModuleNotFoundError` 或导入失败

- 现象：启动时提示 `streamlit`、`langchain`、`chromadb`、`pypdf` 等模块不存在。
- 原因：虚拟环境没激活，或者依赖没有装全。
- 处理：重新激活虚拟环境，再按上面的依赖列表补装。

### 2. Agent 或向量模型初始化失败

- 现象：页面启动时报 Agent 初始化失败，或同步向量库时报模型相关错误。
- 原因：通常是 `DASHSCOPE_API_KEY` 未配置，或者当前网络无法访问 DashScope。
- 处理：重新设置环境变量后重启终端和 Streamlit。

### 3. 页面提示定位失败

- 现象：页面显示“请允许浏览器定位权限”或“定位失败原因”。
- 原因：浏览器没授权定位、`AMAP_API_KEY` 未配置，或高德接口调用失败。
- 处理：先确认浏览器定位权限，再检查高德 Key 是否生效。

### 4. 首次启动特别慢

- 现象：页面长时间停留在“正在检查并同步知识库...”。
- 原因：系统在首次运行时会把知识文件同步进本地 `chroma_db/`。
- 处理：这是正常现象；后续启动一般会更快。

### 5. 报告相关能力异常

- 现象：涉及用户月度数据或报告的回答失败。
- 原因：`data/external/records.csv` 缺失，或者字段格式不符合工具读取要求。
- 处理：确认文件存在，且包含用户 ID、时间等必要字段。

## 测试

如果你要先确认仓库当前基线是否正常，可以在项目根目录运行：

```powershell
pytest
```

我在编写这份 README 时，当前仓库基线测试结果为 `36 passed`。

## 后续可改进项

- 补一份正式的 `requirements.txt` 或 `pyproject.toml`
- 为环境变量提供 `.env.example`
- 为 README 增加运行截图或流程图
