# 外部 RAG 评测数据接入设计

日期：2026-06-09

## 背景

当前项目是面向扫地机器人售后场景的可追踪 RAG 客服 Agent，已有离线评估入口和小规模测试集：

- `data/eval/test_queries.jsonl`
- `rag/eval.py`
- `rag/eval_report.py`
- `make eval-baseline`
- `make eval-enhanced`
- `make eval-report`

现有测试集规模偏小，难以证明检索、路由、引用和 fallback 策略在更大样本上的稳定性。本设计引入两个许可清晰、场景接近的外部 RAG 评测数据集，并保持本地手动下载，不在项目脚本中自动联网下载。

## 数据集选择

### WixQA

来源：https://huggingface.co/datasets/Wix/WixQA

用途：企业客服 RAG 主评测集。该数据集包含 Wix Help Center 知识库 corpus 和多类 QA 数据，QA 样本带有可用于回答的 KB 文档 ID，适合评估检索命中、排序质量和引用 grounding。

本项目优先使用：

- `wix_kb_corpus`
- expert-written QA
- simulated QA

`synthetic` QA 样本数量更大，作为后续扩容选项，不进入第一版默认评测。

### TechQA-RAG-Eval

来源：https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval

用途：技术支持 RAG 补充评测集。字段包括 `id`、`question`、`answer`、`is_impossible`、`contexts`，适合评估技术故障排查、不可回答问题和低置信度 fallback。

### 暂缓接入的数据

IBM 原版 TechQA 暂不接入第一版。原版数据和 TechNotes 规模更大，适合后续做大规模压力测试，但会显著增加下载、存储和适配复杂度。

Kaggle、Twitter 客服数据暂不接入第一版。它们更适合意图分类、对话流转或人工升级评估，但缺少稳定的 KB grounding，不适合作为第一版 RAG 召回评测主集。

## 用户下载责任

项目脚本不自动下载 Hugging Face 数据。用户手动下载后按以下目录放置：

```text
data/
  external/
    wixqa/
      raw/
        kb_corpus/
        expert_written/
        simulated/
    techqa_rag_eval/
      raw/
```

数据文件格式可以是 Hugging Face 导出的 `jsonl`、`json`、`parquet` 或 `csv`。第一版转换脚本优先支持 `jsonl` 和 `json`；如果用户下载的是 `parquet`，文档会说明需要先用 Hugging Face 网页或本地工具导出为 `jsonl`。

## 目标输出

转换脚本生成项目统一评测文件：

```text
data/eval/external_wixqa.jsonl
data/eval/external_techqa_rag_eval.jsonl
data/eval/robot_vacuum_gold.jsonl
```

其中 `robot_vacuum_gold.jsonl` 是预留文件，用于后续人工整理扫地机器人领域黄金集。第一版可以创建空模板或少量示例，不作为默认外部评测依赖。

## 统一样本格式

外部数据转换为兼容现有评估系统的扩展 JSONL schema：

```json
{
  "id": "wixqa-expert-0001",
  "source_dataset": "wixqa",
  "query": "How do I connect a domain to my Wix site?",
  "gold_answer": "Use the referenced Wix Help Center articles to connect or point the domain.",
  "gold_doc_ids": ["article_123"],
  "gold_contexts": ["..."],
  "expected_route": "troubleshooting",
  "expected_question_refs": [],
  "expected_domain_sections": [],
  "tags": ["external", "customer_support", "rag", "wixqa"]
}
```

兼容字段：

- `query`
- `expected_route`
- `expected_question_refs`
- `expected_domain_sections`

扩展字段：

- `id`
- `source_dataset`
- `gold_answer`
- `gold_doc_ids`
- `gold_contexts`
- `is_impossible`
- `tags`

保留兼容字段是为了不破坏 `rag/eval.py` 当前加载逻辑。新增字段用于后续按数据集分组、按 gold evidence 计算检索指标、分析不可回答样本。

## Route 映射

外部数据不完全对应扫地机器人售后 domain，因此第一版采用保守映射：

- WixQA：默认映射为 `policy` 或 `troubleshooting`，由简单关键词规则决定。
- TechQA-RAG-Eval：默认映射为 `troubleshooting`。
- `is_impossible=true`：增加 `fallback_expected` tag，并保留 `expected_route` 为 `troubleshooting`，避免要求现有路由器立即新增独立 route。

关键词规则只作为第一版弱标签，不作为最终业务真值。评估报告必须标注这些 route 标签为自动映射标签，避免把外部数据 route accuracy 解释成领域真实准确率。

## 评估设计

第一版保留现有评估入口，并增加外部数据集维度：

- 原有 `data/eval/test_queries.jsonl` 继续作为扫地机器人领域小集。
- `external_wixqa.jsonl` 作为企业客服 RAG 评测集。
- `external_techqa_rag_eval.jsonl` 作为技术支持 RAG 评测集。

报告按 `source_dataset` 分组输出：

- 样本数
- Route Accuracy
- Recall@K
- MRR@K
- nDCG@K
- Joint Hit@K
- 不可回答样本数量和命中情况
- bad case 分布

如果当前检索服务尚未索引外部 corpus，评估命令应明确失败并提示先运行外部数据准备流程，而不是静默得到全 0 指标。

## 组件边界

### 数据说明文档

新增 `docs/external_eval_datasets.md`，说明：

- 数据集来源和许可
- 下载哪些文件
- 放置目录
- 支持的本地文件格式
- 转换命令
- 评估命令
- 指标解释和限制

### 数据准备脚本

新增 `scripts/prepare_external_eval.py`，只读取本地文件，不自动下载。职责：

- 读取 `data/external/wixqa/raw`
- 读取 `data/external/techqa_rag_eval/raw`
- 校验必要字段
- 生成统一 JSONL
- 输出转换统计
- 对缺字段、空文件、无法识别格式给出明确错误

### 评估加载逻辑

扩展现有加载逻辑，使其能读取带扩展字段的 JSONL。现有四个必需字段仍然保留，避免破坏当前测试集。

### 报告逻辑

扩展 `rag/eval_report.py`，在报告中加入 dataset 分组和 bad case 汇总。第一版不引入 LLM judge，只计算可复现的检索和路由指标。

## 错误处理

数据准备阶段：

- 原始目录不存在：提示用户下载数据并放到指定目录。
- 文件格式不支持：提示转换为 `jsonl` 或 `json`。
- WixQA 缺少 `article_ids`：跳过该样本并计入 skipped。
- TechQA 缺少 `contexts`：样本仍可保留，但 `gold_contexts` 为空并打上 `missing_context` tag。
- `is_impossible=true`：保留样本，打上 `unanswerable` tag。

评估阶段：

- 输入文件为空：直接失败并提示。
- 所有样本缺少 gold evidence：报告标注该数据集不适合检索指标，只输出 route 和样本统计。
- 外部 corpus 未索引：失败并提示先执行外部数据准备和索引步骤。

## 测试策略

第一版测试聚焦转换和报告稳定性：

- 单元测试：WixQA 样本转换。
- 单元测试：TechQA-RAG-Eval 样本转换。
- 单元测试：缺字段和空文件错误提示。
- 单元测试：扩展 schema 仍兼容 `load_test_queries`。
- 集成测试：使用小型 fixture 生成 `external_wixqa.jsonl` 和 `external_techqa_rag_eval.jsonl`。
- 报告测试：确认按 `source_dataset` 分组输出。

不在第一版测试真实 Hugging Face 下载，因为项目选择本地手动下载模式。

## 非目标

第一版不做以下内容：

- 自动联网下载数据。
- 自动登录 Kaggle。
- 接入 Twitter 客服数据。
- 接入 IBM 原版完整 TechQA。
- 引入 LLM judge 进行答案语义打分。
- 大规模扫地机器人网页爬取。
- 改写现有 RAG 主架构。

## 验收标准

设计实现完成后应满足：

- 用户能按文档下载并放置 WixQA 和 TechQA-RAG-Eval。
- 本地转换命令能生成两个外部评测 JSONL。
- 原有 `data/eval/test_queries.jsonl` 不受影响。
- `make eval-baseline` 和 `make eval-enhanced` 仍能用于原始小集。
- 新增外部评测命令能对 WixQA 和 TechQA-RAG-Eval 分别输出报告。
- 报告明确区分领域小集、WixQA、TechQA-RAG-Eval。
- 文档明确说明外部 route 标签是自动映射弱标签。

## 实施顺序

1. 补充数据下载和目录说明文档。
2. 新增本地数据转换脚本和 fixture。
3. 扩展评估输入 schema 的兼容逻辑。
4. 扩展报告分组统计。
5. 增加测试。
6. 运行转换 fixture、评估测试和现有测试。

