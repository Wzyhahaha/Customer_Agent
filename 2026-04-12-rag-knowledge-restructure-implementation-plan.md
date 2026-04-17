# RAG Knowledge Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前统一政策库的两阶段 RAG 改造成“意图分流 + 规则库/排障库分库检索 + 受控回答”的通用型扫地机器人客服系统。

**Architecture:** 保留现有 `question_recall` 作为问法归一与辅助路由层，新建结构化 `policy_rules` 与 `troubleshooting_cases` 两个知识库，并新增轻量 `Query Router` 与独立于 LLM 的 `TypedRetrievalService`。`RagSummarizeService` 只负责“检索结果格式化 + 提示词受控生成”，评估脚本直接依赖检索服务，不再因聊天模型依赖失败而无法运行。

**Tech Stack:** Python 3.12, LangChain, Chroma, Streamlit, built-in `unittest`, JSONL knowledge files, YAML config

---

## File Structure

### New Files

- `data/structured_policies/policy_rules.jsonl`
  第一批规则库知识，覆盖保修期判断、免费维修边界、非保修因素、耗材边界、第三方配件影响、维修方式建议。
- `data/structured_policies/troubleshooting_cases.jsonl`
  第一批排障库知识，覆盖无法联网、无法回充、不出水、吸力下降、地图异常、开机无反应等高频问题。
- `rag/query_router.py`
  轻量查询分流器，负责输出 `policy`、`troubleshooting`、`mixed`、`other`。
- `rag/retrieval_service.py`
  检索聚合层，封装路由、相似问法召回、规则库检索、排障库检索。
- `rag/context_formatter.py`
  将规则库、排障库、相似问法统一格式化为给 LLM 的上下文。
- `tests/__init__.py`
  让 `python -m unittest` 能从 `tests` 包执行。
- `tests/test_query_router.py`
  验证路由器对规则类、排障类、混合类查询的分类结果。
- `tests/test_vector_store_service.py`
  验证结构化知识不被错误切分、老库仍保留原有切分策略。
- `tests/test_retrieval_service.py`
  验证分流后只调用正确知识库、混合问题会并行检索两侧。
- `tests/test_context_formatter.py`
  验证上下文格式按库类型分段输出。
- `tests/test_eval.py`
  验证评估脚本不依赖聊天模型即可运行。

### Modified Files

- `config/chroma.yml`
  新增 `policy_rules` 与 `troubleshooting_cases` 两个 store 配置。
- `rag/vector_store.py`
  为结构化知识库增加“不切分直接入库”的路径。
- `rag/rag_service.py`
  改为依赖 `TypedRetrievalService` 和 `context_formatter`，只保留生成职责。
- `rag/eval.py`
  去掉对 `RagSummarizeService` 的直接依赖，改用检索服务。
- `agent/tools/agent_tools.py`
  如有需要，仅保留 `rag_summarize` 工具接口不变，内部仍返回统一字符串。
- `prompts/rag_summarize.txt`
  增加路由感知和规则类/排障类回答约束。

## Task 1: Build Query Router

**Files:**
- Create: `rag/query_router.py`
- Create: `tests/__init__.py`
- Create: `tests/test_query_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_router.py
import unittest

from rag.query_router import QueryRoute, QueryRouter


class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()

    def test_route_policy_query(self):
        route = self.router.route("这个机器还在保修期吗，能不能免费修？")
        self.assertEqual(route.route, "policy")

    def test_route_troubleshooting_query(self):
        route = self.router.route("机器人连不上 WiFi，APP 也搜不到设备怎么办？")
        self.assertEqual(route.route, "troubleshooting")

    def test_route_mixed_query(self):
        route = self.router.route("机器不出水，如果是正常使用坏的还能保修吗？")
        self.assertEqual(route.route, "mixed")

    def test_route_other_query(self):
        route = self.router.route("谢谢你，问题解决了")
        self.assertEqual(route.route, "other")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_query_router -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.query_router'`

- [ ] **Step 3: Write minimal implementation**

```python
# rag/query_router.py
from dataclasses import dataclass


POLICY_KEYWORDS = {
    "保修", "保修期", "免费修", "免费维修", "在保", "过保", "人为损坏", "非保修",
    "耗材", "第三方", "官方维修", "寄修", "上门维修",
}

TROUBLESHOOTING_KEYWORDS = {
    "怎么办", "不出水", "回不去充电座", "回充", "连不上", "wifi", "吸力下降",
    "地图", "异响", "报错", "开不了机", "故障", "不工作", "暂停",
}


@dataclass(frozen=True)
class QueryRoute:
    route: str
    reason: str


class QueryRouter:
    def route(self, query: str) -> QueryRoute:
        normalized = (query or "").strip().lower()
        if not normalized:
            return QueryRoute(route="other", reason="empty")

        policy_hit = any(keyword in normalized for keyword in POLICY_KEYWORDS)
        troubleshooting_hit = any(keyword in normalized for keyword in TROUBLESHOOTING_KEYWORDS)

        if policy_hit and troubleshooting_hit:
            return QueryRoute(route="mixed", reason="policy+troubleshooting keywords")
        if policy_hit:
            return QueryRoute(route="policy", reason="policy keywords")
        if troubleshooting_hit:
            return QueryRoute(route="troubleshooting", reason="troubleshooting keywords")
        return QueryRoute(route="other", reason="no domain keywords")
```

```python
# tests/__init__.py
"""Test package for RAG knowledge restructure."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_query_router -v`  
Expected: PASS with `Ran 4 tests`

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_query_router.py rag/query_router.py
git commit -m "feat: add query router for typed retrieval"
```

## Task 2: Add Structured Knowledge Stores and No-Split Loading

**Files:**
- Create: `data/structured_policies/policy_rules.jsonl`
- Create: `data/structured_policies/troubleshooting_cases.jsonl`
- Create: `tests/test_vector_store_service.py`
- Modify: `config/chroma.yml`
- Modify: `rag/vector_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_store_service.py
import unittest
from langchain_core.documents import Document

from rag.vector_store import VectorStoreService


class TestVectorStoreService(unittest.TestCase):
    def test_structured_store_skips_splitter(self):
        service = VectorStoreService("policy_rules")
        docs = [Document(page_content="规则正文", metadata={"topic": "保修期判断"})]

        prepared = service._prepare_documents_for_store(docs)

        self.assertEqual(prepared, docs)

    def test_legacy_policy_store_keeps_splitter_path(self):
        service = VectorStoreService("policy_answer")
        docs = [Document(page_content="这是一段很长的正文。" * 50, metadata={})]

        prepared = service._prepare_documents_for_store(docs)

        self.assertGreaterEqual(len(prepared), 1)
        self.assertNotEqual(prepared, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_vector_store_service -v`  
Expected: FAIL with `ValueError: 未找到向量库配置：policy_rules` or `AttributeError: 'VectorStoreService' object has no attribute '_prepare_documents_for_store'`

- [ ] **Step 3: Add the new store configs and structured seed data**

```yaml
# config/chroma.yml
default_store: policy_answer

stores:
  question_recall:
    collection_name: question_recall
    persist_directory: chroma_db/questions
    data_path: data/questions
    md5_hex_store: md5_questions.text
    allow_knowledge_file_type: ["jsonl", "txt", "pdf"]
    k: 5
    mode: question_recall

  policy_answer:
    collection_name: policy_answer
    persist_directory: chroma_db/policies
    data_path: data/policies
    md5_hex_store: md5_policies.text
    allow_knowledge_file_type: ["txt", "pdf", "jsonl"]
    k: 4
    mode: policy_answer

  policy_rules:
    collection_name: policy_rules
    persist_directory: chroma_db/policy_rules
    data_path: data/structured_policies
    md5_hex_store: md5_policy_rules.text
    allow_knowledge_file_type: ["jsonl"]
    k: 4
    mode: policy_rules

  troubleshooting_cases:
    collection_name: troubleshooting_cases
    persist_directory: chroma_db/troubleshooting_cases
    data_path: data/structured_policies
    md5_hex_store: md5_troubleshooting_cases.text
    allow_knowledge_file_type: ["jsonl"]
    k: 4
    mode: troubleshooting_cases

chunk_size: 200
chunk_overlap: 20
separators: ["\n\n", "\n", "。", "；", "，", ".", "!", "?"]
```

```json
{"id":"policy_001","knowledge_type":"policy","scene":"保修判断","topic":"是否在保修期内","user_intents":["在保吗","还保修吗","能免费修吗"],"judgement_factors":["购买时间","发票或订单","激活时间","SN码"],"included":["保内且非人为故障通常可申请保修"],"excluded":["无法仅凭口头描述直接承诺保修"],"required_info":["购买凭证","故障现象"],"response_principle":"先说明判断条件，再说明需核验信息","escalation_condition":["用户无法提供购买信息"],"content":"主题：是否在保修期内\n问法：在保吗；还保修吗；能免费修吗\n判断条件：购买时间、发票或订单、激活时间、SN码\n一般结论：保内且非人为故障通常可申请保修\n限制：无法仅凭口头描述直接承诺保修"}
{"id":"policy_002","knowledge_type":"policy","scene":"免费维修判断","topic":"人为损坏是否属于非保修","user_intents":["进水还能保修吗","摔坏能免费修吗","私拆后还保吗"],"judgement_factors":["故障原因","使用环境","维修历史"],"included":["非人为、非外力、非私拆故障通常按保内规则判断"],"excluded":["进水","跌落","私拆","第三方维修导致的损坏"],"required_info":["损坏原因","故障表现"],"response_principle":"先说明非保修边界，再提示检测结论为准","escalation_condition":["存在安全风险或故障原因不清"],"content":"主题：人为损坏是否属于非保修\n问法：进水还能保修吗；摔坏能免费修吗；私拆后还保吗\n判断条件：故障原因、使用环境、维修历史\n通常非保修：进水、跌落、私拆、第三方维修导致的损坏"}
```

```json
{"id":"trouble_001","knowledge_type":"troubleshooting","scene":"故障排查","symptom":"无法连接 WiFi","aliases":["连不上 WiFi","APP 搜不到设备","配网失败"],"possible_causes":["未连接 2.4G 网络","密码错误","路由器异常"],"user_actions":["确认路由器为 2.4G","重新输入密码","重启路由器和机器人后重新配网"],"do_not_suggest":["不要拆开 WiFi 模块"],"escalation_condition":["多次重试仍失败","疑似模块故障"],"related_parts":["WiFi 模块","路由器"],"priority":"high","content":"症状：无法连接 WiFi\n别名：连不上 WiFi；APP 搜不到设备；配网失败\n可能原因：未连接 2.4G 网络、密码错误、路由器异常\n用户动作：确认 2.4G，重新输入密码，重启路由器和机器人后重新配网\n升级条件：多次重试仍失败或疑似模块故障"}
{"id":"trouble_002","knowledge_type":"troubleshooting","scene":"故障排查","symptom":"机器人不出水","aliases":["拖地不出水","水箱有水但不出水"],"possible_causes":["出水量设置过低","出水口堵塞","水箱未安装到位"],"user_actions":["确认水箱有水","调高出水量","清理出水口并重新安装水箱"],"do_not_suggest":["不要强行拆解水泵"],"escalation_condition":["清理后仍不出水","疑似水泵故障"],"related_parts":["水箱","出水口","水泵"],"priority":"high","content":"症状：机器人不出水\n别名：拖地不出水；水箱有水但不出水\n可能原因：出水量设置过低、出水口堵塞、水箱未安装到位\n用户动作：确认水箱有水，调高出水量，清理出水口并重新安装水箱\n升级条件：清理后仍不出水或疑似水泵故障"}
```

- [ ] **Step 4: Implement no-split loading for structured stores**

```python
# rag/vector_store.py
class VectorStoreService:
    ...
    def _prepare_documents_for_store(self, documents: list[Document]) -> list[Document]:
        if self.store_mode in {"policy_rules", "troubleshooting_cases"}:
            return documents
        return self.spliter.split_documents(documents)

    def load_document(self, allowed_files_path: list[str] | None = None):
        ...
                prepared_documents = self._prepare_documents_for_store(documents)
                if not prepared_documents:
                    logger.warning(f"[加载知识库][{self.store_name}] {path} 处理后为空，跳过")
                    continue

                self._get_vector_store().add_documents(prepared_documents)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库][{self.store_name}] {path} 加载成功")
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m unittest tests.test_vector_store_service -v`  
Expected: PASS with `Ran 2 tests`

- [ ] **Step 6: Commit**

```bash
git add config/chroma.yml data/structured_policies/policy_rules.jsonl data/structured_policies/troubleshooting_cases.jsonl rag/vector_store.py tests/test_vector_store_service.py
git commit -m "feat: add structured knowledge stores"
```

## Task 3: Create Typed Retrieval Service

**Files:**
- Create: `rag/retrieval_service.py`
- Create: `tests/test_retrieval_service.py`
- Modify: `rag/query_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieval_service.py
import unittest
from langchain_core.documents import Document

from rag.retrieval_service import RetrievalBundle, TypedRetrievalService


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.docs


class TestTypedRetrievalService(unittest.TestCase):
    def test_policy_query_only_hits_policy_store(self):
        question = FakeRetriever([Document(page_content="相似问法", metadata={})])
        policy = FakeRetriever([Document(page_content="规则", metadata={"topic": "保修期"})])
        troubleshooting = FakeRetriever([Document(page_content="排障", metadata={"symptom": "不出水"})])

        service = TypedRetrievalService(question_retriever=question, policy_retriever=policy, troubleshooting_retriever=troubleshooting)
        bundle = service.retrieve("这个机器还在保修期吗")

        self.assertEqual(bundle.route.route, "policy")
        self.assertEqual(len(bundle.policy_docs), 1)
        self.assertEqual(len(bundle.troubleshooting_docs), 0)

    def test_mixed_query_hits_both_structured_stores(self):
        question = FakeRetriever([])
        policy = FakeRetriever([Document(page_content="规则", metadata={"topic": "保修期"})])
        troubleshooting = FakeRetriever([Document(page_content="排障", metadata={"symptom": "不出水"})])

        service = TypedRetrievalService(question_retriever=question, policy_retriever=policy, troubleshooting_retriever=troubleshooting)
        bundle = service.retrieve("机器不出水，如果是正常使用坏的还能保修吗")

        self.assertEqual(bundle.route.route, "mixed")
        self.assertEqual(len(bundle.policy_docs), 1)
        self.assertEqual(len(bundle.troubleshooting_docs), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_retrieval_service -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.retrieval_service'`

- [ ] **Step 3: Implement the retrieval bundle and service**

```python
# rag/retrieval_service.py
from dataclasses import dataclass, field

from langchain_core.documents import Document

from rag.query_router import QueryRoute, QueryRouter
from rag.vector_store import VectorStoreService


@dataclass
class RetrievalBundle:
    query: str
    route: QueryRoute
    question_docs: list[Document] = field(default_factory=list)
    policy_docs: list[Document] = field(default_factory=list)
    troubleshooting_docs: list[Document] = field(default_factory=list)


class TypedRetrievalService:
    def __init__(
        self,
        question_retriever=None,
        policy_retriever=None,
        troubleshooting_retriever=None,
        router: QueryRouter | None = None,
    ):
        self.router = router or QueryRouter()
        self.question_retriever = question_retriever or VectorStoreService("question_recall").get_retriever()
        self.policy_retriever = policy_retriever or VectorStoreService("policy_rules").get_retriever()
        self.troubleshooting_retriever = troubleshooting_retriever or VectorStoreService("troubleshooting_cases").get_retriever()

    def retrieve(self, query: str) -> RetrievalBundle:
        route = self.router.route(query)
        question_docs = self.question_retriever.invoke(query)
        policy_docs: list[Document] = []
        troubleshooting_docs: list[Document] = []

        if route.route in {"policy", "mixed"}:
            policy_docs = self.policy_retriever.invoke(query)
        if route.route in {"troubleshooting", "mixed"}:
            troubleshooting_docs = self.troubleshooting_retriever.invoke(query)

        return RetrievalBundle(
            query=query,
            route=route,
            question_docs=question_docs,
            policy_docs=policy_docs,
            troubleshooting_docs=troubleshooting_docs,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_retrieval_service -v`  
Expected: PASS with `Ran 2 tests`

- [ ] **Step 5: Commit**

```bash
git add rag/retrieval_service.py tests/test_retrieval_service.py
git commit -m "feat: add typed retrieval service"
```

## Task 4: Refactor RAG Service to Use Typed Retrieval and Controlled Context Formatting

**Files:**
- Create: `rag/context_formatter.py`
- Create: `tests/test_context_formatter.py`
- Modify: `rag/rag_service.py`
- Modify: `prompts/rag_summarize.txt`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_formatter.py
import unittest
from langchain_core.documents import Document

from rag.context_formatter import format_retrieval_bundle
from rag.retrieval_service import RetrievalBundle
from rag.query_router import QueryRoute


class TestContextFormatter(unittest.TestCase):
    def test_format_bundle_contains_route_and_two_knowledge_sections(self):
        bundle = RetrievalBundle(
            query="机器不出水，如果是正常使用坏的还能保修吗",
            route=QueryRoute(route="mixed", reason="policy+troubleshooting keywords"),
            question_docs=[Document(page_content="相似问法", metadata={"question": "不出水还能保修吗"})],
            policy_docs=[Document(page_content="规则正文", metadata={"topic": "免费维修判断"})],
            troubleshooting_docs=[Document(page_content="排障正文", metadata={"symptom": "机器人不出水"})],
        )

        context = format_retrieval_bundle(bundle)

        self.assertIn("## 路由结果", context)
        self.assertIn("## 规则依据", context)
        self.assertIn("## 排障依据", context)
        self.assertIn("mixed", context)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_formatter -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.context_formatter'`

- [ ] **Step 3: Implement context formatter and wire it into `RagSummarizeService`**

```python
# rag/context_formatter.py
from rag.retrieval_service import RetrievalBundle


def _format_docs(title: str, docs: list) -> str:
    if not docs:
        return f"## {title}\n无"
    rows = []
    for index, doc in enumerate(docs, start=1):
        rows.append(f"【{title}{index}】{doc.page_content}")
    return f"## {title}\n" + "\n\n".join(rows)


def format_retrieval_bundle(bundle: RetrievalBundle) -> str:
    return "\n\n".join(
        [
            "## 路由结果",
            f"route={bundle.route.route}",
            f"reason={bundle.route.reason}",
            _format_docs("相似问法", bundle.question_docs),
            _format_docs("规则依据", bundle.policy_docs),
            _format_docs("排障依据", bundle.troubleshooting_docs),
        ]
    )
```

```python
# rag/rag_service.py
from rag.context_formatter import format_retrieval_bundle
from rag.retrieval_service import TypedRetrievalService


class RagSummarizeService:
    def __init__(self):
        self.retrieval_service = TypedRetrievalService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def build_chain_inputs(self, query: str) -> dict[str, str]:
        bundle = self.retrieval_service.retrieve(query)
        return {
            "input": query,
            "route": bundle.route.route,
            "context": format_retrieval_bundle(bundle),
        }

    def rag_summarize(self, query: str):
        return self.chain.invoke(self.build_chain_inputs(query))
```

```text
你是一个售后客服知识助手。你的任务是根据路由结果，在“规则依据”和“排障依据”的约束下生成最终答复。

输入信息：
1. 用户问题：{input}
2. 路由结果：{route}
3. 检索上下文：{context}

请严格遵守以下规则：
1. 当 `route=policy` 时，优先说明判断条件、边界和需核验的信息，不得直接承诺一定保修。
2. 当 `route=troubleshooting` 时，优先给出低风险排查动作，再说明何时应转人工。
3. 当 `route=mixed` 时，先给排障建议，再补充保修判断条件和边界。
4. 若依据不足，只能明确说“需进一步核验”或“建议转人工/售后检测”，不得编造条件。
5. 只输出中文最终答复正文，不要输出分析过程、不要输出 JSON。
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_context_formatter -v`  
Expected: PASS with `Ran 1 test`

- [ ] **Step 5: Commit**

```bash
git add rag/context_formatter.py rag/rag_service.py prompts/rag_summarize.txt tests/test_context_formatter.py
git commit -m "feat: route rag generation with typed context"
```

## Task 5: Decouple Evaluation From the Chat Model

**Files:**
- Create: `tests/test_eval.py`
- Modify: `rag/eval.py`
- Modify: `data/test_queries.jsonl`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval.py
import unittest
from unittest.mock import patch

import rag.eval as eval_module


class FakeRetrievalService:
    def retrieve(self, query):
        raise RuntimeError("stubbed in later task")


class TestEvalModule(unittest.TestCase):
    def test_eval_module_can_import_without_chat_model(self):
        self.assertTrue(hasattr(eval_module, "load_test_queries"))
        self.assertTrue(hasattr(eval_module, "PolicySectionResolver"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_eval -v`  
Expected: FAIL because importing `rag.eval` still triggers `RagSummarizeService` and `ChatTongyi`

- [ ] **Step 3: Refactor `rag/eval.py` to use `TypedRetrievalService` directly**

```python
# rag/eval.py
from rag.retrieval_service import TypedRetrievalService


def evaluate():
    retrieval_service = TypedRetrievalService()
    resolver = PolicySectionResolver()
    test_queries = load_test_queries()
    ...
        bundle = retrieval_service.retrieve(query)
        question_docs = bundle.question_docs
        retrieved_question_refs = [
            ref for ref in (extract_question_ref(doc) for doc in question_docs) if ref
        ]

        routed_policy_docs = bundle.policy_docs
        routed_troubleshooting_docs = bundle.troubleshooting_docs
        merged_domain_docs = routed_policy_docs + routed_troubleshooting_docs

        retrieved_policy_sections = [
            resolver.resolve(doc) or "(未识别分区)" for doc in merged_domain_docs
        ]
    ...
```

```json
{"name":"是否在保","query":"这个机器还在保修期吗","expected_route":"policy","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":1}],"expected_policy_sections":["保修判断"]}
{"name":"WiFi 故障排查","query":"机器人连不上 WiFi 怎么办","expected_route":"troubleshooting","expected_question_refs":[{"source":"机器人FAQ知识库_合并版.jsonl","id":4}],"expected_policy_sections":["故障排查"]}
{"name":"不出水且问保修","query":"机器不出水，如果是正常使用坏的还能保修吗","expected_route":"mixed","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":12}],"expected_policy_sections":["免费维修判断","故障排查"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_eval -v`  
Expected: PASS with `Ran 1 test`

- [ ] **Step 5: Commit**

```bash
git add rag/eval.py data/test_queries.jsonl tests/test_eval.py
git commit -m "feat: decouple retrieval evaluation from llm"
```

## Task 6: Integration Verification and Knowledge Sync

**Files:**
- Modify: `agent/tools/agent_tools.py` (only if tool wiring needs lazy init adjustment)
- Modify: `app.py` (only if startup should prebuild the two new stores)

- [ ] **Step 1: Write the failing integration check**

```python
# tests/test_retrieval_service.py
    def test_other_query_skips_structured_stores(self):
        question = FakeRetriever([])
        policy = FakeRetriever([])
        troubleshooting = FakeRetriever([])

        service = TypedRetrievalService(question_retriever=question, policy_retriever=policy, troubleshooting_retriever=troubleshooting)
        bundle = service.retrieve("谢谢，已经解决了")

        self.assertEqual(bundle.route.route, "other")
        self.assertEqual(policy.queries, [])
        self.assertEqual(troubleshooting.queries, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_retrieval_service.TestTypedRetrievalService.test_other_query_skips_structured_stores -v`  
Expected: FAIL because the current retrieval service still unconditionally hits stores

- [ ] **Step 3: Implement the minimal integration fix**

```python
# rag/retrieval_service.py
    def retrieve(self, query: str) -> RetrievalBundle:
        route = self.router.route(query)
        question_docs = self.question_retriever.invoke(query)
        policy_docs: list[Document] = []
        troubleshooting_docs: list[Document] = []

        if route.route == "other":
            return RetrievalBundle(
                query=query,
                route=route,
                question_docs=question_docs,
                policy_docs=[],
                troubleshooting_docs=[],
            )

        if route.route in {"policy", "mixed"}:
            policy_docs = self.policy_retriever.invoke(query)
        if route.route in {"troubleshooting", "mixed"}:
            troubleshooting_docs = self.troubleshooting_retriever.invoke(query)

        return RetrievalBundle(
            query=query,
            route=route,
            question_docs=question_docs,
            policy_docs=policy_docs,
            troubleshooting_docs=troubleshooting_docs,
        )
```

```python
# app.py
@st.cache_resource(show_spinner=False)
def initialize_vector_store_once():
    VectorStoreService.ensure_all_vector_stores_synced()
    return True
```

- [ ] **Step 4: Run the targeted test**

Run: `python -m unittest tests.test_retrieval_service.TestTypedRetrievalService.test_other_query_skips_structured_stores -v`  
Expected: PASS

- [ ] **Step 5: Run the full verification suite**

Run: `python -m unittest tests.test_query_router tests.test_vector_store_service tests.test_retrieval_service tests.test_context_formatter tests.test_eval -v`  
Expected: PASS with all tests green

- [ ] **Step 6: Smoke test vector store sync**

Run: `python -c "from rag.vector_store import VectorStoreService; VectorStoreService.ensure_all_vector_stores_synced(); print('sync ok')"`  
Expected: prints `sync ok`

- [ ] **Step 7: Commit**

```bash
git add app.py agent/tools/agent_tools.py rag/retrieval_service.py
git commit -m "feat: complete typed rag integration"
```

## Self-Review

### Spec Coverage

- 双知识体系：Task 2 通过 `policy_rules` 和 `troubleshooting_cases` 落地。
- 意图分流：Task 1 落地 `QueryRouter`。
- 分库检索：Task 3 落地 `TypedRetrievalService`。
- 受控回答：Task 4 修改 `rag_summarize` prompt 与上下文格式。
- 评估解耦：Task 5 将 `rag/eval.py` 改成直接依赖检索服务。
- 线上稳定性验证：Task 6 增加 full suite 和向量库同步 smoke test。

### Placeholder Scan

- 本计划未使用 `TODO`、`TBD`、`之后补`、`类似 Task N`。
- 每个任务均提供了明确文件路径、测试命令、预期失败和预期成功结果。

### Type Consistency

- 路由结果统一使用 `QueryRoute.route`。
- 检索结果统一使用 `RetrievalBundle`，字段固定为 `question_docs`、`policy_docs`、`troubleshooting_docs`。
- 生成层统一通过 `build_chain_inputs()` 接收 `route` 与 `context`。
