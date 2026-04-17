# Maintenance Guides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `维护保养.txt` 的实际 210 条维护知识结构化为 `maintenance_guides`，纳入 typed retrieval 主链路与 `rag.eval` 主评估体系。

**Architecture:** 保持现有 `policy_rules + troubleshooting_cases` 双域结构不变，新增第三个结构化 store `maintenance_guides`。实现上先引入结构化维护数据、维护路由与维护检索，再扩展 context formatter 和 `rag.eval`，最后再调整生成层 Prompt。维护知识统一落入单库，但按三批验收，避免一次性扩展过大导致路由和评估失真。

**Tech Stack:** Python 3.12, unittest, Chroma, LangChain, JSONL, YAML

---

## File Structure

- Create: `data/structured_policies/maintenance_guides.jsonl`
  维护保养 210 条结构化知识源。
- Create: `scripts/build_maintenance_guides.py`
  从 `data/policies/维护保养.txt` 解析并生成 JSONL 的脚本。
- Create: `tests/test_build_maintenance_guides.py`
  验证解析脚本能正确提取 section、topic、frequency 与记录总数。
- Modify: `config/chroma.yml`
  新增 `maintenance_guides` store。
- Modify: `rag/vector_store.py`
  为 `maintenance_guides` 增加文件过滤逻辑。
- Modify: `rag/query_router.py`
  新增 `maintenance` 路由与多域 mixed 判断。
- Modify: `rag/retrieval_service.py`
  增加 `maintenance_docs` 与第三知识域检索逻辑。
- Modify: `rag/context_formatter.py`
  在上下文中新增 `## 维护依据` 段落。
- Modify: `rag/eval.py`
  将 domain 评估扩展为 `policy / troubleshooting / maintenance`。
- Modify: `data/test_queries.jsonl`
  加入维护单域与跨域 mixed 样本。
- Modify: `tests/test_vector_store_service.py`
  增加维护 store 文件过滤测试。
- Modify: `tests/test_query_router.py`
  增加维护类与跨域 mixed 路由测试。
- Modify: `tests/test_retrieval_service.py`
  增加维护检索与 mixed 多域检索测试。
- Modify: `tests/test_context_formatter.py`
  增加维护上下文格式测试。
- Modify: `tests/test_eval.py`
  增加 maintenance domain 统计测试。
- Modify: `prompts/rag_summarize.txt`
  扩展维护类回答约束。
- Modify: `rag/rag_service.py`
  仅在最后接入新的维护上下文。

## Task 1: Build A Parser For 210 Maintenance Records

**Files:**
- Create: `scripts/build_maintenance_guides.py`
- Create: `tests/test_build_maintenance_guides.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_maintenance_guides.py
import unittest

from scripts.build_maintenance_guides import parse_maintenance_text


class TestBuildMaintenanceGuides(unittest.TestCase):
    def test_parse_full_text_returns_210_records(self):
        text = (
            "## 通用基础维护（2条）\n"
            "1. 每日使用后，用干软布擦拭机器人机身外壳。\n"
            "2. 每次使用后，将机器人放回充电座。\n"
            "## 耗材专项维护与更换（1条）\n"
            "1. 主刷：普通家庭3-6个月更换一次。\n"
        )

        records = parse_maintenance_text(text)

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["section"], "通用基础维护")
        self.assertEqual(records[0]["frequency"], "每日")
        self.assertEqual(records[2]["section"], "耗材专项维护与更换")
        self.assertEqual(records[2]["topic"], "主刷更换与维护")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_build_maintenance_guides -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_maintenance_guides'`

- [ ] **Step 3: Write minimal parser implementation**

```python
# scripts/build_maintenance_guides.py
from __future__ import annotations

import json
import re
from pathlib import Path


SECTION_RE = re.compile(r"^##\s+(.+?)（\d+条）\s*$")
ITEM_RE = re.compile(r"^\d+\.\s+(.+)$")

FREQUENCY_RULES = [
    ("每日", ("每日", "每天")),
    ("每次使用后", ("每次使用后", "使用后", "清扫完成后", "拖地完成后")),
    ("每周", ("每周",)),
    ("每月", ("每月",)),
    ("1-3个月", ("1-3个月", "1-2个月", "2-3个月", "3-6个月")),
    ("长期存放前", ("存放前", "长期存放")),
]


def infer_frequency(text: str) -> str:
    for label, keywords in FREQUENCY_RULES:
        if any(keyword in text for keyword in keywords):
            return label
    return "按需"


def infer_topic(section: str, text: str) -> str:
    if "主刷" in text:
        return "主刷更换与维护"
    if "边刷" in text:
        return "边刷更换与维护"
    if "滤网" in text:
        return "滤网清理与更换"
    if "拖布" in text:
        return "拖布清洗与更换"
    if "水箱" in text:
        return "水箱清洁与维护"
    if "尘盒" in text:
        return "尘盒清理与维护"
    if "清洁液" in text:
        return "清洁液使用限制"
    return f"{section}事项"


def parse_maintenance_text(text: str) -> list[dict]:
    current_section = ""
    records: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        section_match = SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1).strip()
            continue
        item_match = ITEM_RE.match(line)
        if not item_match or not current_section:
            continue
        content = item_match.group(1).strip()
        records.append(
            {
                "knowledge_type": "maintenance",
                "section": current_section,
                "topic": infer_topic(current_section, content),
                "applicable_products": "通用",
                "user_intents": [content[:24]],
                "trigger_conditions": [],
                "recommended_actions": [content],
                "do_not_do": [],
                "frequency": infer_frequency(content),
                "consumables": [],
                "escalation_condition": [],
                "content": content,
            }
        )
    return records


def build_jsonl(source_path: str, target_path: str):
    text = Path(source_path).read_text(encoding="utf-8")
    records = parse_maintenance_text(text)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for index, record in enumerate(records, start=1):
            payload = {"id": f"maintenance_{index:03d}", **record}
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_build_maintenance_guides -v`
Expected: PASS with `Ran 1 test`

## Task 2: Generate And Verify The 210-Line Maintenance JSONL

**Files:**
- Create: `data/structured_policies/maintenance_guides.jsonl`
- Create: `scripts/build_maintenance_guides.py`

- [ ] **Step 1: Generate the JSONL file**

Run: `python -c "from scripts.build_maintenance_guides import build_jsonl; build_jsonl('data/policies/维护保养.txt', 'data/structured_policies/maintenance_guides.jsonl'); print('build ok')"`
Expected: prints `build ok`

- [ ] **Step 2: Verify the generated file has 210 records**

Run: `python -c "from pathlib import Path; print(sum(1 for _ in Path('data/structured_policies/maintenance_guides.jsonl').open(encoding='utf-8')))" `
Expected: prints `210`

- [ ] **Step 3: Verify representative records contain required fields**

Run: `python -c "import json; from pathlib import Path; rows=[json.loads(line) for line in Path('data/structured_policies/maintenance_guides.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]; print(rows[0]['section'], rows[0]['topic'], rows[0]['frequency']); print(rows[100]['knowledge_type'])"`
Expected: prints section/topic/frequency for first row and `maintenance`

## Task 3: Add The Maintenance Store And File Filtering

**Files:**
- Modify: `config/chroma.yml`
- Modify: `rag/vector_store.py`
- Modify: `tests/test_vector_store_service.py`

- [ ] **Step 1: Write the failing test**

```python
    def test_maintenance_store_only_keeps_maintenance_file(self):
        service = VectorStoreService("maintenance_guides")

        filtered = service._filter_files_for_store(
            [
                "data/structured_policies/policy_rules.jsonl",
                "data/structured_policies/troubleshooting_cases.jsonl",
                "data/structured_policies/maintenance_guides.jsonl",
            ]
        )

        self.assertEqual(filtered, ["data/structured_policies/maintenance_guides.jsonl"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_vector_store_service.TestVectorStoreService.test_maintenance_store_only_keeps_maintenance_file -v`
Expected: FAIL because `maintenance_guides` is not yet recognized

- [ ] **Step 3: Add the new store config**

```yaml
  maintenance_guides:
    collection_name: maintenance_guides
    persist_directory: chroma_db/maintenance_guides
    data_path: data/structured_policies
    md5_hex_store: md5_maintenance_guides.text
    allow_knowledge_file_type: ["jsonl"]
    k: 4
    mode: maintenance_guides
```

- [ ] **Step 4: Extend file filtering**

```python
        target_basename = {
            "policy_rules": "policy_rules.jsonl",
            "troubleshooting_cases": "troubleshooting_cases.jsonl",
            "maintenance_guides": "maintenance_guides.jsonl",
        }.get(self.store_mode)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m unittest tests.test_vector_store_service -v`
Expected: PASS

## Task 4: Extend Query Routing To Maintenance And Multi-Domain Mixed

**Files:**
- Modify: `rag/query_router.py`
- Modify: `tests/test_query_router.py`

- [x] **Step 1: Write the failing tests**

```python
    def test_route_maintenance_query(self):
        route = self.router.route("拖布怎么清洗，多久换一次？")
        self.assertEqual(route.route, "maintenance")

    def test_route_maintenance_policy_mixed_query(self):
        route = self.router.route("滤网堵了怎么清理，坏了算保修吗？")
        self.assertEqual(route.route, "mixed")

    def test_route_maintenance_troubleshooting_mixed_query(self):
        route = self.router.route("拖布怎么清洗，如果还是不出水要不要送修？")
        self.assertEqual(route.route, "mixed")
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_query_router -v`
Expected: FAIL because maintenance keywords are not routed

- [x] **Step 3: Implement multi-domain matching**

```python
MAINTENANCE_KEYWORDS = {
    "主刷", "边刷", "滤网", "拖布", "尘盒", "水箱", "清洁液",
    "木地板", "地毯", "长期存放", "保养", "维护", "更换一次",
}

class QueryRouter:
    def route(self, query: str) -> QueryRoute:
        normalized = (query or "").strip().lower()
        if not normalized:
            return QueryRoute(route="other", reason="empty")

        hits = []
        if any(keyword in normalized for keyword in POLICY_KEYWORDS):
            hits.append("policy")
        if any(keyword in normalized for keyword in TROUBLESHOOTING_KEYWORDS):
            hits.append("troubleshooting")
        if any(keyword in normalized for keyword in MAINTENANCE_KEYWORDS):
            hits.append("maintenance")

        if len(hits) > 1:
            return QueryRoute(route="mixed", reason="+".join(hits))
        if len(hits) == 1:
            return QueryRoute(route=hits[0], reason=f"{hits[0]} keywords")
        return QueryRoute(route="other", reason="no domain keywords")
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_query_router -v`
Expected: PASS

## Task 5: Extend RetrievalBundle And Retrieval Service

**Files:**
- Modify: `rag/retrieval_service.py`
- Modify: `tests/test_retrieval_service.py`

- [x] **Step 1: Write the failing tests**

```python
    def test_maintenance_query_only_hits_maintenance_store(self):
        question = FakeRetriever([])
        policy = FakeRetriever([])
        troubleshooting = FakeRetriever([])
        maintenance = FakeRetriever([Document(page_content="维护正文", metadata={"section": "耗材专项维护与更换"})])

        service = TypedRetrievalService(
            question_retriever=question,
            policy_retriever=policy,
            troubleshooting_retriever=troubleshooting,
            maintenance_retriever=maintenance,
        )
        bundle = service.retrieve("拖布怎么清洗，多久换一次")

        self.assertEqual(bundle.route.route, "maintenance")
        self.assertEqual(len(bundle.maintenance_docs), 1)
        self.assertEqual(policy.queries, [])
        self.assertEqual(troubleshooting.queries, [])

    def test_mixed_query_hits_maintenance_and_policy(self):
        question = FakeRetriever([])
        policy = FakeRetriever([Document(page_content="规则正文", metadata={"scene": "免费维修判断"})])
        troubleshooting = FakeRetriever([])
        maintenance = FakeRetriever([Document(page_content="维护正文", metadata={"section": "耗材专项维护与更换"})])

        service = TypedRetrievalService(
            question_retriever=question,
            policy_retriever=policy,
            troubleshooting_retriever=troubleshooting,
            maintenance_retriever=maintenance,
        )
        bundle = service.retrieve("滤网堵了怎么清理，坏了算保修吗")

        self.assertEqual(bundle.route.route, "mixed")
        self.assertEqual(len(bundle.policy_docs), 1)
        self.assertEqual(len(bundle.maintenance_docs), 1)
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_retrieval_service -v`
Expected: FAIL because `maintenance_retriever` and `maintenance_docs` do not exist

- [x] **Step 3: Extend the bundle and service**

```python
@dataclass
class RetrievalBundle:
    query: str
    route: "QueryRoute"
    question_docs: list[Document] = field(default_factory=list)
    policy_docs: list[Document] = field(default_factory=list)
    troubleshooting_docs: list[Document] = field(default_factory=list)
    maintenance_docs: list[Document] = field(default_factory=list)
```

```python
    def __init__(
        self,
        question_retriever: Any = None,
        policy_retriever: Any = None,
        troubleshooting_retriever: Any = None,
        maintenance_retriever: Any = None,
        router: "QueryRouter | None" = None,
    ):
        self._maintenance_retriever = maintenance_retriever
```

```python
    @property
    def maintenance_retriever(self) -> BaseRetriever:
        if self._maintenance_retriever is None:
            from rag.vector_store import VectorStoreService
            self._maintenance_retriever = VectorStoreService("maintenance_guides").get_retriever()
        return self._maintenance_retriever
```

```python
        maintenance_docs: list[Document] = []
        if route.route == "maintenance":
            maintenance_docs = self.maintenance_retriever.invoke(query)
        elif route.route == "mixed":
            if "policy" in route.reason:
                policy_docs = self.policy_retriever.invoke(query)
            if "troubleshooting" in route.reason:
                troubleshooting_docs = self.troubleshooting_retriever.invoke(query)
            if "maintenance" in route.reason:
                maintenance_docs = self.maintenance_retriever.invoke(query)
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_retrieval_service -v`
Expected: PASS

## Task 6: Extend Context Formatting For Maintenance

**Files:**
- Modify: `rag/context_formatter.py`
- Modify: `tests/test_context_formatter.py`

- [x] **Step 1: Write the failing test**

```python
    def test_format_bundle_contains_maintenance_section(self):
        bundle = RetrievalBundle(
            query="拖布怎么清洗",
            route=QueryRoute(route="maintenance", reason="maintenance keywords"),
            maintenance_docs=[Document(page_content="维护正文", metadata={"section": "耗材专项维护与更换"})],
        )

        context = format_retrieval_bundle(bundle)

        self.assertIn("## 维护依据", context)
        self.assertIn("维护正文", context)
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_formatter -v`
Expected: FAIL because formatter does not render maintenance docs

- [x] **Step 3: Add the maintenance section**

```python
def format_retrieval_bundle(bundle: RetrievalBundle) -> str:
    return "\n\n".join(
        [
            "## 路由结果",
            f"route={bundle.route.route}",
            f"reason={bundle.route.reason}",
            _format_docs("相似问法", bundle.question_docs),
            _format_docs("规则依据", bundle.policy_docs),
            _format_docs("排障依据", bundle.troubleshooting_docs),
            _format_docs("维护依据", bundle.maintenance_docs),
        ]
    )
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_context_formatter -v`
Expected: PASS

## Task 7: Extend Eval To Maintenance Domain

**Files:**
- Modify: `rag/eval.py`
- Modify: `tests/test_eval.py`
- Modify: `data/test_queries.jsonl`

- [x] **Step 1: Write the failing test**

```python
    def test_policy_section_resolver_uses_section_metadata(self):
        resolver = eval_module.PolicySectionResolver()
        doc = Document(
            page_content="维护正文",
            metadata={"section": "耗材专项维护与更换"},
        )

        self.assertEqual(resolver.resolve(doc), "耗材专项维护与更换")
```

```python
    def test_evaluate_reports_maintenance_domain_metrics(self):
        fake_cases = [
            {
                "name": "maintenance-case",
                "query": "拖布怎么清洗",
                "expected_route": "maintenance",
                "expected_question_refs": [],
                "expected_domain_sections": ["maintenance:耗材专项维护与更换"],
            }
        ]
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_eval -v`
Expected: FAIL because `expected_domain_sections` is not supported

- [x] **Step 3: Extend eval schema and stats**

```python
        required_fields = (
            "query",
            "expected_route",
            "expected_question_refs",
            "expected_domain_sections",
        )
```

```python
        domain_docs = (
            [("policy", doc) for doc in bundle.policy_docs]
            + [("troubleshooting", doc) for doc in bundle.troubleshooting_docs]
            + [("maintenance", doc) for doc in bundle.maintenance_docs]
        )
        retrieved_domain_sections = [
            f"{domain}:{resolver.resolve(doc) or '(未识别分区)'}"
            for domain, doc in domain_docs
        ]
```

- [x] **Step 4: Replace the eval dataset with mixed domain coverage**

```json
{"name":"是否在保","query":"这个机器还在保修期吗","expected_route":"policy","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":1}],"expected_domain_sections":["policy:保修判断"]}
{"name":"WiFi 配网失败","query":"APP搜不到设备，配网失败怎么办","expected_route":"troubleshooting","expected_question_refs":[{"source":"机器人FAQ知识库_合并版.jsonl","id":4}],"expected_domain_sections":["troubleshooting:故障排查"]}
{"name":"拖布清洗","query":"拖布怎么清洗，多久换一次","expected_route":"maintenance","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":31}],"expected_domain_sections":["maintenance:耗材专项维护与更换"]}
{"name":"木地板拖地注意事项","query":"木地板上拖地需要注意什么","expected_route":"maintenance","expected_question_refs":[{"source":"机器人FAQ知识库_合并版.jsonl","id":68}],"expected_domain_sections":["maintenance:环境适配维护"]}
{"name":"滤网堵塞且问保修","query":"滤网堵了怎么清理，坏了算保修吗","expected_route":"mixed","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":32}],"expected_domain_sections":["maintenance:耗材专项维护与更换","policy:免费维修判断"]}
{"name":"拖布清洗后仍不出水","query":"拖布怎么清洗，如果还是不出水要不要送修","expected_route":"mixed","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":52}],"expected_domain_sections":["maintenance:扫拖一体拖地功能专属维护","troubleshooting:故障排查"]}
```

- [x] **Step 5: Run test to verify it passes**

Run: `python -m unittest tests.test_eval -v`
Expected: PASS

## Task 8: Wire Maintenance Into The Generation Layer

**Files:**
- Modify: `prompts/rag_summarize.txt`
- Modify: `rag/rag_service.py`

- [x] **Step 1: Write the failing test**

```python
    def test_build_chain_inputs_contains_maintenance_route(self):
        service = RagSummarizeService.__new__(RagSummarizeService)
        service.retrieval_service = FakeRetrievalServiceForMaintenance()

        payload = service.build_chain_inputs(“拖布怎么清洗”)

        self.assertEqual(payload[“route”], “maintenance”)
        self.assertIn(“## 维护依据”, payload[“context”])
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_formatter tests.test_eval -v`
Expected: FAIL because prompt and service do not yet account for maintenance routes

- [x] **Step 3: Extend the prompt**

```text
1. 当 `route=policy` 时，优先说明判断条件、边界和需核验的信息，不得直接承诺一定保修。
2. 当 `route=troubleshooting` 时，优先给出低风险排查动作，再说明何时应转人工。
3. 当 `route=maintenance` 时，优先给出日常维护动作、频率和禁忌项；若已超出维护边界，明确建议售后检测。
4. 当 `route=mixed` 时，先按用户问题的动作诉求回答，再补充保修或送修边界。
5. 若依据不足，只能明确说”需进一步核验”或”建议转人工/售后检测”，不得编造条件。
```

- [x] **Step 4: Run the targeted tests**

Run: `python -m unittest tests.test_context_formatter tests.test_eval -v`
Expected: PASS

## Task 9: Full Verification And Store Rebuild

**Files:**
- Modify: all files above

- [x] **Step 1: Run the full related test suite**

Run: `python -m unittest tests.test_build_maintenance_guides tests.test_vector_store_service tests.test_query_router tests.test_retrieval_service tests.test_context_formatter tests.test_eval -v`
Expected: PASS

- [x] **Step 2: Generate the maintenance JSONL fresh**

Run: `python -c "from scripts.build_maintenance_guides import build_jsonl; build_jsonl('data/policies/维护保养.txt', 'data/structured_policies/maintenance_guides.jsonl'); print('build ok')"`
Expected: prints `build ok`

- [x] **Step 3: Rebuild the maintenance vector store**

Run: `python -c "from rag.vector_store import VectorStoreService; VectorStoreService('maintenance_guides').ensure_vector_store_synced(); print('maintenance sync ok')"`
Expected: prints `maintenance sync ok`

- [x] **Step 4: Smoke test the eval dataset**

Run: `python -c "import rag.eval as m; data = m.load_test_queries(); print(len(data)); print(sorted({item['expected_route'] for item in data}))"`
Expected: prints total sample count and includes `maintenance`

- [x] **Step 5: Run real eval**

Run: `python -m rag.eval`
Expected: route/domain metrics print successfully and include maintenance samples

## Self-Review

- Spec coverage:
  - 第三知识域 `maintenance_guides` 由 Task 1-3 覆盖。
  - `maintenance` 路由与多域 mixed 由 Task 4 覆盖。
  - 维护检索与上下文格式由 Task 5-6 覆盖。
  - 维护类评估与样本扩展由 Task 7 覆盖。
  - 生成层维护回答约束由 Task 8 覆盖。
- Placeholder scan:
  - 计划未使用 `TODO`、`TBD`、`类似 Task N`。
- Type consistency:
  - 维护知识域统一命名为 `maintenance`。
  - 维护 store 统一命名为 `maintenance_guides`。
  - 检索结果统一字段为 `maintenance_docs`。
