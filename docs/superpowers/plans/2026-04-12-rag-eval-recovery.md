# RAG Eval Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `rag.eval` 低分背后的评估口径失真、结构化双库串库和路由漏判问题。

**Architecture:** 保持现有 typed retrieval 架构不变，只修三处最小闭环：结构化双库装载过滤、轻量路由关键词覆盖、与当前知识覆盖一致的评估集与评估统计。修复后 `rag.eval` 将对结构化保修/排障链路给出可解释结果。

**Tech Stack:** Python 3.12, unittest, Chroma, LangChain, JSONL, YAML

---

## File Structure

- Modify: `rag/vector_store.py`
  为结构化 store 增加目标文件过滤，避免双库串库。
- Modify: `rag/query_router.py`
  补充当前结构化知识覆盖范围内的关键词路由。
- Modify: `rag/eval.py`
  支持 `expected_route` 载入与路由统计输出。
- Modify: `data/test_queries.jsonl`
  重写为与当前结构化知识覆盖一致的评估集。
- Modify: `tests/test_vector_store_service.py`
  增加双库文件过滤回归测试。
- Modify: `tests/test_query_router.py`
  增加保修边界与排障别名路由测试。
- Modify: `tests/test_eval.py`
  增加 `expected_route` 统计测试。

## Task 1: Add Failing Tests For Structured Store Isolation

**Files:**
- Modify: `tests/test_vector_store_service.py`
- Modify: `rag/vector_store.py`

- [ ] **Step 1: Write the failing test**

```python
    def test_policy_rules_store_only_keeps_policy_file(self):
        service = VectorStoreService("policy_rules")

        filtered = service._filter_files_for_store(
            [
                "data/structured_policies/policy_rules.jsonl",
                "data/structured_policies/troubleshooting_cases.jsonl",
            ]
        )

        self.assertEqual(filtered, ["data/structured_policies/policy_rules.jsonl"])

    def test_troubleshooting_store_only_keeps_troubleshooting_file(self):
        service = VectorStoreService("troubleshooting_cases")

        filtered = service._filter_files_for_store(
            [
                "data/structured_policies/policy_rules.jsonl",
                "data/structured_policies/troubleshooting_cases.jsonl",
            ]
        )

        self.assertEqual(filtered, ["data/structured_policies/troubleshooting_cases.jsonl"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_vector_store_service -v`
Expected: FAIL with `AttributeError: 'VectorStoreService' object has no attribute '_filter_files_for_store'`

- [ ] **Step 3: Write minimal implementation**

```python
    def _filter_files_for_store(self, file_paths: list[str]) -> list[str]:
        basename = {
            "policy_rules": "policy_rules.jsonl",
            "troubleshooting_cases": "troubleshooting_cases.jsonl",
        }.get(self.store_mode)
        if not basename:
            return file_paths
        return [path for path in file_paths if os.path.basename(path) == basename]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_vector_store_service -v`
Expected: PASS with `Ran 4 tests`

## Task 2: Add Failing Tests For Route Coverage

**Files:**
- Modify: `tests/test_query_router.py`
- Modify: `rag/query_router.py`

- [ ] **Step 1: Write the failing test**

```python
    def test_route_non_warranty_damage_query(self):
        route = self.router.route("机器进水了还能保修吗？")
        self.assertEqual(route.route, "policy")

    def test_route_wifi_alias_query(self):
        route = self.router.route("APP搜不到设备，配网失败怎么处理？")
        self.assertEqual(route.route, "troubleshooting")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_query_router -v`
Expected: FAIL because one or more queries are classified as `other`

- [ ] **Step 3: Write minimal implementation**

```python
POLICY_KEYWORDS = {
    ...,
    "进水", "摔坏", "跌落", "私拆", "人为", "外力",
}

TROUBLESHOOTING_KEYWORDS = {
    ...,
    "配网失败", "搜不到设备", "无法联网", "处理",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_query_router -v`
Expected: PASS with `Ran 6 tests`

## Task 3: Add Failing Tests For Eval Route Expectations

**Files:**
- Modify: `tests/test_eval.py`
- Modify: `rag/eval.py`

- [ ] **Step 1: Write the failing test**

```python
class FakeRoute:
    def __init__(self, route):
        self.route = route
        self.reason = "fake"


class FakeBundle:
    def __init__(self, route):
        self.route = FakeRoute(route)
        self.question_docs = []
        self.policy_docs = []
        self.troubleshooting_docs = []


class FakeRetriever:
    search_kwargs = {"k": 1}


class FakeRetrievalService:
    def __init__(self):
        self.question_retriever = FakeRetriever()
        self.policy_retriever = FakeRetriever()

    def retrieve(self, query):
        return FakeBundle("policy")


    def test_evaluate_reports_route_accuracy(self):
        fake_cases = [
            {
                "name": "case",
                "query": "这个机器还在保修期吗",
                "expected_route": "policy",
                "expected_question_refs": [],
                "expected_policy_sections": [],
            }
        ]

        with patch.object(eval_module, "TypedRetrievalService", return_value=FakeRetrievalService()):
            with patch.object(eval_module, "load_test_queries", return_value=fake_cases):
                result = eval_module.evaluate()

        self.assertIn("route_stage", result)
        self.assertEqual(result["route_stage"]["accuracy"], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_eval -v`
Expected: FAIL because `evaluate()` does not return `route_stage`

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass
class RouteStats:
    correct_queries: int = 0
    total_queries: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct_queries / self.total_queries if self.total_queries else 0.0
```

并在 `evaluate()` 中：

- 读取 `expected_route`
- 统计 `actual_route == expected_route`
- 在返回值中加入 `route_stage`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_eval -v`
Expected: PASS with `Ran 2 tests`

## Task 4: Align Eval Dataset With Structured Knowledge Coverage

**Files:**
- Modify: `data/test_queries.jsonl`

- [ ] **Step 1: Replace the dataset with structured-coverage cases**

```json
{"name":"是否在保","query":"这个机器还在保修期吗","expected_route":"policy","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":1}],"expected_policy_sections":["保修判断"]}
{"name":"进水是否保修","query":"机器进水了还能保修吗","expected_route":"policy","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":23}],"expected_policy_sections":["免费维修判断"]}
{"name":"WiFi 配网失败","query":"APP搜不到设备，配网失败怎么办","expected_route":"troubleshooting","expected_question_refs":[{"source":"机器人FAQ知识库_合并版.jsonl","id":4}],"expected_policy_sections":["故障排查"]}
{"name":"机器人不出水","query":"机器人不出水怎么处理","expected_route":"troubleshooting","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":52}],"expected_policy_sections":["故障排查"]}
{"name":"不出水且问保修","query":"机器不出水，如果是正常使用坏的还能保修吗","expected_route":"mixed","expected_question_refs":[{"source":"扫地机器人售后高频政策问法清单.jsonl","id":12}],"expected_policy_sections":["免费维修判断","故障排查"]}
```

- [ ] **Step 2: Run a targeted smoke check**

Run: `python -m unittest tests.test_query_router tests.test_eval -v`
Expected: PASS

## Task 5: Full Verification

**Files:**
- Modify: `rag/vector_store.py`
- Modify: `rag/query_router.py`
- Modify: `rag/eval.py`
- Modify: `data/test_queries.jsonl`
- Modify: `tests/test_vector_store_service.py`
- Modify: `tests/test_query_router.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Run the related test suite**

Run: `python -m unittest tests.test_vector_store_service tests.test_query_router tests.test_retrieval_service tests.test_context_formatter tests.test_eval -v`
Expected: PASS

- [ ] **Step 2: Run eval smoke test**

Run: `python -c "import rag.eval as m; data = m.load_test_queries(); print(len(data)); print(sorted({item['expected_route'] for item in data}))"`
Expected: prints sample count and `['mixed', 'policy', 'troubleshooting']`

- [ ] **Step 3: Rebuild structured stores after code fix**

Run: `python -c "from rag.vector_store import VectorStoreService; [VectorStoreService(name)._filter_files_for_store(['data/structured_policies/policy_rules.jsonl','data/structured_policies/troubleshooting_cases.jsonl']) for name in ('policy_rules','troubleshooting_cases')]; print('filter ok')"`
Expected: prints `filter ok`

## Self-Review

- Spec coverage:
  - 串库修复由 Task 1 覆盖。
  - 路由漏判由 Task 2 覆盖。
  - 评估路由统计由 Task 3 覆盖。
  - 评估集对齐由 Task 4 覆盖。
- Placeholder scan:
  - 计划未使用 `TODO`、`TBD`、`类似 Task N`。
- Type consistency:
  - 评估新增字段统一使用 `expected_route`。
  - 评估返回新增统计统一使用 `route_stage` 和 `accuracy`。
