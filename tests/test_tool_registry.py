import pytest
from tools.registry import ToolRegistry
from tools.rag_tool import RagTool
from tools.weather_tool import WeatherTool
from tools.escalation_tool import EscalationTool


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(RagTool())
        assert registry.get("rag_search") is not None

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(RagTool())
        registry.register(WeatherTool())
        specs = registry.list_tools()
        assert len(specs) == 2
        names = [s.name for s in specs]
        assert "rag_search" in names
        assert "weather" in names

    def test_get_nonexistent_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_run_nonexistent_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.run("nonexistent", None, None)

    def test_call_history(self):
        registry = ToolRegistry()
        registry.register(RagTool())
        from tools.base import ToolContext
        ctx = ToolContext(user_id="1001", session_id="s_test")
        result, record = registry.run("rag_search", type("Input", (), {})(), ctx)
        assert record.tool_name == "rag_search"
        assert record.status == "success"
        assert len(registry.get_call_history()) == 1
