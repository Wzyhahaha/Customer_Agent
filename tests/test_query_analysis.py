from rag.query_analysis import QueryAnalyzer, QueryRewriter


def test_query_analyzer_detects_policy_domain_and_risk():
    analysis = QueryAnalyzer().analyze("进水了还能免费保修吗")

    assert analysis.domains == ["policy"]
    assert analysis.intent == "warranty_policy"
    assert "进水" in analysis.keywords
    assert "warranty_commitment" in analysis.risk_flags
    assert analysis.needs_clarification is True


def test_query_analyzer_detects_mixed_domain():
    analysis = QueryAnalyzer().analyze("滤网堵了怎么清理，坏了算保修吗")

    assert analysis.domains == ["maintenance", "policy"]
    assert analysis.intent == "mixed_support"
    assert "滤网" in analysis.keywords
    assert "保修" in analysis.keywords


def test_query_rewriter_expands_troubleshooting_terms():
    analysis = QueryAnalyzer().analyze("APP搜不到设备怎么办")
    rewritten = QueryRewriter().rewrite(analysis)

    assert rewritten.rewritten_queries[0] == "APP搜不到设备怎么办"
    assert "APP 搜不到设备 配网失败 无法连接 WiFi 设备搜索不到" in rewritten.rewritten_queries
    assert rewritten.domains == ["troubleshooting"]


def test_query_rewriter_copies_mutable_analysis_lists():
    analysis = QueryAnalyzer().analyze("进水了还能免费保修吗")
    rewritten = QueryRewriter().rewrite(analysis)

    assert rewritten.domains == analysis.domains
    assert rewritten.keywords == analysis.keywords
    assert rewritten.risk_flags == analysis.risk_flags
    assert rewritten.domains is not analysis.domains
    assert rewritten.keywords is not analysis.keywords
    assert rewritten.risk_flags is not analysis.risk_flags


def test_query_rewriter_matches_expansions_case_insensitively():
    analysis = QueryAnalyzer().analyze("app搜不到设备怎么办")
    rewritten = QueryRewriter().rewrite(analysis)

    assert analysis.domains == ["troubleshooting"]
    assert rewritten.rewritten_queries[0] == "app搜不到设备怎么办"
    assert "APP 搜不到设备 配网失败 无法连接 WiFi 设备搜索不到" in rewritten.rewritten_queries


def test_query_analyzer_handles_empty_string_query():
    analysis = QueryAnalyzer().analyze("")

    assert analysis.original_query == ""
    assert analysis.domains == []
    assert analysis.intent == "general_support"
    assert analysis.rewritten_queries == [""]


def test_query_analyzer_handles_none_query():
    analysis = QueryAnalyzer().analyze(None)

    assert analysis.original_query == ""
    assert analysis.domains == []
    assert analysis.intent == "general_support"
    assert analysis.rewritten_queries == [""]
