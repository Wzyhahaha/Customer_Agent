from __future__ import annotations

from rag.pipeline_types import QueryAnalysis


POLICY_TERMS = {
    "保修": "warranty_policy",
    "免费": "warranty_policy",
    "在保": "warranty_policy",
    "进水": "warranty_policy",
    "摔坏": "warranty_policy",
    "私拆": "warranty_policy",
}

TROUBLESHOOTING_TERMS = {
    "APP搜不到": "connectivity_troubleshooting",
    "搜不到设备": "connectivity_troubleshooting",
    "配网失败": "connectivity_troubleshooting",
    "连不上": "connectivity_troubleshooting",
    "不出水": "water_troubleshooting",
    "回不去充电座": "dock_troubleshooting",
}

MAINTENANCE_TERMS = {
    "滤网": "maintenance",
    "拖布": "maintenance",
    "主刷": "maintenance",
    "边刷": "maintenance",
    "清理": "maintenance",
    "更换": "maintenance",
    "保养": "maintenance",
}

RISK_TERMS = {
    "保修": "warranty_commitment",
    "免费": "warranty_commitment",
    "进水": "warranty_commitment",
    "摔坏": "warranty_commitment",
}


class QueryAnalyzer:
    def analyze(self, query: str | None) -> QueryAnalysis:
        normalized = (query or "").strip()
        lowered = normalized.lower()
        domains: list[str] = []
        keywords: list[str] = []
        intents: list[str] = []
        risk_flags: list[str] = []

        matches = []
        matches.extend(self._matches(lowered, POLICY_TERMS, "policy"))
        matches.extend(self._matches(lowered, TROUBLESHOOTING_TERMS, "troubleshooting"))
        matches.extend(self._matches(lowered, MAINTENANCE_TERMS, "maintenance"))

        for _, domain, term, intent in sorted(matches):
            if domain not in domains:
                domains.append(domain)
            if term not in keywords:
                keywords.append(term)
            if intent not in intents:
                intents.append(intent)

        for term, flag in RISK_TERMS.items():
            if term.lower() in lowered and flag not in risk_flags:
                risk_flags.append(flag)

        intent = self._intent_for(domains, intents)
        needs_clarification = "policy" in domains and any(flag == "warranty_commitment" for flag in risk_flags)

        return QueryAnalysis(
            original_query=normalized,
            domains=domains,
            intent=intent,
            keywords=keywords,
            needs_clarification=needs_clarification,
            risk_flags=risk_flags,
        )

    @staticmethod
    def _matches(
        query: str,
        terms: dict[str, str],
        domain: str,
    ) -> list[tuple[int, str, str, str]]:
        matches = []
        for term, intent in terms.items():
            index = query.find(term.lower())
            if index != -1:
                matches.append((index, domain, term, intent))
        return matches

    @staticmethod
    def _intent_for(domains: list[str], intents: list[str]) -> str:
        if len(domains) > 1:
            return "mixed_support"
        if intents:
            return intents[0]
        return "general_support"


class QueryRewriter:
    EXPANSIONS = {
        "APP搜不到": "APP 搜不到设备 配网失败 无法连接 WiFi 设备搜索不到",
        "搜不到设备": "APP 搜不到设备 配网失败 无法连接 WiFi 设备搜索不到",
        "滤网": "滤网 堵塞 清理 更换 耗材 维护",
        "保修": "保修 在保 免费维修 非人为损坏 购买凭证",
        "进水": "进水 人为损坏 非保修 检测为准",
    }

    def rewrite(self, analysis: QueryAnalysis) -> QueryAnalysis:
        queries = [analysis.original_query]
        lowered = analysis.original_query.lower()
        for term, expanded in self.EXPANSIONS.items():
            if term.lower() in lowered and expanded not in queries:
                queries.append(expanded)

        return QueryAnalysis(
            original_query=analysis.original_query,
            domains=list(analysis.domains),
            intent=analysis.intent,
            keywords=list(analysis.keywords),
            rewritten_queries=queries,
            needs_clarification=analysis.needs_clarification,
            risk_flags=list(analysis.risk_flags),
        )
