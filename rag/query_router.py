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