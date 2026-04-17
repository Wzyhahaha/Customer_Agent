from dataclasses import dataclass


POLICY_KEYWORDS = {
    "保修", "保修期", "免费修", "免费维修", "在保", "过保", "人为损坏", "非保修",
    "耗材", "第三方", "官方维修", "寄修", "上门维修",
    "进水", "摔坏", "跌落", "私拆", "人为", "外力",
}

TROUBLESHOOTING_KEYWORDS = {
    "怎么办", "不出水", "回不去充电座", "回充", "连不上", "wifi", "吸力下降",
    "地图", "异响", "报错", "开不了机", "故障", "不工作", "暂停",
    "配网失败", "搜不到设备", "无法联网",
}

MAINTENANCE_KEYWORDS = {
    "主刷", "边刷", "滤网", "拖布", "尘盒", "水箱", "清洁液",
    "木地板", "地毯", "长期存放", "保养", "维护", "更换一次", "换一次", "清洗",
}

WEAK_MAINTENANCE_KEYWORDS = {"维护"}


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
        strong_maintenance_hit = any(
            keyword in normalized
            for keyword in (MAINTENANCE_KEYWORDS - WEAK_MAINTENANCE_KEYWORDS)
        )
        weak_maintenance_hit = any(keyword in normalized for keyword in WEAK_MAINTENANCE_KEYWORDS)
        maintenance_hit = strong_maintenance_hit or (
            weak_maintenance_hit and not policy_hit and not troubleshooting_hit
        )

        matched_domains = []
        if policy_hit:
            matched_domains.append("policy")
        if troubleshooting_hit:
            matched_domains.append("troubleshooting")
        if maintenance_hit:
            matched_domains.append("maintenance")

        if len(matched_domains) > 1:
            return QueryRoute(route="mixed", reason=f"{'+'.join(matched_domains)} keywords")
        if len(matched_domains) == 1:
            return QueryRoute(route=matched_domains[0], reason=f"{matched_domains[0]} keywords")
        return QueryRoute(route="other", reason="no domain keywords")
