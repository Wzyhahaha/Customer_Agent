from __future__ import annotations

import json
import re
from pathlib import Path


SECTION_RE = re.compile(r"^##\s+(.+?)（\d+条）\s*$")
ITEM_RE = re.compile(r"^\d+\.\s+(.+)$")


FREQUENCY_RULES = [
    ("每次使用前", ("每次使用前", "使用前", "使用水箱前", "每次使用水箱前")),
    ("每次清扫完成后", ("每次清扫完成后", "每次清扫完成", "清扫完成后", "清扫完成")),
    ("每次使用后", ("每次使用后", "使用后", "拖地完成后")),
    ("每周", ("每周",)),
    ("每月", ("每月",)),
    ("定期", ("定期",)),
    ("开封后3个月内", ("开封后3个月内", "开封后3个月")),
    ("6-12个月", ("每6-12个月", "6-12个月")),
    ("3-6个月", ("3-6个月",)),
    ("1-3个月", ("1-3个月", "1-2个月", "2-3个月")),
    ("长期存放前", ("长期存放前", "存放前", "长期存放")),
    ("每日", ("每日", "每天")),
]


def infer_frequency(text: str) -> str:
    for label, keywords in FREQUENCY_RULES:
        if any(keyword in text for keyword in keywords):
            return label
    return "按需"


def infer_applicable_products(section: str, text: str) -> str:
    if any(keyword in text for keyword in ("扫拖一体", "拖地模组", "污水仓", "自动洗拖布", "洗拖布")):
        return "扫拖一体"
    if any(keyword in text for keyword in ("集尘款", "集尘袋", "集尘座", "集尘口")):
        return "集尘款"
    if "扫拖一体拖地功能专属维护" in section:
        return "扫拖一体"
    if "扫地功能专属维护" in section:
        return "扫地机器人"
    return "通用"


def infer_user_intents(section: str, topic: str, text: str) -> list[str]:
    if "长期存放" in topic or "长期存放" in text or "长期存放" in section:
        return ["长期存放前怎么做保养？", "长期存放多久补电一次？"]

    intent_rules = [
        ("集尘口", ["集尘口怎么清理才能不漏灰？", "集尘口多久检查一次密封？"]),
        ("拖地模组", ["拖地模组接口怎么清理？", "拖地模组多久检查一次连接？"]),
        ("充电触点", ["充电触点怎么擦拭才不易氧化？", "充电触点多久检查一次接触状态？"]),
        ("充电座", ["充电座怎么清洁才能保证回充？", "充电座多久检查一次摆放和触点？"]),
        ("传感器", ["传感器怎么清洁才不影响识别？", "传感器多久检查一次灰尘遮挡？"]),
        ("防撞条", ["防撞条开裂后怎么处理？", "防撞条多久检查一次回弹？"]),
        ("驱动轮", ["驱动轮怎么清理毛发和杂物？", "驱动轮多久检查一次转动顺畅？"]),
        ("万向轮", ["万向轮怎么清理才不卡顿？", "万向轮多久检查一次磨损？"]),
        ("电池", ["电池长期不用前怎么保养？", "电池多久补电一次合适？"]),
        ("集尘袋", ["集尘袋怎么更换才不漏灰？", "集尘袋多久检查一次密封？"]),
        ("集尘座", ["集尘座怎么清理出风口？", "集尘座多久检查一次吸力？"]),
        ("红外感应区", ["充电座红外感应区怎么擦拭？", "充电座红外感应区多久清理一次？"]),
        ("电源适配器", ["电源适配器怎么检查发热和异响？", "电源适配器多久检查一次？"]),
        ("APP连接", ["APP连接状态怎么确认正常？", "APP多久检查一次网络连接？"]),
        ("污水仓", ["污水仓怎么冲洗才不留污渍？", "污水仓多久清理一次更合适？"]),
        ("滤网", ["滤网怎么清理和晾干？", "滤网多久更换一次？"]),
        ("拖布", ["拖布怎么清洗和晾干？", "拖布多久更换一次？"]),
        ("尘盒", ["尘盒怎么清理不堵塞？", "尘盒多久倒一次垃圾？"]),
        ("水箱", ["水箱怎么清洁除垢？", "水箱多久检查一次密封？"]),
        ("清洁液", ["清洁液怎么稀释使用？", "清洁液多久用完更合适？"]),
        ("木地板", ["木地板环境怎么控制出水量？", "木地板怎么避免积水和磨损？"]),
        ("地毯", ["地毯环境怎么避免浸湿？", "地毯清扫后怎么清理毛发纤维？"]),
        ("宠物家庭", ["宠物家庭怎么提高清理频率？", "宠物家庭多久清理一次主刷和尘盒？"]),
        ("潮湿环境", ["潮湿环境怎么防霉防锈？", "潮湿环境多久晾干拖布和机身？"]),
        ("固件", ["固件怎么更新更稳妥？", "固件多久检查一次更新？"]),
        ("缓存", ["APP缓存怎么清理才不影响使用？", "缓存多久清理一次合适？"]),
    ]

    for keyword, intents in intent_rules:
        if keyword in topic or keyword in text or keyword in section:
            return intents

    if "环境" in section:
        return ["这种环境怎么维护更稳妥？", "多久需要检查一次相关部件？"]

    return ["这个维护步骤怎么做？", "多久检查一次合适？"]


def infer_topic(section: str, text: str) -> str:
    if "长期存放维护" in section:
        return "长期存放准备"

    topic_rules = [
        ("集尘口", "集尘口清洁与维护"),
        ("拖地模组", "拖地模组接口维护"),
        ("充电触点", "充电触点清洁与维护"),
        ("充电座", "充电座清洁与维护"),
        ("传感器", "传感器清洁与维护"),
        ("防撞条", "防撞条维护"),
        ("驱动轮", "驱动轮清洁与维护"),
        ("万向轮", "万向轮清洁与维护"),
        ("电池", "电池维护"),
        ("集尘袋", "集尘袋维护"),
        ("集尘座", "集尘座清洁与维护"),
        ("污水仓", "污水仓清洁与维护"),
        ("长期存放", "长期存放准备"),
        ("木地板", "木地板环境维护"),
        ("地毯", "地毯环境维护"),
        ("宠物家庭", "宠物家庭环境维护"),
        ("潮湿环境", "潮湿环境维护"),
        ("固件", "固件更新与维护"),
        ("缓存", "缓存清理与维护"),
        ("主刷", "主刷更换与维护"),
        ("边刷", "边刷更换与维护"),
        ("滤网", "滤网清理与更换"),
        ("拖布", "拖布清洗与更换"),
        ("水箱", "水箱清洁与维护"),
        ("尘盒", "尘盒清理与维护"),
        ("清洁液", "清洁液使用限制"),
    ]

    for keyword, topic in topic_rules:
        if keyword in text or keyword in section:
            return topic

    if "环境" in section:
        return f"{section}环境维护"
    if "故障预防" in section:
        return "故障预防维护事项"

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
        topic = infer_topic(current_section, content)
        records.append(
            {
                "knowledge_type": "maintenance",
                "section": current_section,
                "topic": topic,
                "applicable_products": infer_applicable_products(current_section, content),
                "user_intents": infer_user_intents(current_section, topic, content),
                "frequency": infer_frequency(content),
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/build_maintenance_guides.py <source_path> <target_path>")

    build_jsonl(sys.argv[1], sys.argv[2])
