import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import csv
import json
import re
from typing import Any

from langchain.tools import tool
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.runtime_context import (
    get_user_city,
    get_user_weather,
    get_user_temperature,
    get_user_id as get_confirmed_user_id,
)

rag = None

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = [
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
]
external_data = {}
USER_IDS_SET = set(user_ids)


def extract_user_id(text: str) -> str:
    if not text:
        return ""

    candidates = re.findall(r"(?<!\d)\d{4}(?!\d)", text)
    valid_ids = [candidate for candidate in candidates if candidate in USER_IDS_SET]
    if len(valid_ids) == 1:
        return valid_ids[0]
    return ""


def get_rag_service():
    global rag
    if rag is None:
        from rag.rag_service import RagSummarizeService
        rag = RagSummarizeService()
    return rag


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return get_rag_service().rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    user_city = get_user_city()
    user_weather = get_user_weather()
    user_temperature = get_user_temperature()

    if city and user_city and city == user_city and user_weather and user_temperature:
        return f"城市{city}当前天气为{user_weather}，实时温度为{user_temperature}摄氏度"
    if not city and user_weather and user_temperature:
        return f"当前天气为{user_weather}，实时温度为{user_temperature}摄氏度"

    logger.warning(f"[get_weather]未获取到城市{city or user_city or '当前所在地'}对应的实时天气")
    return f"未获取到城市{city or user_city or '当前所在地'}的实时天气信息"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    city = get_user_city()
    if city:
        return city

    logger.warning("[get_user_location]未获取到用户真实城市，返回默认提示")
    return "未获取到城市信息"


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    current_user_id = get_confirmed_user_id()
    if current_user_id:
        return current_user_id

    logger.warning("[get_user_id]未确认客户编号，返回默认提示")
    return "未确认客户编号"

@tool(description="获取当前月份，以纯字符串返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data() -> dict[str, dict[str, dict[str, str]]]:
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")
        with open(external_data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue

                user_id = (row.get("用户ID") or "").strip()
                month = (row.get("时间") or "").strip()
                if not user_id or not month:
                    logger.warning(f"[generate_external_data]跳过缺少关键字段的数据行：{row}")
                    continue

                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][month] = {
                    "特征": (row.get("特征") or "").strip(),
                    "效率": (row.get("清洁效率") or "").strip(),
                    "耗材": (row.get("耗材") or "").strip(),
                    "对比": (row.get("对比") or "").strip(),
                }

    return external_data


def _stringify_external_record(record: Any) -> str:
    if isinstance(record, str):
        return record
    return json.dumps(record, ensure_ascii=False)

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    data = generate_external_data()

    try:
        return _stringify_external_record(data[user_id][month])
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""

@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report() -> str:
    return "fill_context_for_report已调用"


# if __name__ == "__main__":
#     print(fetch_external_data("1005","2025-06"))
