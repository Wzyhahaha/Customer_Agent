import time

import streamlit as st

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import extract_user_id, user_ids
from rag.vector_store import VectorStoreService
from utils.after_sales_service import build_after_sales_message
from utils.amap_service import AMapServiceError, get_location_snapshot
from utils.chat_history import append_user_chat_message, load_user_chat_history
from utils.city_locator_component import locate_city
from utils.issue_handler import (
    PROCESSING,
    is_resolution_message,
    load_user_issue_state,
    mark_cause_inquiry_sent,
    mark_issue_escalated,
    mark_issue_resolved,
    record_assistant_solution,
    should_escalate_to_human,
    should_request_cause,
    start_or_continue_issue,
)
from utils.runtime_context import (
    set_user_city,
    set_user_id,
    set_user_temperature,
    set_user_weather,
)

# 传给 Agent 的上下文只保留最近若干轮，避免会话越来越长。
MAX_CONTEXT_MESSAGES = 12


def get_agent_context_messages() -> list[dict[str, str]]:
    messages = st.session_state.get("message", [])
    return messages[-MAX_CONTEXT_MESSAGES:]


@st.cache_resource(show_spinner=False)
def initialize_vector_store_once():
    # Streamlit 会缓存这个资源，避免每次页面刷新都重新检查/构建向量库。
    VectorStoreService.ensure_all_vector_stores_synced()
    return True


def append_and_render_assistant_message(
    user_id: str,
    message: str,
    persist_as_solution: bool = False,
    mark_as_cause_inquiry: bool = False,
):
    # 统一处理“显示回复 + 写入 session + 持久化 + 更新问题状态”。
    st.chat_message("assistant").markdown(message)
    st.session_state["message"].append({"role": "assistant", "content": message})
    append_user_chat_message(user_id, "assistant", message)
    if persist_as_solution:
        record_assistant_solution(user_id, message)
    if mark_as_cause_inquiry:
        mark_cause_inquiry_sent(user_id, message)


st.title("智扫通机器人智能客服")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = []
if "current_user_id" not in st.session_state:
    st.session_state["current_user_id"] = ""

# 前端组件先拿经纬度，再由后端补全城市、天气、省份等信息。
location_result = locate_city()
if isinstance(location_result, dict):
    if location_result.get("status") == "success":
        latitude = location_result.get("latitude")
        longitude = location_result.get("longitude")
        current_coords = (latitude, longitude)

        if latitude is not None and longitude is not None:
            last_coords = st.session_state.get("user_coords")
            if last_coords != current_coords:
                try:
                    snapshot = get_location_snapshot(float(latitude), float(longitude))
                except AMapServiceError as exc:
                    st.session_state["location_error"] = str(exc)
                except Exception as exc:
                    st.session_state["location_error"] = f"高德定位服务调用失败：{exc}"
                else:
                    st.session_state["user_coords"] = current_coords
                    st.session_state["user_city"] = snapshot["city"]
                    st.session_state["user_province"] = snapshot["province"]
                    st.session_state["user_weather"] = snapshot["weather"]
                    st.session_state["user_temperature"] = snapshot["temperature"]
                    st.session_state["location_error"] = ""
    elif location_result.get("status") == "error":
        st.session_state["location_error"] = (location_result.get("error") or "").strip()

set_user_city(st.session_state.get("user_city", ""))
set_user_weather(st.session_state.get("user_weather", ""))
set_user_temperature(st.session_state.get("user_temperature", ""))
set_user_id(st.session_state.get("current_user_id", ""))

if "agent" not in st.session_state:
    try:
        # Agent 初始化之前，先确保本地知识文件已经同步进 Chroma 向量库。
        with st.spinner("正在检查并同步知识库..."):
            initialize_vector_store_once()
        st.session_state["agent"] = ReactAgent()
    except Exception as exc:
        st.error(f"Agent 初始化失败：{exc}")
        st.stop()

if st.session_state.get("user_city"):
    if st.session_state.get("user_weather"):
        temperature_text = st.session_state.get("user_temperature", "").strip()
        if temperature_text:
            st.caption(
                f"当前定位城市：{st.session_state['user_city']} | 当前天气：{st.session_state['user_weather']} | 当前温度：{temperature_text}°C"
            )
        else:
            st.caption(
                f"当前定位城市：{st.session_state['user_city']} | 当前天气：{st.session_state['user_weather']}"
            )
    else:
        st.caption(f"当前定位城市：{st.session_state['user_city']}")
else:
    st.info("请允许浏览器定位权限，系统会自动获取你当前所在城市。")
    if st.session_state.get("location_error"):
        st.warning(f"定位失败原因：{st.session_state['location_error']}")

current_user_id = st.session_state.get("current_user_id", "")
if current_user_id:
    st.caption(f"当前客户编号：{current_user_id}")
    issue_state = load_user_issue_state(current_user_id)
    active_issue = issue_state.get("active_issue")
    if active_issue:
        st.caption(f"当前问题状态：{active_issue.get('status', PROCESSING)}")
else:
    st.info(f"请先提供客户编号后再提问，支持的编号范围：{user_ids[0]} - {user_ids[-1]}")

for message in st.session_state["message"]:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input()

if prompt:
    current_user_id = st.session_state.get("current_user_id", "")
    if not current_user_id:
        # 用户第一次发言必须先提供客户编号，系统再加载该客户的历史上下文。
        st.chat_message("user").markdown(prompt)
        matched_user_id = extract_user_id(prompt)
        if matched_user_id:
            st.session_state["current_user_id"] = matched_user_id
            st.session_state["message"] = load_user_chat_history(matched_user_id)
            set_user_id(matched_user_id)

            confirm_message = (
                f"已确认您的客户编号为 {matched_user_id}，后续问答会记录到该客户的聊天历史中，请继续提问。"
            )
            st.chat_message("assistant").markdown(confirm_message)
            st.session_state["message"].append({"role": "assistant", "content": confirm_message})
        else:
            invalid_message = "请先提供有效的客户编号。客户编号必须是 user_ids 中的一个，例如 1001。"
            st.chat_message("assistant").markdown(invalid_message)
        st.rerun()

    st.chat_message("user").markdown(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    append_user_chat_message(current_user_id, "user", prompt)

    # 每条用户消息都要归档到“当前问题单”，供后续判断已解决/升级人工。
    issue = start_or_continue_issue(current_user_id, prompt)

    if is_resolution_message(prompt):
        mark_issue_resolved(current_user_id, prompt)
        append_and_render_assistant_message(
            current_user_id,
            "已为您将该问题标记为已解决。如果还有其他问题，可以继续咨询。",
        )
        st.rerun()

    if should_escalate_to_human(issue):
        escalated_issue = mark_issue_escalated(current_user_id)
        summary_text = ""
        if escalated_issue:
            summary_text = escalated_issue.get("summary", "")
        after_sales_message = build_after_sales_message(
            st.session_state.get("user_province", ""),
            summary_text,
        )
        append_and_render_assistant_message(current_user_id, after_sales_message)
        st.rerun()

    if should_request_cause(issue):
        cause_message = (
            "已为您将该问题标记为处理中。"
            "请先告诉我导致这个问题发生的原因、出现前做过什么操作，以及目前的具体现象，"
            "我会先帮您排除障碍。"
        )
        append_and_render_assistant_message(
            current_user_id,
            cause_message,
            mark_as_cause_inquiry=True,
        )
        st.rerun()

    response_chunks = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(get_agent_context_messages())

        def capture(generator, cache_list):
            # Agent 返回增量文本块；这里一边缓存完整回答，一边模拟逐字输出。
            for chunk in generator:
                if not chunk:
                    continue
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        try:
            st.chat_message("assistant").write_stream(capture(res_stream, response_chunks))
        except Exception as exc:
            st.error(f"生成回复时出错：{exc}")
        else:
            full_response = "".join(response_chunks).strip()
            if full_response:
                append_and_render_assistant_message(
                    current_user_id,
                    full_response,
                    persist_as_solution=True,
                )
            st.rerun()
