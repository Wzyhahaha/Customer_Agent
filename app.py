# import os,sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time

import streamlit as st
from agent.react_agent import ReactAgent
from agent.tools.agent_tools import extract_user_id, user_ids
from rag.vector_store import VectorStoreService
from utils.amap_service import AMapServiceError, get_location_snapshot
from utils.chat_history import append_user_chat_message, load_user_chat_history
from utils.city_locator_component import locate_city
from utils.runtime_context import set_user_city, set_user_weather, set_user_temperature, set_user_id


MAX_CONTEXT_MESSAGES = 12


def get_agent_context_messages() -> list[dict[str, str]]:
    messages = st.session_state.get("message", [])
    return messages[-MAX_CONTEXT_MESSAGES:]


st.title("智扫通机器人智能客服")
st.divider()

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
        if "vector_store_initialized" not in st.session_state:
            with st.spinner("正在检查并同步知识库......"):
                VectorStoreService().ensure_vector_store_synced()
            st.session_state["vector_store_initialized"] = True
        st.session_state["agent"] = ReactAgent()
    except Exception as exc:
        st.error(f"Agent 初始化失败：{exc}")
        st.stop()

if "message" not in st.session_state:
    st.session_state["message"] = []
if "current_user_id" not in st.session_state:
    st.session_state["current_user_id"] = ""

if st.session_state.get("user_city"):
    if st.session_state.get("user_weather"):
        temperature_text = st.session_state.get("user_temperature", "").strip()
        if temperature_text:
            st.caption(
                f"当前定位城市：{st.session_state['user_city']} | 当前天气：{st.session_state['user_weather']} | 当前温度：{temperature_text}°C"
            )
        else:
            st.caption(f"当前定位城市：{st.session_state['user_city']} | 当前天气：{st.session_state['user_weather']}")
    else:
        st.caption(f"当前定位城市：{st.session_state['user_city']}")
else:
    st.info("请允许浏览器定位权限，系统会自动获取你当前所在城市。")
    if st.session_state.get("location_error"):
        st.warning(f"定位失败原因：{st.session_state['location_error']}")

if st.session_state.get("current_user_id"):
    st.caption(f"当前客户编号：{st.session_state['current_user_id']}")
else:
    st.info(f"请先提供客户编号后再提问，支持的编号范围：{user_ids[0]} - {user_ids[-1]}")

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    current_user_id = st.session_state.get("current_user_id", "")
    if not current_user_id:
        st.chat_message("user").write(prompt)
        matched_user_id = extract_user_id(prompt)
        if matched_user_id:
            st.session_state["current_user_id"] = matched_user_id
            st.session_state["message"] = load_user_chat_history(matched_user_id)
            set_user_id(matched_user_id)

            confirm_message = f"已确认您的客户编号为 {matched_user_id}，后续问答会记录到该客户的聊天历史中，请继续提问。"
            st.chat_message("assistant").write(confirm_message)
            st.session_state["message"].append({"role": "assistant", "content": confirm_message})
        else:
            invalid_message = "请先提供有效的客户编号。客户编号必须是 user_ids 中的一个，例如 1001。"
            st.chat_message("assistant").write(invalid_message)
        st.rerun()

    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})
    append_user_chat_message(current_user_id, "user", prompt)

    response_chunks = []
    with st.spinner("智能客服思考中......"):
        res_stream = st.session_state["agent"].execute_stream(get_agent_context_messages())

        def capture(generator, cache_list):
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
                st.session_state["message"].append({"role":"assistant","content":full_response})
                append_user_chat_message(current_user_id, "assistant", full_response)
            st.rerun()
