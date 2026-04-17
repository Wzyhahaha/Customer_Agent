import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import Callable
from langchain.agents.middleware import AgentState, dynamic_prompt, wrap_tool_call,before_model
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts,load_report_prompts
from langgraph.runtime import Runtime

@wrap_tool_call
def monitor_tool(
    request:ToolCallRequest,
    handler:Callable[[ToolCallRequest],ToolMessage | Command]
):
    # 统一记录工具调用日志，并在特定工具执行后更新运行时上下文。
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        if request.tool_call['name'] == "fill_context_for_report":
            # 动态 prompt 是否切换到“报告模式”，由这个上下文字段控制。
            request.runtime.context["report"] = True
        return result 
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise
    
@before_model
def log_before_model(
    state:AgentState,
    runtime:Runtime,
):
    # 每轮进模型前记录消息数量和最后一条消息，便于调试 Agent 行为。
    messages = state.get("messages") or []
    logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息")

    if not messages:
        return None

    last_message = messages[-1]
    content = getattr(last_message, "content", "")
    if isinstance(content, str):
        safe_content = content.strip()
    else:
        safe_content = str(content)
    logger.debug(f"[log_before_model]{type(last_message)}| {safe_content}")

    return None



@dynamic_prompt
def report_prompt_switch(request):
    # 同一个 Agent 通过动态 prompt 同时支持“客服问答”和“报告生成”。
    is_report = request.runtime.context.get("report",False)
    if is_report:
        return load_report_prompts()
    
    return load_system_prompts()
