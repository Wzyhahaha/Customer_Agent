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
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        if request.tool_call['name'] == "fill_context_for_report":
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
    is_report = request.runtime.context.get("report",False)
    if is_report:
        return load_report_prompts()
    
    return load_system_prompts()
