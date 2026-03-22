import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from langchain.agents import create_agent
from model.factory import chat_model
from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            # state_schema=load_system_prompts(),
            tools=[rag_summarize,
                   get_weather,
                   get_user_location,
                   get_user_id,
                   get_current_month,
                   fetch_external_data,
                   fill_context_for_report],
            middleware=[monitor_tool,log_before_model,report_prompt_switch]
        ) 

    @staticmethod
    def _stringify_message_content(content) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

        return str(content or "")

    def execute_stream(self, messages):
        if isinstance(messages, str):
            input_messages = [{"role": "user", "content": messages}]
        else:
            input_messages = messages

        input_dict = {
            "messages": input_messages
        }

        previous_content = ""
        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report":False}):
            messages = chunk.get("messages") or []
            if not messages:
                continue

            latest_message = messages[-1]
            if getattr(latest_message, "type", None) != "ai":
                continue

            current_content = self._stringify_message_content(latest_message.content)
            if not current_content:
                continue

            if current_content.startswith(previous_content):
                delta = current_content[len(previous_content):]
            else:
                delta = current_content

            previous_content = current_content
            if delta:
                yield delta


# if __name__ == '__main__':
#     agent = ReactAgent()

#     for chunk in agent.execute_stream("给我生成我的使用报告"):
#         print(chunk,end="",flush=True)
