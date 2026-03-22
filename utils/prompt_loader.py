import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.config_handler import prompt_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompt_conf["main_prompt_path"])
    except KeyError:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise
    
    try:
        with open(system_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_system_prompt]解析系统提示词出错，{str(e)}")
        raise

def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompt_conf["rag_summarize_prompt_path"])
    except KeyError:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise
    
    try:
        with open(rag_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_rag_prompt]解析RAG提示词出错，{str(e)}")
        raise

def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompt_conf["report_prompt_path"])
    except KeyError:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise
    
    try:
        with open(report_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_report_prompt]解析报告提示词出错，{str(e)}")
        raise
    

if __name__ == '__main__':
    print(load_system_prompts())
