import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.path_tool import get_abs_path
import yaml

def load_yaml_config(config_path:str, encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.safe_load(f) or {}

def load_rag_config(config_path:str=get_abs_path("config/rag.yml"),encoding:str="utf-8"):
    return load_yaml_config(config_path, encoding)
    
def load_chroma_config(config_path:str=get_abs_path("config/chroma.yml"),encoding:str="utf-8"):
    return load_yaml_config(config_path, encoding)
    
def load_prompt_config(config_path:str=get_abs_path("config/prompt.yml"),encoding:str="utf-8"):
    return load_yaml_config(config_path, encoding)
    
def load_agent_config(config_path:str=get_abs_path("config/agent.yml"),encoding:str="utf-8"):
    return load_yaml_config(config_path, encoding)
    
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompt_conf = load_prompt_config()
agent_conf = load_agent_config()

if __name__ == "__main__":
    print(rag_conf["chat_model_name"])
