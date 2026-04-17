from abc import ABC,abstractmethod
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import ChatTongyi,BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_conf

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self):
        # 聊天模型名称从 rag.yml 读取，方便后续切换不同大模型。
        return ChatTongyi(model=rag_conf["chat_model_name"])

class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        # 向量化模型与聊天模型解耦，便于分别调整检索效果和生成效果。
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])
    
# 模块导入时直接实例化，其他模块拿来即可用。
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
