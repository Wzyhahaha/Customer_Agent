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
        return ChatTongyi(model=rag_conf["chat_model_name"])

class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])
    
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()