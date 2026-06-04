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


_chat_model: BaseChatModel | None = None
_embed_model: Embeddings | None = None


def get_chat_model() -> BaseChatModel:
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatModelFactory().generator()
    return _chat_model


def get_embed_model() -> Embeddings:
    global _embed_model
    if _embed_model is None:
        _embed_model = EmbeddingsFactory().generator()
    return _embed_model


class _LazyModel:
    def __init__(self, getter):
        self._getter = getter

    def __getattr__(self, name):
        return getattr(self._getter(), name)


# 兼容旧导入名；新代码优先使用 get_chat_model/get_embed_model。
chat_model = _LazyModel(get_chat_model)
embed_model = _LazyModel(get_embed_model)
