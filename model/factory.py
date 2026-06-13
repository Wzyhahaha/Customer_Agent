from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from sentence_transformers import SentenceTransformer
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Embeddings | BaseChatModel:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        provider = rag_conf.get("chat_model_provider", "dashscope")
        model_name = rag_conf["chat_model_name"]

        if provider == "deepseek":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )

        # default: dashscope
        from langchain_community.chat_models.tongyi import ChatTongyi

        return ChatTongyi(model=model_name)


class LocalBgeM3Embeddings(Embeddings):
    def __init__(self, model_path: str | os.PathLike[str]):
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"本地 bge-m3 模型目录不存在：{resolved_path}")
        self.model = SentenceTransformer(str(resolved_path), trust_remote_code=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        provider = rag_conf.get("embedding_model_provider", "dashscope")
        if provider == "local_bge_m3":
            default_path = Path.home() / "Desktop" / "my-mini-swe" / "models" / "bge-m3"
            return LocalBgeM3Embeddings(rag_conf.get("embedding_model_path", str(default_path)))

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
