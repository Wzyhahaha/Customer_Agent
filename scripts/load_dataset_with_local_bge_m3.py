from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL_PATH = Path.home() / "Desktop" / "my-mini-swe" / "models" / "bge-m3"


class LocalBgeM3Embeddings(Embeddings):
    """LangChain embedding wrapper for a locally downloaded BGE-M3 model."""

    def __init__(self, model_path: Path):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 sentence-transformers，请先在当前 Python 环境安装："
                "pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(str(model_path), trust_remote_code=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用桌面本地 bge-m3 模型把项目数据集写入 Chroma 向量数据库。"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"本地 bge-m3 模型目录，默认：{DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--store",
        action="append",
        help="只处理指定知识库；可重复传入。不传则处理全部知识库。",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="只做增量同步，不清空已有 Chroma 库。已有库若不是 bge-m3 维度可能会失败。",
    )
    return parser.parse_args()


def replace_project_embedding_model(embedding_model: Embeddings) -> None:
    import model.factory as model_factory

    model_factory._embed_model = embedding_model


def load_stores(store_names: list[str], *, rebuild: bool) -> None:
    from rag.vector_store import VectorStoreService

    for store_name in store_names:
        service = VectorStoreService(store_name)
        try:
            if rebuild:
                print(f"[{store_name}] 清空旧向量库并重新写入数据集")
                service._reset_vector_store()
            else:
                print(f"[{store_name}] 增量同步数据集")
            service.ensure_vector_store_synced()
            print(f"[{store_name}] 完成")
        finally:
            service.release()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()

    if not model_path.is_dir():
        print(f"本地 bge-m3 模型目录不存在：{model_path}", file=sys.stderr)
        print("请确认模型放在 Desktop/my-mini-swe/models/bge-m3，或用 --model-path 指定实际目录。", file=sys.stderr)
        return 1

    from utils.config_handler import chroma_conf

    configured_stores = list((chroma_conf.get("stores") or {}).keys())
    unknown_stores = sorted(set(args.store or []) - set(configured_stores))
    if unknown_stores:
        print(f"未知知识库：{', '.join(unknown_stores)}", file=sys.stderr)
        print(f"可用知识库：{', '.join(configured_stores)}", file=sys.stderr)
        return 1

    store_names = args.store or configured_stores
    if not store_names:
        print("config/chroma.yml 中没有可处理的 stores 配置。", file=sys.stderr)
        return 1

    print(f"加载本地 bge-m3 模型：{model_path}")
    replace_project_embedding_model(LocalBgeM3Embeddings(model_path))
    load_stores(store_names, rebuild=not args.sync_only)
    print("全部处理完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
