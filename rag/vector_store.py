import gc
import os
import shutil
import sqlite3
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import (
    get_file_md5_hex,
    jsonl_loader,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self, store_name: str | None = None):
        # store_name 对应 config/chroma.yml 里的一类知识库配置。
        self.store_name = store_name or chroma_conf.get("default_store", "policy_answer")
        store_conf = (chroma_conf.get("stores") or {}).get(self.store_name)
        if not store_conf:
            raise ValueError(f"未找到向量库配置：{self.store_name}")

        self.store_mode = store_conf.get("mode", self.store_name)
        self.persist_directory = get_abs_path(store_conf["persist_directory"])
        self.data_path = get_abs_path(store_conf["data_path"])
        self.md5_hex_store = get_abs_path(store_conf["md5_hex_store"])
        self.collection_name = store_conf["collection_name"]
        self.k = int(store_conf.get("k", 3))
        self.allowed_types = tuple(store_conf.get("allow_knowledge_file_type", ["txt", "pdf"]))
        self.vector_store = None

        # 所有文档统一分片后再入库，便于不同格式知识源走同一套流程。
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def _prepare_documents_for_store(self, documents: list[Document]) -> list[Document]:
        """根据 store 类型决定是否切分文档。

        policy_rules 和 troubleshooting_cases 使用结构化知识，直接入库不切分。
        其他 store 类型使用传统的分片逻辑。
        """
        if self.store_mode in {"policy_rules", "troubleshooting_cases"}:
            return documents
        return self.spliter.split_documents(documents)

    @classmethod
    def ensure_all_vector_stores_synced(cls):
        # 应用启动时批量同步所有向量库，避免首问时才发现知识库未就绪。
        for store_name in (chroma_conf.get("stores") or {}):
            service = cls(store_name)
            try:
                service.ensure_vector_store_synced()
            finally:
                service.release()

    def _create_vector_store(self):
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=embed_model,
            persist_directory=self.persist_directory,
        )

    def _get_vector_store(self):
        if self.vector_store is None:
            self.vector_store = self._create_vector_store()
        return self.vector_store

    def _get_collection_count(self) -> int:
        # 直接查看 Chroma 的 sqlite 数据量，快速判断库是否为空。
        sqlite_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return 0

        conn = sqlite3.connect(sqlite_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM embeddings")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0
        finally:
            conn.close()

    def _list_allowed_files(self) -> list[str]:
        return list(
            listdir_with_allowed_type(
                self.data_path,
                self.allowed_types,
            )
        )

    def _rebuild_md5_store_from_files(self, allowed_files_path: list[str] | None = None):
        md5_values = []
        for path in (allowed_files_path or self._list_allowed_files()):
            md5_hex = get_file_md5_hex(path)
            if md5_hex and md5_hex not in md5_values:
                md5_values.append(md5_hex)

        with open(self.md5_hex_store, "w", encoding="utf-8") as f:
            for md5_hex in md5_values:
                f.write(md5_hex + "\n")

    def _load_file_documents(self, read_path: str) -> list[Document]:
        if read_path.endswith(".txt"):
            return txt_loader(read_path)
        if read_path.endswith(".pdf"):
            return pdf_loader(read_path)
        if read_path.endswith(".jsonl"):
            return self._normalize_jsonl_documents(jsonl_loader(read_path))
        return []

    def _normalize_jsonl_documents(self, documents: list[Document]) -> list[Document]:
        normalized_docs: list[Document] = []
        for doc in documents:
            metadata = dict(doc.metadata or {})
            question = str(metadata.get("question") or "").strip()
            answer = str(metadata.get("answer") or "").strip()
            answer_hint = str(metadata.get("answer_hint") or "").strip()
            intent = str(metadata.get("intent") or "").strip()
            category = str(metadata.get("category") or "").strip()

            if self.store_mode == "question_recall":
                # 相似问法库侧重语义召回，把意图/分类拼进正文更容易命中。
                parts = [question or doc.page_content]
                if intent:
                    parts.append(f"意图：{intent}")
                if category:
                    parts.append(f"分类：{category}")
                page_content = "\n".join(part for part in parts if part)
            else:
                # 政策库侧重回答依据，因此保留问题、答案和处理提示。
                parts = []
                if question:
                    parts.append(f"问题：{question}")
                if answer:
                    parts.append(f"答案：{answer}")
                if answer_hint:
                    parts.append(f"处理提示：{answer_hint}")
                if not parts:
                    parts.append(doc.page_content)
                page_content = "\n".join(parts)

            normalized_docs.append(Document(page_content=page_content, metadata=metadata))
        return normalized_docs

    def release(self):
        self.vector_store = None
        gc.collect()

    def _reset_vector_store(self):
        self.release()
        if os.path.isdir(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        if os.path.exists(self.md5_hex_store):
            os.remove(self.md5_hex_store)
        os.makedirs(self.persist_directory, exist_ok=True)

    def get_retriever(self):
        return self._get_vector_store().as_retriever(search_kwargs={"k": self.k})

    def ensure_vector_store_synced(self):
        # 同步规则：
        # 1. 库不存在 -> 全量构建
        # 2. 库为空 -> 重建
        # 3. 库已存在 -> 基于 md5 做增量导入
        sqlite_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        has_md5_store = os.path.exists(self.md5_hex_store)
        document_count = self._get_collection_count()
        allowed_files_path = self._list_allowed_files()

        if not os.path.exists(sqlite_path):
            logger.info(f"[向量库同步][{self.store_name}] 检测到向量库不存在，开始构建知识库")
            os.makedirs(self.persist_directory, exist_ok=True)
            if os.path.exists(self.md5_hex_store):
                os.remove(self.md5_hex_store)
            self.load_document(allowed_files_path)
            return

        if document_count == 0:
            logger.info(f"[向量库同步][{self.store_name}] 检测到向量库为空，开始重建知识库")
            if os.path.exists(self.md5_hex_store):
                os.remove(self.md5_hex_store)
            try:
                self._reset_vector_store()
            except PermissionError as exc:
                logger.warning(
                    f"[向量库同步][{self.store_name}] 重置空向量库时文件被占用，改为在现有库上继续构建：{exc}"
                )
                self.release()
                os.makedirs(self.persist_directory, exist_ok=True)
            self.load_document(allowed_files_path)
            return

        if not has_md5_store:
            logger.warning(f"[向量库同步][{self.store_name}] 检测到 md5 记录缺失，开始重建 md5 记录")
            self._rebuild_md5_store_from_files(allowed_files_path)

        logger.info(f"[向量库同步][{self.store_name}] 检测到已有向量库，开始检查新增文件")
        self.load_document(allowed_files_path)

    def load_document(self, allowed_files_path: list[str] | None = None):
        def check_md5_hex(md5_for_check):
            if not md5_for_check:
                return False
            if not os.path.exists(self.md5_hex_store):
                open(self.md5_hex_store, "w", encoding="utf-8").close()
                return False
            with open(self.md5_hex_store, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_check):
            if not md5_for_check:
                return
            with open(self.md5_hex_store, "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        for path in (allowed_files_path or self._list_allowed_files()):
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库][{self.store_name}] {path} 已入库，跳过")
                continue

            try:
                # 不同来源的知识文件先转为 Document，再统一分片和写入向量库。
                documents = self._load_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库][{self.store_name}] {path} 没有有效内容，跳过")
                    continue

                # 根据 store 类型决定是否切分文档
                prepared_documents = self._prepare_documents_for_store(documents)
                if not prepared_documents:
                    logger.warning(f"[加载知识库][{self.store_name}] {path} 处理后为空，跳过")
                    continue

                self._get_vector_store().add_documents(prepared_documents)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库][{self.store_name}] {path} 加载成功")
            except Exception as exc:
                logger.error(f"[加载知识库][{self.store_name}] {path} 加载失败：{str(exc)}", exc_info=True)
                continue


if __name__ == "__main__":
    VectorStoreService.ensure_all_vector_stores_synced()
