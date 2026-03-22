from langchain_chroma import Chroma
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shutil

import chromadb
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader,txt_loader,listdir_with_allowed_type,get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document

class VectorStoreService:
    def __init__(self):
        self.persist_directory = get_abs_path(chroma_conf["persist_directory"])
        self.data_path = get_abs_path(chroma_conf["data_path"])
        self.md5_hex_store = get_abs_path(chroma_conf["md5_hex_store"])
        self.collection_name = chroma_conf["collection_name"]
        self.vector_store = self._create_vector_store()

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = chroma_conf["chunk_size"],
            chunk_overlap = chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function = len
        )

    def _create_vector_store(self):
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=embed_model,
            persist_directory=self.persist_directory
        )

    def _get_collection_count(self) -> int:
        sqlite_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return 0

        client = chromadb.PersistentClient(path=self.persist_directory)
        collections = client.list_collections()
        for collection in collections:
            if collection.name == self.collection_name:
                return collection.count()
        return 0

    def _reset_vector_store(self):
        if os.path.isdir(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        if os.path.exists(self.md5_hex_store):
            os.remove(self.md5_hex_store)
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = self._create_vector_store()

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":chroma_conf["k"]})

    def ensure_vector_store_synced(self):
        sqlite_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        has_md5_store = os.path.exists(self.md5_hex_store)
        document_count = self._get_collection_count()

        if not os.path.exists(sqlite_path) or document_count == 0:
            logger.info("[向量库同步]检测到向量库为空或不存在，开始重建知识库")
            self._reset_vector_store()
            self.load_document()
            return

        if not has_md5_store:
            logger.warning("[向量库同步]检测到向量库存在但 md5.text 缺失，为避免重复入库，开始重建知识库")
            self._reset_vector_store()
            self.load_document()
            return

        logger.info("[向量库同步]检测到已有向量库，开始检查 data 目录下的新文件")
        self.load_document()
    
    def load_document(self):

        def check_md5_hex(md5_for_check):
            if not md5_for_check:
                return False
            if not os.path.exists(self.md5_hex_store):
                open(self.md5_hex_store,"w",encoding="utf-8").close()
                return False
            with open(self.md5_hex_store,"r",encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
                return False
        def save_md5_hex(md5_for_check):
            if not md5_for_check:
                return
            with open(self.md5_hex_store,"a",encoding="utf-8") as f:
                f.write(md5_for_check+"\n")
        
        def get_file_documents(read_path):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            return []
        
        allowed_files_path:list[str] = listdir_with_allowed_type(
            self.data_path,
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue
            try:
                documents:list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_docunment:list[Document] = self.spliter.split_documents(documents)

                if not split_docunment:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_docunment)

                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}内容加载失败：{str(e)}",exc_info=True)
                continue


        
if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    print("MD5文件路径：", get_abs_path(chroma_conf["md5_hex_store"]))
    for r in res:
        print(r.page_content)
        print("-"*20)
