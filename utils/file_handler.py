import hashlib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.logger_handler import logger


def get_file_md5_hex(filepath):
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件 {filepath} 不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径 {filepath} 不是文件")
        return None

    md5_obj = hashlib.md5()
    chunk_size = 4096
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception:
        logger.error(f"[md5计算]文件 {filepath} 计算 md5 失败", exc_info=True)
        return None


def listdir_with_allowed_type(path, allowed_types):
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path} 不是文件夹")
        return tuple()

    # 把配置里的扩展名统一规范成 .txt / .pdf / .jsonl 这种格式。
    normalized_types = tuple(f".{item.lstrip('.')}" for item in allowed_types)
    for filename in os.listdir(path):
        if filename.endswith(normalized_types):
            files.append(os.path.join(path, filename))
    return tuple(files)


def pdf_loader(filepath):
    loader = PyPDFLoader(filepath)
    return loader.load()


def txt_loader(filepath):
    return TextLoader(filepath, encoding="utf-8").load()


def jsonl_loader(filepath):
    documents: list[Document] = []
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning(f"[jsonl_loader]{filepath}:{line_no} 不是合法 JSON，已跳过")
                    continue

                if not isinstance(record, dict):
                    logger.warning(f"[jsonl_loader]{filepath}:{line_no} 不是对象结构，已跳过")
                    continue

                # 除正文外的字段都放进 metadata，方便后续检索和格式化上下文。
                metadata = {
                    key: value
                    for key, value in record.items()
                    if key != "content"
                }
                metadata["source"] = filepath
                metadata["line_no"] = line_no

                page_content = str(record.get("content") or record.get("question") or "").strip()
                if not page_content:
                    logger.warning(f"[jsonl_loader]{filepath}:{line_no} 缺少 content/question，已跳过")
                    continue

                documents.append(Document(page_content=page_content, metadata=metadata))
    except Exception:
        logger.error(f"[jsonl_loader]读取 {filepath} 失败", exc_info=True)
        return []

    return documents
