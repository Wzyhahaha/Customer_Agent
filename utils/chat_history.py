import json
from datetime import datetime
from typing import Any

from utils.path_tool import get_abs_path


CHAT_HISTORY_PATH = get_abs_path("data/chat_history.json")


def _load_all_histories() -> dict[str, list[dict[str, Any]]]:
    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return {}


def _save_all_histories(histories: dict[str, list[dict[str, Any]]]) -> None:
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)


def load_user_chat_history(user_id: str) -> list[dict[str, str]]:
    histories = _load_all_histories()
    history = histories.get(user_id, [])
    messages: list[dict[str, str]] = []
    for item in history:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def append_user_chat_message(user_id: str, role: str, content: str) -> None:
    role = (role or "").strip()
    content = (content or "").strip()
    if role not in {"user", "assistant"} or not content:
        return

    histories = _load_all_histories()
    histories.setdefault(user_id, []).append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_all_histories(histories)
