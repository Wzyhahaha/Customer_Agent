import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from utils.path_tool import get_abs_path


ISSUE_STATUS_PATH = get_abs_path("data/issue_status.json")
PROCESSING = "处理中"
RESOLVED = "已解决"
ESCALATED = "升级人工"
RESOLUTION_KEYWORDS = (
    "已解决",
    "解决了",
    "问题解决了",
    "好了",
    "可以了",
    "没问题了",
    "搞定了",
    "处理好了",
    "恢复正常了",
)


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_all_issue_states() -> dict[str, dict[str, Any]]:
    try:
        with open(ISSUE_STATUS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return {}


def _save_all_issue_states(states: dict[str, dict[str, Any]]) -> None:
    with open(ISSUE_STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(states, f, ensure_ascii=False, indent=2)


def _ensure_user_bucket(states: dict[str, dict[str, Any]], user_id: str) -> dict[str, Any]:
    bucket = states.setdefault(user_id, {"active_issue_id": "", "issues": []})
    bucket.setdefault("active_issue_id", "")
    bucket.setdefault("issues", [])
    return bucket


def _find_issue(bucket: dict[str, Any], issue_id: str) -> dict[str, Any] | None:
    for issue in bucket.get("issues", []):
        if issue.get("issue_id") == issue_id:
            return issue
    return None


def load_user_issue_state(user_id: str) -> dict[str, Any]:
    states = _load_all_issue_states()
    bucket = _ensure_user_bucket(states, user_id)
    active_issue = _find_issue(bucket, bucket.get("active_issue_id", ""))
    return {
        "active_issue_id": bucket.get("active_issue_id", ""),
        "active_issue": active_issue,
        "issues": bucket.get("issues", []),
    }


def is_resolution_message(text: str) -> bool:
    normalized = (text or "").strip()
    return any(keyword in normalized for keyword in RESOLUTION_KEYWORDS)


def start_or_continue_issue(user_id: str, user_message: str) -> dict[str, Any]:
    states = _load_all_issue_states()
    bucket = _ensure_user_bucket(states, user_id)
    active_issue = _find_issue(bucket, bucket.get("active_issue_id", ""))

    if not active_issue or active_issue.get("status") != PROCESSING:
        active_issue = {
            "issue_id": uuid4().hex,
            "status": PROCESSING,
            "created_at": _now_text(),
            "updated_at": _now_text(),
            "user_messages": [],
            "assistant_messages": [],
            "summary": "",
        }
        bucket["issues"].append(active_issue)
        bucket["active_issue_id"] = active_issue["issue_id"]

    active_issue["user_messages"].append(
        {
            "content": (user_message or "").strip(),
            "timestamp": _now_text(),
        }
    )
    active_issue["updated_at"] = _now_text()

    _save_all_issue_states(states)
    return active_issue


def record_assistant_solution(user_id: str, assistant_message: str) -> dict[str, Any] | None:
    states = _load_all_issue_states()
    bucket = _ensure_user_bucket(states, user_id)
    active_issue = _find_issue(bucket, bucket.get("active_issue_id", ""))
    if not active_issue or active_issue.get("status") != PROCESSING:
        return None

    active_issue["assistant_messages"].append(
        {
            "content": (assistant_message or "").strip(),
            "timestamp": _now_text(),
        }
    )
    active_issue["updated_at"] = _now_text()
    _save_all_issue_states(states)
    return active_issue


def get_active_issue(user_id: str) -> dict[str, Any] | None:
    state = load_user_issue_state(user_id)
    return state.get("active_issue")


def should_escalate_to_human(issue: dict[str, Any] | None) -> bool:
    if not issue or issue.get("status") != PROCESSING:
        return False
    user_turns = len(issue.get("user_messages", []))
    assistant_turns = len(issue.get("assistant_messages", []))
    return assistant_turns >= 2 and user_turns >= 3


def _truncate_text(text: str, max_len: int = 80) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_issue_summary(issue: dict[str, Any]) -> str:
    first_question = ""
    latest_feedback = ""
    if issue.get("user_messages"):
        first_question = issue["user_messages"][0].get("content", "")
        latest_feedback = issue["user_messages"][-1].get("content", "")

    solution_parts = []
    for index, item in enumerate(issue.get("assistant_messages", [])[:2], start=1):
        content = _truncate_text(item.get("content", ""))
        if content:
            solution_parts.append(f"方案{index}：{content}")

    summary_parts = [
        f"问题首述：{_truncate_text(first_question) or '无'}",
        f"最近反馈：{_truncate_text(latest_feedback) or '无'}",
    ]
    summary_parts.extend(solution_parts)
    summary_parts.append("当前状态：升级人工")
    return "；".join(summary_parts)


def mark_issue_resolved(user_id: str, resolution_message: str) -> dict[str, Any] | None:
    states = _load_all_issue_states()
    bucket = _ensure_user_bucket(states, user_id)
    active_issue = _find_issue(bucket, bucket.get("active_issue_id", ""))
    if not active_issue:
        return None

    active_issue["status"] = RESOLVED
    active_issue["resolution_message"] = (resolution_message or "").strip()
    active_issue["updated_at"] = _now_text()
    bucket["active_issue_id"] = ""
    _save_all_issue_states(states)
    return active_issue


def mark_issue_escalated(user_id: str) -> dict[str, Any] | None:
    states = _load_all_issue_states()
    bucket = _ensure_user_bucket(states, user_id)
    active_issue = _find_issue(bucket, bucket.get("active_issue_id", ""))
    if not active_issue:
        return None

    active_issue["status"] = ESCALATED
    active_issue["summary"] = build_issue_summary(active_issue)
    active_issue["updated_at"] = _now_text()
    bucket["active_issue_id"] = ""
    _save_all_issue_states(states)
    return active_issue
