import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_chat_returns_valid_response():
    response = client.post(
        "/chat",
        json={
            "user_id": "1001",
            "message": "我的机器进水了还能保修吗？",
            "debug": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "session_id" in data
    assert "trace_id" in data
    assert "issue_status" in data
    assert "citations" in data
    assert "need_human_escalation" in data


def test_chat_with_location():
    response = client.post(
        "/chat",
        json={
            "user_id": "1001",
            "message": "今天天气怎么样",
            "location": {"city": "杭州", "weather": "小雨", "temperature": "22"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"].startswith("s_")
    assert data["trace_id"].startswith("trace_")
