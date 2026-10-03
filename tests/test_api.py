import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["llm_model"] and r.json()["store_name"]


def test_nlp_analyze():
    r = client.post("/nlp/analyze", json={"message": "Plan an office for 30 people under 5 lakh"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "OFFICE_PLAN"
    assert body["entities"]["headcount"] == 30 and body["entities"]["budget_inr"] == 500_000


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "x" * 2001}, {}])
def test_bad_input_rejected(payload):
    assert client.post("/nlp/analyze", json=payload).status_code == 422


def test_chat_rejects_bad_session_id():
    assert client.post("/chat", json={"session_id": "../etc", "message": "hi"}).status_code == 422


def test_ticket_lookup(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    tid = db.save_ticket({"product": None, "issue_type": "OTHER", "urgency": "LOW", "summary": "s",
                          "status": "OPEN", "message": "m"}, path)
    assert client.get(f"/tickets/{tid}").json()["ticket_id"] == tid
    assert client.get("/tickets/T-MISSING").status_code == 404


@pytest.mark.ollama
def test_chat_response_shape():
    r = client.post("/chat", json={"session_id": "api-test", "message": "My Markus chair keeps sinking"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"response", "nlp", "tool_used", "data", "sources", "suggestions", "session_id"}
    assert body["tool_used"] == "create_ticket" and body["data"]["ticket_id"].startswith("T-")


def test_example_photos_are_served():
    r = client.get("/images/office_chair.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/images/../app/main.py").status_code == 404


def test_quote_lookup(tmp_path, monkeypatch):
    path = tmp_path / "q.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    qid = db.save_quote({"tier": "budget", "estimated_total": 1000, "bom": []}, path)
    assert client.get(f"/quotes/{qid}").json()["total"] == 1000
    assert client.get("/quotes/Q-MISSING").status_code == 404
