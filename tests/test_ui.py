import requests
from streamlit.testing.v1 import AppTest

APP = "../ui/streamlit_app.py"  # AppTest resolves paths relative to this test file
NLP = {"text": "x", "intent": "SUPPORT", "confidence": 0.91, "entities": {"products": ["Tahoe"]},
       "urgency": "HIGH", "safety_flag": True}
LINE = {"sku": "D2", "item": "Office desk, 120 cm", "model": "Bekant Desk, 120 cm", "name": "Bekant Desk, 120 cm",
        "category": "desk", "qty": 10, "unit_price_inr": 2000, "line_total_inr": 20000}

RESPONSES = {
    "product_qa": {"data": {"answer": "Unplug it for 5 minutes [1]."},
                   "sources": [{"document": "dimplex.pdf", "page": 14, "score": 0.4}]},
    "plan_office": {"data": {"tier": "standard", "estimated_total": 45400, "budget_inr": 50000, "within_budget": True,
                             "needs_budget": False, "bom": [LINE],
                             "options": [{"tier": "budget", "total": 28500, "within_budget": True},
                                         {"tier": "standard", "total": 45400, "within_budget": True},
                                         {"tier": "premium", "total": 1317860, "within_budget": False}]},
                    "sources": []},
    "create_ticket": {"data": {"ticket_id": "T-ABC12345", "status": "ESCALATED", "urgency": "HIGH",
                               "issue_type": "GAS_ODOR", "product": "Tahoe", "summary": "gas smell"}, "sources": []},
}


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def app(monkeypatch, tool="product_qa", sent=None):
    body = {"response": "reply text", "nlp": NLP, "tool_used": tool, "session_id": "s", **RESPONSES[tool]}

    def post(url, json, **k):
        if sent is not None:
            sent.append(json["message"])
        return FakeResponse(body)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse({"store_name": "Test Store"}))
    return AppTest.from_file(APP, default_timeout=30).run()


def run_turn(monkeypatch, tool):
    at = app(monkeypatch, tool)
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    return at


def test_header_shows_store_name_and_quick_starts(monkeypatch):
    at = app(monkeypatch)
    assert "Test Store" in at.title[0].value
    assert len(at.main.button) == 3  # quick starts; the sidebar has "Start a new chat"
    assert len(at.sidebar.button) == 1


def test_quick_start_button_sends_message(monkeypatch):
    sent = []
    at = app(monkeypatch, "plan_office", sent)
    at.main.button[0].click().run()
    assert sent and "office" in sent[0].lower()


def test_product_qa_shows_sources(monkeypatch):
    at = run_turn(monkeypatch, "product_qa")
    assert any("dimplex.pdf" in m.value and "14" in m.value for m in at.markdown)


def test_plan_shows_options_plain_items_and_indian_prices(monkeypatch):
    at = run_turn(monkeypatch, "plan_office")
    assert [m.value for m in at.metric] == ["₹28,500", "₹45,400", "₹13,17,860"]
    table = at.dataframe[0].value
    assert list(table["Item"]) == ["Office desk, 120 cm"] and list(table["Total"]) == ["₹20,000"]
    assert at.success  # within budget badge


def test_escalated_ticket_is_red(monkeypatch):
    at = run_turn(monkeypatch, "create_ticket")
    assert any("T-ABC12345" in e.value for e in at.error)


def test_sidebar_shows_nlp(monkeypatch):
    at = run_turn(monkeypatch, "create_ticket")
    assert any("SUPPORT" in m.value for m in at.sidebar.markdown)


def test_api_down_shows_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "get", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception  # header falls back when /health is down
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert any("API" in e.value for e in at.error)


def test_chosen_plan_shows_photo_gallery(monkeypatch):
    chosen = {**RESPONSES["plan_office"]["data"], "chosen": True,
              "bom": [{**LINE, "photo": "/images/desk.jpg", "photo_credit": "Example photo: X, CC0"},
                      {**LINE, "sku": "C1", "item": "Office chair", "photo": "/images/office_chair.jpg",
                       "photo_credit": "Example photo: Y, CC BY-SA 4.0"}]}
    body = {"response": "Great choice!", "nlp": NLP, "tool_used": "plan_office", "session_id": "s",
            "data": chosen, "sources": []}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(body))
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse({"store_name": "Test Store"}))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("I'll take standard").run()
    assert not at.exception
    assert len(at.get("image")) == 2
    assert any("CC BY-SA 4.0" in c.value for c in at.caption)


def test_unchosen_plan_has_no_gallery(monkeypatch):
    at = run_turn(monkeypatch, "plan_office")
    assert len(at.get("image")) == 0


def test_choose_buttons_send_the_choice(monkeypatch):
    sent = []
    at = app(monkeypatch, "plan_office", sent)
    at.chat_input[0].set_value("plan for 10").run()
    choose = [b for b in at.button if b.label.startswith("Choose")]
    assert [b.label for b in choose] == ["Choose Basic", "Choose Standard", "Choose Premium"]
    choose[2].click().run()
    assert not at.exception
    assert sent[-1] == "I'll take the Premium option"


def test_quick_replies_under_latest_answer_send_text(monkeypatch):
    sent = []
    body = {"response": "Over budget.", "nlp": NLP, "tool_used": "plan_office", "session_id": "s",
            "data": RESPONSES["plan_office"]["data"], "sources": [],
            "suggestions": ["Plan without a meeting room", "Plan without drawer cabinets"]}

    def post(url, json, **k):
        sent.append(json["message"])
        return FakeResponse(body)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse({"store_name": "Test Store"}))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("plan for 10").run()
    quick = [b for b in at.button if b.label in body["suggestions"]]
    assert len(quick) == 2
    quick[0].click().run()
    assert not at.exception and sent[-1] == "Plan without a meeting room"


def test_quote_shows_confirmation_card(monkeypatch):
    data = {**RESPONSES["plan_office"]["data"], "chosen": True, "quote_id": "Q-1234ABCD"}
    body = {"response": "Thank you!", "nlp": NLP, "tool_used": "plan_office", "session_id": "s",
            "data": data, "sources": [], "suggestions": []}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(body))
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse({"store_name": "Test Store"}))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("ok").run()
    assert any("Q-1234ABCD" in s.value for s in at.success)
    assert not [b for b in at.button if b.label.startswith("Choose")]  # no more choosing after the quote
