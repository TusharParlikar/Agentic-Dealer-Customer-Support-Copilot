import uuid

import pytest

from langchain_core.messages import HumanMessage

from app.genai.graph import chat, history, nlp_node, route

PLAN_START = "i want to furnish a office sugest me furniture"
OPEN_PLAN = {"req": {"headcount": 30, "meeting_rooms": [30], "budget_inr": 600_000, "preferences": [], "storage": True},
             "tier": "budget", "chosen": False, "quote_id": None}


def turn(message, plan=OPEN_PLAN, last_tool="plan_office"):
    return nlp_node({"messages": [HumanMessage(message)], "plan": plan, "tool_used": last_tool})


def test_follow_up_joins_open_plan():
    out = turn("l have 30 emloyee", plan={**OPEN_PLAN, "req": None})
    assert out["plan_action"] == "update" and route(out) == "plan_office"  # alone it looks like low-confidence OTHER


def test_choice_wins_over_classifier():
    out = turn("I'll take the Basic option", last_tool="product_qa")  # classifier says PRODUCT_QA 0.72 for this
    assert (out["plan_action"], out["chosen_tier"]) == ("choose", "budget")
    assert route(out) == "plan_office"


@pytest.mark.parametrize("message", ["ok", "ok go for it", "yes", "Confirm this plan"])
def test_acknowledgement_after_plan_continues_plan(message):
    out = turn(message)
    assert out["plan_action"] == "ack" and route(out) == "plan_office"


def test_acknowledgement_after_other_topic_is_not_a_plan_action():
    assert turn("ok", last_tool="product_qa")["plan_action"] is None


def test_product_question_mid_plan_goes_to_qa_and_keeps_plan():
    out = turn("What is the warranty on the Markus chair?")
    assert out["plan_action"] is None and route(out) == "product_qa"
    assert "plan" not in out  # state keeps the open plan


def test_removal_request_updates_plan():
    assert turn("Plan without a meeting room", last_tool="product_qa")["plan_action"] == "update"


def test_first_reply_greets_with_store_intro():
    from langchain_core.messages import AIMessage
    from app.config import STORE_NAME
    from app.genai.graph import respond_node
    first = respond_node({"messages": [HumanMessage("hi")], "tool_used": None, "response": "How can I help?"})
    assert first["response"].startswith(f"Hello and welcome to {STORE_NAME}")
    assert "How can I help?" in first["response"]
    later = respond_node({"messages": [HumanMessage("hi"), AIMessage("..."), HumanMessage("ok")],
                          "tool_used": None, "response": "Sure."})
    assert later["response"] == "Sure."


def test_every_reply_has_quick_replies():
    from app.genai.graph import respond_node
    out = respond_node({"messages": [HumanMessage("hi")], "tool_used": None, "response": "Hi"})
    assert out["suggestions"]


def test_safety_beats_open_plan():
    out = turn("I smell gas near my fireplace")
    assert out["plan_action"] is None and route(out) == "create_ticket"


@pytest.mark.parametrize("intent, confidence, node", [
    ("PRODUCT_QA", 0.9, "product_qa"),
    ("OFFICE_PLAN", 0.8, "plan_office"),
    ("SUPPORT", 0.7, "create_ticket"),
    ("OTHER", 0.95, "agent"),
    ("SUPPORT", 0.4, "agent"),  # low confidence -> LLM picks the tool
])
def test_route(intent, confidence, node):
    assert route({"nlp": {"intent": intent, "confidence": confidence, "safety_flag": False}}) == node


def test_safety_always_routes_to_ticket():
    assert route({"nlp": {"intent": "PRODUCT_QA", "confidence": 0.3, "safety_flag": True}}) == "create_ticket"


def sid():
    return uuid.uuid4().hex


@pytest.mark.ollama
def test_office_plan_turn():
    r = chat(sid(), "Plan furniture for a 20-person office with one 8-seat meeting room, budget 6 lakh")
    assert r["tool_used"] == "plan_office"
    assert r["data"]["bom"] and r["data"]["estimated_total"] > 0
    assert r["nlp"]["entities"]["headcount"] == 20
    assert r["response"]


@pytest.mark.ollama
def test_gas_smell_turn_escalates_and_warns():
    r = chat(sid(), "I can smell gas from my Tahoe fireplace!")
    assert r["tool_used"] == "create_ticket"
    assert r["data"]["status"] == "ESCALATED"
    assert r["data"]["ticket_id"] in r["response"]
    assert "leave the building" in r["response"]


@pytest.mark.ollama
def test_product_question_turn_has_sources():
    r = chat(sid(), "How do I reset the Dimplex heater after it overheats?")
    assert r["tool_used"] == "product_qa"
    assert any(s["document"] == "dimplex.pdf" for s in r["sources"])


@pytest.mark.ollama
def test_user_reported_conversation_completes_the_plan():
    s = sid()
    assert chat(s, PLAN_START)["data"]["missing"] == ["headcount"]
    r = chat(s, "l have 30 emloyee")
    assert r["tool_used"] == "plan_office"
    assert r["data"]["requirements"]["headcount"] == 30 and r["data"]["bom"]
    r = chat(s, "1000sq feet")
    assert r["tool_used"] == "plan_office" and r["data"]["requirements"]["headcount"] == 30


@pytest.mark.ollama
def test_user_reported_conversation_2_choose_confirm_quote():
    s = sid()
    chat(s, "I want to furnish my office. Can you suggest furniture?")
    r = chat(s, "about 30 peoples and i need meeting room with same count as employees and my budget is aroun 6lakhs")
    assert r["data"]["requirements"]["meeting_rooms"] == [30] and r["data"]["within_budget"] is False
    assert r["data"]["alternatives"] and r["suggestions"]  # fitting plans offered as quick replies
    r = chat(s, r["suggestions"][0])  # tap "Make the meeting room for 12 people"
    assert r["data"]["requirements"]["headcount"] == 30 and r["data"]["within_budget"]
    r = chat(s, "I'll take the Basic option")
    assert r["tool_used"] == "plan_office" and r["data"]["chosen"] and r["data"]["tier"] == "budget"
    r = chat(s, "ok go for it")
    assert r["data"]["quote_id"].startswith("Q-") and r["data"]["quote_id"] in r["response"]
    r = chat(s, "ok")
    assert r["data"]["quote_id"] and "already" in r["response"].lower()


@pytest.mark.ollama
def test_memory_keeps_both_turns():
    s = sid()
    chat(s, "hello")
    chat(s, "What is your return policy?")
    msgs = history(s)
    assert [m.type for m in msgs] == ["human", "ai", "human", "ai"]
    assert history(sid()) == []


@pytest.mark.ollama
def test_choosing_an_option_returns_chosen_plan_with_photos():
    s = sid()
    chat(s, "Plan an office for 10 people, budget 3 lakh")
    r = chat(s, "I'll take the Premium option")
    assert r["tool_used"] == "plan_office"
    assert r["data"]["chosen"] and r["data"]["tier"] == "premium"
    assert all(line["photo"] for line in r["data"]["bom"])
    assert "Great choice" in r["response"]
