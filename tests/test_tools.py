import pandas as pd
import pytest

from app import db
from app.genai.schemas import IssueType, OfficeRequirements, TicketFields
from app.genai.tools import (add_photos, alternatives, compute_bom, create_ticket, detect_choice, escalate,
                             friendly_name, inr, is_ack, merge_requirements, no_promises, photo_key, plan_office,
                             plan_options, plan_reply, product_qa, update_requirements)
from app.ingest.ingest import load_catalog
from app.nlp.pipeline import analyze

ROWS = [  # sku, name, category, price, capacity
    ("D1", "Desk, 100 cm", "desk", 1000, None), ("D2", "Desk, 120 cm", "desk", 2000, None),
    ("D3", "Desk sit/stand, 120 cm", "desk", 3000, None),
    ("C1", "Office chair", "office_chair", 500, None), ("C2", "Office chair mesh", "office_chair", 800, None),
    ("C3", "Office chair pro", "office_chair", 1500, None),
    ("K1", "Conference chair", "conference_chair", 300, None), ("K2", "Conference chair arm", "conference_chair", 600, None),
    ("T4", "Table 4", "meeting_table", 5000, 4), ("T8", "Table 8", "meeting_table", 9000, 8),
    ("T12", "Table 12", "meeting_table", 15000, 12),
    ("S1", "Drawer unit", "storage", 700, None), ("S2", "Drawer unit big", "storage", 1200, None),
]
CATALOG = pd.DataFrame(ROWS, columns=["sku", "name", "category", "price_inr", "capacity"]).astype({"capacity": "Int64"})


def qty(bom, sku):
    return next((line["qty"] for line in bom["bom"] if line["sku"] == sku), 0)


def test_bom_standard_tier_hand_calculated():
    bom = compute_bom(OfficeRequirements(headcount=10, meeting_rooms=[8]), CATALOG)
    assert bom["tier"] == "standard"
    assert (qty(bom, "D2"), qty(bom, "C2"), qty(bom, "S2"), qty(bom, "T8"), qty(bom, "K2")) == (10, 10, 3, 1, 8)
    assert bom["estimated_total"] == 45_400  # 20000 + 8000 + 3600 + 9000 + 4800
    assert bom["within_budget"] is None


def test_bom_downgrades_to_fit_budget():
    bom = compute_bom(OfficeRequirements(headcount=10, meeting_rooms=[8], budget_inr=40_000), CATALOG)
    assert bom["tier"] == "budget"
    assert bom["estimated_total"] == 28_500  # 10000 + 5000 + 2100 + 9000 + 2400
    assert bom["within_budget"] is True


def test_bom_reports_over_budget_when_even_cheapest_is_too_much():
    bom = compute_bom(OfficeRequirements(headcount=10, meeting_rooms=[8], budget_inr=20_000), CATALOG)
    assert bom["tier"] == "budget" and bom["within_budget"] is False


def test_bom_big_room_uses_several_tables():
    bom = compute_bom(OfficeRequirements(headcount=1, meeting_rooms=[20]), CATALOG)
    assert qty(bom, "T12") == 2 and qty(bom, "K2") == 20


def test_bom_sit_stand_preference_and_premium():
    bom = compute_bom(OfficeRequirements(headcount=2, preferences=["sit-stand", "premium"]), CATALOG)
    assert bom["tier"] == "premium"
    assert qty(bom, "D3") == 2 and qty(bom, "C3") == 2


def test_bom_lines_come_from_catalog_only():
    bom = compute_bom(OfficeRequirements(headcount=7, meeting_rooms=[4, 6]), CATALOG)
    prices = dict(zip(CATALOG["sku"], CATALOG["price_inr"]))
    for line in bom["bom"]:
        assert line["unit_price_inr"] == prices[line["sku"]]
        assert line["line_total_inr"] == line["qty"] * line["unit_price_inr"]
    assert bom["estimated_total"] == sum(line["line_total_inr"] for line in bom["bom"])


def test_merge_requirements_trusts_nlp_numbers():
    llm = OfficeRequirements(headcount=300, meeting_rooms=[1], budget_inr=5, preferences=["premium"])
    nlp = {"headcount": 30, "meeting_room_seats": [10], "budget_inr": 500_000}
    merged = merge_requirements(llm, nlp, "30 people, 10-seat room, premium, 5 lakh")
    assert (merged.headcount, merged.meeting_rooms, merged.budget_inr) == (30, [10], 500_000)
    assert merged.preferences == ["premium"]


NO_ENTITIES = {"headcount": None, "meeting_room_seats": [], "budget_inr": None}


def test_merge_requirements_keeps_llm_numbers_found_in_text():
    llm = OfficeRequirements(headcount=12, meeting_rooms=[6], budget_inr=300_000)
    merged = merge_requirements(llm, NO_ENTITIES, "a dozen folks... say 12 of us, a room for six, 3 lakh")
    assert (merged.headcount, merged.meeting_rooms, merged.budget_inr) == (12, [6], 300_000)


def test_merge_requirements_money_digits_dont_ground_seats():
    llm = OfficeRequirements(headcount=30, meeting_rooms=[8], budget_inr=800_000)
    merged = merge_requirements(llm, NO_ENTITIES, "l have 30 emloyee\naround 8 lakh")
    assert (merged.headcount, merged.meeting_rooms, merged.budget_inr) == (30, [], 800_000)


def test_merge_requirements_drops_invented_preferences():
    llm = OfficeRequirements(headcount=12, preferences=["premium", "ergonomic", "sit-stand"])
    assert merge_requirements(llm, NO_ENTITIES, "12 people, standing desks please").preferences == ["sit-stand"]


def test_merge_requirements_drops_invented_llm_numbers():
    llm = OfficeRequirements(headcount=10, meeting_rooms=[10, 15], budget_inr=100_000)
    merged = merge_requirements(llm, NO_ENTITIES, "Can you plan my office furniture?")
    assert (merged.headcount, merged.meeting_rooms, merged.budget_inr) == (0, [], None)


def ticket(urgency="LOW", issue="OTHER"):
    return TicketFields(product="Tahoe", issue_type=issue, urgency=urgency, summary="x")


@pytest.mark.parametrize("llm_urgency, nlp_urgency, safety, status, urgency", [
    ("LOW", "HIGH", True, "ESCALATED", "HIGH"),     # safety always wins
    ("LOW", "HIGH", False, "ESCALATED", "HIGH"),    # NLP HIGH beats LLM
    ("HIGH", "MEDIUM", False, "OPEN", "MEDIUM"),    # LLM alone can't escalate (qwen3 1.7B over-escalates)
    ("MEDIUM", "LOW", False, "OPEN", "MEDIUM"),
    ("LOW", "LOW", False, "OPEN", "LOW"),
])
def test_escalation_rules(llm_urgency, nlp_urgency, safety, status, urgency):
    fields, result = escalate(ticket(llm_urgency), nlp_urgency, safety)
    assert result == status
    assert fields.urgency == urgency


def test_ticket_roundtrip(tmp_path):
    path = tmp_path / "t.db"
    tid = db.save_ticket({"product": "Tahoe", "issue_type": "GAS_ODOR", "urgency": "HIGH", "summary": "gas smell",
                          "status": "ESCALATED", "message": "I smell gas"}, path)
    t = db.get_ticket(tid, path)
    assert t["ticket_id"] == tid and t["status"] == "ESCALATED" and t["created_at"]
    assert db.get_ticket("T-nope", path) is None
    path.unlink()  # fails on Windows if a connection was left open


# ---------- LLM-backed tools (need Ollama + the built index) ----------

@pytest.mark.ollama
def test_plan_office_end_to_end_uses_catalog_prices():
    bom = plan_office(analyze("Plan furniture for a 30-person office with a 10-seat meeting room under ₹5 lakh"))
    req = bom["requirements"]
    assert (req["headcount"], req["meeting_rooms"], req["budget_inr"]) == (30, [10], 500_000)
    prices = dict(zip(load_catalog()["sku"], load_catalog()["price_inr"]))
    assert all(line["unit_price_inr"] == prices[line["sku"]] for line in bom["bom"])
    assert bom["estimated_total"] == sum(line["line_total_inr"] for line in bom["bom"])
    assert bom["within_budget"] == (bom["estimated_total"] <= 500_000)


@pytest.mark.ollama
def test_plan_office_asks_for_headcount_when_missing():
    assert plan_office(analyze("Can you plan my office furniture?"))["missing"] == ["headcount"]


@pytest.mark.ollama
def test_gas_smell_ticket_is_escalated(tmp_path):
    t = create_ticket(analyze("I smell gas near my Tahoe fireplace"), tmp_path / "t.db")
    assert (t["status"], t["urgency"]) == ("ESCALATED", "HIGH")
    assert "safety_message" in t
    assert db.get_ticket(t["ticket_id"], tmp_path / "t.db")["status"] == "ESCALATED"


@pytest.mark.ollama
def test_loose_armrest_ticket_stays_open(tmp_path):
    t = create_ticket(analyze("The armrest on my Markus chair is loose"), tmp_path / "t.db")
    assert t["status"] == "OPEN" and t["urgency"] in ("LOW", "MEDIUM")
    assert t["issue_type"] in IssueType.__members__  # which type is LLM quality: measured in eval/, not here


@pytest.mark.ollama
def test_product_qa_cites_manual_page():
    r = product_qa(analyze("My Dimplex heater overheated and shut off, how do I reset it?"))
    assert {"document": "dimplex.pdf", "page": 14} in [{k: s[k] for k in ("document", "page")} for s in r["sources"]]
    assert "unplug" in r["answer"].lower()


@pytest.mark.ollama
def test_product_qa_price_comes_from_catalog():
    r = product_qa(analyze("How much does the Tahoe fireplace cost?"))
    assert "206,710" in r["answer"] or "206710" in r["answer"]


@pytest.mark.ollama
def test_product_qa_off_topic_not_found():
    assert product_qa(analyze("What is the capital of France?"))["sources"] == []


# ---------- customer-friendly plans: plain names, options, budget question ----------

@pytest.mark.parametrize("n, text", [(1317860, "₹13,17,860"), (45400, "₹45,400"), (500, "₹500"), (12000000, "₹1,20,00,000")])
def test_inr_uses_indian_grouping(n, text):
    assert inr(n) == text


@pytest.mark.parametrize("sku, name", [
    ("D3", "Height-adjustable desk, 120 cm"),
    ("D1", "Office desk, 100 cm"),
    ("T8", "Meeting table for 8 people"),
    ("C1", "Office chair"),
    ("K2", "Meeting-room chair with armrests"),
    ("S1", "Drawer cabinet"),
])
def test_friendly_names_hide_model_codes(sku, name):
    row = CATALOG.set_index("sku").loc[sku]
    assert friendly_name({**row, "name": {"K2": "Conference chair with armrests"}.get(sku, row["name"])}) == name


def test_bom_lines_have_plain_item_and_model():
    line = compute_bom(OfficeRequirements(headcount=1), CATALOG)["bom"][0]
    assert line["item"] and line["model"] == line["name"]


def test_options_without_budget_show_all_tiers_and_ask():
    plan = plan_options(OfficeRequirements(headcount=10, meeting_rooms=[8]), CATALOG)
    assert [(o["tier"], o["total"]) for o in plan["options"]] == [("budget", 28_500), ("standard", 45_400), ("premium", 62_400)]
    assert plan["tier"] == "standard" and plan["needs_budget"] is True


def test_options_with_budget_recommend_best_fit():
    plan = plan_options(OfficeRequirements(headcount=10, meeting_rooms=[8], budget_inr=50_000), CATALOG)
    assert plan["tier"] == "standard" and plan["within_budget"] and plan["needs_budget"] is False
    assert [o["within_budget"] for o in plan["options"]] == [True, True, False]


def test_plan_reply_lists_options_in_plain_words_and_asks_budget():
    text = plan_reply(plan_options(OfficeRequirements(headcount=10, meeting_rooms=[8]), CATALOG))
    for s in ("₹28,500", "₹45,400", "₹62,400", "10 people", "budget"):
        assert s in text
    assert "?" in text  # asks the customer something


def test_plan_reply_with_budget_says_if_it_fits():
    over = plan_reply(plan_options(OfficeRequirements(headcount=10, meeting_rooms=[8], budget_inr=20_000), CATALOG))
    assert "₹20,000" in over and "₹28,500" in over and "above your budget" in over


# ---------- choosing a plan + photos ----------

@pytest.mark.parametrize("text, tier", [
    ("ok I'll go with the standard one", "standard"), ("Premium please", "premium"), ("option 3", "premium"),
    ("the cheapest", "budget"), ("basic is fine", "budget"), ("we are 30 people", None),
])
def test_detect_choice(text, tier):
    assert detect_choice(text) == tier


def test_chosen_tier_is_kept_even_over_budget():
    plan = plan_options(OfficeRequirements(headcount=10, meeting_rooms=[8], budget_inr=50_000), CATALOG, chosen="premium")
    assert (plan["tier"], plan["chosen"], plan["estimated_total"], plan["within_budget"]) == ("premium", True, 62_400, False)


def test_photo_key_per_item_type():
    assert photo_key({"category": "desk", "name": "Skarsta Desk sit/stand, 120x70 cm"}) == "sitstand_desk"
    assert photo_key({"category": "desk", "name": "Micke Desk, 73x50 cm"}) == "desk"
    assert photo_key({"category": "meeting_table", "name": "Bekant Conference table"}) == "meeting_table"


def test_lines_get_photo_and_credit_when_file_exists(tmp_path):
    (tmp_path / "office_chair.jpg").write_bytes(b"x")
    credits = {"office_chair": "Photo: A. Person, CC BY-SA 4.0, Wikimedia Commons"}
    lines = add_photos([{"category": "office_chair", "name": "Markus Office chair"},
                        {"category": "storage", "name": "Alex Drawer unit"}], tmp_path, credits)
    assert lines[0]["photo"] == "/images/office_chair.jpg" and lines[0]["photo_credit"].startswith("Photo:")
    assert lines[1]["photo"] is None


def test_chosen_plan_reply_confirms_and_offers_next_step():
    text = plan_reply(plan_options(OfficeRequirements(headcount=10, meeting_rooms=[8]), CATALOG, chosen="standard"))
    assert "Great choice" in text and "₹45,400" in text and "photo" in text.lower() and "?" in text


# ---------- conversation basics: slot updates, confirmations, alternatives, quotes ----------

@pytest.mark.parametrize("text, ack", [("ok", True), ("ok go for it", True), ("yes please", True), ("Confirm this plan", True),
                                       ("sounds good", True), ("ok but what about warranty on the chair?", False),
                                       ("no", False), ("30 people", False)])
def test_is_ack(text, ack):
    assert is_ack(text) is ack


def test_update_keeps_old_values_and_applies_new_ones():
    req = OfficeRequirements(headcount=30, meeting_rooms=[30], budget_inr=600_000)
    out = update_requirements(req, {"headcount": 40, "meeting_room_seats": [], "budget_inr": None}, "make it 40 people")
    assert (out.headcount, out.meeting_rooms, out.budget_inr, out.storage) == (40, [30], 600_000, True)


@pytest.mark.parametrize("text, rooms, storage", [
    ("Plan without a meeting room", [], True), ("skip the drawer cabinets", [30], False),
    ("no meeting room and no storage", [], False),
])
def test_update_removals(text, rooms, storage):
    req = OfficeRequirements(headcount=30, meeting_rooms=[30], budget_inr=600_000)
    out = update_requirements(req, NO_ENTITIES, text)
    assert (out.meeting_rooms, out.storage) == (rooms, storage)


def test_no_storage_drops_cabinets_from_bom():
    bom = compute_bom(OfficeRequirements(headcount=10, storage=False), CATALOG, "budget")
    assert bom["estimated_total"] == 15_000 and qty(bom, "S1") == 0


def test_alternatives_only_list_plans_that_fit():
    req = OfficeRequirements(headcount=10, meeting_rooms=[20], budget_inr=30_000)
    alts = alternatives(req, CATALOG)
    assert [(a["message"], a["total"]) for a in alts] == [("Plan without a meeting room", 17_100)]
    req = req.model_copy(update={"budget_inr": 40_000})
    assert [a["message"] for a in alternatives(req, CATALOG)] == ["Make the meeting room for 12 people",
                                                                 "Plan without a meeting room"]


def test_over_budget_reply_offers_fitting_alternatives():
    plan = plan_options(OfficeRequirements(headcount=10, meeting_rooms=[20], budget_inr=30_000), CATALOG)
    plan["alternatives"] = alternatives(OfficeRequirements(headcount=10, meeting_rooms=[20], budget_inr=30_000), CATALOG)
    text = plan_reply(plan)
    assert text.startswith("Got it: 10 people, a meeting room for 20, budget ₹30,000.")
    assert "₹17,100" in text and "without a meeting room" in text


def test_quote_roundtrip(tmp_path):
    path = tmp_path / "q.db"
    qid = db.save_quote({"tier": "standard", "estimated_total": 45_400, "bom": []}, path)
    q = db.get_quote(qid, path)
    assert qid.startswith("Q-") and q["tier"] == "standard" and q["total"] == 45_400
    path.unlink()


def test_quote_reply_gives_id_and_honest_next_step():
    plan = plan_options(OfficeRequirements(headcount=10), CATALOG, chosen="standard")
    plan["quote_id"] = "Q-1234ABCD"
    text = plan_reply(plan)
    assert "Q-1234ABCD" in text and "₹" in text and "call you" not in text.lower()


@pytest.mark.parametrize("text, safe", [("Let me check the details for you.", False),
                                        ("I'll calculate the cost for you.", False),
                                        ("We sell desks and chairs.", True)])
def test_promise_guard(text, safe):
    assert (no_promises(text) == text) is safe


def test_room_size_digits_dont_ground_headcount():
    llm = OfficeRequirements(headcount=12, meeting_rooms=[12])
    merged = merge_requirements(llm, {"headcount": None, "meeting_room_seats": [12], "budget_inr": None},
                                "Make the meeting room for 12 people")
    assert (merged.headcount, merged.meeting_rooms) == (0, [12])


def test_within_budget_reply_names_the_recommended_option():
    plan = plan_options(OfficeRequirements(headcount=10, budget_inr=30_000), CATALOG)
    assert plan["tier"] == "budget"
    assert "The **Basic** option fits your budget" in plan_reply(plan)
