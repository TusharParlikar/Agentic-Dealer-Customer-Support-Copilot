import pytest

from app.nlp.normalize import find_amounts, normalize
from app.nlp.ner import extract_entities
from app.nlp.urgency import assess
from app.nlp.intent import predict
from app.nlp.pipeline import analyze


@pytest.mark.parametrize("text, amounts", [
    ("budget is 5 lakh", [500_000]),
    ("under ₹5,00,000 please", [500_000]),
    ("around 2.5L total", [250_000]),
    ("about 10k per chair", [10_000]),
    ("Rs. 45,000 max", [45_000]),
    ("1.2 crore project", [12_000_000]),
    ("INR 75000", [75_000]),
    ("30000 rupees", [30_000]),
    ("30 people and a 10-seat room", []),
    ("model DVCD36FP70N", []),
])
def test_find_amounts(text, amounts):
    assert find_amounts(text) == amounts


def test_normalize_collapses_whitespace_and_fixes_quotes():
    assert normalize("  my   chair’s  broken \n") == "my chair's broken"


def test_entities_for_office_plan():
    e = extract_entities("Plan furniture for a 30-person office in Pune with a 10-seat meeting room under ₹5 lakh")
    assert e["headcount"] == 30
    assert e["meeting_room_seats"] == [10]
    assert e["budget_inr"] == 500_000
    assert e["location"] == ["Pune"]


@pytest.mark.parametrize("text, city", [
    ("Set up our Chennai office for 25 staff", "Chennai"),
    ("a team of 12 in Hyderabad", "Hyderabad"),
    ("startup of 8 developers in bangalore", "Bengaluru"),
])
def test_entities_indian_cities(text, city):
    assert extract_entities(text)["location"] == [city]


@pytest.mark.parametrize("text", ["l have 30 emloyee", "30 employes", "we are 30 emplyees"])
def test_entities_headcount_survives_typos(text):
    assert extract_entities(text)["headcount"] == 30


def test_entities_number_words():
    assert extract_entities("an office for twenty people")["headcount"] == 20
    assert extract_entities("fifteen employees and a meeting room for six")["meeting_room_seats"] == [6]


def test_entities_meeting_rooms_not_counted_as_headcount():
    e = extract_entities("We have 45 employees and need two 8-seat conference rooms and a boardroom for 12 people")
    assert e["headcount"] == 45
    assert sorted(e["meeting_room_seats"]) == [8, 8, 12]


def test_entities_products_any_spelling():
    e = extract_entities("Is the langfjall chair better than the Markus? Also does Tahoe need a vent?")
    assert e["products"] == ["Långfjäll", "Markus", "Tahoe"]


@pytest.mark.parametrize("text, urgency, safety", [
    ("I can smell gas near the fireplace", "HIGH", True),
    ("smoke is coming into the room from the stove", "HIGH", True),
    ("the chair lever is broken", "MEDIUM", False),
    ("need help urgently, glass door cracked", "HIGH", False),
    ("when will my order arrive", "LOW", False),
])
def test_urgency(text, urgency, safety):
    assert assess(text) == (urgency, safety)


@pytest.mark.parametrize("text, intent", [
    ("What is the warranty period on the Dimplex electric fireplace?", "PRODUCT_QA"),
    ("How far should furniture be kept from the Drolet wood stove?", "PRODUCT_QA"),
    ("Does the Bekant desk height adjust?", "PRODUCT_QA"),
    ("Can I return a desk if I don't like it?", "PRODUCT_QA"),
    ("We're setting up an office for 25 staff, what furniture do we need?", "OFFICE_PLAN"),
    ("Suggest desks and chairs for a 12 member startup, budget 3 lakh", "OFFICE_PLAN"),
    ("Need a quote to furnish a new branch: 40 workstations and 2 meeting rooms", "OFFICE_PLAN"),
    ("My office chair keeps sinking when I sit on it", "SUPPORT"),
    ("The pilot light on my gas fireplace won't stay lit", "SUPPORT"),
    ("Delivery guy left and two drawers are missing from the box", "SUPPORT"),
    ("I forgot my account password", "OTHER"),
    ("hi there", "OTHER"),
])
def test_intent_on_unseen_messages(text, intent):
    label, confidence = predict(text)
    assert label == intent
    assert 0 < confidence <= 1


def test_pipeline_result_shape():
    r = analyze("I smell gas from my Tahoe fireplace, please send someone!")
    assert r.intent == "SUPPORT"
    assert r.urgency == "HIGH" and r.safety_flag
    assert r.entities["products"] == ["Tahoe"]


def test_pipeline_no_urgency_for_planning():
    r = analyze("Plan an office for 20 people with budget 4 lakh")
    assert r.intent == "OFFICE_PLAN"
    assert r.urgency is None and not r.safety_flag
    assert r.entities["budget_inr"] == 400_000
