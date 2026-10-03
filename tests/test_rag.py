import pytest

from app.genai.retriever import manuals_for, retrieve


def test_manuals_for_products():
    assert manuals_for(["Tahoe"]) == {"tahoe.pdf"}
    assert manuals_for(["Tahoe", "Dimplex"]) == {"tahoe.pdf", "dimplex.pdf"}
    assert manuals_for(["Markus"]) is None  # IKEA lines have no manual: search everything
    assert manuals_for([]) is None


def _pages(hits):
    return {(d.metadata["source"], d.metadata["page"]) for d, _ in hits}


@pytest.mark.ollama
@pytest.mark.parametrize("question, products, expected", [
    ("The heater overheated and shut off, how do I reset it?", ["Dimplex"], {("dimplex.pdf", 14)}),
    ("How do I light the fireplace?", ["Tahoe"], {("tahoe.pdf", 61), ("tahoe.pdf", 65)}),
    ("What clearance is needed between the stove and combustible walls?", ["Drolet"],
     {("drolet.pdf", p) for p in (31, 32, 33, 34)}),
    ("My standing desk moves down but not up, how do I reset the control box?", [], {("standing_desk.pdf", 4)}),
    ("Are delivery fees refundable if I return something?", [], {("terms_delivery_returns.pdf", 1)}),
])
def test_retrieve_finds_answer_page(question, products, expected):
    hits = retrieve(question, products)
    assert _pages(hits) & expected, _pages(hits)


@pytest.mark.ollama
def test_product_filter_limits_to_its_manual():
    hits = retrieve("what is the range of the remote control", ["Dimplex"])
    assert hits and {d.metadata["source"] for d, _ in hits} == {"dimplex.pdf"}


@pytest.mark.ollama
def test_off_topic_question_returns_nothing():
    assert retrieve("What is the capital of France?", []) == []
