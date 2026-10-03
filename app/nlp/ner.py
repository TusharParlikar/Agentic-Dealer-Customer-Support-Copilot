"""Entities: PRODUCT (spaCy EntityRuler from the catalog), LOCATION (spaCy NER), headcount / meeting-room
seats (regex) and budget (normalizer)."""
import re
from functools import cache

import spacy

from app.ingest.ingest import load_catalog, product_patterns
from app.nlp.normalize import find_amounts

# ponytail: single words only; "twenty five" reads as 20. Add a compound parser if customers write those.
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
NUMBER_WORDS |= {w: str(i * 10) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
NUMBER_WORD = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)

# en_core_web_sm misses many Indian cities; canonical name -> spellings customers use.
CITIES = {
    "Mumbai": ["mumbai", "bombay"], "Delhi": ["delhi", "new delhi"], "Bengaluru": ["bengaluru", "bangalore"],
    "Hyderabad": ["hyderabad"], "Chennai": ["chennai", "madras"], "Kolkata": ["kolkata", "calcutta"],
    "Pune": ["pune"], "Ahmedabad": ["ahmedabad"], "Jaipur": ["jaipur"], "Noida": ["noida"],
    "Gurugram": ["gurugram", "gurgaon"], "Kochi": ["kochi", "cochin"], "Chandigarh": ["chandigarh"],
    "Lucknow": ["lucknow"], "Indore": ["indore"], "Nagpur": ["nagpur"], "Coimbatore": ["coimbatore"],
}
CITY_PATTERNS = [{"label": "GPE", "pattern": [{"LOWER": t} for t in alias.split()], "id": city}
                 for city, aliases in CITIES.items() for alias in aliases]

ROOM = r"(?:meeting|conference|conf|board|discussion|training)\s*rooms?|boardrooms?"
SEAT = r"(?:seat(?:er)?s?|person|people|pax)"
# "two 8-seat conference rooms", "a 10 seater meeting room"
ROOMS_SEATS_FIRST = re.compile(rf"(?:\b(?P<count>\d+)\s+)?\b(?P<seats>\d+)[- ]?{SEAT}\s+(?:{ROOM})", re.I)
# "a boardroom for 12 people", "2 meeting rooms of 8 seats", "conference room for 16"
ROOMS_SEATS_AFTER = re.compile(
    rf"(?:\b(?P<count>\d+)\s+)?(?:{ROOM})\s+(?:for|with|of|seating)\s+(?P<seats>\d+)(?:\s*{SEAT})?(?:\s+each)?", re.I)
# emp?l[a-z]* accepts typos customers actually send: "emloyee", "employes", "emplyees"
HEADCOUNT = re.compile(
    r"\b(?P<n>\d+)[- ]?(?:person|people|ppl|persons|emp?l[a-z]*|staff|members?|pax|workstations?|desks?|agents?"
    r"|developers?|designers?|lawyers?|executives?|users?|seat(?:er)?s?\s+office)\b"
    r"|\bteam of (?P<team>\d+)\b|\boffice (?:of|for) (?P<office>\d+)\b",
    re.IGNORECASE,
)


@cache
def _nlp():
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(product_patterns(load_catalog()) + CITY_PATTERNS)
    return nlp


def _rooms(text: str) -> tuple[list[int], str]:
    """Meeting-room seat counts, plus the text with those phrases removed so they aren't read as headcount."""
    seats = []
    for pattern in (ROOMS_SEATS_FIRST, ROOMS_SEATS_AFTER):
        for m in pattern.finditer(text):
            seats += [int(m["seats"])] * int(m["count"] or 1)
        text = pattern.sub(" ", text)
    return seats, text


def strip_rooms(text: str) -> str:
    """Text without meeting-room phrases: the 12 in "a meeting room for 12 people" is seats, not headcount."""
    return _rooms(NUMBER_WORD.sub(lambda m: NUMBER_WORDS[m[1].lower()], text))[1]


def extract_entities(text: str) -> dict:
    doc = _nlp()(text)
    products = list(dict.fromkeys(e.ent_id_ for e in doc.ents if e.label_ == "PRODUCT"))
    location = list(dict.fromkeys(e.ent_id_ or e.text for e in doc.ents if e.label_ in ("GPE", "LOC")))

    digits = NUMBER_WORD.sub(lambda m: NUMBER_WORDS[m[1].lower()], text)
    seats, rest = _rooms(digits)
    m = HEADCOUNT.search(rest)
    headcount = int(m["n"] or m["team"] or m["office"]) if m else None
    amounts = find_amounts(text)

    return {
        "products": products,
        "headcount": headcount,
        "meeting_room_seats": seats,
        "budget_inr": max(amounts) if amounts else None,
        "location": location,
    }
