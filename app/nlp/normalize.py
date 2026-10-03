"""Text cleanup and Indian-style money parsing ("5 lakh", "₹5,00,000", "10k", "1.2 crore")."""
import re

UNITS = {"k": 1_000, "l": 100_000, "lac": 100_000, "lacs": 100_000, "lakh": 100_000, "lakhs": 100_000,
         "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000}

MONEY = re.compile(
    r"(?P<cur>₹|\brs\.?|\binr)?\s*"
    r"(?<![a-z0-9])(?P<num>\d+(?:,\d+)*(?:\.\d+)?)\s*"
    r"(?P<unit>lakhs?|lacs?|crores?|cr|k|l)?\b"
    r"(?:\s*(?P<suf>rupees|rs|inr)\b)?",
    re.IGNORECASE,
)

QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    return " ".join(text.translate(QUOTES).split())


def _money(text: str) -> list[re.Match]:
    """A bare number ("30 people") is not money; it needs ₹/Rs/INR or a unit."""
    return [m for m in MONEY.finditer(text) if m["cur"] or m["unit"] or m["suf"]]


def find_amounts(text: str) -> list[int]:
    """Money amounts in INR."""
    return [round(float(m["num"].replace(",", "")) * UNITS.get((m["unit"] or "").lower(), 1)) for m in _money(text)]


def strip_amounts(text: str) -> str:
    """Text with money removed, so the 8 in "8 lakh" can't be read as 8 seats."""
    for m in reversed(_money(text)):
        text = text[:m.start()] + " " + text[m.end():]
    return text
