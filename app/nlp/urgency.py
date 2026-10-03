"""Keyword rules for urgency and safety. Rules, not a model: safety must be predictable and auditable."""
import re

# ponytail: no negation handling ("no gas smell" still flags). Over-escalating a safety call is the safe error.
SAFETY = re.compile(
    r"gas (smell|leak|odou?r)|smell(ing|s)? (of )?gas|leak(ing)? gas|carbon monoxide|\bco (alarm|detector)"
    r"|smoke (is )?(coming|in|into|filling|everywhere)|(smoking|leaking smoke|smoke leak)"
    r"|burning smell|smell(s)? (of )?burning|sparks?\b|electric shock|caught fire|on fire",
    re.IGNORECASE,
)
HIGH = re.compile(
    r"urgent|asap|immediately|emergency|right away|danger|injur|hurt"
    r"|glass (door )?(is )?(cracked|broken|shattered)|cracked glass|shattered",
    re.IGNORECASE,
)
MEDIUM = re.compile(
    r"broken|not working|doesn'?t|does not|won'?t|will not|stopped|keeps|sinking|wobbl|squeak|missing"
    r"|damaged|defect|leak|error|fault|stuck|loose|crack|jammed|noisy|dead",
    re.IGNORECASE,
)


def assess(text: str) -> tuple[str, bool]:
    """-> (urgency LOW|MEDIUM|HIGH, safety_flag)."""
    if SAFETY.search(text):
        return "HIGH", True
    if HIGH.search(text):
        return "HIGH", False
    if MEDIUM.search(text):
        return "MEDIUM", False
    return "LOW", False
