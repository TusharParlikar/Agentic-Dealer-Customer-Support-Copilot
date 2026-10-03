from pydantic import BaseModel

from app.nlp.intent import predict
from app.nlp.ner import extract_entities
from app.nlp.normalize import normalize
from app.nlp.urgency import assess


class NLPResult(BaseModel):
    text: str
    intent: str
    confidence: float
    entities: dict
    urgency: str | None = None
    safety_flag: bool = False


def analyze(message: str) -> NLPResult:
    text = normalize(message)
    intent, confidence = predict(text)
    urgency, safety = assess(text)
    return NLPResult(
        text=text,
        intent=intent,
        confidence=round(confidence, 3),
        entities=extract_entities(text),
        # Urgency only matters for support; a safety hit always counts, whatever the intent.
        urgency=urgency if intent == "SUPPORT" or safety else None,
        safety_flag=safety,
    )
