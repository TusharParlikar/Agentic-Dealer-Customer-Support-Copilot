"""Intent classifier: TF-IDF + Logistic Regression.

Training data = public Bitext customer-support messages (remapped to our intents) + hand-written
messages for what Bitext doesn't cover (office planning, product questions, product faults).
"""
import re
from functools import cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union

ROOT = Path(__file__).resolve().parents[2]
BITEXT = ROOT / "store_data" / "raw" / "bitext.csv"
HANDWRITTEN = ROOT / "nlp_data" / "handwritten.csv"
DATASET = ROOT / "nlp_data" / "intents.csv"
MODEL = ROOT / "models" / "intent_clf.joblib"

# Bitext intent -> (our intent, examples to sample)
BITEXT_MAP = {
    # policy questions, answered from the policy PDFs
    "delivery_options": ("PRODUCT_QA", 40), "check_refund_policy": ("PRODUCT_QA", 40),
    "check_cancellation_fee": ("PRODUCT_QA", 20),
    # order / service problems -> ticket
    "complaint": ("SUPPORT", 20), "track_order": ("SUPPORT", 12), "delivery_period": ("SUPPORT", 12),
    "get_refund": ("SUPPORT", 12), "track_refund": ("SUPPORT", 10), "payment_issue": ("SUPPORT", 10),
    "cancel_order": ("SUPPORT", 10), "change_order": ("SUPPORT", 10), "contact_human_agent": ("SUPPORT", 10),
    "contact_customer_service": ("SUPPORT", 10),
    # account, invoices, newsletter... out of scope for this copilot
    **{i: ("OTHER", 12) for i in (
        "create_account", "delete_account", "edit_account", "recover_password", "registration_problems",
        "switch_account", "newsletter_subscription", "check_invoice", "get_invoice", "check_payment_methods",
        "review", "place_order", "change_shipping_address", "set_up_shipping_address")},
}
PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")  # Bitext slots like {{Order Number}}


def build_dataset() -> pd.DataFrame:
    bitext = pd.read_csv(BITEXT)
    parts = []
    for bitext_intent, (intent, n) in BITEXT_MAP.items():
        sample = bitext[bitext["intent"] == bitext_intent].sample(n, random_state=42)
        parts.append(pd.DataFrame({"text": sample["instruction"], "intent": intent, "source": "bitext"}))
    hand = pd.read_csv(HANDWRITTEN).assign(source="handwritten")
    df = pd.concat(parts + [hand], ignore_index=True)
    df["text"] = df["text"].str.replace(PLACEHOLDER, "it", regex=True).str.strip()
    df.to_csv(DATASET, index=False)
    return df


def _model():
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),  # typos, "langfjall"
    )
    return make_pipeline(features, LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))


def train() -> dict:
    df = build_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["intent"], test_size=0.2,
                                              stratify=df["intent"], random_state=42)
    held_out = _model().fit(X_tr, y_tr).predict(X_te)
    metrics = {"n": len(df), "accuracy": accuracy_score(y_te, held_out),
               "macro_f1": f1_score(y_te, held_out, average="macro"),
               "report": classification_report(y_te, held_out)}
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(_model().fit(df["text"], df["intent"]), MODEL)  # final model sees all data
    predict_model.cache_clear()
    return metrics


@cache
def predict_model():
    if not MODEL.exists():
        train()
    return joblib.load(MODEL)


def predict(text: str) -> tuple[str, float]:
    clf = predict_model()
    probs = clf.predict_proba([text])[0]
    best = probs.argmax()
    return str(clf.classes_[best]), float(probs[best])


if __name__ == "__main__":
    m = train()
    print(f"{m['n']} examples | held-out accuracy {m['accuracy']:.3f} | macro-F1 {m['macro_f1']:.3f}")
    print(m["report"])
