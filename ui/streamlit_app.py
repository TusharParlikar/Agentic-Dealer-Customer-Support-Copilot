"""Chat UI for the Dealer Copilot. Talks to the FastAPI backend: uvicorn app.main:app"""
import os
import uuid

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
QUICK_STARTS = [
    ("🪑 Plan my office", "I want to furnish my office. Can you suggest furniture?"),
    ("📘 Product help", "What is the warranty on your furniture?"),
    ("🛠️ Report a problem", "Something I bought is broken and I need help."),
]
TIER_NAMES = {"budget": "Basic", "standard": "Standard", "premium": "Premium"}

st.set_page_config(page_title="Store Assistant", page_icon="🛋️", layout="centered")


def inr(n: int) -> str:
    """₹ with Indian grouping (same rule as app.genai.tools.inr; the UI only talks to the API)."""
    s = str(int(n))
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "₹" + ",".join(([head] if head else []) + groups + [tail])


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.turns = []  # [{"role": "user"|"assistant", "content": str, "result": dict|None}]
    try:
        st.session_state.store_name = requests.get(f"{API_URL}/health", timeout=3).json()["store_name"]
    except (requests.RequestException, KeyError, ValueError):
        st.session_state.store_name = "our store"

st.title(f"🛋️ {st.session_state.store_name}")
st.caption("Office furniture · Fireplaces & stoves · Ask me anything, I'm happy to help")

clicked = None
if not st.session_state.turns:
    st.info("Hi! Type your question below, or tap one of these to get started:")
for col, (label, text) in zip(st.columns(len(QUICK_STARTS)), QUICK_STARTS):
    if col.button(label, width="stretch"):
        clicked = text


def render_plan(data: dict, key: int) -> str | None:
    """Option cards with Choose buttons (until one is chosen), item table, budget badge, photos once chosen.
    Returns the message to send if a Choose button was clicked."""
    clicked = None
    if data.get("quote_id"):
        st.success(f"✅ **Quote {data['quote_id']}** saved: {TIER_NAMES[data['tier']]} plan, "
                   f"{inr(data['estimated_total'])} in total. Mention this number when you call or visit us.")
    if data.get("options") and not data.get("quote_id"):
        cols = st.columns(len(data["options"]))
        for col, o in zip(cols, data["options"]):
            name = TIER_NAMES[o["tier"]]
            col.metric(name, inr(o["total"]))
            tags = (["✅ fits budget"] if o["within_budget"] else []) + (["⭐ recommended"] if o["tier"] == data["tier"] else [])
            col.caption(" · ".join(tags) or " ")
            if not data.get("chosen") and col.button(f"Choose {name}", key=f"choose-{key}-{o['tier']}", width="stretch"):
                clicked = f"I'll take the {name} option"
    st.markdown(f"**{TIER_NAMES.get(data.get('tier'), '')} option: what's included**")
    st.dataframe(pd.DataFrame([{"Item": line["item"], "Qty": line["qty"], "Price each": inr(line["unit_price_inr"]),
                                "Total": inr(line["line_total_inr"]), "Model": line["model"]} for line in data["bom"]]),
                 hide_index=True, width="stretch")
    if data.get("within_budget") is True:
        st.success(f"Fits your budget of {inr(data['budget_inr'])}")
    elif data.get("within_budget") is False:
        st.warning(f"Above your budget of {inr(data['budget_inr'])}")
    if data.get("chosen"):
        render_photos(data["bom"])
    return clicked


def render_photos(bom: list[dict]) -> None:
    """Example photo per item type, shown once the customer has picked an option."""
    items = [line for line in bom if line.get("photo")]
    if not items:
        return
    st.markdown("**What your furniture looks like**")
    for row in range(0, len(items), 3):
        for col, line in zip(st.columns(3), items[row:row + 3]):
            col.image(API_URL + line["photo"], caption=f"{line['item']} × {line['qty']}", width="stretch")
            col.caption(line["photo_credit"])


def render_result(result: dict, key: int) -> str | None:
    tool, data = result.get("tool_used"), result.get("data") or {}
    if tool == "product_qa" and result.get("sources"):
        with st.expander("📄 Where this answer comes from"):
            for s in result["sources"]:
                st.markdown(f"- `{s['document']}` page {s['page']}")
    elif tool == "plan_office" and data.get("bom"):
        return render_plan(data, key)
    elif tool == "create_ticket" and data.get("ticket_id"):
        card = (f"**Request {data['ticket_id']}**: {data['status'].title()}  \n"
                f"{data['summary']}  \n"
                f"Product: {data.get('product') or 'not given'} · Priority: {data['urgency'].title()}")
        (st.error if data["status"] == "ESCALATED" else st.info)(card)


def render_quick_replies(result: dict, key: int) -> str | None:
    """Next-step buttons under the latest answer, phrased as what the customer would say."""
    clicked = None
    options = result.get("suggestions") or []
    for j, (col, text) in enumerate(zip(st.columns(max(len(options), 1)), options)):
        if col.button(text, key=f"quick-{key}-{j}", width="stretch"):
            clicked = text
    return clicked


def render_nlp(nlp: dict) -> None:
    with st.sidebar.expander("🔍 How I understood your last message", expanded=True):
        st.markdown(f"**Intent:** {nlp['intent']} ({nlp['confidence']:.0%})")
        if nlp.get("urgency"):
            st.markdown(f"**Urgency:** {nlp['urgency']}" + (" · ⚠️ safety" if nlp.get("safety_flag") else ""))
        st.json({k: v for k, v in nlp["entities"].items() if v})


for i, turn in enumerate(st.session_state.turns):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["result"]:
            clicked = render_result(turn["result"], i) or clicked
            if i == len(st.session_state.turns) - 1:  # quick replies only for the latest answer
                clicked = render_quick_replies(turn["result"], i) or clicked

typed = st.chat_input("Ask about a product, plan an office, or report a problem")
if prompt := typed or clicked:
    st.session_state.turns.append({"role": "user", "content": prompt, "result": None})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                r = requests.post(f"{API_URL}/chat", json={"session_id": st.session_state.session_id,
                                                           "message": prompt}, timeout=300)
                r.raise_for_status()
                result = r.json()
        except requests.RequestException as e:
            st.error(f"Sorry, I couldn't reach the store's API at {API_URL}. Please try again. ({e})")
        else:
            st.markdown(result["response"])
            render_result(result, len(st.session_state.turns))  # same key it gets in the history loop
            render_quick_replies(result, len(st.session_state.turns))
            st.session_state.turns.append({"role": "assistant", "content": result["response"], "result": result})
            st.session_state.last_nlp = result["nlp"]

if "last_nlp" in st.session_state:
    render_nlp(st.session_state.last_nlp)
if st.sidebar.button("🔄 Start a new chat"):
    for key in ("session_id", "turns", "last_nlp"):
        st.session_state.pop(key, None)
    st.rerun()
