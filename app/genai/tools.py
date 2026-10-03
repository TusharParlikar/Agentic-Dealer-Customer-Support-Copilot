"""The three tools. LLMs only fill structured fields or write prose; numbers, SKUs, prices and
escalation come from deterministic code."""
import json
import math
import re
from functools import cache

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app import db
from app.config import get_llm
from app.genai import prompts
from app.genai.retriever import retrieve
from app.genai.schemas import OfficeRequirements, TicketFields
from app.ingest.ingest import STORE, load_catalog
from app.nlp.ner import NUMBER_WORDS, strip_rooms
from app.nlp.normalize import find_amounts, strip_amounts
from app.nlp.pipeline import NLPResult

CITATION = re.compile(r"\[(\d+)\]")
IMAGES_DIR = STORE / "images"  # example photos, downloaded by app.ingest.fetch_data


# ---------- product_qa: RAG with page citations ----------

def _catalog_facts(products: list[str]) -> str:
    catalog = load_catalog()
    rows = catalog[catalog["product_line"].isin(products)]
    return "\n".join(f"{r['name']} (SKU {r['sku']}): ₹{r['price_inr']:,}" + (f", {r['dims_cm']} cm" if r["dims_cm"] else "")
                     for _, r in rows.iterrows())


def product_qa(nlp: NLPResult) -> dict:
    hits = retrieve(nlp.text, nlp.entities["products"])
    facts = _catalog_facts(nlp.entities["products"])
    if not hits and not facts:
        return {"answer": prompts.NOT_FOUND, "sources": []}

    blocks = [f"[{i}] ({d.metadata['source']} p.{d.metadata['page']})\n{d.page_content}" for i, (d, _) in enumerate(hits, 1)]
    if facts:
        blocks.append(f"[{len(hits) + 1}] (store catalog)\n{facts}")
    answer = get_llm().invoke([SystemMessage(prompts.QA_SYSTEM),
                               HumanMessage("Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {nlp.text}")]).content

    cited = {int(n) for n in CITATION.findall(answer)}
    used = [h for i, h in enumerate(hits, 1) if i in cited] or hits  # small models often forget to cite
    sources = list({(d.metadata["source"], d.metadata["page"]): {"document": d.metadata["source"], "page": d.metadata["page"],
                                                                 "score": round(s, 3)} for d, s in used}.values())
    return {"answer": answer, "sources": sources}


# ---------- plan_office: bill of materials ----------

# ---------- plan_office: bill of materials ----------

TIERS = ["budget", "standard", "premium"]
PREMIUM = re.compile(r"premium|executive|ergonomic|high.?end|luxury", re.I)
BUDGET = re.compile(r"cheap|economy|low.?cost|basic|affordable", re.I)  # not "budget": "budget 3 lakh" is an amount
SIT_STAND = re.compile(r"sit.?stand|standing|height.?adjust", re.I)
PEOPLE_PER_STORAGE = 4


LABELS = {"desk": "Office desk", "office_chair": "Office chair", "conference_chair": "Meeting-room chair",
          "meeting_table": "Meeting table", "storage": "Drawer cabinet", "gas_fireplace": "Gas fireplace",
          "wood_stove": "Wood stove", "electric_fireplace": "Electric fireplace"}
TIER_NAMES = {"budget": ("Basic", "simple, lowest price"), "standard": ("Standard", "good quality, best value"),
              "premium": ("Premium", "top comfort and finish")}


def inr(n: int) -> str:
    """Indian digit grouping: 1317860 -> ₹13,17,860."""
    s = str(int(n))
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "₹" + ",".join(([head] if head else []) + groups + [tail])


def friendly_name(row) -> str:
    """What a customer understands: 'Office desk, 120x40 cm', not 'Bestå Burs Desk, 120x40 cm'."""
    name, category = row["name"].lower(), row["category"]
    if category == "meeting_table":
        return f"Meeting table for {row['capacity']} people"
    label = "Height-adjustable desk" if category == "desk" and "sit/stand" in name else LABELS[category]
    if "armrest" in name:
        label += " with armrests"
    size = re.search(r"(\d+(?:x\d+)*) cm", name)
    return f"{label}, {size[1]} cm" if size else label


def _pick(catalog: pd.DataFrame, category: str, tier: str, prefer: str | None = None) -> pd.Series:
    items = catalog[catalog["category"] == category].sort_values("price_inr", kind="stable")
    if prefer:
        preferred = items[items["name"].str.contains(prefer, case=False, regex=False)]
        items = preferred if len(preferred) else items
    return items.iloc[{"budget": 0, "standard": len(items) // 2, "premium": len(items) - 1}[tier]]


def _tables(catalog: pd.DataFrame, seats: int) -> tuple[pd.Series, int]:
    """Smallest table that seats everyone, else several of the largest."""
    tables = catalog[catalog["category"] == "meeting_table"].sort_values(["capacity", "price_inr"])
    fits = tables[tables["capacity"] >= seats]
    if len(fits):
        return fits.iloc[0], 1
    largest = tables.iloc[-1]
    return largest, math.ceil(seats / int(largest["capacity"]))


def _lines(req: OfficeRequirements, catalog: pd.DataFrame, tier: str) -> list[dict]:
    sit_stand = "sit/stand" if any(SIT_STAND.search(p) for p in req.preferences) else None
    wanted: list[tuple[pd.Series, int]] = []
    if req.headcount:
        wanted += [(_pick(catalog, "desk", tier, sit_stand), req.headcount),
                   (_pick(catalog, "office_chair", tier), req.headcount)]
        if req.storage:
            wanted.append((_pick(catalog, "storage", tier), math.ceil(req.headcount / PEOPLE_PER_STORAGE)))
    for seats in req.meeting_rooms:
        wanted += [_tables(catalog, seats), (_pick(catalog, "conference_chair", tier), seats)]

    lines: dict[str, dict] = {}
    for item, qty in wanted:
        line = lines.setdefault(item["sku"], {"sku": item["sku"], "item": friendly_name(item), "model": item["name"],
                                              "name": item["name"], "category": item["category"],
                                              "qty": 0, "unit_price_inr": int(item["price_inr"])})
        line["qty"] += qty
    for line in lines.values():
        line["line_total_inr"] = line["qty"] * line["unit_price_inr"]
    return list(lines.values())


def compute_bom(req: OfficeRequirements, catalog: pd.DataFrame, tier: str | None = None) -> dict:
    """Start at the preferred tier and step down until the total fits the budget (or nothing cheaper exists).
    A tier the customer explicitly chose is used as-is."""
    start = "premium" if any(PREMIUM.search(p) for p in req.preferences) else \
        "budget" if any(BUDGET.search(p) for p in req.preferences) else "standard"
    for tier in [tier] if tier else reversed(TIERS[:TIERS.index(start) + 1]):
        lines = _lines(req, catalog, tier)
        total = sum(line["line_total_inr"] for line in lines)
        if req.budget_inr is None or total <= req.budget_inr:
            break
    return {
        "tier": tier,
        "requirements": req.model_dump(),
        "bom": lines,
        "estimated_total": total,
        "budget_inr": req.budget_inr,
        "within_budget": None if req.budget_inr is None else total <= req.budget_inr,
    }


CHOICES = [(re.compile(r"\bpremium\b|option\s*3", re.I), "premium"),
           (re.compile(r"\bstandard\b|option\s*2", re.I), "standard"),
           (re.compile(r"\bbasic\b|\bcheapest\b|option\s*1", re.I), "budget")]


def detect_choice(text: str) -> str | None:
    """Which option the customer picked ("I'll take premium", "option 2"), if any."""
    return next((tier for pattern, tier in CHOICES if pattern.search(text)), None)


def photo_key(item) -> str:
    return "sitstand_desk" if item["category"] == "desk" and "sit/stand" in item["name"].lower() else item["category"]


@cache
def _photo_credits() -> dict:
    path = IMAGES_DIR / "credits.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def add_photos(lines: list[dict], images_dir=None, credits: dict | None = None) -> list[dict]:
    """Attach an example photo URL (served by the API at /images) and its credit line to each item."""
    images_dir = images_dir or IMAGES_DIR
    credits = _photo_credits() if credits is None else credits
    for line in lines:
        key = photo_key(line)
        found = (images_dir / f"{key}.jpg").exists()
        line["photo"] = f"/images/{key}.jpg" if found else None
        line["photo_credit"] = credits.get(key, "") if found else ""
    return lines


def plan_options(req: OfficeRequirements, catalog: pd.DataFrame, chosen: str | None = None) -> dict:
    """The recommended (or chosen) plan in full, plus the total of every tier so the customer can compare."""
    plan = compute_bom(req, catalog, chosen)
    plan["chosen"] = chosen is not None
    plan["options"] = []
    for tier in TIERS:
        total = sum(line["line_total_inr"] for line in _lines(req, catalog, tier))
        plan["options"].append({"tier": tier, "total": total,
                                "within_budget": None if req.budget_inr is None else total <= req.budget_inr})
    plan["needs_budget"] = req.budget_inr is None
    return plan


def _rooms_text(rooms: list[int]) -> str:
    if len(rooms) == 1:
        return f"a meeting room for {rooms[0]}"
    return f"{len(rooms)} meeting rooms (for {', '.join(map(str, rooms))})"


def plan_reply(plan: dict) -> str:
    """Plain-language reply for a plan. Written by code, not the LLM: every number must be exact."""
    req = plan["requirements"]
    name = TIER_NAMES[plan["tier"]][0]
    if plan.get("quote_id"):
        if plan.get("quote_repeat"):
            return (f"Your {name} plan is already saved as quote **{plan['quote_id']}** "
                    f"({inr(plan['estimated_total'])}). Is there anything else I can help you with?")
        return (f"Thank you! Your **{name}** plan ({inr(plan['estimated_total'])}) is saved as quote "
                f"**{plan['quote_id']}**.\n\nTo place the order, mention this quote number when you call or visit "
                "us, and our team will confirm delivery and payment with you. Anything else I can help with?")

    who = [f"{req['headcount']} people"] if req["headcount"] else []
    if req["meeting_rooms"]:
        who.append(_rooms_text(req["meeting_rooms"]))
    # Implicit confirmation (conversation-design basics): say back what was understood, so mistakes show early.
    understood = who + ([] if req.get("storage", True) else ["no drawer cabinets"]) + \
        ([f"budget {inr(req['budget_inr'])}"] if req["budget_inr"] else [])
    confirm = f"Got it: {', '.join(understood)}."

    if plan.get("chosen"):
        out = [f"Great choice! Here is your **{name}** plan for {' and '.join(who)}: "
               f"**{inr(plan['estimated_total'])}** in total."]
        if plan["within_budget"] is False:
            out.append(f"Just so you know, this is above your budget of {inr(plan['budget_inr'])}.")
        out += ["", "Below is the full list, with an example photo of each type of furniture so you can see what it "
                    "looks like. Shall I save this plan as a quote for you?"]
        return "\n".join(out)
    includes = "a desk and a chair for every person" + (", shared drawer cabinets" if req.get("storage", True) else "") + \
        (", and a table with chairs for each meeting room" if req["meeting_rooms"] else "")
    out = [confirm, "", f"Here's what I suggest: {includes}.", "", "You can choose from three options:"]
    for o in plan["options"]:
        option_name, hint = TIER_NAMES[o["tier"]]
        tags = (["fits your budget"] if o["within_budget"] else []) + (["recommended"] if o["tier"] == plan["tier"] else [])
        out.append(f"- **{option_name}** ({hint}): **{inr(o['total'])}**" + (f" ({', '.join(tags)})" if tags else ""))
    out.append("")
    if plan["needs_budget"]:
        out.append(f"What budget do you have in mind? Tell me (for example \"5 lakh\") and I'll pick the best option "
                   f"for you. Below are the details of the {name} option.")
    elif plan["within_budget"]:
        out.append(f"The **{name}** option fits your budget. Details are below. Tap **Choose** on the option you "
                   "like, or tell me what to change.")
    elif plan.get("alternatives"):
        out.append(f"Even the Basic option ({inr(plan['options'][0]['total'])}) is above your budget. "
                   "These would fit your budget:")
        out += [f"- {a['message']}: **{inr(a['total'])}**" for a in plan["alternatives"]]
        out.append("\nTap one below, or tell me what you'd like to change.")
    else:
        out.append(f"Even the Basic option ({inr(plan['options'][0]['total'])}) is above your budget. Could you "
                   "plan for fewer people or raise the budget a little?")
    return "\n".join(out)


# ---------- conversation basics: confirmations, edits, alternatives, quick replies ----------

ACK = re.compile(r"^\s*(ok(ay)?|yes|yeah|yep|sure|go (for|ahead)|proceed|confirm|sounds good|perfect|great|done|"
                 r"book it|place (the )?order|let'?s do it)\b", re.I)
NO_ROOM = re.compile(r"\b(no|without|skip|remove|drop|don'?t need)( a| the| any)? meeting ?rooms?", re.I)
NO_STORAGE = re.compile(r"\b(no|without|skip|remove|drop|don'?t need)( the| any)? (drawer|storage|cabinet)", re.I)
SHOW_OPTIONS = re.compile(r"\b(other|all|show|compare)( the)? options\b", re.I)
PROMISE = re.compile(r"\b(let me|i'?ll|i will|i am going to|i'?m going to)\s+(check|calculate|get|look|find|"
                     r"proceed|prepare|process|confirm|arrange)", re.I)
MENU = ("Sorry, I'm not sure I understood. I can help you with three things: plan office furniture with prices, "
        "answer questions about our products and policies, or log a problem with something you bought. "
        "Which one would you like?")
MENU_SUGGESTIONS = ["Plan furniture for my office", "What is the warranty on furniture?", "Something I bought is broken"]


def is_ack(text: str) -> bool:
    """A short go-ahead ("ok", "yes please", "go for it"), not a new question."""
    return bool(ACK.match(text)) and "?" not in text and len(text.split()) <= 6


def no_promises(text: str) -> str:
    """The chat model must answer now, not promise work it can't do ("Let me check the details...")."""
    return MENU if PROMISE.search(text) else text


def update_requirements(req: OfficeRequirements, entities: dict, text: str,
                        preferences: list[str] = ()) -> OfficeRequirements:
    """Slot filling: values in the newest message replace old ones; everything else is kept."""
    upd = {k: v for k, v in (("headcount", entities.get("headcount")),
                             ("meeting_rooms", entities.get("meeting_room_seats")),
                             ("budget_inr", entities.get("budget_inr"))) if v}
    if NO_ROOM.search(text):
        upd["meeting_rooms"] = []
    if NO_STORAGE.search(text):
        upd["storage"] = False
    upd["preferences"] = list(dict.fromkeys([*req.preferences, *preferences]))
    return req.model_copy(update=upd)


def _basic_total(req: OfficeRequirements, catalog: pd.DataFrame) -> int:
    return sum(line["line_total_inr"] for line in _lines(req, catalog, "budget"))


def alternatives(req: OfficeRequirements, catalog: pd.DataFrame) -> list[dict]:
    """Concrete changes that bring the Basic plan within budget, as messages the customer can send back."""
    if req.budget_inr is None or _basic_total(req, catalog) <= req.budget_inr:
        return []
    biggest = int(catalog.loc[catalog["category"] == "meeting_table", "capacity"].max())
    candidates = []
    if any(s > biggest for s in req.meeting_rooms):
        candidates.append((f"Make the meeting room for {biggest} people",
                           req.model_copy(update={"meeting_rooms": [min(s, biggest) for s in req.meeting_rooms]})))
    if req.meeting_rooms:
        candidates.append(("Plan without a meeting room", req.model_copy(update={"meeting_rooms": []})))
    if req.storage:
        candidates.append(("Plan without drawer cabinets", req.model_copy(update={"storage": False})))
    out = []
    for message, alt in candidates:
        total = _basic_total(alt, catalog)
        if total <= req.budget_inr:
            out.append({"message": message, "total": total})
    return out


def suggestions(tool: str | None, data: dict) -> list[str]:
    """Quick replies for the next step, phrased as what a customer would say."""
    if tool == "plan_office":
        if data.get("missing"):
            return ["10 people", "25 people", "50 people"]
        if data.get("quote_id"):
            return ["What is the warranty on furniture?", "How long does delivery take?"]
        if data.get("chosen"):
            return ["Yes, save it as a quote", "Show all options"]
        if data.get("needs_budget"):
            return ["My budget is 3 lakh", "My budget is 5 lakh", "My budget is 10 lakh"]
        if data.get("alternatives"):
            return [a["message"] for a in data["alternatives"]]
        return [] if data["requirements"]["meeting_rooms"] else ["Add a meeting room for 8 people"]
    if tool == "create_ticket":
        return []
    if tool == "product_qa":
        return ["Plan furniture for my office", "Something I bought is broken"]
    return MENU_SUGGESTIONS


def _numbers_in(text: str) -> set[int]:
    digits = {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)}
    words = {int(NUMBER_WORDS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in NUMBER_WORDS}
    return digits | words


def merge_requirements(llm: OfficeRequirements, entities: dict, text: str) -> OfficeRequirements:
    """Regex-extracted numbers win. An LLM number is kept only if the customer actually wrote it:
    small models invent headcounts and budgets for vague requests."""
    nums = _numbers_in(strip_amounts(text))
    people_nums = _numbers_in(strip_rooms(strip_amounts(text)))  # room sizes don't count as headcount
    words = set(re.findall(r"[a-z]+", text.lower()))
    return llm.model_copy(update={
        # "sit-stand" survives if the customer wrote "standing"; "ergonomic" doesn't if they never said it
        "preferences": [p for p in llm.preferences
                        if any(w in words or any(t.startswith(w) for t in words) for w in re.findall(r"[a-z]+", p.lower()))],
        "headcount": entities.get("headcount") or (llm.headcount if llm.headcount in people_nums else 0),
        "meeting_rooms": entities.get("meeting_room_seats") or [s for s in llm.meeting_rooms if s in nums],
        "budget_inr": entities.get("budget_inr") or (llm.budget_inr if llm.budget_inr in find_amounts(text) else None),
    })


def _structured(schema, system: str, text: str):
    """LLM function calling into a pydantic schema; None if the model returns something invalid."""
    try:
        return get_llm().with_structured_output(schema, method="json_schema").invoke(
            [SystemMessage(system), HumanMessage(text)])
    except Exception:  # bad JSON / validation error from a small model: caller falls back to NLP values
        return None


def _read_requirements(nlp: NLPResult) -> OfficeRequirements:
    """What this one message says: LLM function calling, grounded and overridden by NLP numbers."""
    e = nlp.entities
    hints = {k: e[k] for k in ("headcount", "meeting_room_seats", "budget_inr") if e[k]}
    llm = _structured(OfficeRequirements, prompts.PLAN_SYSTEM.format(hints=hints or "none"), nlp.text) \
        or OfficeRequirements(headcount=0)
    return merge_requirements(llm, e, nlp.text)


EMPTY_PLAN = {"req": None, "tier": None, "chosen": False, "quote_id": None}


def plan_turn(nlp: NLPResult, state: dict | None, action: str = "update", chosen: str | None = None,
              db_path=db.DB_PATH) -> tuple[dict, dict]:
    """One turn of the office-plan conversation. `state` is the plan so far (structured, kept by the graph).
    action: update (new details), choose (picked an option), ack ("ok": go ahead with the next step).
    -> (data for the reply, new state)."""
    state = {**EMPTY_PLAN, **(state or {})}
    req = OfficeRequirements(**state["req"]) if state["req"] else OfficeRequirements(headcount=0)
    save = False
    if action == "choose":
        state.update(tier=chosen, chosen=True, quote_id=None)
    elif action == "ack" and state["chosen"]:
        if state["quote_id"]:
            plan = plan_options(req, load_catalog(), state["tier"])
            return {**add_bom_photos(plan), "quote_id": state["quote_id"], "quote_repeat": True}, state
        save = True
    elif action == "ack" and state["tier"]:
        state.update(chosen=True)  # "ok" to the recommended option
    else:
        said = _read_requirements(nlp)
        req = update_requirements(req, {"headcount": said.headcount, "meeting_room_seats": said.meeting_rooms,
                                        "budget_inr": said.budget_inr}, nlp.text, said.preferences)
        if SHOW_OPTIONS.search(nlp.text):
            state.update(chosen=False)
    state["req"] = req.model_dump()
    if not req.headcount and not req.meeting_rooms:
        return {"missing": ["headcount"], "bom": [], "estimated_total": 0}, state

    catalog = load_catalog()
    plan = plan_options(req, catalog, state["tier"] if state["chosen"] else None)
    if not plan["chosen"]:
        plan["alternatives"] = alternatives(req, catalog)
    state["tier"] = plan["tier"]
    if save:
        plan["quote_id"] = state["quote_id"] = db.save_quote(plan, db_path)
    return add_bom_photos(plan), state


def add_bom_photos(plan: dict) -> dict:
    add_photos(plan["bom"])
    return plan


def plan_office(nlp: NLPResult) -> dict:
    """Single-message plan (no conversation state)."""
    return plan_turn(nlp, None)[0]


# ---------- create_ticket: escalation ----------

def escalate(fields: TicketFields, nlp_urgency: str | None, safety: bool) -> tuple[TicketFields, str]:
    """Only the deterministic NLP rules can escalate: safety flag or HIGH urgency. The LLM's urgency just
    picks LOW vs MEDIUM (a 1.7B model rated a loose armrest HIGH). -> (fields, status)."""
    if safety or nlp_urgency == "HIGH":
        urgency = "HIGH"
    else:
        urgency = "MEDIUM" if fields.urgency == "HIGH" else fields.urgency
    status = "ESCALATED" if urgency == "HIGH" else "OPEN"
    return fields.model_copy(update={"urgency": urgency}), status


def create_ticket(nlp: NLPResult, db_path=db.DB_PATH) -> dict:
    products = nlp.entities["products"]
    fields = _structured(TicketFields, prompts.TICKET_SYSTEM.format(products=products or "none"), nlp.text) \
        or TicketFields(product=products[0] if products else None, issue_type="OTHER",
                        urgency=nlp.urgency or "MEDIUM", summary=nlp.text[:200])
    fields, status = escalate(fields, nlp.urgency, nlp.safety_flag)
    ticket = {**fields.model_dump(mode="json"), "status": status, "message": nlp.text}
    ticket["ticket_id"] = db.save_ticket(ticket, db_path)
    if nlp.safety_flag:
        ticket["safety_message"] = prompts.SAFETY_MESSAGE
    return ticket
