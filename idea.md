# Agentic Dealer & Customer Support Copilot — Project Plan

> **Core idea:** Store gives its data → **NLP** understands the query → **GenAI** reasons and generates → **API** serves it → **natural-language output** + structured JSON.

---

## 1. The Pipeline

```text
STORE DATA ──► NLP ──► GEN AI ──► API ──► OUTPUT
  (input)   (understand) (reason +   (serve)  (NL answer
                         generate)             + JSON)
```

| # | Layer | Job | Tech | Produces |
|---|---|---|---|---|
| 1 | **Store Data** | Store uploads catalog + manuals/policies | CSV, PDF | FAISS index + catalog table |
| 2 | **NLP** | Understand the message | spaCy, scikit-learn | `NLPResult` (intent, entities, urgency) |
| 3 | **GenAI** | Route, use tools, write the answer | LangGraph, LangChain, LLM API, FAISS | Answer / BOM / ticket |
| 4 | **API** | Expose everything over REST | FastAPI, Pydantic | JSON response |
| 5 | **Output** | Show it to the user | Streamlit (or any website) | Chat UI |

---

## 2. Scope Lock

| ✅ In (v1) | ❌ Out (v2 ideas) |
|---|---|
| Store data ingestion (CSV + PDFs) | Multi-store / `store_id` |
| NLP layer: intent, NER, urgency | Live inventory, customers, orders |
| LangGraph agent with 3 tools | PostgreSQL, Docker |
| RAG with page-level citations | API keys, rate limiting, RBAC |
| BOM via LLM function calling | Local Ollama models |
| Structured tickets + escalation | Admin dashboard for store data |
| FastAPI + Streamlit client | |

---

## 3. Architecture

```text
┌───────────────────────────────────────────────┐
│ 1. STORE DATA                                 │
│    catalog.csv · manuals & policies (PDF)     │
│    → ingest: FAISS index + catalog table      │
│    → product names feed the NLP EntityRuler   │
└───────────────────────┬───────────────────────┘
                        ▼  user message
┌───────────────────────────────────────────────┐
│ 2. NLP LAYER  (spaCy + scikit-learn)          │
│    normalize → intent → entities → urgency    │
└───────────────────────┬───────────────────────┘
                        ▼  NLPResult
┌───────────────────────────────────────────────┐
│ 3. GEN AI LAYER  (LangGraph + LLM API)        │
│    route by intent → tool → LLM writes reply  │
│    product_qa │ plan_office │ create_ticket   │
└───────────────────────┬───────────────────────┘
                        ▼
┌───────────────────────────────────────────────┐
│ 4. API LAYER  (FastAPI)   POST /chat          │
└───────────────────────┬───────────────────────┘
                        ▼
┌───────────────────────────────────────────────┐
│ 5. OUTPUT  NL answer + intent + entities      │
│            + data (BOM / ticket) + sources    │
│    → Streamlit / any website / mobile app     │
└───────────────────────────────────────────────┘
```

---

## 4. Layer 1 — Store Data

**What the store provides**

| Data | Format | Target | Goes to |
|---|---|---|---|
| Product catalog | `catalog.csv` | 40–60 SKUs | Catalog table + NLP product patterns |
| Manuals (furniture + hearth) | PDF, multi-page | 8–10 docs | FAISS |
| Policies (warranty, delivery, returns) | PDF | 2–3 docs | FAISS |

**Catalog columns:** `sku, name, category, product_line, price_inr, capacity, dims_cm`

**Ingestion (`ingest.py`)**
1. PDFs → `PyPDFLoader` (one Document per page, `metadata["page"]` kept)
2. Chunk → embed (sentence-transformers) → FAISS
3. Catalog → loaded into a pandas table
4. Product names → spaCy `EntityRuler` patterns (so NLP recognizes "Ergo Chair")

**Tip:** generate synthetic manuals with an LLM and export them to PDF. Real manufacturer PDFs are copyrighted, so keep them out of a public repo. Put each section on its own page (specs, assembly, warranty, troubleshooting, safety) so the citations point somewhere useful.

---

## 5. Layer 2 — NLP (Understand)

| Step | What it does | How |
|---|---|---|
| **Normalize** | Clean text, convert amounts ("5 lakh" → 500000, "10k" → 10000) | Regex + custom normalizer |
| **Intent** | `PRODUCT_QA` · `OFFICE_PLAN` · `SUPPORT` · `OTHER` | TF-IDF + Logistic Regression |
| **NER** | `PRODUCT`, `MONEY`, `HEADCOUNT`, `SEATS`, `LOCATION` | spaCy `en_core_web_sm` + `EntityRuler` from catalog |
| **Urgency** | `LOW` / `MEDIUM` / `HIGH` + safety flags (gas smell, smoke) | Keyword rules + small classifier |

**Output → `NLPResult`**
```json
{
  "intent": "OFFICE_PLAN",
  "confidence": 0.93,
  "entities": { "headcount": 30, "meeting_room_seats": [10], "budget_inr": 500000 },
  "urgency": null,
  "safety_flag": false
}
```

**Training data:** 300–500 labeled messages in `nlp_data/intents.csv`. You can draft them with an LLM, then check them by hand.

**Low confidence rule:** `confidence < 0.6` → the GenAI agent picks the tool itself (LLM tool calling as a fallback).

**Why NLP comes before GenAI (interview answer)**
- Routing is fast and cheap, with no LLM call needed to decide the path
- It's measurable: intent accuracy and NER F1 are real metrics
- Extracted entities ground the LLM, so it doesn't guess budgets or headcounts

---

## 6. Layer 3 — GenAI (Reason + Generate)

**LangGraph flow**
```text
nlp_node
   │
router (conditional edge on intent)
   ├─ PRODUCT_QA   → product_qa
   ├─ OFFICE_PLAN  → plan_office
   ├─ SUPPORT      → create_ticket
   └─ OTHER / low confidence → agent (LLM picks a tool or just chats)
   │
respond_node (LLM writes the final natural-language reply)
```
**Memory:** LangGraph checkpointer, `thread_id = session_id`

### Tool 1 — `product_qa` (RAG + citations)
1. Retrieval query = cleaned text. If NLP found a `PRODUCT`, filter chunks to that product's manual.
2. LLM answers **only** from the retrieved chunks.
3. Return the answer + `{document, page}` per chunk used.
4. Below the score threshold → *"I couldn't find this in the store's documents."*

### Tool 2 — `plan_office` (BOM)
1. **Function calling:** the LLM fills `OfficeRequirements`, with NLP entities passed in as hints
   ```python
   class OfficeRequirements(BaseModel):
       headcount: int
       meeting_rooms: list[int] = []   # seats per room
       budget_inr: int | None = None
       preferences: list[str] = []
   ```
2. **Cross-check:** if the NLP normalizer and the LLM disagree on a number, trust the NLP value (regex is deterministic)
3. **Python rules** compute quantities: 1 workstation + 1 chair per person; 1 table + `seats` chairs per meeting room; 1 storage unit per 4 people
4. **Catalog lookup** chooses SKUs → line totals → estimated total → budget check

**Guardrail:** SKUs and prices come only from the catalog. The LLM never writes a number.

### Tool 3 — `create_ticket` (Support + escalation)
1. **Function calling:** the LLM fills `TicketFields` (`product`, `issue_type`, `urgency`, `summary`)
2. **NLP overrides:** an NLP `safety_flag` or `HIGH` urgency always wins over the LLM
3. **Escalation rules (deterministic)**

| Condition | Status |
|---|---|
| `safety_flag` (gas odor, venting smoke) | `ESCALATED` + `HIGH` + safety message |
| `urgency == HIGH` | `ESCALATED` |
| else | `OPEN` |

4. Save to SQLite → return `ticket_id`

**`IssueType` enum**

| Furniture | Hearth | Common |
|---|---|---|
| `MECHANISM_FAILURE` | `IGNITION_FAILURE` | `DELIVERY` |
| `STRUCTURAL_DAMAGE` | `GAS_ODOR` | `MISSING_PARTS` |
| `UPHOLSTERY` | `VENTING_SMOKE` | `OTHER` |
| | `GLASS_DAMAGE` | |

---

## 7. Layer 4 — API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/chat` | Full pipeline: NLP → GenAI → response |
| `POST` | `/nlp/analyze` | NLP layer only (debugging + demo) |
| `GET` | `/tickets/{ticket_id}` | Fetch a ticket |
| `GET` | `/health` | Sanity check |

**Request**
```json
{ "session_id": "abc123", "message": "Plan furniture for a 30-person office with a 10-seat meeting room under ₹5 lakh" }
```

**Response (same shape every time)**
```json
{
  "response": "For 30 people and one 10-seat meeting room, here's a plan within your ₹5 lakh budget...",
  "nlp": {
    "intent": "OFFICE_PLAN",
    "confidence": 0.93,
    "entities": { "headcount": 30, "meeting_room_seats": [10], "budget_inr": 500000 }
  },
  "tool_used": "plan_office",
  "data": { "bom": [], "estimated_total": 472500, "within_budget": true },
  "sources": [],
  "session_id": "abc123"
}
```

---

## 8. Layer 5 — Output (Streamlit)

- Chat via `st.chat_input` / `st.chat_message`, calling `/chat`
- **NLP panel** in the sidebar: intent, confidence, entities. This makes the pipeline visible in demos.
- Render based on `tool_used`:
  - `product_qa` → answer + expandable citations
  - `plan_office` → BOM table + total + budget badge
  - `create_ticket` → ticket card (red if `ESCALATED`)

---

## 9. Tech Stack

| Layer | Tech |
|---|---|
| Store data | pandas, PyPDFLoader |
| NLP | spaCy, scikit-learn (TF-IDF + Logistic Regression) |
| GenAI | LangGraph, LangChain, LLM API with tool calling (OpenAI / Gemini / Groq) |
| RAG | FAISS + sentence-transformers |
| API | FastAPI, Pydantic |
| Output | Streamlit |
| Storage | CSV (catalog), SQLite (tickets) |

---

## 10. Repo Structure

```text
dealer-copilot/
├── store_data/
│   ├── manuals/              # PDFs
│   ├── policies/             # PDFs
│   └── catalog.csv
├── nlp_data/
│   └── intents.csv           # labeled training data
├── models/
│   └── intent_clf.joblib
├── app/
│   ├── ingest/
│   │   └── ingest.py         # PDFs → FAISS, catalog → table, NER patterns
│   ├── nlp/
│   │   ├── normalize.py      # amounts, units, cleanup
│   │   ├── intent.py         # train + predict
│   │   ├── ner.py            # spaCy + EntityRuler
│   │   ├── urgency.py
│   │   └── pipeline.py       # → NLPResult
│   ├── genai/
│   │   ├── graph.py          # nlp_node → router → tools → respond
│   │   ├── tools.py          # product_qa, plan_office, create_ticket
│   │   ├── schemas.py
│   │   └── prompts.py
│   ├── db.py                 # tickets (SQLite)
│   └── main.py               # FastAPI
├── ui/
│   └── streamlit_app.py
├── eval/
│   ├── testset.json
│   └── run_eval.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 11. Build Order

- [ ] **Phase 1 — Store data:** `catalog.csv` + manual/policy PDFs + `ingest.py`
- [ ] **Phase 2 — NLP:** normalizer, train the intent classifier, NER, urgency; test `pipeline.py` on its own
- [ ] **Phase 3 — RAG:** retriever; check that page numbers come back correct
- [ ] **Phase 4 — Tools:** each tool as a plain Python function, tested on its own
- [ ] **Phase 5 — GenAI graph:** `nlp_node` → router → tools → `respond_node` + fallback agent
- [ ] **Phase 6 — API:** `/chat`, `/nlp/analyze`, `/tickets/{id}`, `/health`
- [ ] **Phase 7 — Output:** Streamlit with the NLP panel
- [ ] **Phase 8 — Eval + README:** pipeline diagram + demo GIF

---

## 12. Evaluation

| Layer | Test | Metric |
|---|---|---|
| NLP | Held-out 20% of `intents.csv` | Intent accuracy, macro-F1 |
| NLP | 30 messages with labeled entities | NER precision / recall per type |
| GenAI | 25–30 questions with known doc + page | Citation accuracy |
| GenAI | 5–10 office scenarios, calculated by hand | BOM quantity + cost match |
| GenAI | 10–15 complaints | Ticket field accuracy, escalation correct |
| API | Load test with concurrent requests | Latency (p50/p95), error rate |

---

## 13. Updated Resume Bullet

The NLP layer wasn't in the original bullet, so here's an updated version:

**Agentic Dealer & Customer Support Copilot** | GitHub
*spaCy, scikit-learn, LangGraph, LangChain, RAG, FAISS, FastAPI, Streamlit, LLM APIs* — Oct 2026

- Built a store-data → NLP → GenAI → API pipeline where a spaCy + scikit-learn layer classifies intent and extracts entities (products, budget, headcount, urgency) to route a LangGraph agent across three tools: RAG Q&A over furniture and hearth manuals with page-level citations, catalog-driven office space planning, and support-ticket escalation
- Generates a furniture bill of materials (items, quantities, estimated cost) from natural-language requirements via LLM function calling, and turns complaints into structured tickets (product, issue type, urgency), served through a FastAPI REST API with a Streamlit client

Once eval runs, add one real number (e.g. intent macro-F1 or citation accuracy).

---

## 14. Bullet ↔ Code Map (for interviews)

| Resume claim | Where it lives |
|---|---|
| Store data → NLP → GenAI → API pipeline | `ingest/`, `nlp/`, `genai/`, `main.py` |
| spaCy + scikit-learn intent + entities | `nlp/intent.py`, `nlp/ner.py` |
| Routes a LangGraph agent across three tools | `genai/graph.py` router |
| RAG with page-level citations | `ingest.py` page metadata + `product_qa` |
| Catalog-driven office planning, BOM via function calling | `plan_office` + `OfficeRequirements` |
| Support-ticket escalation, structured tickets | `create_ticket` + escalation rules |
| FastAPI + Streamlit | `main.py`, `ui/streamlit_app.py` |