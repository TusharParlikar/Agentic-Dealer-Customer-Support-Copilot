# Agentic Dealer & Customer Support Copilot

A copilot for a furniture and hearth (fireplace / stove) dealer. A customer message goes through a classic NLP layer
(intent, entities, urgency), then a LangGraph agent that answers from real product manuals with page citations,
plans office furniture from the store catalog, or files a support ticket and escalates safety issues.

Everything runs locally on [Ollama](https://ollama.com). No API keys needed.

```text
STORE DATA ──► NLP ──► GEN AI ──► API ──► OUTPUT
  (input)   (understand) (reason +   (serve)  (NL answer
                         generate)             + JSON)
```

## Status

All 8 phases are done and tested: 172 tests pass, and eval results are below.

| Phase | What |
|---|---|
| 1 | Store data: real catalog + manuals, FAISS index |
| 2 | NLP: normalizer, intent classifier, NER, urgency |
| 3 | RAG retriever with page citations |
| 4 | Tools: product Q&A, office plan (BOM), tickets |
| 5 | LangGraph agent with memory |
| 6 | FastAPI |
| 7 | Streamlit UI |
| 8 | Eval |

## Quick start

```powershell
.venv\Scripts\uvicorn app.main:app                  # terminal 1: API on :8000
.venv\Scripts\streamlit run ui/streamlit_app.py     # terminal 2: UI on :8501
```

Try: *"Plan an office for 12 people with a 6-seat meeting room under 3 lakh"*, *"My Dimplex heater overheated,
how do I reset it?"*, *"I smell gas near my Tahoe fireplace"*.

## Setup

Requires Python 3.14 and Ollama.

```powershell
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m spacy download en_core_web_sm
copy .env.example .env          # set LLM_MODEL / EMBED_MODEL to what `ollama list` shows
.venv\Scripts\python -m app.config               # smoke test: both models answer
.venv\Scripts\python -m app.ingest.fetch_data    # download real data, build catalog.csv
.venv\Scripts\python -m app.ingest.ingest        # build FAISS index (~10 min on CPU)
```

Models are read from `.env` (`LLM_MODEL`, `EMBED_MODEL`, `OLLAMA_BASE_URL`), and `app/config.py` is the only place
that creates them. To change models, edit `.env`.

## Phase 1: Store data

All store data is real and downloaded from public sources; nothing is generated.

| Data | Source | Used for |
|---|---|---|
| Furniture catalog | [IKEA dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-11-03) (TidyTuesday, 3,694 products) | office chairs, conference chairs, desks, conference tables, drawer units |
| Hearth catalog | `store_data/hearth.csv`: 3 real models with retailer prices, source URL per row | gas fireplace, wood stove, electric firebox |
| Example photos | 9 openly licensed photos from [Wikimedia Commons](https://commons.wikimedia.org), one per furniture type, credits in `store_data/images/credits.json` | shown when a customer picks a plan |
| Hearth manuals | Empire Tahoe Clean-Face, Drolet Escape 1800, Dimplex Multi-Fire XHD | RAG |
| Furniture manuals | office chair manual, height-adjustable desk troubleshooting guides | RAG |
| Policies | South Shore 1-year furniture warranty, a retailer's delivery and returns terms | RAG |

`app/ingest/fetch_data.py` downloads everything and builds `store_data/catalog.csv` (66 rows):

- IKEA rows are filtered by product description (e.g. `Desk, 120x70 cm` stays; desk tops and screens are dropped).
- Sizes are parsed from the description, because IKEA's width column is often blank.
- Prices are converted to INR at fixed rates (1 USD = ₹88, 1 SAR = ₹23.47). IKEA prices are from 2020.
- Conference-table seats are estimated at 70 cm of edge per person on both long sides, since IKEA lists no seat count.
- Each hearth row names its manual (`manual` column), so a question about a product can be limited to its manual.

`app/ingest/ingest.py` then:

1. Reads every PDF page (page numbers 1-based, so citations match the printed page) and skips image-only pages.
2. Splits pages into 1,000-character chunks and embeds them with `nomic-embed-text` into FAISS.
3. Writes spaCy EntityRuler patterns for every product line, plus ASCII spellings (`Långfjäll` also matches `langfjall`).

Manuals and policies are third-party documents, so they're git-ignored and never committed. Run `fetch_data` to download them.

## Phase 2: NLP layer

`app/nlp/pipeline.py` turns a message into an `NLPResult`:

```json
{"intent": "OFFICE_PLAN", "confidence": 0.93,
 "entities": {"products": [], "headcount": 30, "meeting_room_seats": [10], "budget_inr": 500000, "location": ["Pune"]},
 "urgency": null, "safety_flag": false}
```

| Step | How |
|---|---|
| Normalize | Cleans whitespace and quotes; parses money: `5 lakh`, `₹5,00,000`, `2.5L`, `10k`, `1.2 crore`, `Rs. 45,000`. A bare number is never money. |
| Intent | TF-IDF (word 1-2 grams + char 2-5 grams) + Logistic Regression, 4 classes |
| Entities | spaCy `en_core_web_sm` + EntityRuler: product lines from the catalog, plus Indian cities. Regex for headcount and meeting-room seats (`two 8-seat rooms` gives `[8, 8]`; room seats are never counted as headcount). |
| Urgency | Keyword rules: `LOW` / `MEDIUM` / `HIGH` + `safety_flag` for gas smell, smoke, sparks, CO alarm. Rules, not a model, so safety stays predictable. |

**Intent training data** (`nlp_data/intents.csv`, 594 messages, built by `python -m app.nlp.intent`):

- 384 messages sampled from the public [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), remapped:
  - refund, delivery and cancellation policy questions become `PRODUCT_QA`
  - complaints, order tracking and refunds become `SUPPORT`
  - account, invoice and newsletter messages become `OTHER`
- 210 hand-written messages in `nlp_data/handwritten.csv`, for what Bitext doesn't cover: 50 office plans, 75 product questions, 60 product faults and 25 small-talk messages.

On a held-out 20%, accuracy is **0.866** and macro-F1 is **0.889**. `OFFICE_PLAN` is perfect. `SUPPORT` is weakest (0.78 F1) because Bitext "complaints" and "policy questions" overlap.

## Phase 3: RAG

`app/genai/retriever.py`: FAISS similarity search (top 4) over manual and policy chunks.

- If the message names a product that has a manual (Tahoe, Drolet, Dimplex), the search covers only that manual. IKEA lines have no manual, so the search covers everything.
- Chunks beyond a distance threshold are dropped, so off-topic questions get *"I couldn't find this in the store's documents."* instead of a made-up answer.

## Phase 4: Tools

| Tool | LLM does | Code does |
|---|---|---|
| `product_qa` | Writes the answer from retrieved chunks + catalog facts, citing `[n]` | Retrieval; maps citations back to `{document, page}`; catalog prices for named products |
| `plan_office` | Fills `OfficeRequirements` via structured output | NLP numbers override the LLM; picks SKUs; prices all 3 tiers; recommends the best fit for the budget; writes the plain-language reply |
| `create_ticket` | Fills `TicketFields` (product, issue type, urgency, summary) | NLP safety flag / `HIGH` urgency override the LLM; escalation; saves to SQLite |

**Bill-of-materials rules:**

- Each person gets 1 desk + 1 office chair, plus 1 drawer unit per 4 people.
- Each meeting room gets the smallest table that seats everyone (or several of the largest) and one conference chair per seat.
- Tiers pick the cheapest, median or most expensive item per category.
- "premium" / "budget" / "sit-stand" in the request set the tier and desk type. If the total is over budget, the plan steps down a tier.

The LLM never writes a SKU or price. If the LLM returns invalid JSON, the tool falls back to the NLP values.

**Escalation:**

| Condition | Status |
|---|---|
| `safety_flag` (gas, smoke, sparks) | `ESCALATED` + `HIGH` + safety message |
| urgency `HIGH` (from NLP or LLM) | `ESCALATED` |
| otherwise | `OPEN` |

## Phase 5: LangGraph agent

```text
nlp ─► route ─┬─ PRODUCT_QA ─► product_qa ────┐
              ├─ OFFICE_PLAN ─► plan_office ──┤
              ├─ SUPPORT / safety ─► create_ticket ─► respond ─► END
              └─ OTHER / confidence < 0.6 ─► agent (LLM tool calling or small talk)
```

- Routing costs no LLM call. A safety hit always goes to `create_ticket`, whatever the intent.
- `respond` writes the final reply with the LLM. The ticket ID and safety message are always included verbatim.
- `product_qa` answers are passed through as-is, since they're already grounded.
- Memory uses a LangGraph checkpointer keyed by `thread_id = session_id`. It's in-memory, so history is lost on restart.
- An office plan stays open across turns. The bot asks *"How many people?"* and the reply *"l have 30 emloyee"* is merged into that plan, instead of going to small talk because it looks like OTHER on its own. A confident product question, a complaint, or any safety issue leaves the plan.

### Conversation design

The office-plan conversation follows common chatbot design guidelines ([Google conversation design](https://developers.google.com/assistant/conversation-design/confirmations?hl=en), [Voiceflow](https://www.voiceflow.com/stories/conversation-design-best-practices), [Haptik](https://docs.haptik.ai/what-are-the-key-guidelines-for-a-better-conversational-design)):

| Practice | How it's done here |
|---|---|
| **Fill in details step by step** | The plan is kept as structured data in the graph state (`plan`: people, rooms, budget, storage, chosen option, quote). Each message only changes what it mentions. |
| **Plan steps checked before intent** | A choice ("I'll take Basic", "option 2"), a go-ahead ("ok", "go for it") or an edit ("without a meeting room") is matched before the classifier runs. The classifier sees one message at a time and rated "I'll take the Basic option" as a product question. |
| **Confirm back what was understood** | Every plan reply starts with "Got it: 30 people, a meeting room for 30, budget ₹6,00,000." |
| **No dead ends** | Over budget, the bot computes concrete changes that fit (smaller meeting room, no meeting room, no cabinets) and offers them as buttons with their prices. |
| **Quick replies** | Every answer has next-step buttons phrased as the customer would say them (`suggestions` in the API response). |
| **Real next step** | "ok" to a chosen plan saves it as a quote (`Q-…`, `GET /quotes/{id}`). The bot tells the customer to mention the number when they call or visit, rather than promising a call-back it can't make. |
| **No empty promises** | If the chat model says "Let me check…" or "I'll calculate…", the reply is replaced with a clear menu of what the bot can do. |
| **Side questions keep the plan** | A warranty question mid-plan is answered from the manuals, and the plan stays open. |

## Phase 6: API

```powershell
.venv\Scripts\uvicorn app.main:app --reload
```

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/chat` | `{session_id, message}` runs the full pipeline. The response always has the same shape: `response, nlp, tool_used, data, sources, session_id` |
| `POST` | `/nlp/analyze` | NLP layer only |
| `GET` | `/tickets/{ticket_id}` | Fetch a ticket (404 if unknown) |
| `GET` | `/health` | Models in use, whether the index is built |

Input is validated: message 1–2000 chars; `session_id` limited to `[A-Za-z0-9_-]{1,64}`.

## Phase 7: Streamlit UI

```powershell
.venv\Scripts\streamlit run ui/streamlit_app.py     # API must be running; API_URL env var overrides localhost:8000
```

Built for customers who aren't technical:

- **Header:** shows the store name (`STORE_NAME` in `.env`) and what the store sells.
- **Quick-start buttons:** *Plan my office*, *Product help*, *Report a problem*.
- **Greeting:** the first reply of every chat says who the store is, what it sells, and that it's happy to help.
- **Office plans:**
  - If the customer hasn't said how many people, how many meeting-room seats or what budget, the bot asks.
  - It then shows **Basic / Standard / Premium** option cards with Indian-format prices (₹13,17,860), marking the option that fits the budget or is recommended.
  - The table lists plain item names ("Office desk, 120x40 cm"); the IKEA model name is kept in a side column.
  - The plan reply is written by code, not the LLM, so every price is exact.
  - Each option card has a **Choose** button. Typing "I'll take premium" or "option 2" works too.
  - Once the customer chooses, the bot confirms the choice and its total, warns if it's above budget, and shows a **photo gallery** with an example photo for each item type. Photos are served by the API at `/images` and credited to their authors.
  - The photos show each type of furniture, not the exact catalog item: the 2020 IKEA product pages no longer exist, and copying IKEA's photos would break their terms.
- **Product answers:** a "Where this answer comes from" panel shows the document and page.
- **Support:** a request card shows the request ID, status and priority. It turns red when escalated.
- **Sidebar:** "How I understood your last message" shows the NLP layer's output (intent, entities, urgency), plus a *Start a new chat* button.

## Tests

```powershell
.venv\Scripts\python -m pytest -q                     # everything (needs Ollama running)
.venv\Scripts\python -m pytest -q -m "not ollama"     # fast, no LLM
```

| Phase | Tests | Result |
|---|---|---|
| 1 | `test_ingest.py`: IKEA filter + seat estimate, page numbering, product patterns, index returns `dimplex.pdf` p.14 for a heater-reset question | 6 passed |
| 2 | `test_nlp.py`: money parsing, entities, cities, number words, typos ("emloyee"), urgency/safety, intent on 12 unseen messages, pipeline | 40 passed |
| 3 | `test_rag.py`: right document and page for 5 real questions, product filter, off-topic returns nothing | 8 passed |
| 4 | `test_tools.py`: BOM totals worked out by hand, budget step-down, prices only from the catalog, invented LLM numbers (including digits inside "8 lakh") and preferences dropped, Basic/Standard/Premium options, plain item names, Indian ₹ format, plain-language reply, escalation table, choosing an option (kept even over budget), photo + credit per item, SQLite, end-to-end tools | 71 passed |
| 5 | `test_graph.py`: routing (safety always goes to a ticket), follow-ups merge into an open plan, each tool end to end, ticket ID + safety text in the reply, a real 3-turn customer conversation, greeting on the first reply only, choosing an option returns the chosen plan with photos, memory | 25 passed |
| 6 | `test_api.py`: all endpoints, input validation, 404, `/chat` response shape, photos served (no path traversal) | 10 passed |
| 7 | `test_ui.py`: Streamlit `AppTest`: store-name header, quick-start buttons, sources, option cards + plain item table, Choose buttons send the choice, photo gallery only after choosing, red escalated request, NLP sidebar, API-down error | 12 passed |
| **All** | `pytest -q` | **172 passed** (about 8 min with Ollama on CPU) |

**What testing caught** (each now has a test):

- The LLM invented a headcount, three rooms and a budget for *"Can you plan my office furniture?"*. LLM numbers are now kept only if they appear in the message.
- The LLM also invented preferences ("ergonomic", "sit-stand"), which priced a budget plan with ₹23k standing desks. LLM preferences are now kept only if the customer used the word.
- The LLM rated a loose armrest `HIGH`, which escalated the ticket. Escalation now comes only from the deterministic NLP rules.
- `with sqlite3.connect()` commits but never closes, so every request leaked a file handle (found when the eval couldn't delete its temp DB on Windows).
- Embedding all 571 chunks in one request crashed the Ollama runner, so the index is now built in batches of 32.
- `en_core_web_sm` missed Chennai and Hyderabad, so an Indian city list was added to the EntityRuler. "twenty people" wasn't read as a number until number words were extended past twelve.

## Phase 8: Eval

```powershell
.venv\Scripts\python -m eval.run_eval      # ~10 min; writes eval/results.md
```

Test cases are in `eval/testset.json`. Every expected page was checked against the real PDF text.

| Layer | Metric | Result |
|---|---|---|
| NLP | Intent accuracy / macro-F1 (held-out 20% of 594) | 0.866 / 0.889 |
| NLP | Entities exact match (20 labelled messages): headcount, budget | 95%, 95% |
| NLP | Entities exact match: meeting rooms, products, location | 100%, 100%, 100% |
| GenAI | Retrieval hit@4: right page in top 4 (25 questions) | 25/25 = 100% |
| GenAI | Citation accuracy: answer cites the right page | 23/25 = 92% |
| GenAI | Office plan: headcount, rooms and budget extracted exactly (8 requests) | 8/8 = 100% |
| GenAI | Ticket issue type / escalation correct (12 complaints) | 12/12, 12/12 |
| Latency | Tool call p50 / p95, qwen3:1.7b on CPU | 8.8 s / 19.8 s on an idle CPU; 21.6 s / 55.0 s with the API and UI servers running |

**Known misses** (full list in `eval/results.md`):

- **Citations (2 of 25).** The LLM cites a neighbouring page: p.3 instead of p.4 for error E01, and p.30/35/36 instead of p.31–34 for stove clearances.
- **Budget with two amounts.** In *"1.5 crore is too much, keep it under 9 lakh"* the larger amount is taken as the budget.
- **Headcount with a word in between.** *"18 Skarsta desks"*: the headcount regex needs the number right next to "desks".

## Limitations

- **Small local model.** qwen3:1.7b keeps everything local and free but invents values. The design limits the damage: the LLM never sets numbers, SKUs, prices or escalation. A bigger model only needs `LLM_MODEL` changed in `.env`.
- **Only office plans carry over between turns.** A follow-up such as "l have 30 emloyee" or "make it 40 people" is merged into the open office plan, with new values winning. Product questions and tickets are still handled one message at a time.
- **In-memory chat history.** It's lost when the API restarts.
- **Catalog prices are approximate.** IKEA prices are from 2020 (SAR), and both currencies are converted at a fixed rate. Conference-table seat counts are estimates.
- **No authentication or rate limiting.** Both are out of scope for v1.
