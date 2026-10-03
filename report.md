# Progress Report: Agentic Dealer & Customer Support Copilot

**As of:** 4 October 2026
**Status:** v1 is working end to end on a local machine. All 8 planned phases are done; 172 tests pass.

## 1. Summary

A chat assistant for a furniture and fireplace store that runs entirely on a local Ollama model (`qwen3:1.7b`). It can:

- **Answer product and policy questions** from real manufacturer manuals, citing the page.
- **Plan office furniture** with Basic / Standard / Premium price options, alternatives that fit the budget, example photos, and a saved quote.
- **Log support tickets**, escalating safety issues such as a gas smell automatically.

A classic NLP layer (spaCy + scikit-learn) reads every message before the LLM does. Numbers, prices, product codes and escalation decisions always come from code, never from the LLM.

## 2. Phase status

| Phase | Planned in `idea.md` | Status | Notes |
|---|---|---|---|
| 1 | Store data + ingest | ✅ Done | Real data only: IKEA catalog, 3 real fireplace/stove models, 8 real PDFs. 571 chunks in FAISS |
| 2 | NLP layer | ✅ Done | Intent accuracy 0.866 / macro-F1 0.889; entities 95–100% |
| 3 | RAG with page citations | ✅ Done | Right page in top 4 for 25/25 test questions |
| 4 | Tools | ✅ Done | Q&A, office plan with options and quotes, tickets with escalation |
| 5 | LangGraph agent | ✅ Done | The plan is kept across messages; quick replies; memory per session |
| 6 | FastAPI | ✅ Done | `/chat`, `/nlp/analyze`, `/tickets/{id}`, `/quotes/{id}`, `/images`, `/health` |
| 7 | Streamlit UI | ✅ Done | Option cards, Choose buttons, quick replies, photo gallery, NLP sidebar |
| 8 | Eval + README | ✅ Done | `eval/run_eval.py`, results below |

## 3. What was built

**Data** (`app/ingest/`):
- **Catalog:** 63 IKEA office items plus 3 real fireplace/stove models with retailer prices.
- **Documents:** 6 manuals and 2 policies, and 9 openly licensed example photos (Wikimedia Commons, credited).
- **Download script:** one command fetches everything. Third-party files are git-ignored.

**NLP** (`app/nlp/`):
- **Money:** parses Indian amounts ("6lakhs", "₹5,00,000", "1.2 crore").
- **Intent:** classifier trained on 384 Bitext messages + 210 hand-written ones.
- **Entities:** spaCy EntityRuler for products and Indian cities, plus rules for headcount and meeting rooms (they handle typos such as "emloyee").
- **Urgency and safety:** rule-based.

**GenAI** (`app/genai/`):
- **Routing:** a LangGraph router sends each message to one of three tools, or to a chat model for anything else.
- **Office plan conversation, built on common chatbot design guidelines (Google conversation design, Voiceflow, Haptik):**
  - The plan is kept as structured details; each message only changes what it mentions.
  - Every plan reply confirms back what was understood.
  - Over budget, it offers alternatives that fit, as buttons.
  - Choices and "ok" are recognised even when the classifier disagrees.
  - Confirming a plan saves a quote with an ID.
- **Safety guards:**
  - The LLM's numbers and preferences are only used if the customer actually wrote them.
  - Only the NLP rules can escalate a ticket.
  - The chat model can't promise actions it can't take.

**Serving:** FastAPI back end, Streamlit front end, SQLite for tickets and quotes.

## 4. Quality

**Tests:** 172 automated tests (`pytest -q`, about 8 min with Ollama on CPU). The fast subset (`-m "not ollama"`, 149 tests) runs in about 20 s.

**Eval** (`eval/results.md`, latest run):

| Metric | Result |
|---|---|
| Intent accuracy / macro-F1 | 0.866 / 0.889 |
| Entities (20 labelled messages) | 95–100% per field |
| Retrieval: right page in top 4 | 25/25 |
| Citation accuracy | 23/25 (92%) |
| Office-plan requirements read correctly | 8/8 |
| Ticket issue type / escalation | 12/12, 12/12 |
| Time per tool call (median / slowest 5%) | 8.8 s / 19.8 s on an idle CPU; 21.6 s / 55 s with the servers running |

## 5. Problems found and fixed

Each one was caught by a test, the eval, or a replayed real conversation, and now has a test.

| Problem | Fix |
|---|---|
| The LLM invented headcount, rooms, budget and preferences for vague requests | LLM values are kept only if they appear in the message. Digits inside money amounts ("8 lakh") and room sizes ("room for 12") are excluded from this check |
| The LLM rated a loose armrest HIGH and escalated it | Only the deterministic NLP rules can escalate |
| SQLite connections were never closed (each request leaked a file handle) | The connection helper now commits and always closes |
| The Ollama embedding runner crashed on 571 chunks in one request | Chunks are embedded in batches of 32 |
| Follow-ups like "l have 30 emloyee" lost the plan | The plan is kept as structured details; headcount rules handle typos |
| "I'll take the Basic option" was classified as a product question | Plan steps (choice, ok, edits) are checked before intent classification |
| The chat model replied "Let me check the details" and did nothing | A guard replaces such replies with a clear menu |
| Plans showed confusing Swedish product names | Plain names ("Office desk, 120x40 cm"); the model name moves to a side column |
| A reply said the Premium option fit the budget when it was Basic | Fixed a variable being overwritten in the reply code |

## 6. Key decisions

- **Local LLM (Ollama) instead of a paid API.** It's free and private; the cost is quality and speed (see remaining work).
- **Real data only.** Third-party manuals and photos are downloaded by script and never committed.
- **Code writes every number.** Plan replies are written by code, not the LLM, so every price is exact.
- **Rules, not a model, for safety.** Gas-smell and smoke detection must be predictable and easy to audit.
- **Generic example photos.** The 2020 IKEA product pages no longer exist, and copying IKEA's images would break their terms, so photos show each type of furniture rather than the exact item.

## 7. Remaining work

### High priority (product and conversation)
- [ ] **Follow-up on quotes.** Quotes are saved, but nobody is notified. Collect contact details with consent and notify sales (email or webhook). Add a status for each quote.
- [ ] **Remember chats after a restart.** Swap the in-memory checkpointer for `SqliteSaver`.
- [ ] **Follow-ups on product questions.** "And what about its warranty?" should carry over the product from the previous question.
- [ ] **Floor area.** Use the office size in sq ft to check that the plan fits the space, and suggest a layout.
- [ ] **A better model.** Try a larger local model (or an API) for product answers and ticket replies. Only `LLM_MODEL` in `.env` changes.
- [ ] **Speed.** Stream responses to the UI. Cache retrieval and catalog lookups. Consider a GPU.
- [ ] **Browser test.** Add a Playwright test of the real UI (Choose buttons, quick replies, photos), plus a demo GIF for the README.

### Medium priority (accuracy)
- [ ] **Citations.** Fix the 2 misses (neighbouring pages): re-rank results, or merge chunks from neighbouring pages.
- [ ] **Entity misses.** Handle a budget with two amounts ("1.5 crore is too much, keep it under 9 lakh"), and a word between number and noun ("18 Skarsta desks").
- [ ] **Number phrases.** Support compound numbers ("twenty five") and Hindi/Hinglish phrasing.
- [ ] **A bigger, harder eval set.** More intents, real customer phrasing, more multi-turn conversations.
- [ ] **Store details.** Replace the placeholder store name (`STORE_NAME`) and contact details with real ones.

### Low priority (engineering, v2 scope from `idea.md`)
- [ ] **CI.** GitHub Actions running the fast test suite on every push.
- [ ] **Packaging.** Docker image for the API, UI and Ollama.
- [ ] **Database.** PostgreSQL instead of SQLite; multiple stores (`store_id`); live inventory, customers and orders.
- [ ] **Access control.** API keys, rate limiting, roles.
- [ ] **Admin tools.** A dashboard to manage the catalog, manuals and quotes.
- [ ] **Dependencies and monitoring.** Move off the deprecated `langchain-community` imports; add structured logging and monitoring.
- [ ] **Licensed product photos** from suppliers, to replace the generic examples.

## 8. How to run

See `README.md` → Setup and Quick start. In short:

1. `pip install -r requirements.txt` and `spacy download en_core_web_sm`
2. `python -m app.ingest.fetch_data`, then `python -m app.ingest.ingest`, then `python -m app.nlp.intent`
3. `uvicorn app.main:app` and `streamlit run ui/streamlit_app.py`
