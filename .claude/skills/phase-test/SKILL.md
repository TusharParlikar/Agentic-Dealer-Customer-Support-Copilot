---
name: phase-test
description: Use when building or changing any phase of the Dealer Copilot (store data, NLP, RAG, tools, graph, API, UI, eval) - how to test each phase before calling it done.
---

# Phase testing for Dealer Copilot

Follow `superpowers:test-driven-development` (failing test first, then code) and
`superpowers:verification-before-completion` (run the command, read the output, then claim).

## Rules

- Tests live in `tests/test_<phase>.py`. Plain pytest, no fixtures beyond what a test needs.
- Pure logic (normalizer, BOM math, escalation rules) gets fast unit tests with no LLM.
- Anything calling Ollama is marked `@pytest.mark.ollama`. Assert on structure and grounded
  facts (SKU exists, page number right, status ESCALATED), never on exact LLM wording.
- A phase is done only when the full suite is green: `.venv\Scripts\python -m pytest -q`.
- Fast loop while coding: `.venv\Scripts\python -m pytest -q -m "not ollama"`.
- Check Ollama is up first: `ollama list` must show the models in `.env`.

## Per-phase checks

| Phase | Test file | Must prove |
|---|---|---|
| 1 Store data | `test_ingest.py` | catalog 40-60 rows, PDFs one section per page, index returns page metadata |
| 2 NLP | `test_nlp.py` | amounts normalized, intent works on unseen messages, entities, safety flag |
| 3 RAG | `test_rag.py` | known question returns the right doc + page; nonsense returns nothing |
| 4 Tools | `test_tools.py` | BOM math, catalog-only SKUs/prices, escalation table, ticket saved |
| 5 Graph | `test_graph.py` | each intent routes to its tool; memory survives two turns |
| 6 API | `test_api.py` | all 4 endpoints, response shape stable |
| 7 UI | `test_ui.py` | Streamlit AppTest renders each `tool_used` view |
| 8 Eval | `eval/run_eval.py` | prints metrics; numbers go into README |

After each phase: update the README section for it with what was built and the real test result.
