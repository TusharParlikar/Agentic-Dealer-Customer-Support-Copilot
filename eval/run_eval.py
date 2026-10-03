"""Evaluate every layer on eval/testset.json and write eval/results.md.

    .venv\\Scripts\\python -m eval.run_eval          # needs Ollama + the built index (~15-25 min on CPU)
"""
import json
import statistics
import tempfile
import time
from pathlib import Path

from app.genai.retriever import retrieve
from app.genai.tools import create_ticket, plan_office, product_qa
from app.nlp.intent import train
from app.nlp.ner import extract_entities
from app.nlp.pipeline import analyze

HERE = Path(__file__).parent
CASES = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))
NER_FIELDS = ["headcount", "meeting_room_seats", "budget_inr", "products", "location"]


def pct(hits: int, n: int) -> str:
    return f"{hits}/{n} = {hits / n:.0%}"


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def main() -> None:
    rows, misses, latencies = [], [], []

    m = train()
    rows += [("NLP", "Intent accuracy (held-out 20%)", f"{m['accuracy']:.3f}"),
             ("NLP", "Intent macro-F1 (held-out 20%)", f"{m['macro_f1']:.3f}")]

    for field in NER_FIELDS:
        ok = 0
        for c in CASES["ner"]:
            got = extract_entities(c["text"])[field]
            got = sorted(got) if isinstance(got, list) else got
            want = sorted(c[field]) if isinstance(c[field], list) else c[field]
            ok += got == want
            if got != want:
                misses.append(f"NER {field}: {c['text']!r} -> {got} (want {want})")
        rows.append(("NLP", f"Entity `{field}` exact match", pct(ok, len(CASES["ner"]))))

    retrieved = cited = 0
    for c in CASES["rag"]:
        want = {tuple(p) for p in c["pages"]}
        got = {(d.metadata["source"], d.metadata["page"]) for d, _ in retrieve(c["q"], c["products"])}
        retrieved += bool(got & want)
        nlp = analyze(c["q"])
        r, secs = timed(product_qa, nlp)
        latencies.append(secs)
        cites = {(s["document"], s["page"]) for s in r["sources"]}
        cited += bool(cites & want)
        if not cites & want:
            misses.append(f"Citation: {c['q']!r} -> {sorted(cites)} (want {sorted(want)})")
    rows += [("GenAI", "Retrieval hit@4 (right page in top 4)", pct(retrieved, len(CASES["rag"]))),
             ("GenAI", "Citation accuracy (answer cites right page)", pct(cited, len(CASES["rag"])))]

    ok = 0
    for c in CASES["plan"]:
        bom, secs = timed(plan_office, analyze(c["text"]))
        latencies.append(secs)
        req = bom.get("requirements", {})
        got = (req.get("headcount"), sorted(req.get("meeting_rooms", [])), req.get("budget_inr"))
        want = (c["headcount"], sorted(c["meeting_rooms"]), c["budget_inr"])
        ok += got == want
        if got != want:
            misses.append(f"Plan: {c['text']!r} -> {got} (want {want})")
    rows.append(("GenAI", "Office plan: requirements extracted exactly", pct(ok, len(CASES["plan"]))))

    issue_ok = esc_ok = 0
    with tempfile.TemporaryDirectory() as tmp:
        for c in CASES["tickets"]:
            t, secs = timed(create_ticket, analyze(c["text"]), Path(tmp) / "eval.db")
            latencies.append(secs)
            issue_ok += t["issue_type"] in c["issue_types"]
            esc_ok += (t["status"] == "ESCALATED") == c["escalated"]
            if t["issue_type"] not in c["issue_types"] or (t["status"] == "ESCALATED") != c["escalated"]:
                misses.append(f"Ticket: {c['text']!r} -> {t['issue_type']}/{t['status']} "
                              f"(want {c['issue_types']}/{'ESCALATED' if c['escalated'] else 'OPEN'})")
    rows += [("GenAI", "Ticket issue_type accuracy", pct(issue_ok, len(CASES["tickets"]))),
             ("GenAI", "Ticket escalation correct", pct(esc_ok, len(CASES["tickets"])))]

    q = statistics.quantiles(latencies, n=20)
    rows.append(("Latency", f"Tool call p50 / p95 ({len(latencies)} calls, local CPU)",
                 f"{statistics.median(latencies):.1f}s / {q[18]:.1f}s"))

    table = "| Layer | Metric | Result |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    report = f"# Eval results\n\n{table}\n\n## Misses\n\n" + "\n".join(f"- {x}" for x in misses or ["none"]) + "\n"
    (HERE / "results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
