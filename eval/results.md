# Eval results

| Layer | Metric | Result |
|---|---|---|
| NLP | Intent accuracy (held-out 20%) | 0.866 |
| NLP | Intent macro-F1 (held-out 20%) | 0.889 |
| NLP | Entity `headcount` exact match | 19/20 = 95% |
| NLP | Entity `meeting_room_seats` exact match | 20/20 = 100% |
| NLP | Entity `budget_inr` exact match | 19/20 = 95% |
| NLP | Entity `products` exact match | 20/20 = 100% |
| NLP | Entity `location` exact match | 20/20 = 100% |
| GenAI | Retrieval hit@4 (right page in top 4) | 25/25 = 100% |
| GenAI | Citation accuracy (answer cites right page) | 23/25 = 92% |
| GenAI | Office plan: requirements extracted exactly | 8/8 = 100% |
| GenAI | Ticket issue_type accuracy | 12/12 = 100% |
| GenAI | Ticket escalation correct | 12/12 = 100% |
| Latency | Tool call p50 / p95 (45 calls, local CPU) | 21.6s / 55.0s |

## Misses

- NER headcount: 'Need 18 Skarsta desks with Flintan chairs' -> None (want 18)
- NER budget_inr: 'three 4 seater meeting rooms and 20 desks, 1.5 crore is too much, keep it under 9 lakh' -> 15000000 (want 900000)
- Citation: 'What does error E01 Position Lost mean on my standing desk?' -> [('standing_desk.pdf', 3)] (want [('standing_desk.pdf', 4), ('standing_desk.pdf', 10)])
- Citation: 'What clearance is needed between the stove and combustible walls?' -> [('drolet.pdf', 30), ('drolet.pdf', 35), ('drolet.pdf', 36), ('tahoe.pdf', 10)] (want [('drolet.pdf', 31), ('drolet.pdf', 32), ('drolet.pdf', 33), ('drolet.pdf', 34)])
