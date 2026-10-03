"""nlp_node -> router -> tool -> respond_node, with an LLM agent for OTHER / low-confidence messages."""
import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import STORE_NAME, get_llm
from app.genai import prompts, tools
from app.nlp.pipeline import NLPResult, analyze

MIN_CONFIDENCE = 0.6
INTENT_TO_TOOL = {"PRODUCT_QA": "product_qa", "OFFICE_PLAN": "plan_office", "SUPPORT": "create_ticket"}


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    nlp: dict
    tool_used: str | None
    data: dict
    sources: list
    response: str
    plan: dict | None  # the office plan so far, structured: req, tier, chosen, quote_id (see tools.plan_turn)
    plan_action: str | None  # update / choose / ack when this message continues the plan
    chosen_tier: str | None  # option the customer just picked: budget / standard / premium
    suggestions: list[str]  # quick replies for the next step


PLAN_KEYS = ("headcount", "meeting_room_seats", "budget_inr")


def _joins_plan(nlp: NLPResult) -> bool:
    """Not clearly about something else (a confident product question or complaint)."""
    return nlp.intent in ("OFFICE_PLAN", "OTHER") or nlp.confidence < MIN_CONFIDENCE


def _plan_action(nlp: NLPResult, plan: dict | None, last_tool: str | None) -> tuple[str | None, str | None]:
    """Is this message a step in the open plan conversation? Checked before the intent classifier, which
    only sees one message: "I'll take the Basic option" alone looks like a product question."""
    if not plan or nlp.safety_flag:
        return None, None
    if chosen := tools.detect_choice(nlp.text):
        return "choose", chosen
    if tools.is_ack(nlp.text):
        return ("ack", None) if last_tool == "plan_office" else (None, None)  # "ok" answers the last bot turn
    edits = tools.NO_ROOM.search(nlp.text) or tools.NO_STORAGE.search(nlp.text) or tools.SHOW_OPTIONS.search(nlp.text)
    if edits or any(nlp.entities[k] for k in PLAN_KEYS) or (last_tool == "plan_office" and _joins_plan(nlp)):
        return "update", None
    return None, None


def nlp_node(state: State) -> dict:
    nlp = analyze(state["messages"][-1].content)
    action, chosen = _plan_action(nlp, state.get("plan"), state.get("tool_used"))
    return {"nlp": nlp.model_dump(), "plan_action": action, "chosen_tier": chosen,
            "tool_used": None, "data": {}, "sources": [], "response": ""}


def route(state: State) -> str:
    nlp = state["nlp"]
    if nlp["safety_flag"]:
        return "create_ticket"  # never let a gas smell end up as small talk
    if state.get("plan_action"):
        return "plan_office"
    if nlp["intent"] == "OTHER" or nlp["confidence"] < MIN_CONFIDENCE:
        return "agent"
    return INTENT_TO_TOOL[nlp["intent"]]


def _run(name: str, state: State) -> dict:
    nlp = NLPResult(**state["nlp"])
    if name == "product_qa":
        r = tools.product_qa(nlp)
        return {"tool_used": name, "data": {"answer": r["answer"]}, "sources": r["sources"]}
    if name == "plan_office":
        data, plan = tools.plan_turn(nlp, state.get("plan"), state.get("plan_action") or "update",
                                     state.get("chosen_tier"))
        return {"tool_used": name, "data": data, "plan": plan}
    return {"tool_used": name, "data": tools.create_ticket(nlp)}


# Tool descriptions the fallback agent chooses from. The bodies never run: _run does the work with the NLP result.
@tool
def product_qa(question: str) -> str:
    """Answer a question about a product, its manual, warranty, delivery or returns."""


@tool
def plan_office(request: str) -> str:
    """Plan office furniture (desks, chairs, meeting tables, storage) and estimate the cost."""


@tool
def create_ticket(complaint: str) -> str:
    """Log a support ticket for a broken, faulty or unsafe product, or a delivery problem."""


def agent_node(state: State) -> dict:
    llm = get_llm().bind_tools([product_qa, plan_office, create_ticket])
    reply = llm.invoke([SystemMessage(prompts.CHAT_SYSTEM), *state["messages"][-10:]])
    if reply.tool_calls:
        return _run(reply.tool_calls[0]["name"], state)
    return {"response": tools.no_promises(reply.content)}


def respond_node(state: State) -> dict:
    name, data = state.get("tool_used"), state.get("data", {})
    if name is None:
        text = state["response"]
    elif name == "product_qa":
        text = data["answer"]  # already grounded and cited; a second LLM pass would only add risk
    elif name == "plan_office":
        text = prompts.PLAN_ASK if data.get("missing") else tools.plan_reply(data)  # exact numbers: code, not LLM
    else:
        text = get_llm().invoke([SystemMessage(prompts.RESPOND_SYSTEM),
                                 HumanMessage(f"Customer: {state['nlp']['text']}\n\nTool result ({name}):\n"
                                              + json.dumps({k: v for k, v in data.items() if k != "safety_message"},
                                                           ensure_ascii=False))]).content  # added verbatim below
        # Ticket id must reach the customer verbatim, whatever the LLM wrote.
        if data["ticket_id"] not in text:
            text += f"\n\nYour ticket ID is {data['ticket_id']} (status: {data['status']})."
    first_turn = not any(isinstance(m, AIMessage) for m in state["messages"])
    if first_turn:
        text = f"{prompts.GREETING.format(store=STORE_NAME)}\n\n{text}"
    if data.get("safety_message"):
        text = f"{data['safety_message']}\n\n{text}"  # safety before anything else, even the greeting
    return {"response": text, "messages": [AIMessage(text)], "suggestions": tools.suggestions(name, data)}


def build_graph():
    g = StateGraph(State)
    g.add_node("nlp", nlp_node)
    g.add_node("product_qa", lambda s: _run("product_qa", s))
    g.add_node("plan_office", lambda s: _run("plan_office", s))
    g.add_node("create_ticket", lambda s: _run("create_ticket", s))
    g.add_node("agent", agent_node)
    g.add_node("respond", respond_node)
    g.add_edge(START, "nlp")
    g.add_conditional_edges("nlp", route, ["product_qa", "plan_office", "create_ticket", "agent"])
    for node in ("product_qa", "plan_office", "create_ticket", "agent"):
        g.add_edge(node, "respond")
    g.add_edge("respond", END)
    # ponytail: in-memory checkpointer, history is lost on restart. Swap for SqliteSaver to persist.
    return g.compile(checkpointer=InMemorySaver())


GRAPH = build_graph()


def chat(session_id: str, message: str) -> dict:
    state = GRAPH.invoke({"messages": [HumanMessage(message)]}, {"configurable": {"thread_id": session_id}})
    return {
        "response": state["response"],
        "nlp": state["nlp"],
        "tool_used": state.get("tool_used"),
        "data": state.get("data", {}),
        "sources": state.get("sources", []),
        "suggestions": state.get("suggestions", []),
        "session_id": session_id,
    }


def history(session_id: str) -> list[BaseMessage]:
    return GRAPH.get_state({"configurable": {"thread_id": session_id}}).values.get("messages", [])
