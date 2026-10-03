from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, db
from app.genai.graph import chat
from app.genai.tools import IMAGES_DIR
from app.ingest.ingest import INDEX_DIR
from app.nlp.pipeline import NLPResult, analyze

app = FastAPI(title="Dealer Copilot", version="1.0")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")  # example furniture photos

Message = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    message: str = Message


class AnalyzeRequest(BaseModel):
    message: str = Message


class ChatResponse(BaseModel):
    response: str
    nlp: dict
    tool_used: str | None
    data: dict
    sources: list
    suggestions: list[str] = []
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    return chat(req.session_id, req.message)


@app.post("/nlp/analyze", response_model=NLPResult)
def analyze_endpoint(req: AnalyzeRequest):
    return analyze(req.message)


@app.get("/tickets/{ticket_id}")
def ticket_endpoint(ticket_id: str):
    ticket = db.get_ticket(ticket_id, db.DB_PATH)
    if ticket is None:
        raise HTTPException(404, f"Ticket {ticket_id} not found")
    return ticket


@app.get("/quotes/{quote_id}")
def quote_endpoint(quote_id: str):
    quote = db.get_quote(quote_id, db.DB_PATH)
    if quote is None:
        raise HTTPException(404, f"Quote {quote_id} not found")
    return quote


@app.get("/health")
def health():
    return {"status": "ok", "store_name": config.STORE_NAME,
            "llm_model": config.LLM_MODEL, "embed_model": config.EMBED_MODEL,
            "index_built": (INDEX_DIR / "index.faiss").exists()}
