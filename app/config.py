import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.environ["LLM_MODEL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
STORE_NAME = os.getenv("STORE_NAME", "Comfort Office & Hearth")


def get_llm(**kwargs) -> ChatOllama:
    # reasoning=False: qwen3 thinks by default, which is slow on a 1.7B model.
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, reasoning=False, **kwargs)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


if __name__ == "__main__":
    reply = get_llm().invoke("Reply with exactly: ok").content
    vec = get_embeddings().embed_query("ergo chair")
    print(f"llm={LLM_MODEL} reply={reply!r}")
    print(f"embed={EMBED_MODEL} dim={len(vec)}")
    assert reply.strip() and len(vec) > 0
