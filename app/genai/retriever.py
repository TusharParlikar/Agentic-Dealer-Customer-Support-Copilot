"""FAISS retrieval over manual/policy pages, optionally limited to the manuals of named products."""
from functools import cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_embeddings
from app.ingest.ingest import INDEX_DIR, load_catalog

# L2 distance on nomic-embed-text vectors; lower = closer. Calibrated on eval/testset.json: answerable
# questions' best chunk scored 0.28-0.67, off-topic questions 0.97-1.26. Re-check if the embed model changes.
MAX_DISTANCE = 0.8


@cache
def _store() -> FAISS:
    return FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)  # our own file


def manuals_for(products: list[str]) -> set[str] | None:
    """Manual files for these product lines, or None when none of them has a manual (search everything)."""
    catalog = load_catalog()
    manuals = set(catalog.loc[catalog["product_line"].isin(products) & (catalog["manual"] != ""), "manual"])
    return manuals or None


def retrieve(query: str, products: list[str], k: int = 4) -> list[tuple[Document, float]]:
    manuals = manuals_for(products)
    hits = _store().similarity_search_with_score(
        query, k=k, fetch_k=50,
        filter=(lambda md: md["source"] in manuals) if manuals else None,
    )
    return [(d, float(s)) for d, s in hits if s <= MAX_DISTANCE]
