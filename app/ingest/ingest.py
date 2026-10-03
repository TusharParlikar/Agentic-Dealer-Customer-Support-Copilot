"""Store data -> FAISS index (manual/policy pages) + catalog table + NER product patterns."""
import unicodedata
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_embeddings

STORE = Path(__file__).resolve().parents[2] / "store_data"
CATALOG = STORE / "catalog.csv"
MANUALS = STORE / "manuals"
POLICIES = STORE / "policies"
INDEX_DIR = STORE / "index"


def load_catalog(path: Path = CATALOG) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"capacity": "Int64"}, keep_default_na=False, na_values={"capacity": [""]})
    return df


def load_pages(names: set[str] | None = None) -> list[Document]:
    """All manual + policy pages, or only the PDFs in `names` (parsing all of them takes ~2 min)."""
    pages = []
    for folder, doc_type in ((MANUALS, "manual"), (POLICIES, "policy")):
        for pdf in sorted(folder.glob("*.pdf")):
            if names and pdf.name not in names:
                continue
            for d in PyPDFLoader(str(pdf)).load():
                if not d.page_content.strip():
                    continue  # image-only page, nothing to retrieve
                d.metadata = {"source": pdf.name, "page": d.metadata["page"] + 1, "doc_type": doc_type}
                pages.append(d)
    return pages


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def product_patterns(catalog: pd.DataFrame) -> list[dict]:
    # One pattern per product line ("Markus", "Tahoe"); "Flintan / Nominell" gives two.
    patterns, seen = [], set()
    for line in catalog["product_line"].unique():
        for part in line.split("/"):
            for spelling in {part.strip().lower(), _ascii(part.strip().lower())}:
                tokens = tuple(spelling.split())
                if tokens and tokens not in seen:
                    seen.add(tokens)
                    patterns.append({"label": "PRODUCT", "pattern": [{"LOWER": t} for t in tokens], "id": line})
    return patterns


def build_index(pages: list[Document], out_dir: Path = INDEX_DIR) -> FAISS:
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(pages)
    # Small batches: one request with every chunk crashed the Ollama embed runner.
    batch = 32
    store = FAISS.from_documents(chunks[:batch], get_embeddings())
    for i in range(batch, len(chunks), batch):
        store.add_documents(chunks[i:i + batch])
        print(f"embedded {min(i + batch, len(chunks))}/{len(chunks)}", flush=True)
    store.save_local(str(out_dir))
    return store


def main() -> None:
    pages = load_pages()
    build_index(pages)
    print(f"{len(pages)} pages indexed into {INDEX_DIR}")


if __name__ == "__main__":
    main()
