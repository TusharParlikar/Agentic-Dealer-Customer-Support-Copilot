from pathlib import Path

import pandas as pd
import pytest

from app.ingest.fetch_data import ikea_rows, table_seats
from app.ingest.ingest import CATALOG, MANUALS, build_index, load_catalog, load_pages, product_patterns


def test_table_seats_estimate():
    assert table_seats(420) == 12
    assert table_seats(280) == 8
    assert table_seats(140) == 4


def test_ikea_rows_keep_only_office_items_and_parse_size():
    raw = pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "name": ["BEKANT", "BEKANT", "VITVAL", "MARKUS"],
        "price": [4240.0, 1000.0, 200.0, 795.0],
        "short_description": ["Conference table,   280x140 cm", "Desk with screen, 160x160 55 cm",
                              "Desk top, 95x45 cm", "Office chair"],
        "link": ["u1", "u2", "u3", "u4"],
    })
    rows = ikea_rows(raw).set_index("sku")
    assert list(rows.index) == ["IK-4", "IK-1"]  # desk top + screen dropped, sorted by price
    assert rows.at["IK-1", "category"] == "meeting_table"
    assert rows.at["IK-1", "capacity"] == 8
    assert rows.at["IK-1", "dims_cm"] == "280x140"
    assert rows.at["IK-4", "price_inr"] > 0


def test_catalog_is_valid():
    df = load_catalog()
    assert len(df) >= 40
    assert {"sku", "name", "category", "product_line", "price_inr", "capacity", "manual", "source_url"} <= set(df.columns)
    assert df["sku"].is_unique
    assert (df["price_inr"] > 0).all()
    assert df.loc[df["category"] == "meeting_table", "capacity"].notna().all()
    for manual in df["manual"][df["manual"] != ""]:
        assert (MANUALS / manual).exists(), manual


def test_pages_are_one_based_and_tagged():
    pages = load_pages({"standing_desk.pdf", "furniture_warranty.pdf"})
    desk = [d.metadata["page"] for d in pages if d.metadata["source"] == "standing_desk.pdf"]
    assert desk == list(range(1, len(desk) + 1))
    kinds = {d.metadata["source"]: d.metadata["doc_type"] for d in pages}
    assert kinds == {"standing_desk.pdf": "manual", "furniture_warranty.pdf": "policy"}
    assert all(d.page_content.strip() for d in pages)


def test_product_patterns_include_ascii_spellings():
    patterns = product_patterns(load_catalog())
    by_text = {" ".join(t["LOWER"] for t in p["pattern"]): p["id"] for p in patterns}
    assert by_text["markus"] == "Markus"
    assert by_text["långfjäll"] == "Långfjäll"
    assert by_text["langfjall"] == "Långfjäll"
    assert by_text["nominell"] == "Flintan / Nominell"
    assert by_text["tahoe"] == "Tahoe"
    assert all(p["label"] == "PRODUCT" for p in patterns)


@pytest.mark.ollama
def test_index_retrieves_right_page(tmp_path: Path):
    store = build_index(load_pages({"dimplex.pdf", "standing_desk.pdf"}), tmp_path)
    hit = store.similarity_search("My electric fireplace heater overheated and shut off. How do I reset it?", k=1)[0]
    assert (hit.metadata["source"], hit.metadata["page"]) == ("dimplex.pdf", 14)
    assert (tmp_path / "index.faiss").exists()
