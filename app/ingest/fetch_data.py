"""Download the real store data and build store_data/catalog.csv.

Manuals and policies are third-party documents: they are downloaded here and
git-ignored, never committed.
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "store_data"

IKEA_CSV = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv"
BITEXT_CSV = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
              "/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

DOCS = {
    "manuals/tahoe.pdf": "https://cdn.shopify.com/s/files/1/0669/3791/3622/files/Tahoe-CF-DEL-II2.pdf",
    "manuals/drolet.pdf": "https://www.drolet.ca/file/45930a_20-03-2019.pdf",
    "manuals/dimplex.pdf": "https://homelectrical.com/sites/default/files/DMP-XHD23L-Manual.pdf",
    "manuals/office_chair.pdf": "https://media-www.canadiantire.ca/manual/product/1680183/customizable-office-chair-product-instruction-manuals-english-updated-12-7bbee24a-ced7-4c12-8b4e-0d003bca90a4.pdf",
    "manuals/standing_desk.pdf": "https://www.roomandboard.com/images/pdf/troubleshooting_guide_height_adjustable_desks.pdf",
    "manuals/ology_desk.pdf": "https://alresources.nd.edu/assets/565018/steelcase_ology_troubleshooting_manual.pdf",
    "policies/furniture_warranty.pdf": "https://cdn.rona.ca/webassets/manuals/330785372_WarrantyGuide.pdf",
    "policies/terms_delivery_returns.pdf": "https://cdn.shopify.com/s/files/1/0168/1304/files/CF_Designer_Terms_Conditions.pdf",
}

# One openly licensed example photo per item type (Wikimedia Commons), shown when a customer picks a plan.
# Generic photos of the type, not the exact catalog item: the UI labels them "Example photo".
PHOTOS = {
    "desk": "File:Workspace (Unsplash).jpg",
    "sitstand_desk": "File:Height adjustable workstation.jpg",
    "office_chair": "File:Buerostuhl (fcm).jpg",
    "conference_chair": "File:Stylish meeting room (Unsplash).jpg",
    "meeting_table": "File:Quiet meeting room (Unsplash).jpg",
    "storage": "File:(1) Steel storage cabinet with multiple drawers-front view (4791860538).jpg",
    "gas_fireplace": "File:NT Typical gas log fireplace (5114230942).jpg",
    "wood_stove": "File:Cast Iron Wood Stove.JPG",
    "electric_fireplace": "File:Dimplex electric fireplace.jpg",
}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "DealerCopilot/1.0 (resume project)"}

# ponytail: fixed rates (Oct 2026, approximate). Swap for a rates API if prices must be live.
USD_TO_INR = 88.0
SAR_TO_INR = USD_TO_INR / 3.75  # SAR is pegged at 3.75 per USD

# IKEA short_description regex -> our category
IKEA_CATEGORIES = {
    r"^Office chair": "office_chair",
    r"^Conference chair": "conference_chair",
    r"^Desk(?: sit/stand)?, \d": "desk",  # skips desk tops, screens, combinations
    r"^Conference table": "meeting_table",
    r"^Drawer unit": "storage",
}
SIZE = r"(\d+)(?:x(\d+))?(?:x(\d+))? cm"  # "140x70 cm", "140 cm" (round)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=300) as r:
        dest.write_bytes(r.read())
    print(f"downloaded {dest.relative_to(ROOT)}")


def download_photos(out: Path = ROOT / "images") -> None:
    """Fetch the PHOTOS (640 px wide) and write credits.json with author + license, as CC BY / BY-SA require."""
    credits_file = out / "credits.json"
    credits = json.loads(credits_file.read_text(encoding="utf-8")) if credits_file.exists() else {}
    for key, title in PHOTOS.items():
        if (out / f"{key}.jpg").exists() and key in credits:
            continue
        query = urlencode({"action": "query", "format": "json", "titles": title, "prop": "imageinfo",
                           "iiprop": "url|extmetadata", "iiurlwidth": 640})
        with urlopen(Request(f"{COMMONS_API}?{query}", headers=UA), timeout=60) as r:
            info = next(iter(json.load(r)["query"]["pages"].values()))["imageinfo"][0]
        meta = info["extmetadata"]
        author = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "unknown")).strip()
        license_ = meta.get("LicenseShortName", {}).get("value", "see source")
        download(info["thumburl"], out / f"{key}.jpg")
        credits[key] = f"Example photo: {author}, {license_}, via Wikimedia Commons"
        time.sleep(3)  # Commons rate-limits fast downloads (HTTP 429)
    credits_file.write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")


def table_seats(length_cm: float) -> int:
    # ponytail: estimate, 70 cm of edge per person on both long sides; IKEA lists no seat count.
    return max(4, 2 * int(length_cm // 70))


def ikea_rows(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["desc"] = df["short_description"].str.split().str.join(" ")
    df["category"] = None
    for pattern, cat in IKEA_CATEGORIES.items():
        df.loc[df["desc"].str.contains(pattern), "category"] = cat
    df = df[df["category"].notna()].sort_values("price").drop_duplicates(["name", "desc"])
    # The width/depth columns are often blank; the description has the real size.
    size = df["desc"].str.extract(SIZE)

    def dims(i):
        return "x".join(v for v in size.loc[i] if pd.notna(v))

    def capacity(i):
        if df.at[i, "category"] != "meeting_table":
            return None
        return table_seats(int(size.at[i, 0]))

    return pd.DataFrame({
        "sku": "IK-" + df["item_id"].astype(str),
        "name": df["name"].str.title() + " " + df["desc"],
        "category": df["category"],
        "product_line": df["name"].str.title(),
        "price_inr": (df["price"] * SAR_TO_INR).round(-1).astype(int),
        "capacity": pd.Series([capacity(i) for i in df.index], index=df.index, dtype="Int64"),
        "dims_cm": [dims(i) for i in df.index],
        "manual": "",
        "source_url": df["link"],
    })


def hearth_rows(hearth: pd.DataFrame) -> pd.DataFrame:
    out = hearth.copy()
    out["price_inr"] = (out.pop("price_usd") * USD_TO_INR).round(-1).astype(int)
    out["capacity"] = out["capacity"].astype("Int64")
    return out


def build_catalog() -> pd.DataFrame:
    furniture = ikea_rows(pd.read_csv(ROOT / "raw" / "ikea.csv", index_col=0))
    hearth = hearth_rows(pd.read_csv(ROOT / "hearth.csv", keep_default_na=False, na_values={"capacity": [""]}))
    catalog = pd.concat([furniture, hearth[furniture.columns]], ignore_index=True).fillna({"dims_cm": "", "manual": ""})
    catalog.to_csv(ROOT / "catalog.csv", index=False)
    return catalog


def main() -> None:
    download(IKEA_CSV, ROOT / "raw" / "ikea.csv")
    download(BITEXT_CSV, ROOT / "raw" / "bitext.csv")
    for rel, url in DOCS.items():
        download(url, ROOT / rel)
    download_photos()
    print(f"catalog: {len(build_catalog())} rows")


if __name__ == "__main__":
    main()
