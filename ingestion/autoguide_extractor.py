"""Extract AutoGuide.com article HTML into `core.schemas.Document` JSON.

Keeps reading order with inline '## Heading' markers (for structure-aware
chunking), renders tables as pipe rows in place, and pulls title / dates
from JSON-LD or meta tags. Ads, navigation, related-article rails, share
bars and comments are removed.

    python ingest.py extract-autoguide
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from core.config import get_settings
from ingestion.cleaner import clean_text

log = logging.getLogger(__name__)

NOISE_SELECTORS = (
    "script, style, noscript, iframe, form, nav, header, footer, aside, svg, button, "
    "[class*=advert], [class^=ad-], [class~=ad], [id^=ad-], [id^=ad_], [class*=related], [class*=newsletter], "
    "[class*=share], [class*=social], [class*=comment], [class*=breadcrumb], [class*=sidebar], "
    "[class*=promo], [class*=subscribe], [aria-hidden=true]"
)
BODY_SELECTORS = ("[itemprop=articleBody]", "article", "main", "[role=main]", "body")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\b\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")


def _json_ld(soup: BeautifulSoup) -> dict:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (TypeError, ValueError):
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for item in items:
            if isinstance(item, dict) and str(item.get("@type", "")).lower() in ("article", "newsarticle", "blogposting", "review"):
                return item
    return {}


def _meta(soup: BeautifulSoup, *names: str) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def _table_rows(table: Tag) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [" ".join(c.get_text(" ", strip=True).split()) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    return rows


def extract_article(html: bytes | str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    ld = _json_ld(soup)
    title = (
        ld.get("headline")
        or _meta(soup, "og:title", "twitter:title")
        or (soup.title.get_text(" ", strip=True) if soup.title else url)
    )
    published = ld.get("datePublished") or _meta(soup, "article:published_time")
    modified = ld.get("dateModified") or _meta(soup, "article:modified_time")

    for node in soup.select(NOISE_SELECTORS):
        if node.name not in ("body", "html"):
            node.decompose()
    body = next((soup.select_one(s) for s in BODY_SELECTORS if soup.select_one(s)), soup)

    blocks: list[str] = []
    headings: list[str] = []
    tables: list[list[list[str]]] = []
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.find_parent("table") is not None and el.name != "table":
            continue
        if el.name == "table":
            rows = _table_rows(el)
            if rows:
                tables.append(rows)
                blocks.append("\n".join(" | ".join(r) for r in rows))
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            if el.name == "h1" and text == title:
                continue
            headings.append(text)
            blocks.append(f"## {text}")
        elif len(text) >= 25 or el.name == "li":
            blocks.append(text)

    text = clean_text("\n".join(blocks))
    text = PHONE.sub("[phone removed]", EMAIL.sub("[email removed]", text))
    return {
        "title": " ".join(str(title).split()),
        "text": text,
        "headings": headings,
        "tables": tables,
        "table_count": len(tables),
        "effective_date": modified or published,
        "published_date": published,
    }


def process_autoguide(raw_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    settings = get_settings()
    raw_dir = raw_dir or settings.data_dir / "raw" / "autoguide"
    out_dir = out_dir or settings.data_dir / "processed" / "autoguide"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = raw_dir / "metadata.jsonl"
    if not metadata.exists():
        return {"processed": 0, "note": "no raw AutoGuide pages; run crawl-autoguide first"}
    processed, too_short = 0, 0
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        raw_path = Path(meta["raw_file"])
        if not raw_path.exists():
            raw_path = raw_dir / Path(meta["raw_file"]).name
        article = extract_article(raw_path.read_bytes(), meta["source_url"])
        if len(article["text"].split()) < 80:
            too_short += 1
            continue
        doc = {**meta, **article, "source_type": "autoguide_article", "version": "v1"}
        (out_dir / f"{meta['doc_id']}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        processed += 1
    log.info("AutoGuide: %d documents processed, %d skipped as too short", processed, too_short)
    return {"processed": processed, "skipped_too_short": too_short, "out_dir": str(out_dir)}
