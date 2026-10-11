"""Extract saved Wikipedia/FuelEconomy HTML; never access the network."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from bs4 import BeautifulSoup

from core.config import get_settings
from core.schemas import Document
from ingestion.cleaner import clean_text
from ingestion.web_sources import HOSTS, PageSeed, SOURCE_TYPES

log = logging.getLogger(__name__)
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def extract_page(html: bytes, metadata: dict) -> Document:
    seed = PageSeed(source=metadata["source"], url=metadata["source_url"], vehicle=metadata.get("vehicle"))
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.select_one("h1") or soup.title
    title = title_node.get_text(" ", strip=True) if title_node else metadata.get("title", "")
    if seed.source == "wikipedia":
        root = soup.select_one("#mw-content-text .mw-parser-output, #mw-content-text, main .mw-parser-output")
        attribution = {"authors": "Wikipedia contributors", "license": "CC BY-SA 4.0",
                       "license_url": LICENSE_URL, "source_url": seed.url,
                       "history_url": seed.url + "?action=history",
                       "changes": "Text extracted and normalized; answers may paraphrase the source."}
    else:
        root = soup.select_one("main, #content, #main-content, #mainContent, #content-area, article")
        attribution = {"authors": "FuelEconomy.gov", "source_url": seed.url}
    if root is None:
        raise ValueError(f"No recognized article content at {seed.url}")
    for node in root.select("script, style, nav, footer, header, aside, form, button, .mw-editsection, "
                            ".navbox, .reflist, .references, .reference, .toc, .hatnote, .metadata, "
                            ".sistersitebox, .printfooter, .catlinks, .noprint"):
        node.decompose()
    blocks, headings, tables = [], [], []
    for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "table", "dl"]):
        if node.find_parent(["table", "dl", "li"]):
            continue
        if node.name == "table":
            rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)]
                    for row in node.find_all("tr") if row.find_parent("table") is node]
            rows = [row for row in rows if any(row)]
            if rows:
                tables.append({"rows": rows})
                blocks.append("\n".join(" | ".join(row) for row in rows))
        else:
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            if node.name.startswith("h"):
                headings.append({"level": int(node.name[1]), "text": text})
                text = "#" * int(node.name[1]) + " " + text
            blocks.append(text)
    text = clean_text("\n\n".join(blocks))
    if len(text.split()) < 30:
        raise ValueError(f"Too little article text at {seed.url}; check HTML selectors")
    # Reviewed scope must be visible in the captured page; never silently tag a wrong car.
    for key, value in (seed.vehicle or {}).items():
        if key in {"make", "model", "year", "drive_train", "body_style"}:
            compact = lambda s: "".join(c for c in str(s).casefold() if c.isalnum())
            if compact(value) not in compact(f"{title} {text}"):
                raise ValueError(f"Manifest {key} is not present in the page: {seed.url}")
    digest = hashlib.sha256(html).hexdigest()
    if metadata.get("content_hash") and metadata["content_hash"] != digest:
        raise ValueError("Raw HTML hash does not match collection metadata")
    return Document(doc_id=metadata["doc_id"], source_url=seed.url, title=title, text=text,
                    source_type=SOURCE_TYPES[seed.source], page_type="html", content_hash=digest,
                    crawl_timestamp=metadata.get("crawl_timestamp"), version=digest[:16],
                    vehicle=seed.vehicle, headings=headings, tables=tables, attribution=attribution)


def process_web_sources(data_dir: Path | None = None, source: str = "all") -> dict:
    data = data_dir or get_settings().data_dir
    written = skipped = 0
    errors = []
    for name in HOSTS:
        if source not in ("all", name):
            continue
        raw_dir = data / "raw" / name
        meta = raw_dir / "metadata.jsonl"
        if not meta.exists():
            continue
        # A refresh supersedes the old snapshot; keep only the latest per source URL.
        latest = {}
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["requested_url"]] = row
        out = data / "processed" / name
        out.mkdir(parents=True, exist_ok=True)
        for row in sorted(latest.values(), key=lambda r: r["doc_id"]):
            target = out / f"{row['doc_id']}.json"
            try:
                raw = (raw_dir / row["raw_file"]).resolve()
                if raw.parent != raw_dir.resolve() or row["source"] != name:
                    raise ValueError("Invalid raw snapshot path or source")
                document = extract_page(raw.read_bytes(), row)
                payload = document.model_dump_json(indent=2)
                if target.exists() and target.read_text(encoding="utf-8") == payload:
                    skipped += 1
                else:
                    target.write_text(payload, encoding="utf-8")
                    written += 1
            except (ValueError, OSError, KeyError) as error:
                # Never keep an obsolete document after a failed refresh.
                target.unlink(missing_ok=True)
                errors.append({"doc_id": row["doc_id"], "error": str(error)})
                log.warning("Skipping %s: %s", row["doc_id"], error)
    return {"written": written, "unchanged": skipped, "errors": errors}
