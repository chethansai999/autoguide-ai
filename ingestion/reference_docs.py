"""Reference documents supplied by the team (ADR-009).

PDFs such as IIHS crash-test protocols and rating guidelines are added by hand,
not scraped: put them in ``data/raw/reference/`` (``python ingest.py add-docs
FILE...`` copies them there) and run the same command to extract them into
``data/processed/reference/<doc_id>.json`` (the ``Document`` shape), which
``rag/chunking.chunk_corpus`` picks up.

Extraction is tuned for publisher-formatted reports:

* cover page -> title, version ("Version III") and date ("February 2026");
* the contents page -> the list of section headings, then removed;
* running headers/footers (lines repeated on most pages) -> removed;
* body lines equal to a contents entry -> ``## Heading`` / ``### Heading``;
* wrapped lines -> re-flowed into paragraphs; "Figure 3" + caption -> one line.

Raw PDFs stay in ``data/raw/`` and are never committed (publisher copyright).
Answers paraphrase them and quote at most 25 words (``trim_long_quotes``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from collections import Counter
from pathlib import Path

from core.config import get_settings
from core.schemas import Document, SourceType

log = logging.getLogger(__name__)

# Publisher metadata keyed by what appears in the PDF's Author field or text.
PUBLISHERS = {
    "IIHS": {
        "name": "Insurance Institute for Highway Safety (IIHS)",
        "short": "IIHS",
        "url": "https://www.iihs.org/ratings/about-our-tests/test-protocols-and-technical-information",
        "match": re.compile(r"\bIIHS\b|Insurance Institute for Highway Safety", re.I),
    },
}

TOC_LINE = re.compile(r"^(?P<title>.+?)\s*\.{3,}\s*(?P<page>\d+)\s*$")
VERSION_LINE = re.compile(r"^Version\s+(?P<v>[IVXLC]+|\d+(?:\.\d+)?)\s*$", re.I)
DATE_LINE = re.compile(
    r"^(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})\s*$"
)
CAPTION_START = re.compile(r"^(Figure|Table)\s+\d+[A-Za-z]?\s*$")
BOILERPLATE = re.compile(r"^(?:©|\(c\)|copyright)?\s*\d{4}\s+Insurance Institute for Highway Safety\s*$|all rights reserved", re.I)
SENTENCE_END = re.compile(r"[.:;?!)\]]\s*$")
MONTHS = {m: i for i, m in enumerate(
    "January February March April May June July August September October November December".split(), start=1)}


def _norm(line: str) -> str:
    return " ".join(line.split()).strip()


def _key(line: str) -> str:
    return re.sub(r"\d+", "#", _norm(line).lower())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _roman_to_int(value: str) -> int | None:
    numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    value = value.upper()
    if not value or any(c not in numerals for c in value):
        return None
    total = 0
    for i, c in enumerate(value):
        n = numerals[c]
        total += -n if i + 1 < len(value) and numerals[value[i + 1]] > n else n
    return total


def detect_publisher(author: str, text: str) -> dict | None:
    for publisher in PUBLISHERS.values():
        if publisher["match"].search(author or "") or publisher["match"].search(text[:5000]):
            return publisher
    return None


def parse_toc(page_lines: list[list[str]], max_pages: int = 4) -> tuple[list[str], set[int]]:
    """Headings listed on the contents page(s) and which pages those are."""
    headings: list[str] = []
    toc_pages: set[int] = set()
    for index, lines in enumerate(page_lines[:max_pages]):
        found = 0
        pending = ""
        for raw in lines:
            line = _norm(raw)
            if not line:
                continue
            match = TOC_LINE.match(f"{pending} {line}".strip() if pending else line)
            if match:
                headings.append(_norm(match.group("title")))
                found += 1
                pending = ""
            elif found and line.isupper() and len(line) > 20:  # a wrapped long entry
                pending = line
            else:
                pending = ""
        if found >= 3:
            toc_pages.add(index)
    return headings, toc_pages


def repeated_lines(page_lines: list[list[str]], min_share: float = 0.4) -> set[str]:
    """Normalised lines that appear on at least ``min_share`` of pages (headers/footers)."""
    counts: Counter = Counter()
    for lines in page_lines:
        counts.update({_key(line) for line in lines if _norm(line)})
    threshold = max(3, int(len(page_lines) * min_share))
    return {key for key, count in counts.items() if count >= threshold}


def _heading_text(title: str) -> str:
    return title.capitalize() if title.isupper() else title


def reflow(lines: list[str], headings: list[str], lookahead: int = 3) -> str:
    """Join wrapped lines into paragraphs; mark headings and figure captions.

    Headings are matched in contents-page order (the next one expected, or one
    of the few after it), so a table cell that happens to read "Torso" is not
    mistaken for the section of that name.
    """
    expected = 0
    lengths = sorted(len(_norm(line)) for line in lines if len(_norm(line)) > 20)
    typical = lengths[len(lengths) // 2] if lengths else 80
    out: list[str] = []
    paragraph: list[str] = []
    caption: str | None = None

    def flush() -> None:
        if paragraph:
            out.append(" ".join(paragraph))
            paragraph.clear()

    for raw in lines:
        line = _norm(raw)
        if not line:
            continue
        upcoming = [h.lower() for h in headings[expected:expected + lookahead]]
        if line.lower() in upcoming:
            heading = headings[expected + upcoming.index(line.lower())]
            expected += upcoming.index(line.lower()) + 1
            flush()
            out.append(f"{'## ' if heading.isupper() else '### '}{_heading_text(heading)}")
            continue
        if caption is not None:  # line after "Figure 3" is its caption
            out.append(f"[{caption}: {line}]")
            caption = None
            continue
        if CAPTION_START.match(line):
            flush()
            caption = line
            continue
        paragraph.append(line)
        # a short line that ends a sentence usually ends the paragraph
        if SENTENCE_END.search(line) and len(line) < 0.85 * typical:
            flush()
    flush()
    return "\n\n".join(out)


# Sections that list many topics in a few lines (revision notes, bibliographies):
# they answer no reader question and crowd out the real content in retrieval.
SKIPPED_SECTIONS = ("document revision history", "revision history", "references")


def drop_sections(text: str, names: tuple[str, ...] = SKIPPED_SECTIONS) -> str:
    """Remove ``## Name`` sections (to the next heading of the same or higher level)."""
    kept, skip_level = [], None
    for block in text.split("\n\n"):
        match = re.match(r"^(#{2,6})\s+(.*)$", block)
        if match:
            level, title = len(match.group(1)), match.group(2).strip().lower()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and any(title == n or title.startswith(n + " ") for n in names):
                skip_level = level
                continue
        if skip_level is None:
            kept.append(block)
    return "\n\n".join(kept)


def extract_reference_pdf(path: Path, source_url: str | None = None) -> dict:
    """Extract one publisher PDF into a ``Document``-shaped dict."""
    import pymupdf

    with pymupdf.open(path) as pdf:
        meta = pdf.metadata or {}
        pages = [page.get_text("text") for page in pdf]
    page_lines = [text.splitlines() for text in pages]
    full_text = "\n".join(pages)

    publisher = detect_publisher(meta.get("author", ""), full_text)
    cover = [_norm(line) for line in page_lines[0] if _norm(line)] if page_lines else []
    version = next((m.group("v") for line in cover[:12] if (m := VERSION_LINE.match(line))), None)
    date_match = next((m for line in cover[:12] if (m := DATE_LINE.match(line))), None)
    published = f"{date_match.group('month')} {date_match.group('year')}" if date_match else None
    effective = f"{date_match.group('year')}-{MONTHS[date_match.group('month')]:02d}" if date_match else None

    title = re.sub(r"[:,]?\s*Version\s+[IVXLC\d.]+\s*$", "", _norm(meta.get("title") or ""), flags=re.I).strip()
    if not title:  # cover lines before the version line
        stop = next((i for i, line in enumerate(cover) if VERSION_LINE.match(line) or line.upper() == "CONTENTS"), 3)
        title = ": ".join(cover[:stop]) or path.stem
    label_bits = [b for b in (f"Version {version}" if version else None, published) if b]
    display_title = f"{publisher['short'] + ' ' if publisher else ''}{title}" + (f" ({', '.join(label_bits)})" if label_bits else "")

    headings, toc_pages = parse_toc(page_lines)
    noise = repeated_lines(page_lines)
    body_lines: list[str] = []
    page_records = []
    for index, lines in enumerate(page_lines):
        if index in toc_pages:
            continue
        kept = [line for line in lines if _norm(line) and _key(line) not in noise
                and not TOC_LINE.match(_norm(line)) and not BOILERPLATE.search(_norm(line))]
        if index == 0:  # cover without a contents list: drop the title block
            kept = [line for line in kept if _norm(line) not in cover[:6]]
        body_lines.extend(kept)
        page_records.append({"page_number": index + 1, "text": "\n".join(kept)})

    text = drop_sections(reflow(body_lines, headings))
    version_number = _roman_to_int(version) if version and version.isalpha() else None
    slug = _slug(title)[:80]
    doc_id = f"{(publisher['short'].lower() + '-') if publisher else ''}{slug}" + (
        f"-v{version_number or _slug(version)}" if version else "")
    return {
        "doc_id": doc_id,
        "source_url": source_url or (publisher["url"] if publisher else f"file:{path.name}"),
        "title": display_title,
        "text": text,
        "page_type": "pdf",
        "source_type": SourceType.REFERENCE.value,
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "effective_date": effective,
        "version": "v1",
        "headings": headings,
        "pages": page_records,
        "publisher": publisher["name"] if publisher else (meta.get("author") or None),
        "document_version": version,
        "published": published,
        "file_name": path.name,
    }


def add_documents(files: list[Path] | None = None, raw_dir: Path | None = None, out_dir: Path | None = None,
                  source_url: str | None = None) -> dict:
    """Copy ``files`` into the raw folder (if given), then extract every PDF there."""
    settings = get_settings()
    raw_dir = raw_dir or settings.data_dir / "raw" / "reference"
    out_dir = out_dir or settings.data_dir / "processed" / "reference"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in files or []:
        file = Path(file)
        if file.suffix.lower() != ".pdf":
            raise ValueError(f"{file} is not a PDF")
        target = raw_dir / re.sub(r"^[0-9a-f]{8}-", "", file.name)  # drop upload-id prefixes
        if file.resolve() != target.resolve():
            shutil.copy2(file, target)

    processed, rows = 0, []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        try:
            doc = extract_reference_pdf(pdf, source_url=source_url)
            Document.model_validate(doc)  # contract check before writing
        except Exception as error:
            log.warning("Skipping %s: %s", pdf.name, error)
            continue
        (out_dir / f"{doc['doc_id']}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        processed += 1
        rows.append({"doc_id": doc["doc_id"], "title": doc["title"], "words": len(doc["text"].split()),
                     "sections": len(doc["headings"])})
    return {"processed": processed, "documents": rows, "out_dir": str(out_dir)}
