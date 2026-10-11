"""Offline pipeline CLI. Each stage reads the previous stage's files, so any
stage can be re-run alone (e.g. re-chunk without re-crawling).

    python ingest.py status                       # what exists at each stage
    python ingest.py crawl --max-pages 180        # AutoGuide.com articles   (WP-01, ADR-008)
    python ingest.py crawl --dry-run              # list what it would fetch, fetch nothing
    python ingest.py extract-autoguide            # raw -> data/processed/autoguide (WP-02)
    python ingest.py add-docs FILE.pdf ...        # team PDFs (e.g. IIHS protocols) -> processed/reference (ADR-009)
    python ingest.py collect-recalls              # NHTSA Recalls API (an API, not scraping)
    python ingest.py extract                      # legacy raw NHTSA pages (not crawled any more)
    python ingest.py normalize-recalls            # API JSON -> documents    (WP-02)
    python ingest.py load-safercar                # CSV -> vehicles table    (WP-08)
    python ingest.py chunk --strategy fixed       # -> data/chunks/fixed     (WP-03)
    python ingest.py index --strategy fixed       # -> ChromaDB + registry   (WP-04)
    python ingest.py all                          # extract..index (no crawling)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core.config import get_settings
from core.observability import configure_logging

log = logging.getLogger("ingest")


def cmd_status(_args) -> None:
    from sqlalchemy import func, select

    from db.models import IndexRegistry, Vehicle
    from db.session import init_db, session_scope

    settings = get_settings()
    data = settings.data_dir
    init_db()
    with session_scope() as db:
        vehicles = db.scalar(select(func.count()).select_from(Vehicle)) or 0
        indexes = [
            {"collection": i.collection, "strategy": i.strategy, "chunks": i.chunk_count, "active": i.is_active}
            for i in db.scalars(select(IndexRegistry)).all()
        ]
    raw_meta = data / "raw" / "metadata.jsonl"
    raw_rows = [json.loads(l) for l in raw_meta.read_text().splitlines() if l.strip()] if raw_meta.exists() else []
    ag_meta = data / "raw" / "autoguide" / "metadata.jsonl"
    report = {
        "web_sources": {name: {
            "raw_snapshots": sum(1 for line in (data / "raw" / name / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()) if (data / "raw" / name / "metadata.jsonl").exists() else 0,
            "processed_docs": len(list((data / "processed" / name).glob("*.json"))),
        } for name in ("wikipedia", "fueleconomy")},
        "autoguide_raw_pages": sum(1 for l in ag_meta.read_text().splitlines() if l.strip()) if ag_meta.exists() else 0,
        "autoguide_processed_docs": len(list((data / "processed" / "autoguide").glob("*.json"))),
        "autoguide_docs_with_tables": sum(
            1 for p in (data / "processed" / "autoguide").glob("*.json")
            if json.loads(p.read_text(encoding="utf-8")).get("table_count", 0) > 0
        ),
        "raw_html": sum(r.get("page_type") == "html" for r in raw_rows),
        "raw_pdf": sum(r.get("page_type") == "pdf" for r in raw_rows),
        "raw_api_files": len(list((data / "raw" / "api").glob("*.json"))),
        "processed_docs": len(list((data / "processed").glob("*.json"))),
        "processed_api_docs": len(list((data / "processed" / "api").glob("*.json"))),
        "reference_docs": len(list((data / "processed" / "reference").glob("*.json"))),  # PDFs (ADR-009)
        "pages_with_tables": sum(
            1 for p in (data / "processed").glob("*.json")
            if json.loads(p.read_text(encoding="utf-8")).get("table_count", 0) > 0
        ),
        "chunks": {
            s: sum(1 for _ in (settings.chunks_dir(s) / "chunks.jsonl").open()) if (settings.chunks_dir(s) / "chunks.jsonl").exists() else 0
            for s in ("fixed", "structure")
        },
        "vehicles": vehicles,
        "indexes": indexes,
        "corpus_minimums": {"html": 150, "pdf": 5, "pages_with_tables": 3},
    }
    print(json.dumps(report, indent=2))


def cmd_crawl(args) -> None:
    """Collect AutoGuide.com; curated additional sources use crawl-web (ADR-010).
    The NHTSA website is not crawled; ingestion/crawler.py is kept for history only."""
    from ingestion.autoguide_crawler import AutoGuideCrawler  # needs CRAWLER_CONTACT_EMAIL

    result = AutoGuideCrawler().crawl(max_pages=args.max_pages, dry_run=args.dry_run)
    if args.dry_run:
        for url in result["would_fetch"]:
            print(url)
        print(f"\n{len(result['would_fetch'])} of {result['total_candidates']} candidate URLs shown (dry run, nothing fetched)")
    else:
        print(json.dumps(result, indent=2))


def cmd_extract_autoguide(_args) -> None:
    from ingestion.autoguide_extractor import process_autoguide

    print(json.dumps(process_autoguide(), indent=2))


def cmd_crawl_web(args) -> None:
    from ingestion.web_sources import WebSourceCrawler, load_manifest

    if args.max_pages < 1:
        raise SystemExit("--max-pages must be positive")
    pages = load_manifest(args.manifest, args.source)
    if args.dry_run:
        print(json.dumps({"offline_preview": True, "robots_checked": False,
                          "pages": [p.model_dump() for p in pages[:args.max_pages]]}, indent=2))
        return
    print(json.dumps(WebSourceCrawler().crawl(pages, args.max_pages, args.refresh), indent=2))


def cmd_extract_web(args) -> None:
    from ingestion.web_extractor import process_web_sources

    result = process_web_sources(source=getattr(args, "source", "all"))
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit("Some web snapshots could not be extracted; review errors before indexing")


def cmd_collect_recalls(args) -> None:
    from ingestion.api_collector import NHTSAAPICollector

    NHTSAAPICollector().collect()


def cmd_extract(_args) -> None:
    from ingestion.extractor import process_documents

    process_documents()


def cmd_normalize_recalls(_args) -> None:
    from ingestion.api_normalizer import process_api_documents

    process_api_documents()


def cmd_load_safercar(args) -> None:
    from db.session import init_db, session_scope
    from structured.safercar import load_safercar

    init_db()
    with session_scope() as db:
        count = load_safercar(db, args.csv)
    print(f"Loaded {count} vehicles")


def cmd_chunk(args) -> None:
    from rag.chunking import chunk_corpus

    for strategy in _strategies(args.strategy):
        print(json.dumps(chunk_corpus(strategy), indent=2))


def cmd_index(args) -> None:
    from db.session import init_db, session_scope
    from rag.vector_store import build_index

    init_db()
    for strategy in _strategies(args.strategy):
        with session_scope() as db:
            print(json.dumps(build_index(db, strategy), indent=2))


def cmd_add_docs(args) -> None:
    """Team-supplied PDFs (ADR-009): copied to data/raw/reference, then extracted."""
    from pathlib import Path

    from ingestion.reference_docs import add_documents

    print(json.dumps(add_documents([Path(f) for f in args.files], source_url=args.url), indent=2))


def cmd_all(args) -> None:
    cmd_extract_autoguide(args)
    cmd_extract_web(args)
    if (get_settings().data_dir / "raw" / "reference").exists():
        args.files, args.url = [], None
        cmd_add_docs(args)
    cmd_normalize_recalls(args)
    cmd_load_safercar(args)
    cmd_chunk(args)
    cmd_index(args)


def _strategies(value: str) -> list[str]:
    return ["fixed", "structure"] if value == "both" else [value]


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    crawl = sub.add_parser("crawl")
    crawl.add_argument("--max-pages", type=int, default=180)
    crawl.add_argument("--dry-run", action="store_true")
    crawl.set_defaults(func=cmd_crawl)
    sub.add_parser("extract-autoguide").set_defaults(func=cmd_extract_autoguide)
    from ingestion.web_sources import MANIFEST

    web = sub.add_parser("crawl-web", help="Collect reviewed Wikipedia/FuelEconomy pages")
    web.add_argument("--source", choices=["all", "wikipedia", "fueleconomy"], default="all")
    web.add_argument("--manifest", type=Path, default=MANIFEST)
    web.add_argument("--max-pages", type=int, default=20)
    web.add_argument("--dry-run", action="store_true", help="Offline manifest preview; no robots or pages fetched")
    web.add_argument("--refresh", action="store_true")
    web.set_defaults(func=cmd_crawl_web)
    extract_web = sub.add_parser("extract-web")
    extract_web.add_argument("--source", choices=["all", "wikipedia", "fueleconomy"], default="all")
    extract_web.set_defaults(func=cmd_extract_web)
    add_docs = sub.add_parser("add-docs", help="add PDFs you were given (not scraped); re-extracts all of data/raw/reference")
    add_docs.add_argument("files", nargs="*", help="PDF files to copy in first (optional)")
    add_docs.add_argument("--url", default=None, help="citation URL; default: the publisher's page")
    add_docs.set_defaults(func=cmd_add_docs)
    sub.add_parser("collect-recalls").set_defaults(func=cmd_collect_recalls)
    sub.add_parser("extract").set_defaults(func=cmd_extract)
    sub.add_parser("normalize-recalls").set_defaults(func=cmd_normalize_recalls)
    load = sub.add_parser("load-safercar")
    load.add_argument("--csv", default=None)
    load.set_defaults(func=cmd_load_safercar)
    for name, func in (("chunk", cmd_chunk), ("index", cmd_index), ("all", cmd_all)):
        p = sub.add_parser(name)
        p.add_argument("--strategy", choices=["fixed", "structure", "both"], default=get_settings().chunk_strategy)
        if name == "all":
            p.add_argument("--csv", default=None)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
