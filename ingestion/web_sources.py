"""Bounded, robots-aware collection of approved vehicle pages (ADR-010).

No link discovery: every page and its vehicle scope must be reviewed in the
manifest. Dry runs are offline. Actual collection requires an honest contact.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit

import requests
from protego import Protego
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config import ROOT, get_settings
from core.schemas import SourceType

log = logging.getLogger(__name__)
MANIFEST = ROOT / "config" / "vehicle_web_sources.json"
HOSTS = {"wikipedia": "en.wikipedia.org", "fueleconomy": "www.fueleconomy.gov"}
SOURCE_TYPES = {"wikipedia": SourceType.WIKIPEDIA.value, "fueleconomy": SourceType.FUEL_ECONOMY.value}
MAX_BYTES = 10 * 1024 * 1024
STOP_CODES = {403, 429, 503}


class CollectionStopped(RuntimeError):
    """A site declined collection or its robots policy could not be checked."""


def normalize_url(url: str, source: str) -> str:
    """Strict allowlist; never turn a manifest into an arbitrary URL fetcher."""
    parsed = urlsplit(url.strip())
    host = HOSTS.get(source)
    if not host or parsed.scheme != "https" or parsed.netloc != host:
        raise ValueError(f"Only https://{host or 'approved hosts'} is allowed")
    path = unquote(parsed.path)
    if any(c in path for c in ("\\", "%", "\x00")) or any(p in (".", "..") for p in path.split("/")):
        raise ValueError("Unsafe URL path")
    if source == "wikipedia":
        slug = path.removeprefix("/wiki/")
        if not path.startswith("/wiki/") or not slug or ":" in slug or "/" in slug or parsed.query:
            raise ValueError("Only English Wikipedia article URLs are allowed")
        query = ""
    elif path == "/feg/Find.do":
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        query_map = dict(pairs)
        if len(pairs) != 2 or query_map.get("action") != "sbs" or not re.fullmatch(r"[0-9]+", query_map.get("id", "")):
            raise ValueError("FuelEconomy vehicle URLs require action=sbs and one numeric id")
        query = urlencode(sorted(pairs))
    else:
        if not re.fullmatch(r"/feg/[A-Za-z0-9_-]+\.shtml", path) or parsed.query:
            raise ValueError("Only FuelEconomy /feg/*.shtml guides or reviewed vehicle pages are allowed")
        query = ""
    return urlunsplit(("https", host, parsed.path, query, ""))


class PageSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    url: str
    # Vehicle pages need explicit provenance; a model article can omit year.
    vehicle: dict[str, str | int] | None = None
    title: str = ""
    purpose: str = ""

    @model_validator(mode="after")
    def validate_scope(self) -> "PageSeed":
        self.url = normalize_url(self.url, self.source)
        is_vehicle = self.source == "wikipedia" or urlsplit(self.url).path == "/feg/Find.do"
        if is_vehicle and (not self.vehicle or not self.vehicle.get("make") or not self.vehicle.get("model")):
            raise ValueError("Vehicle pages require reviewed make/model metadata")
        if self.source == "fueleconomy" and is_vehicle and not (self.vehicle or {}).get("year"):
            raise ValueError("FuelEconomy vehicle pages require the reviewed model year")
        if self.vehicle and "year" in self.vehicle:
            year = self.vehicle["year"]
            if isinstance(year, bool) or not str(year).isdigit() or not 1886 <= int(year) <= 2100:
                raise ValueError("Invalid vehicle year")
            self.vehicle["year"] = int(year)
        return self


def load_manifest(path: Path = MANIFEST, source: str = "all") -> list[PageSeed]:
    pages = [PageSeed.model_validate(row) for row in json.loads(Path(path).read_text(encoding="utf-8"))["pages"]]
    unique = {(page.source, page.url): page for page in pages}
    if len(unique) != len(pages):
        raise ValueError("Duplicate URLs in vehicle source manifest")
    return sorted((p for p in pages if source == "all" or p.source == source), key=lambda p: (p.source, p.url))


class WebSourceCrawler:
    def __init__(self, out_dir: Path | None = None, session: requests.Session | None = None,
                 delay: float = 1.0, contact_email: str | None = None):
        email = (contact_email or get_settings().crawler_contact_email or "").strip()
        if not re.fullmatch(r"[^\s<>@]+@[^\s<>@]+\.[A-Za-z]{2,}", email) or email.rsplit("@", 1)[-1].lower() in {"example.com", "example.org", "example.net"}:
            raise ValueError("Set CRAWLER_CONTACT_EMAIL to a real team email before collection")
        self.ua = f"AutoGuideAI-AcademicBot/1.0 (MSAI 633 RAG project; contact: {email})"
        self.out_dir = out_dir or get_settings().data_dir / "raw"
        self.session = session or requests.Session()
        self.session.trust_env = False  # never use environment proxies
        self.session.headers.update({"User-Agent": self.ua, "Accept": "text/html,text/plain;q=0.9"})
        self.delay = max(1.0, delay)
        self.last_request: float | None = None
        self.robots: dict[str, Protego] = {}
        self.stopped = False

    def _get(self, url: str) -> tuple[requests.Response, bytes]:
        if self.stopped:
            raise CollectionStopped("Collection has stopped")
        if self.last_request is not None:
            time.sleep(max(0.0, self.delay - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        response = self.session.get(url, timeout=30, allow_redirects=False, stream=True)
        try:
            if response.status_code in STOP_CODES:
                self.stopped = True
                raise CollectionStopped(f"HTTP {response.status_code}; collection stopped")
            parts, size = [], 0
            for part in response.iter_content(chunk_size=65536):
                size += len(part)
                if size > MAX_BYTES:
                    raise CollectionStopped("Response exceeds 10 MiB")
                parts.append(part)
            return response, b"".join(parts)
        finally:
            response.close()

    def _robots(self, source: str) -> Protego:
        if source not in self.robots:
            url = f"https://{HOSTS[source]}/robots.txt"
            response, body = self._get(url)
            text = body.decode("utf-8", errors="replace")
            if response.status_code != 200 or not re.search(r"(?im)^\s*user-agent\s*:", text) or "<html" in text.lower():
                self.stopped = True
                raise CollectionStopped(f"Cannot verify robots.txt for {HOSTS[source]}")
            policy = Protego.parse(text)
            self.delay = max(self.delay, float(policy.crawl_delay(self.ua) or 0))
            directory = self.out_dir / source
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "robots.txt").write_bytes(body)
            self.robots[source] = policy
        return self.robots[source]

    def _page(self, seed: PageSeed) -> tuple[str, bytes]:
        policy = self._robots(seed.source)
        url = seed.url
        for _ in range(5):
            if not policy.can_fetch(url, self.ua):
                raise CollectionStopped(f"robots.txt disallows {url}")
            response, body = self._get(url)
            if response.status_code in {301, 302, 303, 307, 308}:
                # Validate each hop before any request; never follow cross-host redirects.
                url = normalize_url(urljoin(url, response.headers.get("Location", "")), seed.source)
                continue
            if response.status_code != 200:
                raise CollectionStopped(f"HTTP {response.status_code} for {url}")
            if "text/html" not in response.headers.get("Content-Type", "").lower():
                raise CollectionStopped(f"Expected HTML at {url}")
            return url, body
        raise CollectionStopped("Too many redirects")

    def crawl(self, pages: list[PageSeed], max_pages: int = 20, refresh: bool = False) -> dict:
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        saved = skipped = 0
        for seed in pages[:max_pages]:
            directory = self.out_dir / seed.source
            metadata = directory / "metadata.jsonl"
            old = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()] if metadata.exists() else []
            if not refresh and any(row.get("requested_url", row["source_url"]) == seed.url for row in old):
                skipped += 1
                continue
            try:
                url, body = self._page(seed)
                # A redirect must not silently transfer a vehicle identity to another article/id.
                if url != seed.url and seed.vehicle:
                    raise CollectionStopped("Vehicle page redirected; review the destination in the manifest")
            except Exception:
                self.stopped = True
                raise
            digest = hashlib.sha256(body).hexdigest()
            doc_id = hashlib.sha256(seed.url.encode()).hexdigest()[:20]
            directory.mkdir(parents=True, exist_ok=True)
            raw_path = directory / f"{doc_id}-{digest[:16]}.html"
            raw_path.write_bytes(body)
            row = {"doc_id": doc_id, "source_url": url, "requested_url": seed.url,
                   "source": seed.source, "source_type": SOURCE_TYPES[seed.source],
                   "page_type": "html", "raw_file": raw_path.name, "content_hash": digest,
                   "crawl_timestamp": datetime.now(timezone.utc).isoformat(), "vehicle": seed.vehicle,
                   "title": seed.title, "user_agent": self.ua}
            with metadata.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            saved += 1
            log.info("Saved %s", url)
        return {"saved": saved, "skipped": skipped}
