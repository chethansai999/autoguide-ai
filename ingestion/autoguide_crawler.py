"""Polite crawler for www.autoguide.com (the project's only scraped website, ADR-008).

Rules it enforces (do not weaken them):
  * robots.txt is fetched first and obeyed for OUR user agent, with wildcard
    support (protego). If robots.txt can't be read, nothing is crawled.
  * An honest User-Agent that names the project and a contact email.
  * At most 1 request per second (or the site's Crawl-delay, if larger).
  * Only www.autoguide.com over https; no query strings; no login/account,
    search, comment or shop pages; stop on 403/429/503.
  * Raw bytes, URL, timestamp and SHA-256 are saved before any extraction.

Discovery: the sitemaps listed in robots.txt, filtered to safety-related
articles by keywords in the URL (config/autoguide_crawl.json).

    python ingest.py crawl-autoguide --max-pages 180
    python ingest.py crawl-autoguide --dry-run      # list what it WOULD fetch
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

from core.config import ROOT, get_settings

log = logging.getLogger(__name__)

BASE = "https://www.autoguide.com"
HOST = "www.autoguide.com"
CONFIG = ROOT / "config" / "autoguide_crawl.json"
TIMEOUT = 30
MAX_BYTES = 10 * 1024 * 1024

# Never fetched, whatever robots.txt says.
EXCLUDED_PATH_PARTS = (
    "/account", "/login", "/signin", "/register", "/search", "/comment", "/shop",
    "/ajax", "/api/", "/notifications", "/logging", "/delete", "/members", "/forum",
)


class CrawlStopped(RuntimeError):
    """The site asked us to stop (403/429/503) or robots.txt is unreadable."""


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def user_agent() -> str:
    email = (get_settings().crawler_contact_email or "").strip()
    if not email or "@" not in email or email.endswith("@example.com"):
        raise ValueError("Set CRAWLER_CONTACT_EMAIL in .env to a real team email before crawling.")
    return f"AutoGuideAI-Academic/1.0 (MSAI 633 student RAG project; contact: {email})"


def normalize_url(url: str) -> str | None:
    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != HOST or parsed.query:
        return None
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        return None
    path = parsed.path.lower()
    if any(part in path for part in EXCLUDED_PATH_PARTS):
        return None
    return parsed._replace(query="", fragment="").geturl()


def is_relevant(url: str, config: dict) -> bool:
    slug = urlparse(url).path.lower()
    if any(re.search(p, slug) for p in config.get("exclude_patterns", [])):
        return False
    return any(k in slug for k in config.get("include_keywords", []))


class AutoGuideCrawler:
    def __init__(self, out_dir: Path | None = None, delay: float | None = None, session: requests.Session | None = None):
        settings = get_settings()
        self.out_dir = out_dir or settings.data_dir / "raw" / "autoguide"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.out_dir / "metadata.jsonl"
        self.config = load_config()
        self.ua = user_agent()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.ua, "Accept": "text/html,application/xml;q=0.9"})
        self.delay = max(1.0, delay or self.config.get("delay_seconds", 1.0))
        self._last = 0.0
        self.robots = None
        self.seen_urls, self.seen_hashes = self._load_seen()

    # ------------------------------------------------------------ plumbing
    def _load_seen(self) -> tuple[set[str], set[str]]:
        urls, hashes = set(), set()
        if self.metadata_file.exists():
            for line in self.metadata_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    urls.add(row["source_url"])
                    hashes.add(row["content_hash"])
        return urls, hashes

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last = time.monotonic()

    def _get(self, url: str) -> requests.Response:
        self._wait()
        response = self.session.get(url, timeout=TIMEOUT, allow_redirects=False)
        if response.status_code in (403, 429, 503):
            raise CrawlStopped(f"{url} returned {response.status_code}; stopping politely.")
        return response

    def load_robots(self) -> None:
        from protego import Protego

        response = self._get(f"{BASE}/robots.txt")
        if response.status_code != 200:
            raise CrawlStopped(f"robots.txt returned {response.status_code}; not crawling.")
        self.robots = Protego.parse(response.text)
        crawl_delay = self.robots.crawl_delay(self.ua)
        if crawl_delay:
            self.delay = max(self.delay, float(crawl_delay))
        log.info("robots.txt loaded; delay %.1fs", self.delay)

    def allowed(self, url: str) -> bool:
        return self.robots is not None and self.robots.can_fetch(url, self.ua)

    # ------------------------------------------------------------ discovery
    def sitemap_urls(self) -> list[str]:
        sitemaps = list(self.robots.sitemaps) if self.robots else []
        sitemaps += [s for s in self.config.get("extra_sitemaps", []) if s not in sitemaps]
        found: list[str] = []
        queue = list(dict.fromkeys(sitemaps))
        visited = set()
        while queue and len(visited) < self.config.get("max_sitemaps", 200):
            sitemap = queue.pop(0)
            if sitemap in visited or urlparse(sitemap).hostname != HOST or not self.allowed(sitemap):
                continue
            visited.add(sitemap)
            response = self._get(sitemap)
            if response.status_code != 200:
                continue
            body = response.content
            if sitemap.endswith(".gz") or body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            soup = BeautifulSoup(body, "xml")
            if soup.find("sitemapindex"):
                queue += [loc.get_text(strip=True) for loc in soup.select("sitemap > loc")]
            else:
                found += [loc.get_text(strip=True) for loc in soup.select("url > loc")]
        log.info("Read %d sitemaps, %d URLs listed", len(visited), len(found))
        return found

    def candidates(self) -> list[str]:
        urls = []
        for raw in self.sitemap_urls() + self.config.get("seed_urls", []):
            url = normalize_url(raw)
            if url and is_relevant(url, self.config) and self.allowed(url) and url not in urls:
                urls.append(url)
        return urls

    # ------------------------------------------------------------ fetching
    def fetch(self, url: str) -> dict | None:
        if not self.allowed(url):
            log.info("robots.txt disallows %s", url)
            return None
        response = self._get(url)
        if response.status_code in (301, 302, 303, 307, 308):
            target = normalize_url(requests.compat.urljoin(url, response.headers.get("Location", "")))
            if not target or not self.allowed(target):
                return None
            url, response = target, self._get(target)
        if response.status_code != 200 or "text/html" not in response.headers.get("Content-Type", ""):
            return None
        if len(response.content) > MAX_BYTES:
            return None
        return {"url": url, "content": response.content, "content_type": response.headers.get("Content-Type", "")}

    def save(self, page: dict) -> bool:
        digest = hashlib.sha256(page["content"]).hexdigest()
        if digest in self.seen_hashes:
            return False
        doc_id = "AG-" + hashlib.sha256(page["url"].encode()).hexdigest()[:14]
        raw_path = self.out_dir / f"{doc_id}.html"
        raw_path.write_bytes(page["content"])
        row = {
            "doc_id": doc_id,
            "source_url": page["url"],
            "content_hash": digest,
            "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
            "page_type": "html",
            "content_type": page["content_type"],
            "raw_file": str(raw_path),
            "size_bytes": len(page["content"]),
            "source_site": "autoguide.com",
        }
        with self.metadata_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        self.seen_urls.add(page["url"])
        self.seen_hashes.add(digest)
        return True

    def crawl(self, max_pages: int = 180, dry_run: bool = False) -> dict:
        self.load_robots()
        urls = [u for u in self.candidates() if u not in self.seen_urls]
        log.info("%d relevant, unseen URLs", len(urls))
        if dry_run:
            return {"would_fetch": urls[:max_pages], "total_candidates": len(urls)}
        saved, skipped = 0, 0
        for url in urls:
            if saved >= max_pages:
                break
            page = self.fetch(url)
            if page and self.save(page):
                saved += 1
                log.info("[%d/%d] %s", saved, max_pages, url)
            else:
                skipped += 1
        return {"saved": saved, "skipped": skipped, "candidates": len(urls), "out_dir": str(self.out_dir)}
