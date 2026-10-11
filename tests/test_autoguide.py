"""AutoGuide.com crawler + extractor, fully offline (fake HTTP session)."""
import json

import pytest

from ingestion import autoguide_crawler as ag
from ingestion.autoguide_extractor import extract_article

ROBOTS = """User-agent: GPTBot
Disallow: /

User-agent: *
Disallow: /search/
Disallow: /*?*size=
Disallow: /news/private-
Sitemap: https://www.autoguide.com/sitemap-index.xml
"""
INDEX = """<?xml version="1.0"?><sitemapindex><sitemap><loc>https://www.autoguide.com/sitemap-1.xml</loc></sitemap></sitemapindex>"""
SITEMAP = """<?xml version="1.0"?><urlset>
<url><loc>https://www.autoguide.com/car-news/what-is-automatic-emergency-braking-12345.html</loc></url>
<url><loc>https://www.autoguide.com/car-news/new-sports-car-revealed-777.html</loc></url>
<url><loc>https://www.autoguide.com/news/private-airbag-recall-1.html</loc></url>
<url><loc>https://www.autoguide.com/car-news/blind-spot-monitoring-explained-2.html?utm=x</loc></url>
<url><loc>https://other.example.com/safety-1.html</loc></url>
</urlset>"""
ARTICLE = """<html><head><title>AEB | AutoGuide</title>
<script type="application/ld+json">{"@type":"NewsArticle","headline":"What Is Automatic Emergency Braking?","datePublished":"2024-05-01","dateModified":"2025-02-03"}</script>
</head><body><nav>Home Reviews News</nav>
<article><h1>What Is Automatic Emergency Braking?</h1>
<div class="ad-slot">Buy now!</div>
<p>Automatic emergency braking uses cameras and radar to watch the road ahead and can brake on its own.</p>
<h2>How It Works</h2>
<p>When a collision looks imminent and the driver has not reacted, the system applies the brakes automatically. Call 555-123-4567.</p>
<table><tr><th>System</th><th>Sensor</th></tr><tr><td>AEB</td><td>Radar</td></tr></table>
<div class="related-articles"><p>Ten best SUVs you should buy this year right now</p></div>
</article><footer>Copyright</footer></body></html>"""


class Resp:
    def __init__(self, status=200, text="", ctype="text/html", headers=None):
        self.status_code, self.text, self.content = status, text, text.encode()
        self.headers = {"Content-Type": ctype, **(headers or {})}


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, timeout=None, allow_redirects=False):
        self.calls.append(url)
        return self.routes.get(url, Resp(404))


@pytest.fixture
def crawler(tmp_path, monkeypatch, data_dir):
    monkeypatch.setenv("CRAWLER_CONTACT_EMAIL", "team@university.edu")
    from core import config
    config.get_settings.cache_clear()
    monkeypatch.setattr(ag.time, "sleep", lambda s: None)
    routes = {
        "https://www.autoguide.com/robots.txt": Resp(text=ROBOTS, ctype="text/plain"),
        "https://www.autoguide.com/sitemap-index.xml": Resp(text=INDEX, ctype="application/xml"),
        "https://www.autoguide.com/sitemap-1.xml": Resp(text=SITEMAP, ctype="application/xml"),
        "https://www.autoguide.com/car-news/what-is-automatic-emergency-braking-12345.html": Resp(text=ARTICLE),
    }
    session = FakeSession(routes)
    yield ag.AutoGuideCrawler(out_dir=tmp_path / "raw", session=session), session
    config.get_settings.cache_clear()


def test_requires_real_contact_email(monkeypatch, data_dir):
    monkeypatch.setenv("CRAWLER_CONTACT_EMAIL", "your_team_email@example.com")
    from core import config
    config.get_settings.cache_clear()
    with pytest.raises(ValueError):
        ag.user_agent()
    config.get_settings.cache_clear()


def test_normalize_url_rejects_other_hosts_queries_and_private_paths():
    assert ag.normalize_url("https://www.autoguide.com/a.html#x") == "https://www.autoguide.com/a.html"
    assert ag.normalize_url("https://www.autoguide.com/a.html?size=2") is None
    assert ag.normalize_url("http://www.autoguide.com/a.html") is None
    assert ag.normalize_url("https://evil.com/a.html") is None
    assert ag.normalize_url("https://www.autoguide.com/account/settings") is None


def test_dry_run_respects_robots_scope_and_relevance(crawler):
    c, session = crawler
    result = c.crawl(max_pages=10, dry_run=True)
    assert result["would_fetch"] == ["https://www.autoguide.com/car-news/what-is-automatic-emergency-braking-12345.html"]
    assert not any("emergency-braking" in u for u in session.calls if u.endswith(".html"))  # nothing fetched


def test_crawl_saves_raw_with_hash_and_dedupes(crawler):
    c, _ = crawler
    first = c.crawl(max_pages=10)
    assert first["saved"] == 1
    rows = [json.loads(l) for l in c.metadata_file.read_text().splitlines()]
    assert rows[0]["source_url"].endswith("12345.html") and len(rows[0]["content_hash"]) == 64
    again = c.crawl(max_pages=10)
    assert again["saved"] == 0


def test_stops_on_429(crawler):
    c, session = crawler
    session.routes["https://www.autoguide.com/sitemap-1.xml"] = Resp(429)
    with pytest.raises(ag.CrawlStopped):
        c.crawl(max_pages=10)


def test_robots_unreadable_means_no_crawl(crawler):
    c, session = crawler
    session.routes["https://www.autoguide.com/robots.txt"] = Resp(404)
    with pytest.raises(ag.CrawlStopped):
        c.crawl(max_pages=10)


def test_extractor_keeps_structure_and_drops_noise():
    doc = extract_article(ARTICLE, "https://www.autoguide.com/x.html")
    assert doc["title"] == "What Is Automatic Emergency Braking?"
    assert doc["effective_date"] == "2025-02-03"
    assert "## How It Works" in doc["text"]
    assert "System | Sensor" in doc["text"] and doc["table_count"] == 1
    assert "Buy now" not in doc["text"] and "Ten best SUVs" not in doc["text"] and "Home Reviews" not in doc["text"]
    assert "555-123-4567" not in doc["text"]
