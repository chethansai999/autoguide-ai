import hashlib
import json
import os
import time

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

BASE_URL = "https://www.nhtsa.gov"

CONTACT_EMAIL = os.getenv("CRAWLER_CONTACT_EMAIL", "").strip()

if (
    not CONTACT_EMAIL
    or "@" not in CONTACT_EMAIL
    or CONTACT_EMAIL.endswith("@example.com")
):
    raise ValueError(
        "Please set a valid CRAWLER_CONTACT_EMAIL in your .env file."
    )

USER_AGENT = (
    f"AutoGuideAI-Research/1.0 "
    f"(Academic RAG Project; contact: {CONTACT_EMAIL})"
)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = RAW_DIR / "metadata.jsonl"

REQUEST_DELAY = 1.0

MAX_FILE_BYTES = 20 * 1024 * 1024

REQUEST_TIMEOUT = 30

MAX_REDIRECTS = 5


# --------------------------------------------------
# 2. NHTSA Crawler
# --------------------------------------------------

class NHTSACrawler:

    def __init__(self):

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": USER_AGENT
        })

        self.robot_parser = RobotFileParser()

        self.robot_parser.set_url(
            f"{BASE_URL}/robots.txt"
        )

        self.request_delay = REQUEST_DELAY

        self.last_request_time = 0.0

        self.visited_urls = set()

        self.content_hashes = set()

        self._load_existing_metadata()

        self._load_robots()


    # --------------------------------------------------
    # 3. Rate limiting
    # --------------------------------------------------

    def _wait_for_rate_limit(self):

        elapsed = time.monotonic() - self.last_request_time

        if elapsed < self.request_delay:

            time.sleep(
                self.request_delay - elapsed
            )

        self.last_request_time = time.monotonic()


    # --------------------------------------------------
    # 4. Load and check robots.txt
    # --------------------------------------------------

    def _load_robots(self):

        robots_url = f"{BASE_URL}/robots.txt"

        print("Checking robots.txt...")

        self._wait_for_rate_limit()

        try:

            response = self.session.get(
                robots_url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False
            )

            response.raise_for_status()

            self.robot_parser.parse(
                response.text.splitlines()
            )

            crawl_delay = self.robot_parser.crawl_delay(
                USER_AGENT
            )

            if crawl_delay is None:

                crawl_delay = self.robot_parser.crawl_delay(
                    "*"
                )

            if crawl_delay is not None:

                self.request_delay = max(
                    REQUEST_DELAY,
                    float(crawl_delay)
                )

            print("robots.txt loaded successfully.")

            print(
                f"Request delay: {self.request_delay} seconds"
            )

        except requests.RequestException as error:

            raise RuntimeError(
                f"Unable to verify robots.txt: {error}. "
                "Crawling stopped."
            )


    # --------------------------------------------------
    # 5. Load previously downloaded metadata
    # --------------------------------------------------

    def _load_existing_metadata(self):

        if not METADATA_FILE.exists():
            return

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if not line.strip():
                    continue

                try:

                    record = json.loads(line)

                    self.visited_urls.add(
                        record["source_url"]
                    )

                    self.content_hashes.add(
                        record["content_hash"]
                    )

                except (json.JSONDecodeError, KeyError):

                    print(
                        "Warning: Skipping invalid metadata record."
                    )


    # --------------------------------------------------
    # 6. Normalize URLs
    # --------------------------------------------------

    def normalize_url(self, url):

        url, _ = urldefrag(url)

        parsed = urlparse(url)

        if parsed.scheme != "https":
            return None

        if parsed.hostname != "www.nhtsa.gov":
            return None

        if parsed.username or parsed.password:
            return None

        if parsed.port not in (None, 443):
            return None

        # Skip URLs with query parameters.
        if parsed.query:
            return None

        return parsed._replace(
            fragment="",
            query=""
        ).geturl()


    # --------------------------------------------------
    # 7. Check whether URL is allowed
    # --------------------------------------------------

    def is_allowed(self, url):

        normalized_url = self.normalize_url(url)

        if normalized_url is None:
            return False

        parsed = urlparse(normalized_url)

        path = parsed.path.lower()

        # Exclude pages that may contain personal data
        # or require authentication.
        excluded_patterns = [
            "/login",
            "/signin",
            "/sign-in",
            "/account",
            "/directory",
            "/staff",
            "/people",
            "/employee",
            "/personnel",
            "/contact-directory"
        ]

        if any(
            pattern in path
            for pattern in excluded_patterns
        ):
            return False

        return self.robot_parser.can_fetch(
            USER_AGENT,
            normalized_url
        )


    # --------------------------------------------------
    # 8. Download HTML or PDF documents
    # --------------------------------------------------

    def fetch_page(self, url):

        normalized_url = self.normalize_url(url)

        if normalized_url is None:
            return None

        if not self.is_allowed(normalized_url):

            print(
                f"Skipped disallowed URL: {normalized_url}"
            )

            return None

        current_url = normalized_url

        for redirect_count in range(MAX_REDIRECTS + 1):

            if not self.is_allowed(current_url):

                print(
                    f"Redirect target not allowed: {current_url}"
                )

                return None

            self._wait_for_rate_limit()

            try:

                response = self.session.get(
                    current_url,
                    timeout=REQUEST_TIMEOUT,
                    stream=True,
                    allow_redirects=False
                )

                # Handle redirects manually.
                if response.status_code in (
                    301, 302, 303, 307, 308
                ):

                    location = response.headers.get(
                        "Location"
                    )

                    response.close()

                    if not location:
                        return None

                    redirected_url = urljoin(
                        current_url,
                        location
                    )

                    redirected_url = self.normalize_url(
                        redirected_url
                    )

                    if redirected_url is None:
                        return None

                    current_url = redirected_url

                    continue

                if response.status_code in (429, 503):

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    print(
                        f"Server requested reduced traffic: "
                        f"{response.status_code}"
                    )

                    if retry_after:

                        print(
                            f"Retry-After: {retry_after}"
                        )

                    response.close()

                    # Stop this crawl rather than repeatedly
                    # requesting a busy server.
                    raise RuntimeError(
                        "NHTSA returned a rate-limit or "
                        "service-unavailable response. "
                        "Stop crawling and retry later."
                    )

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if "text/html" in content_type:

                    page_type = "html"

                elif "application/pdf" in content_type:

                    page_type = "pdf"

                else:

                    response.close()
                    return None

                content_length = response.headers.get(
                    "Content-Length"
                )

                if content_length:

                    if int(content_length) > MAX_FILE_BYTES:

                        print(
                            f"File too large: {current_url}"
                        )

                        response.close()
                        return None

                content_chunks = []

                total_bytes = 0

                for chunk in response.iter_content(
                    chunk_size=65536
                ):

                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if total_bytes > MAX_FILE_BYTES:

                        print(
                            f"Download exceeded size limit: "
                            f"{current_url}"
                        )

                        response.close()
                        return None

                    content_chunks.append(chunk)

                response.close()

                content = b"".join(content_chunks)

                if not content:
                    return None

                # Verify PDF signature.
                if page_type == "pdf":

                    if not content.startswith(b"%PDF-"):

                        print(
                            f"Invalid PDF skipped: {current_url}"
                        )

                        return None

                return {
                    "content": content,
                    "content_type": content_type,
                    "page_type": page_type,
                    "url": current_url
                }

            except requests.RequestException as error:

                print(
                    f"Request failed: {current_url}: {error}"
                )

                return None

        print(
            f"Maximum redirects exceeded: {url}"
        )

        return None


    # --------------------------------------------------
    # 9. Save raw documents and metadata
    # --------------------------------------------------

    def save_document(self, page):

        content = page["content"]

        content_hash = hashlib.sha256(
            content
        ).hexdigest()

        if content_hash in self.content_hashes:

            print(
                f"Duplicate content skipped: {page['url']}"
            )

            return False

        source_url = page["url"]

        document_id = hashlib.sha256(
            source_url.encode("utf-8")
        ).hexdigest()[:16]

        extension = (
            ".pdf"
            if page["page_type"] == "pdf"
            else ".html"
        )

        filename = f"{document_id}{extension}"

        filepath = RAW_DIR / filename

        # Save original bytes before extracting text.
        filepath.write_bytes(content)

        metadata = {
            "doc_id": document_id,
            "source_url": source_url,
            "content_hash": content_hash,
            "crawl_timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "page_type": page["page_type"],
            "content_type": page["content_type"],
            "raw_file": str(filepath),
            "size_bytes": len(content)
        }

        with open(
            METADATA_FILE,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(metadata) + "\n"
            )

        self.visited_urls.add(source_url)

        self.content_hashes.add(content_hash)

        print(
            f"Saved {page['page_type'].upper()}: "
            f"{source_url}"
        )

        return True


    # --------------------------------------------------
    # 10. Extract links from HTML
    # --------------------------------------------------

    def extract_links(self, page):

        if page["page_type"] != "html":
            return []

        soup = BeautifulSoup(
            page["content"],
            "html.parser"
        )

        links = []

        for anchor in soup.find_all(
            "a",
            href=True
        ):

            link = urljoin(
                page["url"],
                anchor["href"]
            )

            normalized_link = self.normalize_url(
                link
            )

            if normalized_link is None:
                continue

            if self.is_allowed(normalized_link):

                links.append(normalized_link)

        # Remove duplicate links while preserving order.
        return list(dict.fromkeys(links))


    # --------------------------------------------------
    # 11. Crawl website
    # --------------------------------------------------

    def crawl(self, start_url, max_pages=10):

        normalized_start = self.normalize_url(
            start_url
        )

        if normalized_start is None:

            raise ValueError(
                "Invalid starting URL."
            )

        queue = deque([normalized_start])

        queued_urls = {normalized_start}

        attempted_urls = set()

        saved_count = 0

        print("\nStarting NHTSA crawler...")

        while queue and saved_count < max_pages:

            url = queue.popleft()

            if url in attempted_urls:
                continue

            attempted_urls.add(url)

            if url in self.visited_urls:
                continue

            page = self.fetch_page(url)

            if page is None:
                continue

            # Discover links from the downloaded HTML.
            links = self.extract_links(page)

            if self.save_document(page):

                saved_count += 1

            for link in links:

                if (
                    link not in self.visited_urls
                    and link not in attempted_urls
                    and link not in queued_urls
                ):

                    queue.append(link)

                    queued_urls.add(link)

            print(
                f"Progress: {saved_count}/{max_pages}"
            )

        print("\n--------------------------------")

        print("NHTSA Crawling Completed")

        print(
            f"New documents saved: {saved_count}"
        )

        print(
            f"Unique content hashes: "
            f"{len(self.content_hashes)}"
        )

        print("--------------------------------")


# --------------------------------------------------
# 12. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    crawler = NHTSACrawler()

    crawler.crawl(
        start_url="https://www.nhtsa.gov/",
        max_pages=10
    )