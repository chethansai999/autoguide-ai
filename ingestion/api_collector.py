import hashlib
import json
import time

from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"

OUTPUT_DIR = Path("data/raw/api")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY = 1.0


class NHTSAAPICollector:

    def __init__(self):
        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": "AutoGuideAI-AcademicResearch/1.0",
            "Accept": "application/json"
        })

        self.last_request_time = 0.0

    def wait_for_rate_limit(self):
        elapsed = time.monotonic() - self.last_request_time

        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        self.last_request_time = time.monotonic()

    def fetch_recalls(self, make, model, year):

        params = {
            "make": make,
            "model": model,
            "modelYear": year
        }

        self.wait_for_rate_limit()

        try:
            response = self.session.get(
                BASE_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            print(
                f"{year} {make} {model}: "
                f"{data.get('Count', 0)} recalls"
            )

            return {
                "data": data,
                "source_url": response.url,
                "retrieved_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

        except requests.RequestException as error:
            print(f"API request failed: {error}")
            return None

    def save_recalls(self, result, make, model, year):

        if result is None:
            return

        data = result["data"]

        raw_json = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True
        )

        content_hash = hashlib.sha256(
            raw_json.encode("utf-8")
        ).hexdigest()

        filename = (
            f"{year}_{make}_{model}"
            .lower()
            .replace(" ", "_")
        )

        output_file = OUTPUT_DIR / f"{filename}.json"

        document = {
            "source": "NHTSA Recalls API",
            "source_url": result["source_url"],
            "retrieved_at": result["retrieved_at"],
            "content_hash": content_hash,
            "vehicle": {
                "make": make,
                "model": model,
                "year": year
            },
            "record_count": data.get("Count", 0),
            "records": data.get("results", [])
        }

        output_file.write_text(
            json.dumps(
                document,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(f"Saved: {output_file}")

    def collect(self):

        vehicles = [
            ("Honda", "Accord", 2020),
            ("Toyota", "Camry", 2020),
            ("Ford", "F-150", 2020),
            ("Honda", "Civic", 2021),
            ("Toyota", "Corolla", 2021)
        ]

        for make, model, year in vehicles:

            result = self.fetch_recalls(
                make,
                model,
                year
            )

            self.save_recalls(
                result,
                make,
                model,
                year
            )

        print("\nNHTSA API collection completed.")


if __name__ == "__main__":

    collector = NHTSAAPICollector()

    collector.collect()