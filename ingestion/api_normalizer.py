import hashlib
import json
from pathlib import Path

from ingestion.cleaner import clean_text


RAW_API_DIR = Path("data/raw/api")
PROCESSED_API_DIR = Path("data/processed/api")

PROCESSED_API_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def normalize_recall(record, source_url, vehicle):
    """Convert an NHTSA recall record into a searchable document."""

    campaign_number = record.get(
        "NHTSACampaignNumber", ""
    )

    component = record.get("Component", "")
    summary = record.get("Summary", "")
    consequence = record.get("Consequence", "")
    remedy = record.get("Remedy", "")
    notes = record.get("Notes", "")

    make = record.get(
        "Make", vehicle.get("make", "")
    )

    model = record.get(
        "Model", vehicle.get("model", "")
    )

    year = record.get(
        "ModelYear", vehicle.get("year", "")
    )

    title = (
        f"{year} {make} {model} "
        f"- {component} Recall"
    )

    text = f"""
    Vehicle: {year} {make} {model}
    NHTSA Campaign Number: {campaign_number}
    Component: {component}

    Recall Summary:
    {summary}

    Safety Consequence:
    {consequence}

    Recommended Remedy:
    {remedy}

    Additional Notes:
    {notes}
    """

    text = clean_text(text)

    # Use the source and recall identifier for a stable ID.
    unique_key = (
        f"{source_url}|{campaign_number}|"
        f"{year}|{make}|{model}"
    )

    document_id = hashlib.sha256(
        unique_key.encode("utf-8")
    ).hexdigest()[:20]

    return {
        "id": document_id,
        "title": title,
        "text": text,
        "url": source_url,
        "source_type": "nhtsa_recalls_api",
        "campaign_number": campaign_number,
        "vehicle": {
            "make": make,
            "model": model,
            "year": year
        },
        "headings": [
            "Recall Summary",
            "Safety Consequence",
            "Recommended Remedy",
            "Additional Notes"
        ]
    }


def process_api_documents():

    files = sorted(
        RAW_API_DIR.glob("*.json")
    )

    if not files:
        print("No API JSON files found.")
        return

    processed_count = 0

    for file_path in files:

        with file_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        source_url = data.get("source_url", "")
        vehicle = data.get("vehicle", {})

        records = data.get("records", [])

        for record in records:

            document = normalize_recall(
                record,
                source_url,
                vehicle
            )

            output_file = (
                PROCESSED_API_DIR /
                f"{document['id']}.json"
            )

            output_file.write_text(
                json.dumps(
                    document,
                    indent=2,
                    ensure_ascii=False
                ),
                encoding="utf-8"
            )

            processed_count += 1

        print(
            f"Processed {len(records)} recalls "
            f"from {file_path.name}"
        )

    print("\nAPI normalization completed.")
    print(
        f"Total normalized documents: "
        f"{processed_count}"
    )


if __name__ == "__main__":
    process_api_documents()