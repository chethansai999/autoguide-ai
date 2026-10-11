import json
from pathlib import Path
from ingestion.cleaner import normalize_document
import pymupdf
from bs4 import BeautifulSoup


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def extract_html(file_path):
    """Extract text, headings, and tables from HTML."""

    html = Path(file_path).read_bytes()
    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Untitled"
    )

    # Remove unnecessary website elements.
    for element in soup.select(
        "script, style, nav, footer, header, aside, "
        "form, noscript"
    ):
        element.decompose()

    main_content = soup.find("main")

    if main_content is None:
        main_content = soup.body or soup

    headings = []

    for heading in main_content.find_all(
        ["h1", "h2", "h3", "h4"]
    ):
        headings.append({
            "level": heading.name,
            "text": heading.get_text(" ", strip=True)
        })

    tables = []

    for table in main_content.find_all("table"):

        rows = []

        for tr in table.find_all("tr"):

            cells = [
                cell.get_text(" ", strip=True)
                for cell in tr.find_all(["th", "td"])
            ]

            if cells:
                rows.append(cells)

        if rows:
            tables.append(rows)

    # Preserve headings and tables in the extracted text.
    text = main_content.get_text(
        "\n",
        strip=True
    )

    return {
        "title": title,
        "text": text,
        "headings": headings,
        "tables": tables,
        "table_count": len(tables)
    }


def extract_pdf(file_path):
    """Extract text from PDF pages."""

    document = pymupdf.open(file_path)

    pages = []

    full_text = []

    try:
        for page_number, page in enumerate(
            document,
            start=1
        ):

            page_text = page.get_text("text")

            pages.append({
                "page_number": page_number,
                "text": page_text
            })

            full_text.append(page_text)

    finally:
        document.close()

    return {
        "title": Path(file_path).stem,
        "text": "\n".join(full_text),
        "pages": pages,
        "headings": [],
        "tables": [],
        "table_count": 0
    }


def process_documents():

    metadata_path = RAW_DIR / "metadata.jsonl"

    if not metadata_path.exists():
        print("No raw metadata found.")
        return

    processed_count = 0

    with metadata_path.open(
        "r",
        encoding="utf-8"
    ) as metadata_file:

        for line in metadata_file:

            if not line.strip():
                continue

            metadata = json.loads(line)

            raw_file = Path(metadata["raw_file"])

            if not raw_file.exists():
                print(f"Missing file: {raw_file}")
                continue

            try:
                if metadata["page_type"] == "html":
                    extracted = extract_html(raw_file)

                elif metadata["page_type"] == "pdf":
                    extracted = extract_pdf(raw_file)

                else:
                    continue

                result = {
                    **metadata,
                    **extracted
                }

                result = normalize_document(result)

                output_file = (
                    PROCESSED_DIR /
                    f"{metadata['doc_id']}.json"
                )

                output_file.write_text(
                    json.dumps(
                        result,
                        indent=2,
                        ensure_ascii=False
                    ),
                    encoding="utf-8"
                )

                processed_count += 1

                print(
                    f"Processed: {metadata['doc_id']}"
                )

            except Exception as error:
                print(
                    f"Failed to process "
                    f"{metadata['doc_id']}: {error}"
                )

    print(
        f"\nTotal documents processed: "
        f"{processed_count}"
    )


if __name__ == "__main__":
    process_documents()