import re


def clean_text(text):
    """Clean extracted document text."""

    if not text:
        return ""

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters.
    text = text.replace("\x00", "")

    # Normalize spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_document(document):
    """Normalize an extracted document."""

    document["text"] = clean_text(
        document.get("text", "")
    )

    if "pages" in document:
        for page in document["pages"]:
            page["text"] = clean_text(
                page.get("text", "")
            )

    return document