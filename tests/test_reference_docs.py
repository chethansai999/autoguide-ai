"""Team-supplied reference PDFs (ADR-009), tested on a small generated PDF.

The real IIHS PDFs are copyrighted and not committed; this fixture copies
their layout: a cover with version and date, a contents page with dot
leaders, a running header/footer on every page, and figure captions.
"""

from __future__ import annotations

import json

import pytest

pymupdf = pytest.importorskip("pymupdf")

from core.schemas import Document, SourceType  # noqa: E402
from ingestion.reference_docs import add_documents, extract_reference_pdf  # noqa: E402

FOOTER = "© 2025 Insurance Institute for Highway Safety"
HEADER = "Widget Overlap Test Protocol (Version II)"


def make_pdf(path):
    pages = [
        ["Widget Overlap Crashworthiness Evaluation", "Crash Test Protocol", "Version II", "May 2025", "CONTENTS",
         "OVERVIEW ............................................. 2", "Test speed ......................................... 2",
         "RESULTS ............................................. 3", "Torso ............................................... 3",
         FOOTER],
        [FOOTER, HEADER, "May 2025 — 2", "OVERVIEW",
         "This protocol describes a widget overlap crash test conducted by the",
         "Institute at a fixed speed against a rigid barrier.",
         "Test speed", "The test is run at 64.4 km/h with 25 percent overlap.",
         "Figure 1", "Barrier layout seen from above"],
        [FOOTER, HEADER, "May 2025 — 3", "RESULTS",
         "Table 1. Body regions Head Torso Pelvis",
         "Torso", "Torso injury is measured with rib deflection sensors."],
        [FOOTER, HEADER, "May 2025 — 4", "Further notes on the barrier face and its mounting."],
    ]
    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        for i, line in enumerate(lines):
            page.insert_text((72, 72 + 18 * i), line, fontsize=10)
    pdf.set_metadata({"title": "Widget Overlap Crashworthiness Evaluation: Crash Test Protocol", "author": "IIHS"})
    pdf.save(path)
    pdf.close()


@pytest.fixture()
def widget_pdf(tmp_path):
    path = tmp_path / "1a2b3c4d-widget_overlap_test_protocol.pdf"
    make_pdf(path)
    return path


def test_extracts_metadata_headings_and_drops_layout_noise(widget_pdf):
    doc = extract_reference_pdf(widget_pdf)
    Document.model_validate(doc)
    assert doc["title"] == "IIHS Widget Overlap Crashworthiness Evaluation: Crash Test Protocol (Version II, May 2025)"
    assert doc["document_version"] == "II" and doc["published"] == "May 2025" and doc["effective_date"] == "2025-05"
    assert doc["source_type"] == SourceType.REFERENCE.value
    assert doc["source_url"].startswith("https://www.iihs.org/")
    assert doc["doc_id"] == "iihs-widget-overlap-crashworthiness-evaluation-crash-test-protocol-v2"
    text = doc["text"]
    assert "## Overview" in text and "### Test speed" in text and "## Results" in text
    assert "......" not in text and "May 2025 —" not in text and FOOTER not in text and HEADER not in text
    assert "conducted by the Institute at a fixed speed" in text  # wrapped line re-flowed
    assert "[Figure 1: Barrier layout seen from above]" in text


def test_table_cell_with_a_heading_name_is_not_a_heading(widget_pdf):
    text = extract_reference_pdf(widget_pdf)["text"]
    assert "Body regions Head Torso Pelvis" in text  # the table line stays text
    assert text.count("### Torso") == 1  # only the real section heading


def test_add_documents_copies_extracts_and_feeds_chunking(widget_pdf, tmp_path):
    from rag.chunking import chunk_corpus

    raw, out = tmp_path / "raw", tmp_path / "processed"
    result = add_documents([widget_pdf], raw_dir=raw, out_dir=out)
    assert result["processed"] == 1
    assert (raw / "widget_overlap_test_protocol.pdf").exists()  # upload-id prefix dropped
    saved = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["publisher"].startswith("Insurance Institute for Highway Safety")
    stats = chunk_corpus("fixed", processed_dirs=[out], out_dir=tmp_path / "chunks")
    assert stats["documents"] == 1 and stats["chunks"] >= 1


def test_non_pdf_is_rejected(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        add_documents([bad], raw_dir=tmp_path / "raw", out_dir=tmp_path / "out")


def test_revision_history_and_references_are_dropped():
    from ingestion.reference_docs import drop_sections

    text = ("## Document revision history\n\nAdded a camera.\n\n## Overview\n\nThe test runs at 64 km/h.\n\n"
            "### References (Appendix A)\n\nSmith 2020.\n\n## Appendix B\n\nParallax.")
    out = drop_sections(text)
    assert "Added a camera" not in out and "Smith 2020" not in out
    assert "## Overview" in out and "64 km/h" in out and "## Appendix B" in out and "Parallax" in out
