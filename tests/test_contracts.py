"""Lock the shared data contracts (core/schemas.py)."""
from core.schemas import Chunk, Document, VehicleSlots
from rag.chunking import make_chunk_id


def test_document_accepts_extractor_shape():
    doc = Document.model_validate({"doc_id": "abc", "source_url": "https://www.nhtsa.gov/x", "text": "t", "title": "T"})
    assert doc.doc_id == "abc" and doc.source_url.endswith("/x")


def test_document_accepts_api_normalizer_shape():
    doc = Document.model_validate({"id": "r1", "url": "https://api.nhtsa.gov/r", "text": "t", "vehicle": {"make": "HONDA"}})
    assert doc.doc_id == "r1" and doc.vehicle["make"] == "HONDA"


def test_chunk_record_has_retrieval_keys():
    chunk = Chunk(chunk_id="D:v1:fixed:001", doc_id="D", version="v1", strategy="fixed", chunk_index=1,
                  text="x", title="T", source_url="https://www.nhtsa.gov/d", token_count=1)
    record = chunk.to_record()
    assert {"id", "text", "title", "url"} <= set(record)


def test_chunk_id_format_matches_proposal():
    assert make_chunk_id("NHTSA001", "v1", "fixed", 1) == "NHTSA001:v1:fixed:001"


def test_slot_merge_newer_wins_and_model_change_drops_variant():
    old = VehicleSlots(make="FORD", model="F-150 SUPERCAB", model_year=2019, drive_train="4X4")
    assert old.merged_with(VehicleSlots(model_year=2020)).model_dump(exclude_none=True) == {
        "make": "FORD", "model": "F-150 SUPERCAB", "model_year": 2020, "drive_train": "4X4"}
    switched = old.merged_with(VehicleSlots(model="RANGER"))
    assert switched.model == "RANGER" and switched.drive_train is None and switched.model_year == 2019


def test_web_source_contracts_preserve_attribution():
    from core.schemas import Citation, Evidence, Intent, SourceType
    from rag.chunking import FixedSizeChunker

    credit = {"authors": "Wikipedia contributors", "license": "CC BY-SA 4.0"}
    doc = Document(doc_id="wiki", source_url="https://en.wikipedia.org/wiki/Toyota_Camry",
                   text="Toyota Camry history.", source_type=SourceType.WIKIPEDIA.value, attribution=credit)
    chunk = FixedSizeChunker().chunk(doc)[0]
    assert chunk.to_record()["attribution"] == credit
    evidence = Evidence(ref=1, evidence_id=chunk.chunk_id, kind="chunk", title="Camry",
                        url=doc.source_url, text=doc.text, attribution=chunk.attribution)
    citation = Citation(ref=1, title=evidence.title, url=evidence.url, attribution=evidence.attribution)
    assert citation.model_dump()["attribution"] == credit
    assert Intent("vehicle_info") == Intent.VEHICLE_INFO
    assert SourceType("fueleconomy_page") == SourceType.FUEL_ECONOMY
