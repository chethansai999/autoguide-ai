import pytest

from core.schemas import Document
from rag.chunking import FixedSizeChunker, StructureAwareChunker

DOC = Document(doc_id="D1", source_url="https://www.nhtsa.gov/d1", title="T",
               text="## Air Bags\n" + "word " * 900 + "\n## Seat Belts\n" + "belt " * 50)


def test_fixed_chunks_are_deterministic_and_overlap():
    chunker = FixedSizeChunker(size=200, overlap=50, version="v1")
    a, b = chunker.chunk(DOC), chunker.chunk(DOC)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert a[0].chunk_id == "D1:v1:fixed:001" and len(a) > 3
    assert all(c.token_count <= 260 for c in a)


@pytest.mark.xfail(raises=NotImplementedError, reason="WP-03: remove this marker when implemented")
def test_structure_chunks_carry_heading_path():
    chunks = StructureAwareChunker(version="v1").chunk(DOC)
    assert chunks[0].heading_path[0] == "Air Bags"
    assert chunks[-1].heading_path[0] == "Seat Belts"
    assert all(c.chunk_id.startswith("D1:v1:structure:") for c in chunks)


@pytest.mark.xfail(raises=NotImplementedError, reason="WP-03: remove this marker when implemented")
def test_structure_never_merges_sections():
    chunks = StructureAwareChunker(version="v1").chunk(DOC)
    assert not any("belt" in c.text and "word" in c.text for c in chunks)
