from rag.retrieval import (
    BM25Retriever,
    chroma_metadata,
    normalize_chunk,
    reciprocal_rank_fusion,
    tokenize
)

CHUNKS = [
    normalize_chunk({
        "id": "c1",
        "title": "Airbag Recall",
        "text": "The driver airbag inflator may rupture during deployment."
    }),
    normalize_chunk({
        "id": "c2",
        "title": "Brake Recall",
        "text": "Anti-lock brake system (ABS) module may fail."
    }),
    normalize_chunk({
        "id": "c3",
        "title": "Rearview Camera",
        "text": "The rearview camera image may not display."
    })
]


def test_tokenize_removes_stopwords_and_keeps_hyphens():
    assert tokenize("What is the F-150 ABS recall?") == [
        "f-150", "abs", "recall"
    ]


def test_bm25_ranks_keyword_match_first():
    results = BM25Retriever(CHUNKS).search("ABS brake failure")

    assert results[0][0] == "c2"


def test_bm25_returns_nothing_for_stopword_only_query():
    assert BM25Retriever(CHUNKS).search("what is the") == []


def test_rrf_rewards_chunks_found_by_both_retrievers():
    bm25 = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
    dense = [("d", 0.9), ("b", 0.8), ("c", 0.7)]

    fused = [chunk_id for chunk_id, _ in reciprocal_rank_fusion([bm25, dense])]

    # "b" is 2nd in both lists, so it beats "a" (1st in only one list).
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([[("a", 1.0)], [("a", 1.0)]], k=60))

    assert abs(fused["a"] - 2 / 61) < 1e-9


def test_normalize_chunk_accepts_doc_id_and_source_url():
    chunk = normalize_chunk({
        "doc_id": "NHTSA001",
        "text": "x",
        "source_url": "https://www.nhtsa.gov"
    })

    assert chunk["id"] == "NHTSA001"
    assert chunk["url"] == "https://www.nhtsa.gov"


def test_chroma_metadata_flattens_nested_values():
    metadata = chroma_metadata({
        "id": "x",
        "text": "y",
        "title": "t",
        "vehicle": {"make": "Honda", "year": 2020},
        "headings": ["a", "b"]
    })

    assert metadata == {
        "title": "t",
        "vehicle_make": "Honda",
        "vehicle_year": 2020
    }
