from collections import Counter

from finetune.synth_data import TAGS, generate


def test_synthetic_data_is_aligned_and_split_without_leakage(vocab):
    rows = generate(vocab, n=600, seed=1)
    assert len(rows) >= 300
    for row in rows:
        assert len(row["tokens"]) == len(row["tags"])
        assert set(row["tags"]) <= set(TAGS)
    assert set(Counter(r["intent"] for r in rows)) >= {"vehicle_lookup", "recall_lookup", "concept_question", "out_of_scope"}
    train = {(r["slots"]["make"], r["slots"]["model"]) for r in rows if r["split"] == "train" and r["slots"]["model"]}
    test = {(r["slots"]["make"], r["slots"]["model"]) for r in rows if r["split"] == "test" and r["slots"]["model"]}
    assert not (train & test)
