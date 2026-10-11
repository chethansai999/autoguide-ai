from evaluation.metrics import bootstrap_ci, mcnemar_exact, mrr, ndcg_at_k, recall_at_k, refusal_rates


def test_retrieval_metrics():
    ranked = ["a", "b", "a", "c"]
    assert recall_at_k(ranked, ["b", "c"], 2) == 0.5
    assert recall_at_k(ranked, ["b", "c"], 3) == 1.0  # duplicates collapse
    assert mrr(ranked, ["c"]) == 1 / 3
    assert ndcg_at_k(["x", "a"], ["a"], 10) < 1.0 and ndcg_at_k(["a"], ["a"]) == 1.0
    assert recall_at_k(ranked, [], 5) is None


def test_stats():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01
    lo, hi = bootstrap_ci([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    assert 0 <= lo <= 0.7 <= hi <= 1


def test_refusal_rates():
    rows = [{"answerable": False, "refused": True}, {"answerable": False, "refused": False},
            {"answerable": True, "refused": True}, {"answerable": True, "refused": False}]
    assert refusal_rates(rows) == {"correct_refusal_rate": 0.5, "over_refusal_rate": 0.5}
