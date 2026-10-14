import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from dedup_bench.evaluation.metrics import (
    adjusted_rand_index,
    b_cubed_scores,
    blocking_scores,
    cluster_scores,
    count_pairs,
    pairwise_scores,
)


def series(labels: list[str]) -> pd.Series:
    return pd.Series(labels, index=[f"r{i}" for i in range(len(labels))])


def test_count_pairs() -> None:
    assert count_pairs(pd.Series([1, 2, 3, 4])) == 0 + 1 + 3 + 6


def test_pairwise_scores_hand_computed() -> None:
    # True entities {r0, r1, r2}, {r3, r4}, {r5}: 3 + 1 = 4 true pairs.
    true = series(["a", "a", "a", "b", "b", "c"])
    # Predicted {r0, r1}, {r2, r3, r4}, {r5}: 1 + 3 = 4 predicted pairs, of
    # which (r0, r1) and (r3, r4) are correct.
    predicted = series(["x", "x", "y", "y", "y", "z"])
    scores = pairwise_scores(predicted, true)
    assert scores["precision"] == pytest.approx(2 / 4)
    assert scores["recall"] == pytest.approx(2 / 4)
    assert scores["f1"] == pytest.approx(0.5)


def test_pairwise_scores_ignore_label_names_and_order() -> None:
    true = series(["a", "a", "b"])
    predicted = series(["9", "9", "1"]).iloc[::-1]
    assert pairwise_scores(predicted, true) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_pairwise_scores_all_singletons() -> None:
    true = series(["a", "a", "b"])
    predicted = series(["1", "2", "3"])
    assert pairwise_scores(predicted, true) == {"precision": 1.0, "recall": 0.0, "f1": 0.0}


def test_pairwise_scores_one_big_cluster() -> None:
    true = series(["a", "a", "b", "c"])
    predicted = series(["x", "x", "x", "x"])
    scores = pairwise_scores(predicted, true)
    assert scores["precision"] == pytest.approx(1 / 6)
    assert scores["recall"] == 1.0


def test_pairwise_scores_reject_mismatched_records() -> None:
    with pytest.raises(ValueError, match="same"):
        pairwise_scores(series(["a", "b"]), series(["a", "b", "c"]))


def test_b_cubed_scores_hand_computed() -> None:
    # Same clustering as the pairwise test. Per record (precision, recall):
    # r0, r1: (2/2, 2/3); r2: (1/3, 1/3); r3, r4: (2/3, 2/2); r5: (1, 1).
    # Both averages are 7/9.
    true = series(["a", "a", "a", "b", "b", "c"])
    predicted = series(["x", "x", "y", "y", "y", "z"])
    scores = b_cubed_scores(predicted, true)
    assert scores["precision"] == pytest.approx(7 / 9)
    assert scores["recall"] == pytest.approx(7 / 9)
    assert scores["f1"] == pytest.approx(7 / 9)


def test_b_cubed_scores_all_singletons() -> None:
    # Precision 1 for every record; recall 1/2, 1/2 and 1.
    scores = b_cubed_scores(series(["1", "2", "3"]), series(["a", "a", "b"]))
    assert scores["precision"] == 1.0
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["f1"] == pytest.approx(0.8)


def test_b_cubed_scores_one_big_cluster() -> None:
    # Precision 2/4, 2/4, 1/4, 1/4; recall 1 for every record.
    scores = b_cubed_scores(series(["x"] * 4), series(["a", "a", "b", "c"]))
    assert scores["precision"] == pytest.approx(0.375)
    assert scores["recall"] == 1.0


def test_b_cubed_scores_ignore_label_names_and_order() -> None:
    true = series(["a", "a", "b"])
    predicted = series(["9", "9", "1"]).iloc[::-1]
    assert b_cubed_scores(predicted, true) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_cluster_scores_hand_computed() -> None:
    # Same clustering as the pairwise test: true {r0, r1, r2}, {r3, r4}, {r5};
    # predicted {r0, r1}, {r2, r3, r4}, {r5}. Only {r5} equals a true entity:
    # 1 of 3 predicted clusters, 1 of 3 entities.
    true = series(["a", "a", "a", "b", "b", "c"])
    predicted = series(["x", "x", "y", "y", "y", "z"])
    scores = cluster_scores(predicted, true)
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["recall"] == pytest.approx(1 / 3)
    assert scores["f1"] == pytest.approx(1 / 3)


def test_cluster_scores_all_singletons() -> None:
    # Predicted {r0}, {r1}, {r2}; true {r0, r1}, {r2}. Only {r2} is correct:
    # precision 1/3, recall 1/2, F1 2 (1/3)(1/2) / (5/6) = 0.4.
    scores = cluster_scores(series(["1", "2", "3"]), series(["a", "a", "b"]))
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["recall"] == pytest.approx(1 / 2)
    assert scores["f1"] == pytest.approx(0.4)


def test_cluster_scores_one_big_cluster() -> None:
    # One predicted cluster of four records; no true entity has all four.
    scores = cluster_scores(series(["x"] * 4), series(["a", "a", "b", "c"]))
    assert scores == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_cluster_scores_ignore_label_names_and_order() -> None:
    true = series(["a", "a", "b"])
    predicted = series(["9", "9", "1"]).iloc[::-1]
    assert cluster_scores(predicted, true) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_cluster_scores_reject_mismatched_records() -> None:
    with pytest.raises(ValueError, match="same"):
        cluster_scores(series(["a", "b"]), series(["a", "b", "c"]))


def test_adjusted_rand_index_hand_computed() -> None:
    # Same clustering as the pairwise test, 6 records so 15 pairs: 2 pairs
    # together in both, 4 in the predicted, 4 in the true clustering.
    # expected = 4 * 4 / 15 = 16/15, max = (4 + 4) / 2 = 4,
    # ARI = (2 - 16/15) / (4 - 16/15) = (14/15) / (44/15) = 7/22.
    true = series(["a", "a", "a", "b", "b", "c"])
    predicted = series(["x", "x", "y", "y", "y", "z"])
    assert adjusted_rand_index(predicted, true) == pytest.approx(7 / 22)


def test_adjusted_rand_index_matches_scikit_learn() -> None:
    true = series(list("aaabbbbccdeeeff"))
    predicted = series(list("xxyyyyzzzzwwvvv"))
    expected = adjusted_rand_score(true.to_numpy(), predicted.to_numpy())
    assert adjusted_rand_index(predicted, true) == pytest.approx(expected)


def test_adjusted_rand_index_identical_and_undefined() -> None:
    true = series(["a", "a", "b"])
    assert adjusted_rand_index(series(["9", "9", "1"]).iloc[::-1], true) == 1.0
    # All singletons in both: no pairs anywhere, so the index is undefined.
    assert adjusted_rand_index(series(["1", "2", "3"]), series(["a", "b", "c"])) == 1.0


def test_blocking_scores_hand_computed() -> None:
    # Entities {r0, r1, r2}, {r3, r4}, {r5}: 4 true pairs among 15 pairs.
    # Of the 5 candidates, (r0, r1), (r1, r2) and (r3, r4) are true matches.
    true = series(["a", "a", "a", "b", "b", "c"])
    left = pd.Series(["r0", "r1", "r2", "r3", "r4"])
    right = pd.Series(["r1", "r2", "r3", "r4", "r5"])
    scores = blocking_scores(left, right, true)
    assert scores["pair_completeness"] == pytest.approx(3 / 4)
    assert scores["reduction_ratio"] == pytest.approx(1 - 5 / 15)
    assert scores["pair_quality"] == pytest.approx(3 / 5)
    assert scores["n_candidates"] == 5
    # (r0, r2) is not a candidate but is reached through r1; (r2, r3) and
    # (r4, r5) join different entities, so they add nothing.
    assert scores["reachable_pair_completeness"] == 1.0


def test_reachable_pair_completeness_follows_only_matching_chains() -> None:
    # Entity {r0, r1, r2, r3}: 6 true pairs. Candidates (r0, r1), (r1, r2)
    # match and reach (r0, r1), (r1, r2), (r0, r2): 3 of 6. r3 is linked only
    # through the non-matching (r3, r4), which reaches nothing.
    true = series(["a", "a", "a", "a", "b"])
    left = pd.Series(["r0", "r1", "r3"])
    right = pd.Series(["r1", "r2", "r4"])
    scores = blocking_scores(left, right, true)
    assert scores["pair_completeness"] == pytest.approx(2 / 6)
    assert scores["reachable_pair_completeness"] == pytest.approx(3 / 6)


def test_blocking_scores_no_candidates() -> None:
    true = series(["a", "a", "b"])
    scores = blocking_scores(pd.Series([], dtype=str), pd.Series([], dtype=str), true)
    assert scores["pair_completeness"] == 0.0
    assert scores["reduction_ratio"] == 1.0
