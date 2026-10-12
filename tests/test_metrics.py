import pandas as pd
import pytest

from dedup_bench.evaluation.metrics import b_cubed_scores, count_pairs, pairwise_scores


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
