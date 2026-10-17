import numpy as np
import pandas as pd
import pytest
from test_probabilistic import FIELDS, match_mask, people, same_city_pairs

from dedup_bench import experiment, learned
from dedup_bench.learned import (
    active_learning_order,
    active_learning_scores,
    classifier,
    match_probability,
    oracle_labels,
    supervised_scores,
)

FOREST = {"n_estimators": 20, "max_depth": None, "class_weight": "balanced"}


@pytest.fixture(autouse=True)
def empty_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(learned, "_ORDERS", {})
    monkeypatch.setattr(learned, "_SCORES", {})


def test_oracle_labels_hand_computed() -> None:
    truth = pd.Series({"a": "e1", "b": "e1", "c": "e2"})
    pairs = pd.DataFrame({"left_id": ["a", "a", "b"], "right_id": ["b", "c", "c"]})
    assert oracle_labels(pairs, truth).tolist() == [True, False, False]


def test_one_class_training_gives_zero_probability() -> None:
    fitted = classifier("random_forest", FOREST, 0).fit(np.zeros((3, 1)), [False] * 3)
    assert match_probability(fitted, np.ones((2, 1))).tolist() == [0.0, 0.0]


def test_a_larger_budget_extends_a_smaller_ones_order() -> None:
    x = np.linspace(0, 1, 200)[:, None]
    oracle = x[:, 0] > 0.5
    short = active_learning_order(x, oracle, FOREST, 10, 2, 25, seed=3)
    long = active_learning_order(x, oracle, FOREST, 10, 2, 60, seed=3)
    assert len(short) == 25 and len(set(long)) == 60
    assert (long[:25] == short).all()
    assert len(active_learning_order(x[:7], oracle[:7], FOREST, 10, 2, 60, seed=3)) == 7


def test_queries_go_to_the_decision_boundary() -> None:
    # Matches are x > 0.5; the 6 queries after the 10 random labels close in on 0.5.
    x = np.linspace(0, 1, 401)[:, None]
    order = active_learning_order(x, x[:, 0] > 0.5, FOREST, 10, 1, 16, seed=0)
    assert np.median(np.abs(x[order[10:], 0] - 0.5)) < 0.02


def test_only_non_matches_so_far_queries_the_most_similar_pairs() -> None:
    x = np.linspace(0, 1, 50)[:, None]
    order = active_learning_order(x, np.zeros(50, dtype=bool), FOREST, 5, 3, 20, seed=1)
    rest = np.setdiff1d(np.arange(50), order[:5])
    assert sorted(order[5:]) == sorted(rest[-15:])


@pytest.mark.parametrize("model", ["random_forest", "hist_gradient_boosting"])
def test_supervised_scores_matches_above_non_matches(model: str) -> None:
    train, train_truth = people(60, seed=3)
    target, truth = people(20, seed=4)
    train_pairs, pairs = same_city_pairs(train), same_city_pairs(target)
    params = FOREST if model == "random_forest" else {"max_iter": 50, "class_weight": "balanced"}
    probability = supervised_scores(
        model, train, train_pairs, train_truth, target, pairs, FIELDS, params, 0
    )
    matches = match_mask(pairs, truth)
    assert probability[matches].mean() > probability[~matches].mean() + 0.5


def test_active_learning_scores_matches_above_non_matches() -> None:
    train, train_truth = people(60, seed=3)
    target, truth = people(20, seed=4)
    train_pairs, pairs = same_city_pairs(train), same_city_pairs(target)
    settings = {**FOREST, "label_budget": 40, "initial_labels": 10, "query_batch_size": 5}
    probability = active_learning_scores(
        train, train_pairs, train_truth, target, pairs, FIELDS, settings, 0
    )
    matches = match_mask(pairs, truth)
    assert probability[matches].mean() > probability[~matches].mean() + 0.5


def test_label_budgets_run_largest_first_in_their_own_folders() -> None:
    budgets = [
        f["label_budget"] for f in experiment.factor_settings("active_learning.random_forest")
    ]
    assert budgets == sorted(budgets, reverse=True) and len(budgets) > 1
    assert experiment.factor_settings("graph_clustering.center") == [{}]
    path = experiment.results_dir("d", "m", 2, {"label_budget": 50})
    assert path.parts[-3:] == ("m", "label_budget50", "seed2")
