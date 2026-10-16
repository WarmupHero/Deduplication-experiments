import itertools
import random

import numpy as np
import pandas as pd
import pytest

from dedup_bench import probabilistic
from dedup_bench.probabilistic import (
    ecm_scores,
    pair_probabilities,
    splink_comparisons,
    splink_scores,
)

FIRST = ["anna", "boris", "chloe", "dimitri", "elena", "fotis", "georgia", "hector", "irene"]
LAST = ["papas", "nikolaou", "smith", "jones", "garcia", "muller", "rossi", "dupont", "novak"]
CITIES = ["athens", "patras", "volos", "chania"]
FIELDS = {"name": "jaro_winkler", "city": "exact", "phone": "levenshtein"}
CONFIG = {"match_fields": FIELDS, "em_blocking": ["city", "substr(name, 1, 2)"]}
SPLINK = {
    "jaro_winkler_thresholds": [0.9, 0.7],
    "levenshtein_thresholds": [1, 2],
    "u_max_pairs": 100000,
    "max_iterations": 25,
    "em_convergence": 0.0001,
}
ECM = {"binarize": 0.8, "max_iter": 100, "atol": 0.0001}


def people(n_entities: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Two records per person: the second has a name typo and sometimes a phone typo."""
    rng = random.Random(seed)
    rows, truth = [], {}
    for entity in range(n_entities):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        city = rng.choice(CITIES)
        phone = "".join(rng.choice("0123456789") for _ in range(7))
        cut = rng.randrange(1, len(name) - 1)
        typo_name = name[:cut] + name[cut + 1 :]
        typo_phone = phone if rng.random() < 0.5 else phone[:-1] + str((int(phone[-1]) + 1) % 10)
        for copy, (n, p) in enumerate([(name, phone), (typo_name, typo_phone)]):
            record_id = f"s{seed}e{entity}c{copy}"
            rows.append({"record_id": record_id, "name": n, "city": city, "phone": p})
            truth[record_id] = f"e{entity}"
    return pd.DataFrame(rows), pd.Series(truth)


def same_city_pairs(records: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (a, b, 0.0)
        for _, group in records.groupby("city")
        for a, b in itertools.combinations(sorted(group["record_id"]), 2)
    ]
    return pd.DataFrame(rows, columns=["left_id", "right_id", "similarity"])


def match_mask(pairs: pd.DataFrame, truth: pd.Series) -> np.ndarray:
    return truth[pairs["left_id"]].to_numpy() == truth[pairs["right_id"]].to_numpy()


@pytest.fixture(autouse=True)
def empty_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probabilistic, "_MODELS", {})
    monkeypatch.setattr(probabilistic, "_SCORES", {})


def test_splink_comparisons_follow_the_field_list() -> None:
    comparisons = splink_comparisons(FIELDS, [0.9, 0.7], [1, 2])
    described = [c.get_comparison("duckdb").as_dict() for c in comparisons]
    assert [d["output_column_name"] for d in described] == ["name", "city", "phone"]
    labels = [
        [level.get("label_for_charts") for level in d["comparison_levels"]] for d in described
    ]
    assert "Jaro-Winkler distance of name >= 0.9" in labels[0]
    assert "Jaro-Winkler distance of name >= 0.7" in labels[0]
    assert "Exact match on city" in labels[1]
    assert any("Levenshtein" in label and "<= 2" in label for label in labels[2])


def test_splink_comparisons_reject_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown comparison"):
        splink_comparisons({"name": "cosine"}, [0.9], [1])


def test_pair_probabilities_align_whichever_way_round() -> None:
    predictions = pd.DataFrame({"l": ["b", "c"], "r": ["a", "a"], "p": [0.9, 0.2]})
    pairs = pd.DataFrame({"left_id": ["a", "a"], "right_id": ["c", "b"]})
    assert pair_probabilities(predictions, pairs, "l", "r", "p").tolist() == [0.2, 0.9]
    with pytest.raises(ValueError, match="not scored"):
        pair_probabilities(
            predictions, pd.DataFrame({"left_id": ["x"], "right_id": ["y"]}), "l", "r", "p"
        )


def test_ecm_scores_matches_above_non_matches() -> None:
    train, _ = people(40, seed=1)
    target, truth = people(20, seed=2)
    pairs = same_city_pairs(target)
    probability = ecm_scores(train, same_city_pairs(train), target, pairs, CONFIG, ECM)
    assert probability.shape == (len(pairs),)
    assert ((0 <= probability) & (probability <= 1)).all()
    matches = match_mask(pairs, truth)
    assert probability[matches].mean() > probability[~matches].mean() + 0.5


def test_splink_scores_matches_above_non_matches() -> None:
    train, _ = people(60, seed=3)
    target, truth = people(20, seed=4)
    pairs = same_city_pairs(target)
    probability = splink_scores(train, target, pairs, CONFIG, SPLINK, seed=0)
    assert probability.shape == (len(pairs),)
    assert ((0 <= probability) & (probability <= 1)).all()
    matches = match_mask(pairs, truth)
    assert probability[matches].mean() > probability[~matches].mean() + 0.5


def test_splink_trains_once_per_train_split(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = probabilistic.train_splink

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(probabilistic, "train_splink", counting)
    train, _ = people(60, seed=3)
    target, _ = people(10, seed=5)
    pairs = same_city_pairs(target)
    first = splink_scores(train, target, pairs, CONFIG, SPLINK, seed=0)
    second = splink_scores(train, target, pairs, CONFIG, SPLINK, seed=0)
    assert len(calls) == 1 and (first == second).all()
