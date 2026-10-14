import numpy as np
import pandas as pd
import pytest

from dedup_bench import smoke


def test_middle_takes_the_later_middle_for_even_lengths() -> None:
    assert smoke.middle([1, 2, 3]) == 2
    assert smoke.middle([1, 2, 3, 4]) == 3
    assert smoke.middle(["only"]) == "only"


def test_load_settings_resolves_lists_and_ranges() -> None:
    # Ranges crashed the smoke test before 3b53327. The threshold range 0.3 to
    # 0.95 in steps of 0.05 has 14 points; the middle is the 8th, 0.65.
    settings = smoke.load_settings()
    assert settings["graph_clustering"]["connected_components"]["threshold"] == pytest.approx(0.65)
    assert settings["blocking"]["tfidf_knn"]["top_k"] == 25
    assert settings["baselines"]["exact_key"]["token_sort"] is True
    assert "description" not in settings and "description" not in settings["blocking"]
    for models in settings.values():
        for params in models.values():
            assert not any(isinstance(value, dict) for value in params.values())


def test_smoke_datasets_come_from_the_smoke_config() -> None:
    assert set(smoke.SMOKE_DATASETS) == {"febrl3", "restaurants"}
    assert "phone" in smoke.SMOKE_DATASETS["restaurants"]["fields"]


def test_clusters_from_pairs_and_groups() -> None:
    ids = pd.Index(["a", "b", "c", "d"])
    from_pairs = smoke.clusters_from_pairs(ids, np.array([0]), np.array([2]))
    assert from_pairs["a"] == from_pairs["c"] and from_pairs.nunique() == 3
    from_groups = smoke.clusters_from_groups(ids, [["b", "d"]])
    assert from_groups["b"] == from_groups["d"] and from_groups.nunique() == 3


def test_fingerprint_ignores_label_names_and_order() -> None:
    first = pd.Series([0, 0, 1], index=["a", "b", "c"])
    renamed = pd.Series(["y", "x", "x"], index=["c", "b", "a"])
    different = pd.Series([0, 1, 1], index=["a", "b", "c"])
    assert smoke.fingerprint(first) == smoke.fingerprint(renamed)
    assert smoke.fingerprint(first) != smoke.fingerprint(different)


def test_tfidf_knn_pairs_finds_the_near_duplicate() -> None:
    texts = pd.Series(["apple pie", "apple pie!", "zebra crossing"])
    left, right, similarity = smoke.tfidf_knn_pairs(texts)
    best = np.argmax(similarity)
    assert (left[best], right[best]) == (0, 1)
    assert (left < right).all()
