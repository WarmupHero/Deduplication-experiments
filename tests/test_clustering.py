import pandas as pd

from dedup_bench.clustering.graph import components, connected_components

IDS = pd.Index(["a", "b", "c", "d", "e"])


def groups(labels: pd.Series) -> set[frozenset[str]]:
    return {frozenset(ids) for ids in labels.groupby(labels).groups.values()}


def test_components_follow_chains_and_keep_singletons() -> None:
    # a-b and b-c chain a, b and c together; d and e are in no pair.
    labels = components(IDS, pd.Series(["a", "b"]), pd.Series(["b", "c"]))
    assert labels.index.equals(IDS)
    assert groups(labels) == {frozenset("abc"), frozenset("d"), frozenset("e")}


def test_connected_components_drops_pairs_below_the_threshold() -> None:
    pairs = pd.DataFrame(
        {"left_id": ["a", "b", "d"], "right_id": ["b", "c", "e"], "similarity": [0.9, 0.4, 0.5]}
    )
    # At 0.5, b-c (0.4) is dropped and d-e (0.5, equal to the threshold) is kept.
    labels = connected_components(IDS, pairs, threshold=0.5)
    assert groups(labels) == {frozenset("ab"), frozenset("c"), frozenset("de")}
    assert len(groups(connected_components(IDS, pairs, threshold=0.95))) == 5
