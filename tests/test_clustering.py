import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from dedup_bench.clustering.graph import (
    center,
    components,
    connected_components,
    hierarchical,
    leiden_cpm,
    markov,
    merge_center,
    ranked_pairs,
)

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


def pairs_of(*triples: tuple[str, str, float]) -> pd.DataFrame:
    left, right, similarity = zip(*triples)
    return pd.DataFrame({"left_id": left, "right_id": right, "similarity": similarity})


ABCD = pd.Index(["A", "B", "C", "D"])
CHAIN = pairs_of(("A", "B", 0.9), ("B", "C", 0.8), ("C", "D", 0.7))
# Two dense triangles joined by one weak pair, and a record g with no pair.
TRIANGLES_IDS = pd.Index(list("abcdefg"))
TRIANGLES = pairs_of(
    ("a", "b", 0.9), ("a", "c", 0.9), ("b", "c", 0.9),
    ("d", "e", 0.9), ("d", "f", 0.9), ("e", "f", 0.9),
    ("c", "d", 0.35),
)  # fmt: skip
TWO_GROUPS = {frozenset("abc"), frozenset("def"), frozenset("g")}
MARKOV = {
    "expansion": 2,
    "inflation": 2.0,
    "pruning_threshold": 0.001,
    "max_iterations": 100,
    "tolerance": 1e-6,
}


def test_ranked_pairs_order_and_threshold() -> None:
    pairs = pairs_of(("B", "C", 0.9), ("A", "D", 0.4), ("A", "B", 0.9))
    assert ranked_pairs(pairs, threshold=0.5) == [("A", "B"), ("B", "C")]


def test_center_breaks_a_chain_into_centre_clusters() -> None:
    # A-B: A becomes a centre, B joins it. B-C: B is a member, not a centre,
    # so C does not join. C-D: C becomes a centre, D joins it.
    labels = center(ABCD, CHAIN, threshold=0.5)
    assert groups(labels) == {frozenset("AB"), frozenset("CD")}
    assert labels.to_dict() == {"A": "A", "B": "A", "C": "C", "D": "C"}
    assert groups(connected_components(ABCD, CHAIN, threshold=0.5)) == {frozenset("ABCD")}


def test_center_lets_records_join_a_centre_they_are_paired_with() -> None:
    # A-B and A-C both join A; B-C comes last and changes nothing.
    pairs = pairs_of(("A", "B", 0.9), ("A", "C", 0.8), ("B", "C", 0.7))
    assert groups(center(ABCD, pairs, threshold=0.5)) == {frozenset("ABC"), frozenset("D")}


def test_center_breaks_ties_by_record_id() -> None:
    # Equal similarities: (A, B) is processed before (B, C), so A is the
    # centre and C, paired only with the member B, stays alone. Processing
    # (B, C) first would have made B the centre of all three.
    pairs = pairs_of(("B", "C", 0.9), ("A", "B", 0.9))
    assert groups(center(ABCD, pairs, threshold=0.5)) == {
        frozenset("AB"),
        frozenset("C"),
        frozenset("D"),
    }


def test_center_drops_pairs_below_the_threshold() -> None:
    assert len(groups(center(ABCD, CHAIN, threshold=0.95))) == 4


def test_merge_center_merges_clusters_linked_through_a_centre() -> None:
    # A and C become centres; A-C links two centres, so Merge-Center merges
    # the clusters and Center does not.
    pairs = pairs_of(("A", "B", 0.9), ("C", "D", 0.8), ("A", "C", 0.7))
    assert groups(merge_center(ABCD, pairs, threshold=0.5)) == {frozenset("ABCD")}


def test_merge_center_does_not_merge_through_two_members() -> None:
    # B and D are members, not centres, so B-D merges nothing.
    pairs = pairs_of(("A", "B", 0.9), ("C", "D", 0.8), ("B", "D", 0.7))
    assert groups(merge_center(ABCD, pairs, threshold=0.5)) == {frozenset("AB"), frozenset("CD")}
    assert groups(merge_center(ABCD, CHAIN, threshold=0.5)) == {frozenset("AB"), frozenset("CD")}


def test_markov_splits_two_dense_groups_joined_by_a_weak_pair() -> None:
    labels = markov(TRIANGLES_IDS, TRIANGLES, threshold=0.3, **MARKOV)
    assert groups(labels) == TWO_GROUPS
    assert groups(connected_components(TRIANGLES_IDS, TRIANGLES, 0.3)) == {
        frozenset("abcdef"),
        frozenset("g"),
    }


def test_leiden_splits_two_dense_groups_and_is_seeded() -> None:
    settings = {"objective_function": "CPM", "resolution": 0.5, "beta": 0.01, "n_iterations": -1}
    first = leiden_cpm(TRIANGLES_IDS, TRIANGLES, threshold=0.3, seed=7, **settings)
    again = leiden_cpm(TRIANGLES_IDS, TRIANGLES, threshold=0.3, seed=7, **settings)
    assert groups(first) == TWO_GROUPS
    assert first.equals(again)


def test_hierarchical_linkages_hand_computed() -> None:
    # A-B at distance 0.1, B-C at 0.15, A-C not a candidate (distance 1).
    # After A and B merge, {A, B} to C is (1 + 0.15) / 2 = 0.575 with average
    # linkage and max(1, 0.15) = 1 with complete linkage.
    ids = pd.Index(["A", "B", "C"])
    pairs = pairs_of(("A", "B", 0.9), ("B", "C", 0.85))
    two = {frozenset("AB"), frozenset("C")}
    assert groups(hierarchical(ids, pairs, threshold=0.5, method="average")) == two  # cut 0.5
    assert groups(hierarchical(ids, pairs, threshold=0.4, method="average")) == {frozenset("ABC")}
    assert groups(hierarchical(ids, pairs, threshold=0.4, method="complete")) == two


def test_hierarchical_per_group_equals_one_global_run() -> None:
    rng = np.random.default_rng(0)
    ids = pd.Index([f"r{i:02d}" for i in range(30)])
    left, right = rng.integers(0, 30, 60), rng.integers(0, 30, 60)
    keep = left < right
    pairs = pairs_of(
        *zip(ids[left[keep]], ids[right[keep]], rng.uniform(0.2, 1.0, keep.sum()))
    ).drop_duplicates(["left_id", "right_id"])
    for method in ("average", "complete"):
        matrix = np.ones((30, 30))
        np.fill_diagonal(matrix, 0)
        i, j = ids.get_indexer(pairs["left_id"]), ids.get_indexer(pairs["right_id"])
        matrix[i, j] = matrix[j, i] = 1 - pairs["similarity"].to_numpy()
        tree = linkage(squareform(matrix, checks=False), method=method)
        whole = pd.Series(fcluster(tree, t=1 - 0.5, criterion="distance"), index=ids)
        assert groups(hierarchical(ids, pairs, threshold=0.5, method=method)) == groups(whole)


def test_hierarchical_rejects_centroid() -> None:
    with pytest.raises(ValueError, match="average or complete"):
        hierarchical(ABCD, CHAIN, threshold=0.5, method="centroid")
