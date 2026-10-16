import numpy as np
import pandas as pd
import pytest

from dedup_bench import embeddings
from dedup_bench.clustering.embedding import agglomerative, distance_matrix, hdbscan
from dedup_bench.clustering.graph import hierarchical


def frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["left_id", "right_id", "similarity"])


def groups_of(labels: pd.Series) -> set[frozenset[str]]:
    return {frozenset(ids) for ids in labels.groupby(labels).groups.values()}


def test_distance_matrix_hand_computed() -> None:
    # r0-r1 identical texts (similarity 1), r1-r2 similarity 0.6, no r0-r2 pair.
    members = pd.Index(["r0", "r1", "r2"])
    matrix = distance_matrix(members, frame([("r0", "r1", 1.0), ("r1", "r2", 0.6)])).toarray()
    assert matrix[1, 2] == pytest.approx(0.4) and matrix[2, 1] == pytest.approx(0.4)
    assert (matrix == matrix.T).all()
    # The identical pair keeps an edge: a tiny positive distance, not a missing 0.
    assert 0 < matrix[0, 1] < 1e-6
    # No candidate pair between r0 and r2, and nothing on the diagonal.
    assert matrix[0, 2] == 0 and (np.diag(matrix) == 0).all()


def test_distance_matrix_stores_every_edge() -> None:
    members = pd.Index(["r0", "r1", "r2"])
    matrix = distance_matrix(members, frame([("r0", "r1", 1.0), ("r1", "r2", 0.6)]))
    assert matrix.nnz == 4


def test_hdbscan_splits_two_tight_groups_joined_by_a_weak_pair() -> None:
    ids = pd.Index([f"r{i}" for i in range(7)])
    pairs = frame(
        [
            ("r0", "r1", 0.97), ("r1", "r2", 0.96), ("r0", "r2", 0.95),
            ("r3", "r4", 0.97), ("r4", "r5", 0.96), ("r3", "r5", 0.95),
            ("r2", "r3", 0.10),
        ]
    )  # fmt: skip
    labels = hdbscan(ids, pairs, 2, 1, "leaf", 0.0)
    assert groups_of(labels) == {
        frozenset({"r0", "r1", "r2"}),
        frozenset({"r3", "r4", "r5"}),
        frozenset({"r6"}),
    }


def test_hdbscan_keeps_separate_pairs_as_clusters() -> None:
    # Two pairs with no candidate pair between them, and an unpaired record:
    # as in one run over all records, each pair is its own cluster.
    ids = pd.Index(["r0", "r1", "r2", "r3", "r4"])
    pairs = frame([("r0", "r1", 1.0), ("r2", "r3", 0.8)])
    labels = hdbscan(ids, pairs, 2, 1, "leaf", 0.0)
    assert groups_of(labels) == {
        frozenset({"r0", "r1"}),
        frozenset({"r2", "r3"}),
        frozenset({"r4"}),
    }


def test_bridge_groups_connects_every_group_beyond_any_cosine_distance() -> None:
    from scipy.sparse.csgraph import connected_components

    from dedup_bench.clustering.embedding import BRIDGE_DISTANCE, bridge_groups

    ids = pd.Index(["r0", "r1", "r2", "r3", "r4"])
    matrix = bridge_groups(distance_matrix(ids, frame([("r0", "r1", 1.0), ("r2", "r3", 0.8)])))
    assert connected_components(matrix, directed=False)[0] == 1
    added = matrix.toarray()[matrix.toarray() > 1]
    assert len(added) == 4 and (added == BRIDGE_DISTANCE).all()  # 2 bridges, both directions


def test_agglomerative_is_hierarchical_with_the_distance_converted() -> None:
    ids = pd.Index(["r0", "r1", "r2", "r3"])
    pairs = frame([("r0", "r1", 0.9), ("r1", "r2", 0.7), ("r2", "r3", 0.4)])
    for linkage in ("average", "complete"):
        expected = hierarchical(ids, pairs, threshold=0.75, method=linkage)
        assert agglomerative(ids, pairs, linkage, 0.25).equals(expected)


def test_embed_encodes_each_distinct_text_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_encode(texts: pd.Series, model_name: str, batch_size: int) -> np.ndarray:
        calls.append(texts.tolist())
        return np.eye(len(texts), 4, dtype=np.float32)

    monkeypatch.setattr(embeddings, "encode", fake_encode)
    monkeypatch.setattr(embeddings, "_CACHE", {})
    first = embeddings.embed(pd.Series(["a", "b", "a"]), "m", 8)
    second = embeddings.embed(pd.Series(["b", "c"]), "m", 8)
    assert calls == [["a", "b"], ["c"]]
    assert (first[0] == first[2]).all() and (second[0] == first[1]).all()


def test_embedding_pairs_scores_by_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = {"x": [1.0, 0.0], "y": [0.6, 0.8], "z": [-1.0, 0.0]}

    def fake_encode(texts: pd.Series, model_name: str, batch_size: int) -> np.ndarray:
        return np.array([vectors[t] for t in texts], dtype=np.float32)

    monkeypatch.setattr(embeddings, "encode", fake_encode)
    monkeypatch.setattr(embeddings, "_CACHE", {})
    texts = pd.Series({"r0": "x", "r1": "y", "r2": "z"})
    pairs = frame([("r0", "r1", 0.0), ("r0", "r2", 0.0)])
    scored = embeddings.embedding_pairs(texts, pairs, "m", 8)
    # cos(x, y) = 0.6; cos(x, z) = -1 is clipped to 0 like the TF-IDF scores.
    assert scored["similarity"].tolist() == pytest.approx([0.6, 0.0])
    assert scored[["left_id", "right_id"]].equals(pairs[["left_id", "right_id"]])
