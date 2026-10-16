"""
Clustering of candidate pairs scored by embedding similarity.

Input: a split's record IDs and its frozen candidate pairs, whose
``similarity`` is the cosine similarity of the two records' sentence
embeddings (``embeddings.embedding_pairs``).

Output: the predicted cluster of every record. Clusters only grow along
candidate pairs; records HDBSCAN calls noise, and records in no pair, are
singletons.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_components
from sklearn.cluster import HDBSCAN

from dedup_bench.clustering.graph import hierarchical

# Distance of the edges that join separate groups of candidate pairs: beyond
# any cosine distance (at most 1) and any cluster_selection_epsilon, so the
# groups behave as if infinitely far apart.
BRIDGE_DISTANCE = 2.0


def distance_matrix(record_ids: pd.Index, pairs: pd.DataFrame) -> csr_matrix:
    """
    Sparse cosine-distance matrix of the candidate pairs.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        Record IDs; row and column ``i`` is ``record_ids[i]``.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of the candidate pairs.

    Returns
    -------
    scipy.sparse.csr_matrix of float, shape (n_records, n_records)
        Symmetric; entry (i, j) is 1 - similarity when (i, j) is a candidate
        pair, and absent otherwise (HDBSCAN treats absent entries as
        unreachable). A stored distance is never exactly 0, so identical
        records keep their edge.

    Notes
    -----
    Processing:
    1. Map each pair's record IDs to positions in ``record_ids``.
    2. Store 1 - similarity in both directions, raised to a tiny positive
       minimum.
    """
    i = record_ids.get_indexer(pairs["left_id"])
    j = record_ids.get_indexer(pairs["right_id"])
    # A stored 0 can be read as "no edge", which would cut identical records apart.
    distance = np.maximum(1 - pairs["similarity"].to_numpy(), 1e-9)
    n = len(record_ids)
    return csr_matrix(
        (np.concatenate([distance, distance]), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(n, n),
    )


def bridge_groups(matrix: csr_matrix) -> csr_matrix:
    """
    Join the separate groups of a sparse distance matrix into one graph.

    scikit-learn's HDBSCAN accepts a sparse distance matrix only if it forms
    one connected graph. Chaining one record of each group to the next at
    ``BRIDGE_DISTANCE`` gives the same clustering as groups infinitely far
    apart: every group splits off the root before any real distance matters.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix of float, shape (n_records, n_records)
        Symmetric distance matrix, possibly in several connected groups.

    Returns
    -------
    scipy.sparse.csr_matrix of float, shape (n_records, n_records)
        The matrix plus symmetric bridging edges; unchanged if already connected.

    Notes
    -----
    Processing:
    1. Label the connected groups and take the first record of each.
    2. Add an edge of ``BRIDGE_DISTANCE`` between consecutive representatives.
    """
    n_groups, labels = scipy_components(matrix, directed=False)
    if n_groups < 2:
        return matrix
    _, first = np.unique(labels, return_index=True)
    i, j = first[:-1], first[1:]
    bridges = np.full(len(i), BRIDGE_DISTANCE)
    extra = csr_matrix(
        (np.concatenate([bridges, bridges]), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=matrix.shape,
    )
    return (matrix + extra).tocsr()


def hdbscan(
    record_ids: pd.Index,
    pairs: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
) -> pd.Series:
    """
    HDBSCAN on the sparse cosine distances of the candidate pairs.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and embedding ``similarity`` of each pair.
    min_cluster_size, min_samples : int
        HDBSCAN's density settings (2 and 1 in the config: a duplicate pair
        must be able to form a cluster).
    cluster_selection_method : str
        ``leaf`` or ``eom``.
    cluster_selection_epsilon : float
        Clusters closer than this cosine distance are merged.

    Returns
    -------
    pd.Series of str, shape (n_records,)
        Cluster label of every record, indexed by record ID; noise points are
        singletons labelled by their own record ID.

    Notes
    -----
    Processing:
    1. Build the distance matrix of all candidate pairs and bridge its groups.
    2. Run HDBSCAN once with precomputed distances.
    3. Label clustered records by their cluster; noise stays a singleton.
    """
    matrix = bridge_groups(distance_matrix(record_ids, pairs))
    found = HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        copy=False,
    ).fit_predict(matrix)
    labels = pd.Series(record_ids.astype(str), index=record_ids)
    clustered = found >= 0
    labels[clustered] = [f"c{c}" for c in found[clustered]]
    return labels


def agglomerative(
    record_ids: pd.Index, pairs: pd.DataFrame, linkage: str, distance_threshold: float
) -> pd.Series:
    """
    Agglomerative clustering of the embedding-scored pairs.

    The same algorithm as the graph rung's ``hierarchical``, given embedding
    similarities instead of TF-IDF ones, so the two differ only in the score.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and embedding ``similarity`` of each pair.
    linkage : str
        ``average`` or ``complete``.
    distance_threshold : float
        Clusters merge while their linkage cosine distance is at most this.

    Returns
    -------
    pd.Series of str, shape (n_records,)
        Cluster label of every record, indexed by record ID.

    Notes
    -----
    Processing:
    1. Call ``hierarchical`` with similarity threshold 1 - distance_threshold.
    """
    return hierarchical(record_ids, pairs, threshold=1 - distance_threshold, method=linkage)
