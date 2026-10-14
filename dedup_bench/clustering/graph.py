"""
Graph clustering of the scored candidate pairs.

Input: a split's records and candidate pairs with a similarity score.

Output: the predicted cluster of every record. Every method first drops the
pairs below its similarity threshold; records in no kept pair are singletons.
"""

import random

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, identity
from scipy.sparse.csgraph import connected_components as scipy_components
from scipy.spatial.distance import squareform


def components(record_ids: pd.Index, left: pd.Series, right: pd.Series) -> pd.Series:
    """
    Connected components of a set of record pairs.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs.
    left, right : pd.Series of str, shape (n_pairs,)
        Record IDs of the two records of each pair.

    Returns
    -------
    pd.Series of int, shape (n_records,)
        Component label of every record, indexed by record ID.

    Notes
    -----
    Processing:
    1. Map the IDs to positions, build an undirected sparse graph and label
       its connected components with SciPy.
    """
    n = len(record_ids)
    graph = coo_matrix(
        (np.ones(len(left)), (record_ids.get_indexer(left), record_ids.get_indexer(right))),
        shape=(n, n),
    )
    _, labels = scipy_components(graph, directed=False)
    return pd.Series(labels, index=record_ids)


def connected_components(record_ids: pd.Index, pairs: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Connected components of the candidate pairs at or above a similarity threshold.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    threshold : float
        Pairs with a lower similarity are dropped.

    Returns
    -------
    pd.Series of int, shape (n_records,)
        Cluster label of every record, indexed by record ID.

    Notes
    -----
    Processing:
    1. Keep the pairs at or above the threshold.
    2. Label the connected components of the kept pairs.
    """
    kept = pairs[pairs["similarity"] >= threshold]
    return components(record_ids, kept["left_id"], kept["right_id"])


def ranked_pairs(pairs: pd.DataFrame, threshold: float) -> list[tuple[str, str]]:
    """
    The pairs at or above the threshold, most similar first.

    Parameters
    ----------
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    threshold : float
        Pairs with a lower similarity are dropped.

    Returns
    -------
    list of (str, str)
        ``(left_id, right_id)`` of the kept pairs in processing order.

    Notes
    -----
    Processing:
    1. Keep the pairs at or above the threshold.
    2. Sort by similarity, highest first; ties by left and then right record
       ID, so the order (and every clustering built on it) is deterministic.
    """
    kept = pairs[pairs["similarity"] >= threshold]
    kept = kept.sort_values(
        ["similarity", "left_id", "right_id"], ascending=[False, True, True], kind="stable"
    )
    return list(zip(kept["left_id"], kept["right_id"]))


def center(record_ids: pd.Index, pairs: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Center clustering (Haveliwala et al., 2000; Hassanzadeh et al., 2009).

    Pairs are processed from most to least similar. Every cluster is one
    centre record plus members that each were paired directly with that
    centre, so a cluster cannot grow into a long chain.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    threshold : float
        Pairs with a lower similarity are dropped.

    Returns
    -------
    pd.Series of str, shape (n_records,)
        The centre of every record's cluster, indexed by record ID; records in
        no cluster are their own centre.

    Notes
    -----
    Processing:
    1. Rank the kept pairs with ``ranked_pairs``.
    2. For each pair (u, v) in order: if neither is assigned, u becomes a
       centre and v joins it; if exactly one is assigned and it is a centre,
       the other joins that centre; otherwise skip the pair.
    3. Unassigned records are singletons.
    """
    centre_of: dict[str, str] = {}
    for u, v in ranked_pairs(pairs, threshold):
        cu, cv = centre_of.get(u), centre_of.get(v)
        if cu is None and cv is None:
            centre_of[u] = centre_of[v] = u
        elif cu is None and cv == v:
            centre_of[u] = v
        elif cv is None and cu == u:
            centre_of[v] = u
    return pd.Series([centre_of.get(r, r) for r in record_ids], index=record_ids)


def merge_center(record_ids: pd.Index, pairs: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Merge-Center clustering (Hassanzadeh et al., 2009).

    Like ``center``, but when a pair links a centre to a record of another
    cluster, the two clusters merge. Clusters are less fragmented than with
    Center, while a chain of members that are not centres still cannot link
    clusters, as it does with connected components.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    threshold : float
        Pairs with a lower similarity are dropped.

    Returns
    -------
    pd.Series of str, shape (n_records,)
        A cluster label (one of its centres) per record ID.

    Notes
    -----
    Processing:
    1. Rank the kept pairs with ``ranked_pairs``.
    2. Assign records as ``center`` does, tracking which records are centres.
    3. When both records are already assigned to different clusters and at
       least one is a centre, merge the two clusters (union-find over centres).
    """
    centre_of: dict[str, str] = {}
    parent: dict[str, str] = {}

    def root(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for u, v in ranked_pairs(pairs, threshold):
        cu, cv = centre_of.get(u), centre_of.get(v)
        if cu is None and cv is None:
            centre_of[u] = centre_of[v] = parent[u] = u
        elif cu is None:
            if centre_of[v] == v:
                centre_of[u] = v
        elif cv is None:
            if centre_of[u] == u:
                centre_of[v] = u
        elif (centre_of[u] == u or centre_of[v] == v) and root(cu) != root(cv):
            parent[root(cv)] = root(cu)
    return pd.Series(
        [root(centre_of[r]) if r in centre_of else r for r in record_ids], index=record_ids
    )


def similarity_matrix(record_ids: pd.Index, pairs: pd.DataFrame, threshold: float) -> csr_matrix:
    """
    Symmetric sparse matrix of the similarities of the pairs above the threshold.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs; row and column i is ``record_ids[i]``.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    threshold : float
        Pairs with a lower similarity are dropped.

    Returns
    -------
    scipy.sparse.csr_matrix of float, shape (n_records, n_records)
        Entry (i, j) and (j, i) hold the similarity of a kept pair.

    Notes
    -----
    Processing:
    1. Keep the pairs at or above the threshold and map their IDs to positions.
    2. Fill both triangles.
    """
    kept = pairs[pairs["similarity"] >= threshold]
    i = record_ids.get_indexer(kept["left_id"])
    j = record_ids.get_indexer(kept["right_id"])
    weights = kept["similarity"].to_numpy(dtype=float)
    n = len(record_ids)
    return coo_matrix(
        (np.concatenate([weights, weights]), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(n, n),
    ).tocsr()


def markov(
    record_ids: pd.Index,
    pairs: pd.DataFrame,
    threshold: float,
    expansion: int,
    inflation: float,
    pruning_threshold: float,
    max_iterations: int,
    tolerance: float,
) -> pd.Series:
    """
    Markov clustering, MCL (van Dongen, 2000), on sparse matrices.

    Simulates random walks on the similarity graph: expansion spreads flow
    along paths, inflation strengthens strong flows and weakens weak ones.
    Flow stays inside densely linked groups and drains from the thin links
    between them, so chains break where connected components would follow
    them.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    threshold : float
        Pairs with a lower similarity are dropped.
    expansion : int
        Power the flow matrix is raised to in each step.
    inflation : float
        Power applied to every entry in each step; higher gives smaller
        clusters.
    pruning_threshold : float
        Flow below this is set to zero after each step, keeping the matrix sparse.
    max_iterations : int
        Upper bound on the number of steps.
    tolerance : float
        Stop when no entry changes by more than this in one step.

    Returns
    -------
    pd.Series of int, shape (n_records,)
        Cluster label of every record (the record its flow ends at), indexed
        by record ID.

    Notes
    -----
    Processing:
    1. Build the similarity matrix, add self-loops of weight 1 and normalise
       every column to sum to 1.
    2. Repeat: expand (matrix power), inflate (entry-wise power), prune small
       entries and renormalise, until the matrix stops changing or
       ``max_iterations`` is reached.
    3. Each record joins the row that holds most of its column's flow (its
       attractor); records with the same attractor form a cluster.
    """
    n = len(record_ids)
    flow = (similarity_matrix(record_ids, pairs, threshold) + identity(n, format="csr")).tocsc()

    def normalise(m: csc_matrix) -> csc_matrix:
        return csc_matrix(m.multiply(1 / np.asarray(m.sum(axis=0)).ravel()))

    flow = normalise(flow)
    for _ in range(max_iterations):
        previous = flow
        flow = normalise(flow**expansion)
        flow = flow.power(inflation)
        flow.data[flow.data < pruning_threshold] = 0
        flow.eliminate_zeros()
        flow = normalise(flow)
        if abs(flow - previous).max() <= tolerance:
            break
    return pd.Series(np.asarray(flow.argmax(axis=0)).ravel(), index=record_ids)


def leiden_cpm(
    record_ids: pd.Index,
    pairs: pd.DataFrame,
    threshold: float,
    objective_function: str,
    resolution: float,
    beta: float,
    n_iterations: int,
    seed: int,
) -> pd.Series:
    """
    Leiden community detection (Traag et al., 2019) with python-igraph.

    With the CPM objective, a group of records is a community when its
    average internal similarity exceeds ``resolution``, so the resolution is
    on the similarity scale.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    threshold : float
        Pairs with a lower similarity are dropped.
    objective_function, resolution, beta, n_iterations
        Passed to ``igraph.Graph.community_leiden`` (``n_iterations`` -1 runs
        until the partition is stable).
    seed : int
        Seeds Python's ``random`` module, which igraph draws from.

    Returns
    -------
    pd.Series of int, shape (n_records,)
        Community of every record, indexed by record ID.

    Notes
    -----
    Processing:
    1. Build an undirected graph of the kept pairs, weighted by similarity.
    2. Seed and run Leiden.
    """
    import igraph

    kept = pairs[pairs["similarity"] >= threshold]
    edges = zip(record_ids.get_indexer(kept["left_id"]), record_ids.get_indexer(kept["right_id"]))
    graph = igraph.Graph(n=len(record_ids), edges=list(edges))
    graph.es["weight"] = kept["similarity"].to_list()
    random.seed(seed)
    partition = graph.community_leiden(
        objective_function=objective_function,
        weights="weight",
        resolution=resolution,
        beta=beta,
        n_iterations=n_iterations,
    )
    return pd.Series(partition.membership, index=record_ids)


def hierarchical(
    record_ids: pd.Index, pairs: pd.DataFrame, threshold: float, method: str
) -> pd.Series:
    """
    Agglomerative clustering on 1 - similarity, cut at 1 - threshold.

    Candidate pairs have distance 1 - similarity and all other pairs distance
    1, so clusters only grow along candidate pairs. Each connected group of
    pairs at or above the threshold is clustered on its own: with the
    ``average`` or ``complete`` linkage, two groups are at distance above
    1 - threshold from each other, so they would never merge in one global
    run either, and the dense distance matrices stay small.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    threshold : float
        Clusters merge while their linkage distance is at most 1 - threshold.
    method : str
        SciPy linkage, ``average`` or ``complete``.

    Returns
    -------
    pd.Series of str, shape (n_records,)
        Cluster label of every record, indexed by record ID.

    Raises
    ------
    ValueError
        If ``method`` is not ``average`` or ``complete``.

    Notes
    -----
    Processing:
    1. Find the connected groups of the pairs at or above the threshold.
    2. For each group of two or more records: build its dense distance matrix
       from all its candidate pairs, link it and cut at 1 - threshold.
    3. Label each cluster by its group and number within the group.
    """
    if method not in ("average", "complete"):
        raise ValueError(f"method must be average or complete, got {method!r}")
    groups = connected_components(record_ids, pairs, threshold)
    labels = pd.Series(record_ids.astype(str), index=record_ids)
    distance = pd.Series(
        1 - pairs["similarity"].to_numpy(),
        index=pd.MultiIndex.from_arrays([pairs["left_id"], pairs["right_id"]]),
    )
    for group, members in groups.groupby(groups).groups.items():
        if len(members) < 2:
            continue
        members = pd.Index(members)
        matrix = np.ones((len(members), len(members)))
        np.fill_diagonal(matrix, 0)
        inside = distance[
            distance.index.get_level_values(0).isin(members)
            & distance.index.get_level_values(1).isin(members)
        ]
        i = members.get_indexer(inside.index.get_level_values(0))
        j = members.get_indexer(inside.index.get_level_values(1))
        matrix[i, j] = matrix[j, i] = inside.to_numpy()
        tree = linkage(squareform(matrix, checks=False), method=method)
        cut = fcluster(tree, t=1 - threshold, criterion="distance")
        labels[members] = [f"{group}-{c}" for c in cut]
    return labels
