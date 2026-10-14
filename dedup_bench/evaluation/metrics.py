"""
Quality metrics for a predicted clustering of records into entities.

Input: two membership vectors over the same records, the predicted cluster
and the true entity of every record, as pandas Series indexed by record ID.

Output: precision, recall and F1 scores (pairwise, B-cubed and cluster-level),
and the adjusted Rand index. Pairwise scores count every pair of records placed
in the same cluster, so an entity with k records contributes k (k - 1) / 2
pairs.

Blocking is scored separately, from the candidate pairs and the true entities:
pair completeness (direct and reachable), reduction ratio and pair quality.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def count_pairs(sizes: pd.Series) -> int:
    """
    Count the record pairs inside clusters of the given sizes.

    Parameters
    ----------
    sizes : pd.Series of int, shape (n_clusters,)
        Number of records in each cluster.

    Returns
    -------
    int
        The sum of k (k - 1) / 2 over all cluster sizes k.

    Notes
    -----
    Processing:
    1. Apply k (k - 1) / 2 to every size and add the results.
    """
    sizes = sizes.astype("int64")
    return int((sizes * (sizes - 1) // 2).sum())


def pairwise_scores(predicted: pd.Series, true: pd.Series) -> dict[str, float]:
    """
    Pairwise precision, recall and F1 of a predicted clustering.

    A pair of records is a predicted match if both records share a predicted
    cluster, and a true match if both share a true entity.

    Parameters
    ----------
    predicted : pd.Series, shape (n_records,)
        Predicted cluster label of each record, indexed by record ID.
    true : pd.Series, shape (n_records,)
        True entity label of each record, indexed by the same record IDs.

    Returns
    -------
    dict of str to float
        ``precision``, ``recall`` and ``f1``. A score whose denominator is
        zero is reported as 1.0 when its numerator is also zero (nothing
        predicted and nothing to find), so an all-singleton prediction on
        all-singleton data scores perfectly.

    Raises
    ------
    ValueError
        If the two Series do not cover exactly the same record IDs.

    Notes
    -----
    Processing:
    1. Align the two Series on record ID and check they match.
    2. Count predicted pairs from the predicted cluster sizes, true pairs from
       the entity sizes, and correctly predicted pairs from the sizes of the
       (cluster, entity) intersections.
    3. Derive precision, recall and F1 from the three counts.
    """
    if predicted.index.has_duplicates or set(predicted.index) != set(true.index):
        raise ValueError("predicted and true must cover the same, unique record IDs")
    frame = pd.DataFrame({"predicted": predicted, "true": true.reindex(predicted.index)})

    predicted_pairs = count_pairs(frame.groupby("predicted").size())
    true_pairs = count_pairs(frame.groupby("true").size())
    correct_pairs = count_pairs(frame.groupby(["predicted", "true"]).size())

    precision = correct_pairs / predicted_pairs if predicted_pairs else 1.0
    recall = correct_pairs / true_pairs if true_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def b_cubed_scores(predicted: pd.Series, true: pd.Series) -> dict[str, float]:
    """
    B-cubed precision, recall and F1 of a predicted clustering.

    Every record is scored on its own and the scores are averaged, so each
    record weighs the same however large its entity is (pairwise scores, by
    contrast, are dominated by large entities). For a record r with predicted
    cluster C(r) and true entity E(r):

    - precision(r) = |C(r) ∩ E(r)| / |C(r)|
    - recall(r) = |C(r) ∩ E(r)| / |E(r)|

    Parameters
    ----------
    predicted : pd.Series, shape (n_records,)
        Predicted cluster label of each record, indexed by record ID.
    true : pd.Series, shape (n_records,)
        True entity label of each record, indexed by the same record IDs.

    Returns
    -------
    dict of str to float
        ``precision``, ``recall`` (the averages over all records) and ``f1``
        (their harmonic mean).

    Raises
    ------
    ValueError
        If the two Series do not cover exactly the same record IDs.

    Notes
    -----
    Processing:
    1. Align the two Series on record ID and check they match.
    2. For every record, count the records in its predicted cluster, in its
       true entity, and in the intersection of the two.
    3. Average the per-record precision and recall, and combine them into F1.
    """
    if predicted.index.has_duplicates or set(predicted.index) != set(true.index):
        raise ValueError("predicted and true must cover the same, unique record IDs")
    frame = pd.DataFrame({"predicted": predicted, "true": true.reindex(predicted.index)})

    # Each record gets the size of its predicted cluster, its true entity and
    # their intersection; transform keeps the sizes aligned to the records.
    cluster_size = frame.groupby("predicted")["true"].transform("size")
    entity_size = frame.groupby("true")["predicted"].transform("size")
    overlap = frame.groupby(["predicted", "true"])["true"].transform("size")
    precision = float((overlap / cluster_size).mean())
    recall = float((overlap / entity_size).mean())

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cluster_scores(predicted: pd.Series, true: pd.Series) -> dict[str, float]:
    """
    Cluster-level precision, recall and F1 of a predicted clustering.

    A predicted cluster is correct only if it holds exactly the records of one
    true entity, no more and no fewer. Singletons count like any other cluster.

    - precision = correct clusters / predicted clusters
    - recall = correct clusters / true entities

    Parameters
    ----------
    predicted : pd.Series, shape (n_records,)
        Predicted cluster label of each record, indexed by record ID.
    true : pd.Series, shape (n_records,)
        True entity label of each record, indexed by the same record IDs.

    Returns
    -------
    dict of str to float
        ``precision``, ``recall`` and ``f1``.

    Raises
    ------
    ValueError
        If the two Series do not cover exactly the same record IDs.

    Notes
    -----
    Processing:
    1. Align the two Series on record ID and check they match.
    2. Count the predicted clusters that equal a true entity.
    3. Divide by the numbers of predicted clusters and true entities, and
       combine into F1.
    """
    if predicted.index.has_duplicates or set(predicted.index) != set(true.index):
        raise ValueError("predicted and true must cover the same, unique record IDs")
    frame = pd.DataFrame({"predicted": predicted, "true": true.reindex(predicted.index)})

    # A (cluster, entity) overlap is a correct cluster when it is the whole
    # cluster and the whole entity.
    overlap = frame.groupby(["predicted", "true"]).size()
    cluster_size = frame.groupby("predicted").size()
    entity_size = frame.groupby("true").size()
    whole_cluster = (
        overlap.to_numpy() == cluster_size[overlap.index.get_level_values(0)].to_numpy()
    )
    whole_entity = overlap.to_numpy() == entity_size[overlap.index.get_level_values(1)].to_numpy()
    correct = int((whole_cluster & whole_entity).sum())
    n_predicted, n_true = len(cluster_size), len(entity_size)

    precision = correct / n_predicted
    recall = correct / n_true
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def adjusted_rand_index(predicted: pd.Series, true: pd.Series) -> float:
    """
    Adjusted Rand index (Hubert and Arabie, 1985) of a predicted clustering.

    The share of record pairs on which the two clusterings agree (together in
    both, or apart in both), corrected for chance: 1 for identical
    clusterings, about 0 for a random one, and negative when worse than
    random.

    Parameters
    ----------
    predicted : pd.Series, shape (n_records,)
        Predicted cluster label of each record, indexed by record ID.
    true : pd.Series, shape (n_records,)
        True entity label of each record, indexed by the same record IDs.

    Returns
    -------
    float
        The index. When it is undefined (both clusterings all singletons, or
        both one cluster) it is 1.0, as in scikit-learn.

    Raises
    ------
    ValueError
        If the two Series do not cover exactly the same record IDs.

    Notes
    -----
    Processing:
    1. Align the two Series on record ID and check they match.
    2. Count the pairs together in both clusterings (``both``), in the
       predicted one (``a``) and in the true one (``b``), and all pairs.
    3. ARI = (both - expected) / (max - expected), with
       expected = a b / all pairs and max = (a + b) / 2.
    """
    if predicted.index.has_duplicates or set(predicted.index) != set(true.index):
        raise ValueError("predicted and true must cover the same, unique record IDs")
    frame = pd.DataFrame({"predicted": predicted, "true": true.reindex(predicted.index)})

    both = count_pairs(frame.groupby(["predicted", "true"]).size())
    a = count_pairs(frame.groupby("predicted").size())
    b = count_pairs(frame.groupby("true").size())
    n = len(frame)
    expected = a * b / (n * (n - 1) / 2) if n > 1 else 0.0
    maximum = (a + b) / 2
    if maximum == expected:
        return 1.0
    return (both - expected) / (maximum - expected)


def blocking_scores(left: pd.Series, right: pd.Series, true: pd.Series) -> dict[str, float]:
    """
    How well a set of candidate pairs covers the true matches.

    - pair completeness (blocking recall) = true matches among the candidates
      / all true matches
    - reachable pair completeness = true matches joined by a chain of
      candidate pairs that are true matches / all true matches; what a perfect
      matcher followed by transitive clustering could recover
    - reduction ratio = 1 - candidates / all record pairs
    - pair quality = true matches among the candidates / candidates

    Parameters
    ----------
    left, right : pd.Series of str, shape (n_pairs,)
        Record IDs of the two records of each candidate pair; each unordered
        pair once.
    true : pd.Series, shape (n_records,)
        True entity of every record, indexed by record ID.

    Returns
    -------
    dict of str to float
        ``pair_completeness``, ``reachable_pair_completeness``,
        ``reduction_ratio``, ``pair_quality`` and ``n_candidates``. A ratio
        whose denominator is zero is 1.0, as in ``pairwise_scores``.

    Notes
    -----
    Processing:
    1. Look up both records' entities and count the candidates that match.
    2. Join the records along the matching candidates into connected
       components; each lies inside one entity, so the pairs within the
       components are the reachable true matches.
    3. Count the true matches from the entity sizes and all pairs from the
       record count, and derive the ratios.
    """
    n_candidates = len(left)
    matching = true[left].to_numpy() == true[right].to_numpy()
    found = int(matching.sum())
    left_at = true.index.get_indexer(left[matching])
    right_at = true.index.get_indexer(right[matching])
    graph = coo_matrix((np.ones(found), (left_at, right_at)), shape=(len(true), len(true)))
    _, component = connected_components(graph, directed=False)
    reachable = count_pairs(pd.Series(component).value_counts())
    true_pairs = count_pairs(true.value_counts())
    all_pairs = len(true) * (len(true) - 1) // 2
    return {
        "pair_completeness": found / true_pairs if true_pairs else 1.0,
        "reachable_pair_completeness": reachable / true_pairs if true_pairs else 1.0,
        "reduction_ratio": 1 - n_candidates / all_pairs if all_pairs else 1.0,
        "pair_quality": found / n_candidates if n_candidates else 1.0,
        "n_candidates": n_candidates,
    }
