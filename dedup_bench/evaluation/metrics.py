"""
Quality metrics for a predicted clustering of records into entities.

Input: two membership vectors over the same records, the predicted cluster
and the true entity of every record, as pandas Series indexed by record ID.

Output: precision, recall and F1 scores. Pairwise scores count every pair of
records placed in the same cluster, so an entity with k records contributes
k (k - 1) / 2 pairs.
"""

import pandas as pd


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
