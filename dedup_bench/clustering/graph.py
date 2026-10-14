"""
Graph clustering of the scored candidate pairs.

Input: a split's records and candidate pairs with a similarity score.

Output: the predicted cluster of every record. Records in no kept pair are
singletons.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as scipy_components


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
