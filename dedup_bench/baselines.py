"""
Baselines every other method must beat.

Input: a split's normalised record texts and its frozen candidate pairs.

Output: the predicted cluster of every record.
"""

import numpy as np
import pandas as pd

from dedup_bench.clustering.graph import components


def exact_keys(texts: pd.Series, token_sort: bool) -> pd.Series:
    """
    The comparison key of every record.

    Parameters
    ----------
    texts : pd.Series of str, shape (n_records,)
        Normalised text per record ID; nulls allowed.
    token_sort : bool
        Sort the words, so that word order does not matter
        (``smith john`` and ``john smith`` get the same key).

    Returns
    -------
    pd.Series of str, shape (n_records,)
        The key per record ID; null for an empty text, which matches nothing.

    Notes
    -----
    Processing:
    1. Sort each text's words when ``token_sort`` is set.
    2. Turn empty texts into nulls.
    """
    keys = texts.astype("string")
    if token_sort:
        keys = keys.str.split().map(lambda words: " ".join(sorted(words)), na_action="ignore")
    keys = keys.astype("string")
    return keys.mask(keys == "")


def equal_keys(keys: pd.Series, pairs: pd.DataFrame) -> np.ndarray:
    """
    Whether the two records of each pair have the same, non-null key.

    Parameters
    ----------
    keys : pd.Series of str, shape (n_records,)
        Key per record ID, from ``exact_keys``.
    pairs : pd.DataFrame, shape (n_pairs, 2+)
        ``left_id`` and ``right_id`` of each pair.

    Returns
    -------
    np.ndarray of bool, shape (n_pairs,)
        True where both keys are equal; False where they differ or either is
        null.

    Notes
    -----
    Processing:
    1. Look up both keys and compare them as pandas strings, whose missing
       values compare as unknown, then count unknown as False.
    """
    left = keys[pairs["left_id"]].reset_index(drop=True)
    right = keys[pairs["right_id"]].reset_index(drop=True)
    return (left == right).fillna(False).to_numpy(dtype=bool)


def exact_key(texts: pd.Series, pairs: pd.DataFrame, token_sort: bool) -> pd.Series:
    """
    Cluster the candidate pairs whose two records have identical keys.

    Parameters
    ----------
    texts : pd.Series of str, shape (n_records,)
        Normalised text per record ID of the split.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    token_sort : bool
        See ``exact_keys``.

    Returns
    -------
    pd.Series of int, shape (n_records,)
        Cluster label of every record, indexed by record ID.

    Notes
    -----
    Processing:
    1. Compute every record's key.
    2. Keep the candidate pairs whose keys are equal and not null.
    3. Label the connected components of the kept pairs.
    """
    kept = pairs[equal_keys(exact_keys(texts, token_sort), pairs)]
    return components(pd.Index(texts.index), kept["left_id"], kept["right_id"])
