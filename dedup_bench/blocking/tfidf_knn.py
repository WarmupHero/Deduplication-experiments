"""
TF-IDF nearest-neighbour blocking.

Input: one text per record.

Output: candidate pairs, each record paired with its ``top_k`` most similar
records by cosine similarity of TF-IDF character n-grams. The neighbours are
found once for the largest ``top_k`` and cut down for smaller ones, so trying
several values costs one search.
"""

import numpy as np
import pandas as pd


def nearest_neighbours(
    texts: pd.Series,
    k: int,
    analyzer: str,
    ngram_range: list[int],
    sublinear_tf: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Each record's ``k`` most similar records by TF-IDF cosine similarity.

    Parameters
    ----------
    texts : pd.Series of str, shape (n_records,)
        One text per record.
    k : int
        Neighbours per record; fewer when there are not ``k`` other records.
    analyzer, ngram_range, sublinear_tf
        ``TfidfVectorizer`` settings (``blocking.tfidf_knn`` in
        ``configs/models.json``).

    Returns
    -------
    neighbours : np.ndarray of int, shape (n_records, min(k, n_records - 1))
        Positions of each record's neighbours, most similar first.
    similarity : np.ndarray of float, same shape
        Their cosine similarities.

    Notes
    -----
    Processing:
    1. Vectorise the texts, fitting the vocabulary and IDF on these texts only.
    2. Query every record's nearest neighbours by cosine distance, excluding
       the record itself.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(texts) - 1)
    if k < 1:
        return np.empty((len(texts), 0), dtype=int), np.empty((len(texts), 0))
    vectors = TfidfVectorizer(
        analyzer=analyzer, ngram_range=tuple(ngram_range), sublinear_tf=sublinear_tf
    ).fit_transform(texts)
    # kneighbors without X leaves each point out of its own neighbours.
    distances, neighbours = (
        NearestNeighbors(n_neighbors=k, metric="cosine").fit(vectors).kneighbors()
    )
    return neighbours, 1 - distances


def neighbour_pairs(neighbours: np.ndarray, similarity: np.ndarray, top_k: int) -> pd.DataFrame:
    """
    Unordered candidate pairs from each record's ``top_k`` nearest neighbours.

    Parameters
    ----------
    neighbours, similarity : np.ndarray, shape (n_records, k)
        Output of ``nearest_neighbours``, with ``k >= top_k`` or all other
        records.
    top_k : int
        Neighbours per record to keep.

    Returns
    -------
    pd.DataFrame, shape (n_pairs, 3)
        ``left`` and ``right``, the positions of the two records
        (``left < right``), and their ``similarity``; each pair once.

    Notes
    -----
    Processing:
    1. Keep the first ``top_k`` neighbours of every record.
    2. Order each pair's positions, so a pair found from both ends is the
       same row, and drop the duplicates.
    """
    neighbours, similarity = neighbours[:, :top_k], similarity[:, :top_k]
    query = np.repeat(np.arange(len(neighbours)), neighbours.shape[1])
    found = neighbours.ravel()
    pairs = pd.DataFrame(
        {
            "left": np.minimum(query, found),
            "right": np.maximum(query, found),
            "similarity": similarity.ravel(),
        }
    )
    return pairs.drop_duplicates(["left", "right"]).reset_index(drop=True)
