"""
Sentence embeddings of record texts, and candidate pairs scored by them.

Input: record texts (one string per record) and a Hugging Face model ID;
``minishlab/`` models load with model2vec, all others with
sentence-transformers.

Output: unit-length embedding vectors, and candidate pairs whose similarity
is the cosine similarity of their two records' embeddings. Encoding uses no
labels, so each distinct text is encoded once per process and reused across
splits, seeds and tuning trials.
"""

import numpy as np
import pandas as pd

# Embedding of every text encoded so far in this process, per model.
_CACHE: dict[str, dict[str, np.ndarray]] = {}


def encode(texts: pd.Series, model_name: str, batch_size: int) -> np.ndarray:
    """
    Embed texts with a model2vec or sentence-transformers model, L2-normalised.

    Parameters
    ----------
    texts : pd.Series of str, shape (n_records,)
        One text per record.
    model_name : str
        Hugging Face model ID; ``minishlab/`` models load with model2vec.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    np.ndarray of float32, shape (n_records, dim)
        Unit-length embeddings.

    Notes
    -----
    Processing:
    1. Load the model with the matching library (downloaded on first use).
    2. Encode and normalise every row to unit length.
    """
    if model_name.startswith("minishlab/"):
        from model2vec import StaticModel

        vectors = StaticModel.from_pretrained(model_name).encode(
            texts.tolist(), batch_size=batch_size
        )
    else:
        from sentence_transformers import SentenceTransformer

        vectors = SentenceTransformer(model_name, device="cpu").encode(
            texts.tolist(), batch_size=batch_size
        )
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def embed(texts: pd.Series, model_name: str, batch_size: int) -> np.ndarray:
    """
    Embed texts, encoding only those not already cached for the model.

    Parameters
    ----------
    texts : pd.Series of str, shape (n_records,)
        One text per record.
    model_name : str
        Hugging Face model ID.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    np.ndarray of float32, shape (n_records, dim)
        Unit-length embeddings, in the order of ``texts``.

    Notes
    -----
    Processing:
    1. Encode the distinct texts missing from the model's cache and add them.
    2. Stack the cached vectors in the order of ``texts``.
    """
    cache = _CACHE.setdefault(model_name, {})
    missing = [text for text in pd.unique(texts) if text not in cache]
    if missing:
        cache.update(zip(missing, encode(pd.Series(missing), model_name, batch_size)))
    return np.stack([cache[text] for text in texts])


def embedding_pairs(
    texts: pd.Series, pairs: pd.DataFrame, model_name: str, batch_size: int
) -> pd.DataFrame:
    """
    Re-score candidate pairs by the cosine similarity of their records' embeddings.

    Parameters
    ----------
    texts : pd.Series of str, shape (n_records,)
        Each record's text, indexed by record ID.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and ``similarity`` of each candidate pair.
    model_name : str
        Hugging Face model ID.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    pd.DataFrame, shape (n_pairs, 3)
        The same pairs, with ``similarity`` replaced by the embeddings' cosine
        similarity, clipped to [0, 1] like the TF-IDF scores.

    Notes
    -----
    Processing:
    1. Embed the texts (unit vectors, so a dot product is the cosine).
    2. Take the row-wise dot product of each pair's two vectors.
    """
    vectors = embed(texts, model_name, batch_size)
    left = vectors[texts.index.get_indexer(pairs["left_id"])]
    right = vectors[texts.index.get_indexer(pairs["right_id"])]
    similarity = np.einsum("ij,ij->i", left, right).clip(0, 1)
    return pairs.assign(similarity=similarity.astype(float))
