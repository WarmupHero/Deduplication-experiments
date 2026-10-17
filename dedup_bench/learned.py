"""
Pair classifiers that learn from labels: active learning and the supervised ceiling.

Input: a split's normalised records and frozen candidate pairs, the train
split's records, pairs and truth, and the dataset's ``match_fields`` from
``configs/datasets.json``.

Output: a match probability for every candidate pair of the target split.
Both methods describe a pair by the same features: the per-field similarities
ECM uses (``probabilistic.comparison_vectors``) plus the pair's TF-IDF
similarity. Labels come only from train: the supervised ceiling sees every
train pair's label, the active learner only the ones it asks a simulated
oracle for. Training does not depend on the match threshold the experiment
tunes, so labelled sets and pair scores are cached per process and reused
across tuning trials.
"""

from typing import Any

import numpy as np
import pandas as pd

from dedup_bench.probabilistic import comparison_vectors, content_key

# Labelled-pair orders and pair scores so far in this process, by content key.
_ORDERS: dict[str, np.ndarray] = {}
_SCORES: dict[str, np.ndarray] = {}

# Settings of the active learner that steer the querying, not the classifier.
QUERY_SETTINGS = ("label_budget", "initial_labels", "query_batch_size")


def pair_features(
    records: pd.DataFrame, pairs: pd.DataFrame, match_fields: dict[str, str]
) -> np.ndarray:
    """
    Feature matrix of a split's candidate pairs.

    Parameters
    ----------
    records : pd.DataFrame, shape (n_records, n_columns)
        The split's records.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        Its candidate pairs with their TF-IDF ``similarity``.
    match_fields : dict of str to str
        Fields to compare and how.

    Returns
    -------
    np.ndarray of float, shape (n_pairs, n_fields + 1)
        Per-field similarities in [0, 1], then the TF-IDF similarity.

    Notes
    -----
    Processing:
    1. Take the cached comparison vectors and append the TF-IDF similarity.
    """
    vectors = comparison_vectors(records, pairs, match_fields).to_numpy(dtype=float)
    return np.column_stack([vectors, pairs["similarity"].to_numpy(dtype=float)])


def oracle_labels(pairs: pd.DataFrame, truth: pd.Series) -> np.ndarray:
    """
    The simulated oracle's answer for every pair: do its records share an entity?

    Parameters
    ----------
    pairs : pd.DataFrame, shape (n_pairs, >= 2)
        Pairs with ``left_id`` and ``right_id``.
    truth : pd.Series of str, shape (n_records,)
        True entity per record ID.

    Returns
    -------
    np.ndarray of bool, shape (n_pairs,)
        True where both records belong to the same entity.

    Notes
    -----
    Processing:
    1. Look up both records' entities and compare.
    """
    return truth[pairs["left_id"]].to_numpy() == truth[pairs["right_id"]].to_numpy()


def classifier(model: str, params: dict[str, Any], seed: int) -> Any:
    """
    An unfitted scikit-learn classifier.

    Parameters
    ----------
    model : str
        ``random_forest`` or ``hist_gradient_boosting``.
    params : dict
        Its settings, by scikit-learn argument name.
    seed : int
        ``random_state``.

    Returns
    -------
    sklearn classifier
        ``RandomForestClassifier`` (on all cores) or ``HistGradientBoostingClassifier``.

    Raises
    ------
    ValueError
        If the model is unknown.

    Notes
    -----
    Processing:
    1. Build the named classifier with the settings and seed.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    if model == "random_forest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
    if model == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params, random_state=seed)
    raise ValueError(f"unknown model {model!r}")


def match_probability(fitted: Any, features: np.ndarray) -> np.ndarray:
    """
    Probability of the match class, also when training saw only one class.

    Parameters
    ----------
    fitted : sklearn classifier
        Fitted on boolean labels.
    features : np.ndarray of float, shape (n_pairs, n_features)
        Pairs to score.

    Returns
    -------
    np.ndarray of float, shape (n_pairs,)
        Match probability per pair (constant 0 or 1 after one-class training).

    Notes
    -----
    Processing:
    1. Pick the column of class True, or a constant if it never occurred.
    """
    classes = list(fitted.classes_)
    if True not in classes:
        return np.zeros(len(features))
    return fitted.predict_proba(features)[:, classes.index(True)]


def active_learning_order(
    features: np.ndarray,
    oracle: np.ndarray,
    params: dict[str, Any],
    initial_labels: int,
    query_batch_size: int,
    budget: int,
    seed: int,
) -> np.ndarray:
    """
    Which pool pairs the active learner asks about, in the order it asks.

    Parameters
    ----------
    features : np.ndarray of float, shape (n_pool, n_features)
        Features of the unlabelled pool (train's candidate pairs).
    oracle : np.ndarray of bool, shape (n_pool,)
        The oracle's answers; only the entries of queried pairs are read.
    params : dict
        Random forest settings.
    initial_labels, query_batch_size : int
        Random pairs labelled first, and pairs queried per round.
    budget : int
        Total labels to use (capped at the pool size).
    seed : int
        Seed of the random picks and the forest.

    Returns
    -------
    np.ndarray of int, shape (min(budget, n_pool),)
        Pool indices in query order. A larger budget extends a smaller one's
        order, so results for every budget come from one run.

    Notes
    -----
    Processing:
    1. Label ``initial_labels`` pairs from a seeded permutation of the pool.
    2. While labels remain, fit the forest on the labelled pairs and query the
       ``query_batch_size`` unlabelled pairs whose match probability is
       closest to 0.5. While only one class has been seen, a forest cannot
       rank uncertainty, so query the pairs most likely of the other class:
       the most similar (mean feature) if all were non-matches, else the least.
       Ties are broken at random, never by row order.
    """
    rng = np.random.default_rng(seed)
    budget = min(budget, len(features))
    order = list(rng.permutation(len(features))[: min(initial_labels, budget)])
    unlabelled = np.ones(len(features), dtype=bool)
    unlabelled[order] = False
    while len(order) < budget:
        size = min(query_batch_size, budget - len(order))
        pool = np.flatnonzero(unlabelled)
        known = oracle[order]
        if known.all() or not known.any():
            similarity = features[pool].mean(axis=1)
            priority = similarity if known.all() else -similarity
        else:
            fitted = classifier("random_forest", params, seed).fit(features[order], known)
            priority = np.abs(match_probability(fitted, features[pool]) - 0.5)
        query = pool[np.lexsort((rng.random(len(pool)), priority))[:size]]
        order.extend(query)
        unlabelled[query] = False
    return np.array(order, dtype=int)


def active_learning_scores(
    train: pd.DataFrame,
    train_pairs: pd.DataFrame,
    train_truth: pd.Series,
    target: pd.DataFrame,
    pairs: pd.DataFrame,
    match_fields: dict[str, str],
    settings: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """
    Match probabilities from a random forest trained on the pairs it chose to label.

    Parameters
    ----------
    train, target : pd.DataFrame
        The train records and the records of the split to score.
    train_pairs, pairs : pd.DataFrame
        The candidate pairs of train (the pool) and of the target split.
    train_truth : pd.Series of str
        Train's true entities, which only the oracle reads.
    match_fields : dict of str to str
        Fields to compare and how.
    settings : dict
        ``QUERY_SETTINGS`` plus the random forest's settings.
    seed : int
        Seed of the querying and the forest.

    Returns
    -------
    np.ndarray of float, shape (n_pairs,)
        Match probability of each target candidate pair.

    Notes
    -----
    Processing:
    1. Compute the pool's features and the oracle's answers.
    2. Get the query order, reusing a cached longer one when possible.
    3. Fit the forest on the first ``label_budget`` labelled pairs and score
       the target's pairs.
    """
    params = {k: v for k, v in settings.items() if k not in QUERY_SETTINGS}
    budget = settings["label_budget"]
    pool = pair_features(train, train_pairs, match_fields)
    oracle = oracle_labels(train_pairs, train_truth)
    order_key = content_key(
        "active", train["record_id"], train_pairs[["left_id", "right_id"]], match_fields,
        params, settings["initial_labels"], settings["query_batch_size"], seed,
    )  # fmt: skip
    cached = _ORDERS.get(order_key)
    if cached is None or len(cached) < min(budget, len(pool)):
        _ORDERS[order_key] = cached = active_learning_order(
            pool, oracle, params, settings["initial_labels"], settings["query_batch_size"],
            budget, seed,
        )  # fmt: skip
    score_key = content_key(order_key, budget, target["record_id"], pairs[["left_id", "right_id"]])
    if score_key not in _SCORES:
        labelled = cached[:budget]
        fitted = classifier("random_forest", params, seed).fit(pool[labelled], oracle[labelled])
        _SCORES[score_key] = match_probability(fitted, pair_features(target, pairs, match_fields))
    return _SCORES[score_key]


def supervised_scores(
    model: str,
    train: pd.DataFrame,
    train_pairs: pd.DataFrame,
    train_truth: pd.Series,
    target: pd.DataFrame,
    pairs: pd.DataFrame,
    match_fields: dict[str, str],
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """
    Match probabilities from a classifier trained on every labelled train pair.

    Parameters
    ----------
    model : str
        ``random_forest`` or ``hist_gradient_boosting``.
    train, target : pd.DataFrame
        The train records and the records of the split to score.
    train_pairs, pairs : pd.DataFrame
        The candidate pairs of train and of the target split.
    train_truth : pd.Series of str
        Train's true entities, the source of the labels.
    match_fields : dict of str to str
        Fields to compare and how.
    params : dict
        The classifier's settings.
    seed : int
        ``random_state``.

    Returns
    -------
    np.ndarray of float, shape (n_pairs,)
        Match probability of each target candidate pair.

    Notes
    -----
    Processing:
    1. Fit the classifier on train's pair features and labels.
    2. Score the target's pairs; only the scores are cached, as fitted
       forests on large pools take gigabytes.
    """
    key = content_key(
        model, train["record_id"], train_pairs[["left_id", "right_id"]], match_fields, params,
        seed, target["record_id"], pairs[["left_id", "right_id"]],
    )  # fmt: skip
    if key not in _SCORES:
        fitted = classifier(model, params, seed).fit(
            pair_features(train, train_pairs, match_fields),
            oracle_labels(train_pairs, train_truth),
        )
        _SCORES[key] = match_probability(fitted, pair_features(target, pairs, match_fields))
    return _SCORES[key]
