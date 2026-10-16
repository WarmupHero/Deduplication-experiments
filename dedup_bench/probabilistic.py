"""
Fellegi-Sunter models trained without labels: Splink and recordlinkage's ECM.

Input: a split's normalised records and frozen candidate pairs, the train
records to learn from, and the dataset's field lists from
``configs/datasets.json``: ``match_fields`` (field -> ``jaro_winkler``,
``levenshtein`` or ``exact``) and, for Splink, ``em_blocking``.

Output: a match probability for every candidate pair of the target split.
Neither model sees a label: both learn how often each field agrees between
matches and non-matches from the train records alone. Training does not depend
on the match threshold the experiment tunes, so trained models and scores are
cached per process and reused across tuning trials.
"""

import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

# Trained models and pair scores so far in this process, by content key.
_MODELS: dict[str, Any] = {}
_SCORES: dict[str, np.ndarray] = {}

# recordlinkage's name for each string comparison of match_fields.
RECORDLINKAGE_METHODS = {"jaro_winkler": "jarowinkler", "levenshtein": "levenshtein"}


def content_key(*parts: Any) -> str:
    """
    A short hash identifying the given values, for the caches.

    Parameters
    ----------
    *parts : Any
        JSON-serialisable values, or DataFrames / Series (hashed by content).

    Returns
    -------
    str
        The first 16 hex digits of a SHA-256 over all parts.

    Notes
    -----
    Processing:
    1. Hash pandas objects with ``pd.util.hash_pandas_object`` and everything
       else as sorted JSON, then hash the concatenation.
    """
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, pd.DataFrame | pd.Series):
            digest.update(pd.util.hash_pandas_object(part, index=False).to_numpy().tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def field_frame(records: pd.DataFrame, match_fields: dict[str, str]) -> pd.DataFrame:
    """
    The record IDs and match fields, with missing values as None.

    Parameters
    ----------
    records : pd.DataFrame, shape (n_records, n_columns)
        Normalised records with ``record_id``.
    match_fields : dict of str to str
        Fields to compare and how.

    Returns
    -------
    pd.DataFrame, shape (n_records, 1 + n_fields)
        ``record_id`` and the match fields as Python objects.

    Notes
    -----
    Processing:
    1. Select the columns and replace pandas nulls with None.
    """
    frame = records[["record_id", *match_fields]].astype(object)
    return frame.where(frame.notna(), None)


def splink_comparisons(
    match_fields: dict[str, str],
    jaro_winkler_thresholds: list[float],
    levenshtein_thresholds: list[int],
) -> list[Any]:
    """
    One Splink comparison per match field.

    Parameters
    ----------
    match_fields : dict of str to str
        Field -> ``jaro_winkler``, ``levenshtein`` or ``exact``.
    jaro_winkler_thresholds : list of float
        Similarity levels for ``jaro_winkler`` fields, e.g. [0.9, 0.7].
    levenshtein_thresholds : list of int
        Edit-distance levels for ``levenshtein`` fields, e.g. [1, 2].

    Returns
    -------
    list of splink comparison creators
        In the order of ``match_fields``.

    Raises
    ------
    ValueError
        If a field's comparison is not one of the three names.

    Notes
    -----
    Processing:
    1. Map each comparison name to Splink's ``JaroWinklerAtThresholds``,
       ``LevenshteinAtThresholds`` or ``ExactMatch`` for that field.
    """
    import splink.comparison_library as cl

    builders = {
        "jaro_winkler": lambda field: cl.JaroWinklerAtThresholds(field, jaro_winkler_thresholds),
        "levenshtein": lambda field: cl.LevenshteinAtThresholds(field, levenshtein_thresholds),
        "exact": cl.ExactMatch,
    }
    unknown = set(match_fields.values()) - set(builders)
    if unknown:
        raise ValueError(f"unknown comparison(s) {sorted(unknown)}; use {sorted(builders)}")
    return [builders[how](field) for field, how in match_fields.items()]


def train_splink(
    records: pd.DataFrame,
    match_fields: dict[str, str],
    em_blocking: list[str],
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """
    Train a Splink model on records without labels.

    Parameters
    ----------
    records : pd.DataFrame, shape (n_records, n_columns)
        The train records.
    match_fields : dict of str to str
        Fields to compare and how.
    em_blocking : list of str
        Field names or SQL expressions; one EM session per rule.
    settings : dict
        ``jaro_winkler_thresholds``, ``levenshtein_thresholds``,
        ``u_max_pairs``, ``max_iterations`` and ``em_convergence``.
    seed : int
        Seed of the random pair sampling for u.

    Returns
    -------
    dict
        The trained model, as Splink's settings dictionary.

    Notes
    -----
    Processing:
    1. Register the records with DuckDB and build the comparisons.
    2. Estimate u by seeded random sampling.
    3. Run one EM session per blocking rule for m; the first also estimates
       the prior probability that two random records match.
    """
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    db_api = DuckDBAPI()
    linker = Linker(
        db_api.register(field_frame(records, match_fields)),
        SettingsCreator(
            link_type="dedupe_only",
            unique_id_column_name="record_id",
            comparisons=splink_comparisons(
                match_fields,
                settings["jaro_winkler_thresholds"],
                settings["levenshtein_thresholds"],
            ),
            blocking_rules_to_generate_predictions=[],
            max_iterations=settings["max_iterations"],
            em_convergence=settings["em_convergence"],
        ),
        log_level=logging.WARNING,
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=settings["u_max_pairs"], seed=seed)
    for number, rule in enumerate(em_blocking):
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on(rule),
            populate_probability_two_random_records_match_from_trained_values=number == 0,
        )
    return linker.misc.save_model_to_json()


def pair_probabilities(
    predictions: pd.DataFrame, pairs: pd.DataFrame, left: str, right: str, column: str
) -> np.ndarray:
    """
    Align scored pairs to the candidate pairs, whichever way round they were scored.

    Parameters
    ----------
    predictions : pd.DataFrame
        Scored pairs with record IDs in ``left`` and ``right`` and a ``column``.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        The candidate pairs (``left_id``, ``right_id``).
    left, right, column : str
        Column names in ``predictions``.

    Returns
    -------
    np.ndarray of float, shape (n_pairs,)
        The score of each candidate pair, in the order of ``pairs``.

    Raises
    ------
    ValueError
        If a candidate pair has no score.

    Notes
    -----
    Processing:
    1. Key every scored pair by its two IDs in sorted order and look up each
       candidate pair the same way.
    """
    keys = pd.Index(
        [f"{min(a, b)}|{max(a, b)}" for a, b in zip(predictions[left], predictions[right])]
    )
    scores = pd.Series(predictions[column].to_numpy(), index=keys)
    wanted = [f"{min(a, b)}|{max(a, b)}" for a, b in zip(pairs["left_id"], pairs["right_id"])]
    found = scores.reindex(wanted)
    if found.isna().any():
        raise ValueError(f"{int(found.isna().sum())} candidate pairs were not scored")
    return found.to_numpy(dtype=float)


def score_splink(
    model: dict[str, Any],
    records: pd.DataFrame,
    pairs: pd.DataFrame,
    match_fields: dict[str, str],
) -> np.ndarray:
    """
    Score candidate pairs with a trained Splink model.

    Parameters
    ----------
    model : dict
        A trained model from ``train_splink``.
    records : pd.DataFrame, shape (n_records, n_columns)
        The split's records.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        Its frozen candidate pairs.
    match_fields : dict of str to str
        Fields the model compares.

    Returns
    -------
    np.ndarray of float, shape (n_pairs,)
        Match probability of each candidate pair.

    Notes
    -----
    Processing:
    1. Load the model on the split's records.
    2. Register the candidate pairs as Splink's blocked pairs, so exactly
       these pairs are scored.
    3. Predict and align the probabilities to ``pairs``.
    """
    from splink import DuckDBAPI, Linker

    if pairs.empty:
        return np.zeros(0)
    db_api = DuckDBAPI()
    linker = Linker(
        db_api.register(field_frame(records, match_fields)), model, log_level=logging.WARNING
    )
    blocked = pd.DataFrame(
        {
            "match_key": "0",
            "join_key_l": pairs["left_id"].to_numpy(),
            "join_key_r": pairs["right_id"].to_numpy(),
        }
    )
    linker.table_management.register_blocked_pairs_for_predict(db_api.register(blocked))
    predictions = linker.inference.predict().as_pandas_dataframe()
    return pair_probabilities(
        predictions, pairs, "record_id_l", "record_id_r", "match_probability"
    )


def splink_scores(
    train: pd.DataFrame,
    target: pd.DataFrame,
    pairs: pd.DataFrame,
    dataset_config: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """
    Splink match probabilities of a split's candidate pairs, trained on train.

    Parameters
    ----------
    train : pd.DataFrame
        The train records.
    target : pd.DataFrame
        The records of the split to score.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        The target's candidate pairs.
    dataset_config : dict
        The dataset's entry in ``configs/datasets.json``.
    settings : dict
        The Splink settings except the match threshold.
    seed : int
        Seed of the u sampling.

    Returns
    -------
    np.ndarray of float, shape (n_pairs,)
        Match probability of each candidate pair.

    Notes
    -----
    Processing:
    1. Train (or reuse) the model for these train records, fields and settings.
    2. Score (or reuse the scores of) the target's candidate pairs.
    """
    fields = dataset_config["match_fields"]
    model_key = content_key(
        "splink", train["record_id"], fields, dataset_config["em_blocking"], settings, seed
    )
    if model_key not in _MODELS:
        _MODELS[model_key] = train_splink(
            train, fields, dataset_config["em_blocking"], settings, seed
        )
    score_key = content_key(model_key, target["record_id"], pairs[["left_id", "right_id"]])
    if score_key not in _SCORES:
        _SCORES[score_key] = score_splink(_MODELS[model_key], target, pairs, fields)
    return _SCORES[score_key]


def ecm_features(
    records: pd.DataFrame, pairs: pd.DataFrame, match_fields: dict[str, str]
) -> pd.DataFrame:
    """
    recordlinkage comparison vectors of candidate pairs, one column per field.

    Parameters
    ----------
    records : pd.DataFrame, shape (n_records, n_columns)
        The split's records.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        Its candidate pairs.
    match_fields : dict of str to str
        Fields to compare and how.

    Returns
    -------
    pd.DataFrame, shape (n_pairs, n_fields)
        Similarity in [0, 1] per field (exact fields 0 or 1; missing values
        0), indexed by (``left_id``, ``right_id``) in the order of ``pairs``.

    Notes
    -----
    Processing:
    1. Index the pairs and add one string or exact comparison per field.
    2. Compute the comparison vectors.
    """
    import recordlinkage

    frame = field_frame(records, match_fields).set_index("record_id")
    index = pd.MultiIndex.from_arrays([pairs["left_id"], pairs["right_id"]])
    compare = recordlinkage.Compare()
    for field, how in match_fields.items():
        if how == "exact":
            compare.exact(field, field, missing_value=0, label=field)
        else:
            compare.string(
                field, field, method=RECORDLINKAGE_METHODS[how], missing_value=0, label=field
            )
    return compare.compute(index, frame).reindex(index)


def ecm_scores(
    train: pd.DataFrame,
    train_pairs: pd.DataFrame,
    target: pd.DataFrame,
    pairs: pd.DataFrame,
    dataset_config: dict[str, Any],
    settings: dict[str, Any],
) -> np.ndarray:
    """
    ECM match probabilities of a split's candidate pairs, trained on train's pairs.

    Parameters
    ----------
    train, target : pd.DataFrame
        The train records and the records of the split to score.
    train_pairs, pairs : pd.DataFrame
        The candidate pairs of train and of the target split.
    dataset_config : dict
        The dataset's entry in ``configs/datasets.json``.
    settings : dict
        ``binarize``, ``max_iter`` and ``atol``.

    Returns
    -------
    np.ndarray of float, shape (n_pairs,)
        Match probability of each target candidate pair.

    Notes
    -----
    Processing:
    1. Compute (or reuse) the comparison vectors of both splits' pairs.
    2. Fit ECM on train's vectors at the given binarize level, without labels.
    3. Return its match probabilities for the target's vectors.
    """
    import recordlinkage

    fields = dataset_config["match_fields"]
    vectors = {}
    for name, records, split_pairs in (("train", train, train_pairs), ("target", target, pairs)):
        key = content_key(
            "ecm", records["record_id"], split_pairs[["left_id", "right_id"]], fields
        )
        if key not in _MODELS:
            _MODELS[key] = ecm_features(records, split_pairs, fields)
        vectors[name] = _MODELS[key]
    if vectors["target"].empty:
        return np.zeros(0)
    classifier = recordlinkage.ECMClassifier(**settings)
    classifier.fit(vectors["train"])
    return classifier.prob(vectors["target"]).to_numpy(dtype=float)
