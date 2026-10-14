"""
Entity-disjoint train / validation / test split of a dataset's records.

Input: the true entity of every record, and the split shares and seeds from
``configs/experiment.json``.

Output: the split name (``train``, ``validation`` or ``test``) of every
record. Whole entities are assigned to one split, so no entity spans two, and
the splits hold close to their shares of the records. The split is computed,
not saved: the same seed always gives the same split.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from dedup_bench.config import EXPERIMENT_CONFIG

SPLITS = ("train", "validation", "test")


def load_experiment() -> dict[str, Any]:
    """
    Read ``configs/experiment.json``.

    Returns
    -------
    dict
        The datasets, seeds and split shares.

    Notes
    -----
    Processing:
    1. Load the JSON file.
    """
    return json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))


def entity_split(truth: pd.Series, seed: int, shares: dict[str, float]) -> pd.Series:
    """
    Assign whole entities to train, validation and test.

    Parameters
    ----------
    truth : pd.Series of str, shape (n_records,)
        True entity of every record, indexed by record ID.
    seed : int
        Seed of the entity shuffle.
    shares : dict of str to float
        Share of the records for each name in ``SPLITS``; must add up to 1.

    Returns
    -------
    pd.Series of str, shape (n_records,)
        Split name of every record, indexed like ``truth``.

    Raises
    ------
    ValueError
        If ``shares`` does not name exactly the three splits or does not add
        up to 1.

    Notes
    -----
    Processing:
    1. Shuffle the entities with the seed (sorted first, so the order of
       ``truth`` does not matter).
    2. Lay the shuffled entities end to end on a line of length 1, each taking
       its share of the records.
    3. Put each entity in the split whose interval holds its midpoint, so a
       large entity goes where most of its records fall.
    """
    if set(shares) != set(SPLITS) or not np.isclose(sum(shares.values()), 1.0):
        raise ValueError(f"shares must give {SPLITS} and add up to 1, got {shares}")
    sizes = truth.value_counts().sort_index()
    sizes = sizes.iloc[np.random.default_rng(seed).permutation(len(sizes))]

    ends = sizes.cumsum() / len(truth)
    midpoints = ends - sizes / len(truth) / 2
    boundaries = np.cumsum([shares[name] for name in SPLITS])[:-1]
    split_of_entity = pd.Series(
        np.array(SPLITS)[np.searchsorted(boundaries, midpoints, side="right")],
        index=sizes.index,
    )
    return pd.Series(split_of_entity[truth].to_numpy(), index=truth.index)
