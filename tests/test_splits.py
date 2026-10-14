import numpy as np
import pandas as pd
import pytest

from dedup_bench.splits import SPLITS, entity_split, load_experiment

SHARES = {"train": 0.6, "validation": 0.2, "test": 0.2}


def random_truth(n_entities: int, seed: int) -> pd.Series:
    sizes = np.random.default_rng(seed).integers(1, 5, n_entities)
    entities = np.repeat([f"e{i}" for i in range(n_entities)], sizes)
    return pd.Series(entities, index=[f"r{i}" for i in range(len(entities))])


def test_singleton_entities_split_exactly() -> None:
    # Ten one-record entities have midpoints 0.05, 0.15, ..., 0.95: six fall
    # below 0.6, two between 0.6 and 0.8, two above.
    truth = pd.Series([f"e{i}" for i in range(10)], index=[f"r{i}" for i in range(10)])
    counts = entity_split(truth, seed=0, shares=SHARES).value_counts()
    assert counts.to_dict() == {"train": 6, "validation": 2, "test": 2}


def test_no_entity_spans_two_splits() -> None:
    truth = random_truth(500, seed=1)
    split = entity_split(truth, seed=3, shares=SHARES)
    assert split.index.equals(truth.index)
    assert (
        pd.DataFrame({"entity": truth, "split": split}).groupby("entity")["split"].nunique() == 1
    ).all()


def test_record_shares_are_close_to_the_target() -> None:
    truth = random_truth(2000, seed=2)
    shares = entity_split(truth, seed=0, shares=SHARES).value_counts(normalize=True)
    for name in SPLITS:
        assert shares[name] == pytest.approx(SHARES[name], abs=0.01)


def test_same_seed_same_split_whatever_the_record_order() -> None:
    truth = random_truth(300, seed=4)
    split = entity_split(truth, seed=7, shares=SHARES)
    shuffled = truth.sample(frac=1, random_state=0)
    assert entity_split(shuffled, seed=7, shares=SHARES).reindex(truth.index).equals(split)
    assert not entity_split(truth, seed=8, shares=SHARES).equals(split)


def test_shares_must_name_the_splits_and_add_up_to_one() -> None:
    truth = random_truth(10, seed=0)
    with pytest.raises(ValueError, match="add up to 1"):
        entity_split(truth, seed=0, shares={"train": 0.8, "test": 0.2})
    with pytest.raises(ValueError, match="add up to 1"):
        entity_split(truth, seed=0, shares={"train": 0.6, "validation": 0.2, "test": 0.3})


def test_experiment_config_uses_60_20_20() -> None:
    assert load_experiment()["split"] == SHARES
