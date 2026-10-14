import json

import numpy as np
import pandas as pd
import pytest

from dedup_bench.blocking.tfidf_knn import nearest_neighbours, neighbour_pairs
from dedup_bench.config import DATASETS_CONFIG, DATASETS_SMOKE_CONFIG, PROCESSED_DATA_DIR
from dedup_bench.splits import load_experiment

TFIDF = {"analyzer": "char_wb", "ngram_range": [2, 4], "sublinear_tf": True}


def pair_set(pairs: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(pairs["left"], pairs["right"]))


def test_neighbour_pairs_keeps_one_sided_neighbours() -> None:
    # r2's nearest neighbour is r0, but r0's is r1: the pair (0, 2) is found
    # only from r2's side and must still be kept.
    neighbours = np.array([[1], [0], [0]])
    similarity = np.array([[0.9], [0.9], [0.5]])
    pairs = neighbour_pairs(neighbours, similarity, top_k=1)
    assert pair_set(pairs) == {(0, 1), (0, 2)}
    assert pairs.set_index(["left", "right"]).loc[(0, 2), "similarity"] == 0.5


def test_neighbour_pairs_cuts_to_top_k() -> None:
    neighbours = np.array([[1, 2], [0, 2], [1, 0]])
    similarity = np.array([[0.9, 0.1], [0.9, 0.8], [0.8, 0.1]])
    assert pair_set(neighbour_pairs(neighbours, similarity, top_k=1)) == {(0, 1), (1, 2)}
    assert pair_set(neighbour_pairs(neighbours, similarity, top_k=2)) == {(0, 1), (0, 2), (1, 2)}


def test_nearest_neighbours_finds_the_near_duplicate() -> None:
    texts = pd.Series(["apple pie", "apple pie!", "zebra crossing"])
    found, similarity = nearest_neighbours(texts, k=5, **TFIDF)
    assert found.shape == (3, 2)
    assert found[0, 0] == 1 and found[1, 0] == 0
    assert similarity[0, 0] > 0.5 > similarity[0, 1]


def test_nearest_neighbours_of_a_single_record() -> None:
    found, similarity = nearest_neighbours(pd.Series(["only"]), k=5, **TFIDF)
    assert found.shape == similarity.shape == (1, 0)


def test_dataset_configs_cover_the_experiment() -> None:
    datasets = json.loads(DATASETS_CONFIG.read_text(encoding="utf-8"))
    smoke = json.loads(DATASETS_SMOKE_CONFIG.read_text(encoding="utf-8"))
    assert set(load_experiment()["datasets"]) <= set(datasets)
    assert all(
        spec["fields"] and spec["blocking"] for n, spec in smoke.items() if n != "description"
    )


@pytest.mark.parametrize(
    "name",
    [n for n in json.loads(DATASETS_CONFIG.read_text(encoding="utf-8")) if n != "description"],
)
def test_text_fields_exist_in_prepared_records(name: str) -> None:
    path = PROCESSED_DATA_DIR / name / "records.parquet"
    if not path.exists():
        pytest.skip(f"{name} not prepared; run make data")
    fields = json.loads(DATASETS_CONFIG.read_text(encoding="utf-8"))[name]["text_fields"]
    assert set(fields) <= set(pd.read_parquet(path).columns)
