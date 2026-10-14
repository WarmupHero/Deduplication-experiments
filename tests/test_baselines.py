import pandas as pd

from dedup_bench import experiment
from dedup_bench.baselines import exact_key, exact_keys

TEXTS = pd.Series(
    ["john smith", "smith john", "john smith", "jane doe", None, None],
    index=["r0", "r1", "r2", "r3", "r4", "r5"],
)
# r0 and r2 share a key but are not a candidate pair; r4 and r5 are both empty.
PAIRS = pd.DataFrame(
    {
        "left_id": ["r0", "r1", "r3", "r4"],
        "right_id": ["r1", "r2", "r4", "r5"],
        "similarity": [0.9, 0.9, 0.1, 0.0],
    }
)


def groups(labels: pd.Series) -> set[frozenset[str]]:
    return {frozenset(ids) for ids in labels.groupby(labels).groups.values()}


def test_exact_keys_with_and_without_token_sort() -> None:
    texts = pd.Series(["john smith", "smith john", "", None])
    assert exact_keys(texts, token_sort=False).iloc[:2].tolist() == ["john smith", "smith john"]
    assert exact_keys(texts, token_sort=True).iloc[:2].tolist() == ["john smith", "john smith"]
    assert exact_keys(texts, token_sort=True).iloc[2:].isna().all()


def test_exact_key_without_token_sort_matches_only_identical_candidate_pairs() -> None:
    # No candidate pair has identical texts: r0 and r2 do, but are not a pair.
    labels = exact_key(TEXTS, PAIRS, token_sort=False)
    assert labels.index.equals(TEXTS.index)
    assert len(groups(labels)) == 6


def test_exact_key_with_token_sort_chains_reordered_names() -> None:
    # r0-r1 and r1-r2 now have equal keys, so r0, r1 and r2 form one cluster;
    # the two empty texts never match.
    labels = exact_key(TEXTS, PAIRS, token_sort=True)
    assert groups(labels) == {
        frozenset({"r0", "r1", "r2"}),
        frozenset({"r3"}),
        frozenset({"r4"}),
        frozenset({"r5"}),
    }


def test_run_exact_key_scores_pairs_by_key_equality() -> None:
    records = pd.DataFrame({"record_id": TEXTS.index})
    target = experiment.SplitData(records, PAIRS, TEXTS, truth=None)
    prediction = experiment.run_exact_key(target, target, {"token_sort": True}, seed=0)
    assert prediction.pairs["score"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert prediction.pairs.columns.tolist() == ["left_id", "right_id", "score"]
