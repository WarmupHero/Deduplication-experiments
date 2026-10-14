import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import pytest

from dedup_bench import experiment
from dedup_bench.blocking import candidates
from dedup_bench.config import MODELS_CONFIG
from dedup_bench.dataset import normalise_records
from dedup_bench.tuning import candidates as candidates_of
from dedup_bench.tuning import load_settings, load_space

METHOD = "graph_clustering.connected_components"


@pytest.fixture
def tiny_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    # 40 entities of two near-identical records each, plus their candidate sets.
    names = [hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in range(40)]
    texts = [t for name in names for t in (name, name + "!")]
    ids = [f"r{i:03d}" for i in range(len(texts))]
    processed = tmp_path / "processed" / "tiny"
    processed.mkdir(parents=True)
    records = pd.DataFrame({"record_id": ids, "source": "tiny", "name": texts})
    records.to_parquet(processed / "records.parquet")
    normalise_records(records, []).to_parquet(processed / "records_normalised.parquet")
    entities = [f"e{i // 2}" for i in range(len(texts))]
    pd.DataFrame({"record_id": ids, "original_id": ids, "entity_id": entities}).to_parquet(
        processed / "truth.parquet"
    )
    datasets_config = tmp_path / "datasets.json"
    datasets_config.write_text(json.dumps({"tiny": {"text_fields": ["name"]}}))

    monkeypatch.setattr("dedup_bench.dataset.PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr("dedup_bench.dataset.DATASETS_CONFIG", datasets_config)
    monkeypatch.setattr(candidates, "INTERIM_DATA_DIR", tmp_path / "interim")
    monkeypatch.setattr(experiment, "RESULTS_DIR", tmp_path / "results")
    grid = {**load_settings(), "strategy": "grid"}
    monkeypatch.setattr(experiment, "load_settings", lambda: grid)
    candidates.build_candidates("tiny", seed=0)
    return "tiny"


def test_run_writes_results_that_evaluate_reproduces(
    tiny_dataset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []

    def spy(train, target, params, seed):
        seen.append((train.truth is not None, target.truth is None))
        return experiment.run_connected_components(train, target, params, seed)

    monkeypatch.setitem(experiment.METHODS, METHOD, spy)
    result = experiment.run_experiment(tiny_dataset, METHOD, seed=0)

    out_dir = experiment.results_dir(tiny_dataset, METHOD, 0)
    assert {p.name for p in out_dir.iterdir()} == {
        "result.json",
        "membership.parquet",
        "pairs.parquet",
    }
    # Grid search runs every threshold in the config once, then the test run.
    n_grid = len(
        candidates_of(load_space("graph_clustering", "connected_components")["threshold"])
    )
    assert len(result["trials"]) == n_grid
    assert len(seen) == n_grid + 1
    assert all(train_has_truth and target_blind for train_has_truth, target_blind in seen)

    membership = pd.read_parquet(out_dir / "membership.parquet")
    assert result["test"]["n_entities"] * 2 == len(membership)
    assert result["test"]["b_cubed_f1"] == 1.0
    experiment.evaluate(tiny_dataset, METHOD, 0)


def test_evaluate_detects_a_changed_membership(tiny_dataset: str) -> None:
    experiment.run_experiment(tiny_dataset, METHOD, seed=0)
    path = experiment.results_dir(tiny_dataset, METHOD, 0) / "membership.parquet"
    membership = pd.read_parquet(path)
    membership["cluster"] = [str(i) for i in range(len(membership))]
    membership.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="differ"):
        experiment.evaluate(tiny_dataset, METHOD, 0)


def blocking_config(tmp_path: Path, target: float) -> Path:
    models = json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))
    models["blocking"]["tfidf_knn"]["target_reachable_pair_completeness"] = [target]
    path = tmp_path / f"models_{target}.json"
    path.write_text(json.dumps(models), encoding="utf-8")
    return path


def test_top_k_is_the_smallest_reaching_the_target(
    tiny_dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every duplicate pair is each record's nearest neighbour, so the smallest
    # top_k already reaches any target up to 1.0.
    monkeypatch.setattr(candidates, "MODELS_CONFIG", blocking_config(tmp_path, 1.0))
    report = candidates.build_candidates(tiny_dataset, seed=0)
    assert report["top_k"] == 10
    assert report["splits"]["validation"]["reachable_pair_completeness"] == 1.0
    assert set(report["validation_by_top_k"]) == {"10", "25", "50"}


def test_top_k_falls_back_to_the_largest_when_no_value_reaches_the_target(
    tiny_dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(candidates, "MODELS_CONFIG", blocking_config(tmp_path, 1.01))
    assert candidates.build_candidates(tiny_dataset, seed=0)["top_k"] == 50


def test_candidate_pairs_never_cross_splits(tiny_dataset: str) -> None:
    splits = experiment.load_splits(tiny_dataset, seed=0)
    for name, split in splits.items():
        ids = set(split.records["record_id"])
        assert set(split.pairs["left_id"]) | set(split.pairs["right_id"]) <= ids, name
        assert candidates.candidates_dir(tiny_dataset, 0).name == "seed0"
    all_ids = [set(s.records["record_id"]) for s in splits.values()]
    assert sum(map(len, all_ids)) == len(set().union(*all_ids)) == 80


def test_load_splits_needs_the_candidate_sets(tiny_dataset: str) -> None:
    with pytest.raises(FileNotFoundError, match="make candidates"):
        experiment.load_splits(tiny_dataset, seed=1)


def test_evaluate_clusters_hand_computed() -> None:
    # The clustering of the metric tests: pairwise 1/2, B-cubed 7/9,
    # cluster-level 1/3, ARI 7/22, 3 predicted clusters and 3 entities.
    ids = [f"r{i}" for i in range(6)]
    truth = pd.Series(["a", "a", "a", "b", "b", "c"], index=ids)
    clusters = pd.Series(["x", "x", "y", "y", "y", "z"], index=ids)
    scores = experiment.evaluate_clusters(clusters, truth)
    assert scores["pairwise_f1"] == pytest.approx(1 / 2)
    assert scores["b_cubed_f1"] == pytest.approx(7 / 9)
    assert scores["cluster_f1"] == pytest.approx(1 / 3)
    assert scores["ari"] == pytest.approx(7 / 22)
    assert (scores["n_clusters"], scores["n_entities"]) == (3, 3)
    assert len(scores) == 12


def test_peak_memory_sees_a_large_allocation() -> None:
    import psutil

    before_mb = psutil.Process().memory_info().rss / 2**20
    with experiment.PeakMemory() as memory:
        block = bytearray(200 * 2**20)
        time.sleep(0.5)
    del block
    assert memory.peak_mb >= before_mb + 150


def test_provenance_names_the_commit_and_hashes_every_config() -> None:
    info = experiment.provenance()
    assert len(info["git_commit"]) == 40
    assert isinstance(info["git_dirty"], bool)
    assert {"models.json", "tuning.json", "datasets.json", "experiment.json"} <= set(
        info["config_sha256"]
    )


@pytest.mark.parametrize("method", sorted(experiment.METHODS))
def test_every_method_runs_through_the_runner(tiny_dataset: str, method: str) -> None:
    # The middle candidate of each setting, as in the smoke test.
    section, model = method.split(".")
    space = load_space(section, model)
    setting = {name: (c := candidates_of(spec))[len(c) // 2] for name, spec in space.items()}
    splits = experiment.load_splits(tiny_dataset, seed=0)
    target = experiment.replace(splits["validation"], truth=None)
    prediction = experiment.METHODS[method](splits["train"], target, setting, 0)
    assert prediction.clusters.index.equals(pd.Index(target.records["record_id"]))
    assert prediction.pairs.columns.tolist() == ["left_id", "right_id", "score"]
