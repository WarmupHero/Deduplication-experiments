import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dedup_bench import experiment
from dedup_bench.blocking import candidates
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
    pd.DataFrame({"record_id": ids, "source": "tiny", "name": texts}).to_parquet(
        processed / "records.parquet"
    )
    entities = [f"e{i // 2}" for i in range(len(texts))]
    pd.DataFrame({"record_id": ids, "original_id": ids, "entity_id": entities}).to_parquet(
        processed / "truth.parquet"
    )
    datasets_config = tmp_path / "datasets.json"
    datasets_config.write_text(json.dumps({"tiny": {"text_fields": ["name"]}}))

    monkeypatch.setattr("dedup_bench.dataset.PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(candidates, "INTERIM_DATA_DIR", tmp_path / "interim")
    monkeypatch.setattr(candidates, "DATASETS_CONFIG", datasets_config)
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
