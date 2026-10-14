"""
Run one method of the benchmark on one dataset and seed: tune, test and save.

Input: a prepared dataset (``make data``), its candidate sets for the seed
(``make candidates``), the method's search space from ``configs/models.json``
and the tuning settings from ``configs/tuning.json``.

Each tuning trial runs the method with train and validation and scores it on
validation with the tuning objective; the best setting then runs once on test.
A method gets the truth of train only, never of the split it predicts.

Output: in ``reports/results/<dataset>/<method>/seed<seed>/``:

- ``result.json``: the best setting and its validation score, every trial,
  the test metrics, runtime and peak memory, the chosen blocking ``top_k``,
  and the git commit and config hashes the run used
- ``membership.parquet``: ``record_id`` and predicted ``cluster`` on test
- ``pairs.parquet``: ``left_id``, ``right_id`` and the method's ``score`` of
  every test candidate pair

``evaluate`` recomputes the test metrics from ``membership.parquet`` and the
truth, so every number in ``result.json`` can be checked against the files.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Annotated, Any, Self

from loguru import logger
import pandas as pd
import typer

from dedup_bench.blocking.candidates import candidates_dir
from dedup_bench.clustering.graph import connected_components
from dedup_bench.config import CONFIGS_DIR, PROJ_ROOT, RESULTS_DIR
from dedup_bench.dataset import load_prepared
from dedup_bench.evaluation.metrics import (
    adjusted_rand_index,
    b_cubed_scores,
    cluster_scores,
    pairwise_scores,
)
from dedup_bench.splits import SPLITS, entity_split, load_experiment
from dedup_bench.tuning import load_settings, load_space, tune

app = typer.Typer()


@dataclass
class SplitData:
    """
    One split of a dataset as a method sees it.

    Attributes
    ----------
    records : pd.DataFrame, shape (n_records, n_columns)
        The split's normalised records.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        Its frozen candidate pairs: ``left_id``, ``right_id``, ``similarity``.
    truth : pd.Series of str or None, shape (n_records,)
        True entity per record ID; None when the method must not see it.
    """

    records: pd.DataFrame
    pairs: pd.DataFrame
    truth: pd.Series | None


@dataclass
class Prediction:
    """
    A method's output on one split.

    Attributes
    ----------
    clusters : pd.Series, shape (n_records,)
        Predicted cluster of every record, indexed by record ID.
    pairs : pd.DataFrame, shape (n_pairs, 3)
        ``left_id``, ``right_id`` and the method's ``score`` of each pair.
    """

    clusters: pd.Series
    pairs: pd.DataFrame


# A method takes train (with truth), the split to predict (without truth), a
# setting from its search space and the seed.
Method = Callable[[SplitData, SplitData, dict[str, Any], int], Prediction]


def run_connected_components(
    train: SplitData, target: SplitData, params: dict[str, Any], seed: int
) -> Prediction:
    """
    Connected components of the candidate pairs above a TF-IDF similarity threshold.

    Parameters
    ----------
    train : SplitData
        Unused; nothing is learned.
    target : SplitData
        The split to cluster.
    params : dict
        ``threshold``.
    seed : int
        Unused; the method is deterministic.

    Returns
    -------
    Prediction
        The components, and the pairs scored by their similarity.

    Notes
    -----
    Processing:
    1. Run ``clustering.graph.connected_components`` on the target's pairs.
    """
    ids = pd.Index(target.records["record_id"])
    clusters = connected_components(ids, target.pairs, params["threshold"])
    return Prediction(clusters, target.pairs.rename(columns={"similarity": "score"}))


# Methods by their ``<section>.<model>`` name in configs/models.json.
METHODS: dict[str, Method] = {
    "graph_clustering.connected_components": run_connected_components,
}


def load_splits(dataset: str, seed: int) -> dict[str, SplitData]:
    """
    Load a dataset's three splits with their frozen candidate pairs.

    Parameters
    ----------
    dataset : str
        Dataset name.
    seed : int
        Split seed; its candidate sets must exist.

    Returns
    -------
    dict of str to SplitData
        ``train``, ``validation`` and ``test``, each with its truth.

    Raises
    ------
    FileNotFoundError
        If the candidate sets were not built (run ``make candidates``).

    Notes
    -----
    Processing:
    1. Load the normalised records and truth, and recompute the seed's entity
       split.
    2. Read each split's candidate pairs.
    """
    records, truth = load_prepared(dataset, normalised=True)
    split = entity_split(truth, seed, load_experiment()["split"])
    splits = {}
    for name in SPLITS:
        path = candidates_dir(dataset, seed) / f"candidates_{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run make candidates")
        ids = split.index[split == name]
        splits[name] = SplitData(
            records[records["record_id"].isin(ids)].reset_index(drop=True),
            pd.read_parquet(path),
            truth[ids],
        )
    return splits


def evaluate_clusters(clusters: pd.Series, truth: pd.Series) -> dict[str, float]:
    """
    Every clustering metric of the benchmark.

    Parameters
    ----------
    clusters : pd.Series, shape (n_records,)
        Predicted cluster per record ID.
    truth : pd.Series, shape (n_records,)
        True entity per record ID.

    Returns
    -------
    dict of str to float
        ``pairwise_*``, ``b_cubed_*`` and ``cluster_*`` precision, recall and
        F1, ``ari``, and the predicted and true cluster counts
        (``n_clusters``, ``n_entities``).

    Notes
    -----
    Processing:
    1. Compute each metric family and prefix its keys.
    """
    scores: dict[str, float] = {}
    for prefix, scorer in [
        ("pairwise", pairwise_scores),
        ("b_cubed", b_cubed_scores),
        ("cluster", cluster_scores),
    ]:
        scores |= {f"{prefix}_{k}": v for k, v in scorer(clusters, truth).items()}
    scores["ari"] = adjusted_rand_index(clusters, truth)
    scores["n_clusters"] = int(clusters.nunique())
    scores["n_entities"] = int(truth.nunique())
    return scores


class PeakMemory:
    """
    Track the peak resident memory of this process and its children.

    Use as a context manager; ``peak_mb`` holds the highest combined resident
    memory sampled ten times a second, which also counts worker processes
    such as pyJedAI's Ray workers.
    """

    def __enter__(self) -> Self:
        import psutil

        self.peak_mb = 0.0
        self._stop = threading.Event()
        process = psutil.Process()

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    members = [process, *process.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in members)
                    self.peak_mb = max(self.peak_mb, rss / 2**20)
                except psutil.Error:
                    pass
                self._stop.wait(0.1)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def provenance() -> dict[str, Any]:
    """
    The code and configuration a run used.

    Returns
    -------
    dict
        ``git_commit``, ``git_dirty`` (uncommitted changes to tracked files
        outside ``reports/results/``) and the SHA-256 of every file in
        ``configs/``.

    Notes
    -----
    Processing:
    1. Ask git for the commit and working-tree status.
    2. Hash the config files.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        # Results are regenerated by the run itself, so they never make the code dirty.
        "git_dirty": bool(
            git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!reports/results")
        ),
        "config_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(CONFIGS_DIR.glob("*.json"))
        },
    }


def results_dir(dataset: str, method: str, seed: int) -> Path:
    """
    Folder of one run's result files.

    Parameters
    ----------
    dataset, method : str
        Dataset and method name.
    seed : int
        Seed.

    Returns
    -------
    Path
        ``reports/results/<dataset>/<method>/seed<seed>``.

    Notes
    -----
    Processing:
    1. Join the results folder, dataset, method and seed.
    """
    return RESULTS_DIR / dataset / method / f"seed{seed}"


def run_experiment(
    dataset: str, method: str, seed: int, fixed: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Tune a method on validation, run its best setting on test and save the results.

    Parameters
    ----------
    dataset : str
        Dataset name.
    method : str
        Key of ``METHODS``.
    seed : int
        Split and method seed.
    fixed : dict or None, default None
        Values for the method's settings listed in ``not_tuned``.

    Returns
    -------
    dict
        The contents written to ``result.json``.

    Notes
    -----
    Processing:
    1. Load the search space, tuning settings and the three splits.
    2. Tune: every trial runs the method on validation (with train available)
       and scores the tuning objective on validation.
    3. Run the best setting once on test and score every metric.
    4. Write ``result.json``, ``membership.parquet`` and ``pairs.parquet``.
    """
    section, model = method.split(".")
    space = load_space(section, model, fixed)
    settings = load_settings()
    splits = load_splits(dataset, seed)
    predict = METHODS[method]
    train, validation, test = (splits[name] for name in SPLITS)

    def objective(params: dict[str, Any]) -> float:
        prediction = predict(train, replace(validation, truth=None), params, seed)
        return evaluate_clusters(prediction.clusters, validation.truth)[settings["objective"]]

    start = time.perf_counter()
    with PeakMemory() as memory:
        tuning = tune(space, objective, settings)
        test_start = time.perf_counter()
        prediction = predict(train, replace(test, truth=None), tuning.best_params, seed)
        test_seconds = time.perf_counter() - test_start
    clusters = prediction.clusters.astype(str)

    blocking = json.loads((candidates_dir(dataset, seed) / "blocking.json").read_text("utf-8"))
    result = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "top_k": blocking["top_k"],
        "objective": settings["objective"],
        "strategy": tuning.strategy,
        "best_params": tuning.best_params,
        "validation_score": tuning.best_score,
        "test": evaluate_clusters(clusters, test.truth),
        "seconds_total": time.perf_counter() - start,
        "seconds_test": test_seconds,
        "peak_mb": round(memory.peak_mb),
        "trials": tuning.trials,
        **provenance(),
    }

    out_dir = results_dir(dataset, method, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"record_id": clusters.index, "cluster": clusters.to_numpy()}).to_parquet(
        out_dir / "membership.parquet", index=False
    )
    prediction.pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), "utf-8")
    return result


@app.command()
def run(
    dataset: Annotated[str, typer.Argument(help="Dataset to run on.")],
    method: Annotated[str, typer.Argument(help="Method, as <section>.<model>.")],
    seeds: Annotated[
        list[int] | None, typer.Option("--seed", help="Seeds (repeatable; default: all).")
    ] = None,
) -> None:
    """
    Tune and test one method on one dataset for the given seeds.

    \f
    Parameters
    ----------
    dataset : str
        Dataset name.
    method : str
        Key of ``METHODS``.
    seeds : list of int or None
        Seeds; None means those in ``configs/experiment.json``.

    Returns
    -------
    None
        Writes the result files and logs the main test scores.

    Raises
    ------
    typer.BadParameter
        If the method is unknown.

    Notes
    -----
    Processing:
    1. Run ``run_experiment`` for every seed and log its test scores.
    """
    if method not in METHODS:
        raise typer.BadParameter(f"unknown method {method!r}; known: {sorted(METHODS)}")
    for seed in seeds or load_experiment()["seeds"]:
        result = run_experiment(dataset, method, seed)
        test = result["test"]
        logger.info(
            f"{dataset} {method} seed {seed}: best {result['best_params']}, "
            f"validation {result['objective']} {result['validation_score']:.3f}, "
            f"test B3 F1 {test['b_cubed_f1']:.3f}, pairwise F1 {test['pairwise_f1']:.3f}, "
            f"cluster F1 {test['cluster_f1']:.3f}, ARI {test['ari']:.3f}, "
            f"{result['seconds_total']:.1f} s, {result['peak_mb']} MB"
        )


@app.command()
def evaluate(
    dataset: Annotated[str, typer.Argument(help="Dataset.")],
    method: Annotated[str, typer.Argument(help="Method, as <section>.<model>.")],
    seed: Annotated[int, typer.Argument(help="Seed.")],
) -> None:
    """
    Recompute a run's test metrics from its membership file and check result.json.

    \f
    Parameters
    ----------
    dataset, method : str
        Dataset and method name.
    seed : int
        Seed.

    Returns
    -------
    None
        Logs whether the recomputed metrics match.

    Raises
    ------
    ValueError
        If a recomputed metric differs from the one in ``result.json``.

    Notes
    -----
    Processing:
    1. Read the membership file and the test truth.
    2. Recompute every metric and compare with ``result.json``.
    """
    out_dir = results_dir(dataset, method, seed)
    membership = pd.read_parquet(out_dir / "membership.parquet")
    clusters = pd.Series(membership["cluster"].to_numpy(), index=membership["record_id"])
    saved = json.loads((out_dir / "result.json").read_text("utf-8"))["test"]
    recomputed = evaluate_clusters(clusters, load_prepared(dataset)[1][clusters.index])
    differ = {k: (saved[k], v) for k, v in recomputed.items() if abs(saved[k] - v) > 1e-12}
    if differ:
        raise ValueError(f"metrics differ from result.json (saved, recomputed): {differ}")
    logger.success(f"{out_dir}: all {len(recomputed)} metrics match result.json")


if __name__ == "__main__":
    app()
