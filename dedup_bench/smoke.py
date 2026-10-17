"""
Smoke test: run every library of the benchmark once on small datasets.

This checks that the environment works (compiled dependencies, model
downloads, library APIs) and that each method produces a sensible clustering.
It is not a benchmark: there is no train / validation / test split, and
settings are not tuned. Model hyperparameters come from ``configs/models.json``
(the middle candidate of each list or range); only the plain call settings a
library needs to run at all are fixed here.

Input: prepared datasets in ``data/processed/<dataset>/`` (run ``make data``
first), by default Febrl 3 and the restaurants, with the fields each library
uses from ``configs/datasets_smoke.json``.

Output: a table on the console with, for every dataset and check, the run
time, the number of predicted clusters, pairwise precision / recall / F1 and a
fingerprint of the clustering (identical fingerprints mean identical
clusterings).

Every check runs in its own Python process, so the memory it used (PyTorch
models, pyJedAI's Ray workers) is released before the next one starts, and a
crash in one check is reported without stopping the others.
"""

from collections.abc import Callable
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from typing import Annotated, Any

from loguru import logger
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import typer

from dedup_bench.blocking.tfidf_knn import nearest_neighbours, neighbour_pairs
from dedup_bench.config import DATASETS_SMOKE_CONFIG, MODELS_CONFIG
from dedup_bench.dataset import load_prepared, record_texts
from dedup_bench.embeddings import encode
from dedup_bench.evaluation.metrics import pairwise_scores
from dedup_bench.tuning import candidates, parse_space

app = typer.Typer()

SEED = 42

# Per dataset, the fields each library compares and its blocking keys.
SMOKE_DATASETS: dict[str, dict[str, list[str]]] = {
    name: spec
    for name, spec in json.loads(DATASETS_SMOKE_CONFIG.read_text(encoding="utf-8")).items()
    if name != "description"
}

# A check takes the records, the true entities and the dataset settings, and
# returns one or more named clusterings (cluster label per record ID).
Check = Callable[[pd.DataFrame, pd.Series, dict[str, list[str]]], dict[str, pd.Series]]


def middle(values: list[Any]) -> Any:
    """
    Pick the middle candidate of a hyperparameter grid.

    Parameters
    ----------
    values : list
        Candidate values from ``configs/models.json``.

    Returns
    -------
    Any
        ``values[len(values) // 2]``.

    Notes
    -----
    Processing:
    1. Index the list at its midpoint (the later of the two for even lengths).
    """
    return values[len(values) // 2]


def load_settings() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Read ``configs/models.json`` and reduce every grid to its middle candidate.

    Parameters
    ----------
    None
        Reads ``MODELS_CONFIG``.

    Returns
    -------
    dict
        ``section -> model -> hyperparameter -> value``, without the
        ``description`` entries.

    Notes
    -----
    Processing:
    1. Load the JSON file.
    2. Drop ``description`` keys, expand each list or range to its candidates and
       take ``middle`` of them.
    """
    config = json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))
    return {
        section: {
            model: {name: middle(candidates(spec)) for name, spec in parse_space(params).items()}
            for model, params in models.items()
            if model != "description"
        }
        for section, models in config.items()
        if section != "description"
    }


SETTINGS = load_settings()


def seed_everything(seed: int) -> None:
    """
    Seed Python's ``random`` (also used by igraph) and NumPy's global generator.

    Parameters
    ----------
    seed : int
        The seed.

    Returns
    -------
    None

    Notes
    -----
    Processing:
    1. Seed ``random`` and ``numpy.random``.
    """
    random.seed(seed)
    np.random.seed(seed)


def clusters_from_pairs(record_ids: pd.Index, left: np.ndarray, right: np.ndarray) -> pd.Series:
    """
    Connected components of a set of matched pairs, as cluster labels.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs; records in no pair become singletons.
    left, right : np.ndarray of int, shape (n_pairs,)
        Positions (into ``record_ids``) of the two records of each matched pair.

    Returns
    -------
    pd.Series of int, shape (n_records,)
        Component label of every record, indexed by record ID.

    Notes
    -----
    Processing:
    1. Build an undirected sparse graph from the pairs and label its
       connected components with SciPy.
    """
    n = len(record_ids)
    graph = coo_matrix((np.ones(len(left)), (left, right)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return pd.Series(labels, index=record_ids)


def clusters_from_groups(record_ids: pd.Index, groups: list[Any]) -> pd.Series:
    """
    Cluster labels from a list of record groups; ungrouped records are singletons.

    Parameters
    ----------
    record_ids : pd.Index of str, shape (n_records,)
        All record IDs.
    groups : list of iterables of str
        Each item lists the record IDs of one predicted cluster.

    Returns
    -------
    pd.Series of int, shape (n_records,)
        Cluster label of every record, indexed by record ID.

    Notes
    -----
    Processing:
    1. Give every record its own label, then overwrite the members of each
       group with the group's label.
    """
    labels = pd.Series(np.arange(len(record_ids)), index=record_ids)
    for number, group in enumerate(groups):
        labels[list(group)] = len(record_ids) + number
    return labels


def fingerprint(clusters: pd.Series) -> str:
    """
    A short hash that identifies a clustering independent of its label names.

    Parameters
    ----------
    clusters : pd.Series, shape (n_records,)
        Cluster label per record ID.

    Returns
    -------
    str
        The first 10 hex digits of the SHA-256 of the sorted list of sorted
        clusters.

    Notes
    -----
    Processing:
    1. Group record IDs by label, sort within and across groups, and hash.
    """
    groups = sorted(sorted(map(str, ids)) for ids in clusters.groupby(clusters).groups.values())
    return hashlib.sha256(json.dumps(groups).encode()).hexdigest()[:10]


def tfidf_knn_pairs(texts: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Candidate pairs from TF-IDF character n-grams and nearest neighbours.

    Parameters
    ----------
    texts : pd.Series of str, shape (n_records,)
        One text per record.

    Returns
    -------
    left, right : np.ndarray of int, shape (n_pairs,)
        Positions of the two records of each candidate pair (left < right).
    similarity : np.ndarray of float, shape (n_pairs,)
        Cosine similarity of each pair.

    Notes
    -----
    Processing:
    1. Find each record's ``top_k`` nearest neighbours with the
       ``blocking.tfidf_knn`` settings.
    2. Keep each unordered pair once.
    """
    params = dict(SETTINGS["blocking"]["tfidf_knn"])
    top_k = params.pop("top_k")
    params.pop("target_reachable_pair_completeness")
    pairs = neighbour_pairs(*nearest_neighbours(texts, top_k, **params), top_k)
    return pairs["left"].to_numpy(), pairs["right"].to_numpy(), pairs["similarity"].to_numpy()


def check_graph_clustering(
    records: pd.DataFrame, truth: pd.Series, spec: dict[str, list[str]]
) -> dict[str, pd.Series]:
    """
    Graph clustering of the TF-IDF candidate graph: connected components and Leiden.

    Parameters
    ----------
    records : pd.DataFrame
        Prepared records.
    truth : pd.Series
        True entity per record ID (unused; part of the check signature).
    spec : dict
        ``SMOKE_DATASETS`` entry with the fields to compare.

    Returns
    -------
    dict of str to pd.Series
        ``connected_components`` and ``leiden_cpm`` clusterings.

    Notes
    -----
    Processing:
    1. Build TF-IDF candidate pairs.
    2. Connected components of the pairs above the configured threshold.
    3. Leiden (python-igraph) with the CPM objective on the pairs above its
       threshold, weighted by similarity.
    """
    import igraph

    texts = record_texts(records, spec["fields"])
    left, right, similarity = tfidf_knn_pairs(texts)
    ids = texts.index

    cc = SETTINGS["graph_clustering"]["connected_components"]
    keep = similarity >= cc["threshold"]
    components = clusters_from_pairs(ids, left[keep], right[keep])

    leiden = dict(SETTINGS["graph_clustering"]["leiden_cpm"])
    keep = similarity >= leiden.pop("threshold")
    graph = igraph.Graph(n=len(ids), edges=list(zip(left[keep], right[keep])))
    graph.es["weight"] = similarity[keep]
    partition = graph.community_leiden(weights="weight", **leiden)
    return {
        "connected_components": components,
        "leiden_cpm": pd.Series(partition.membership, index=ids),
    }


def check_embeddings(
    records: pd.DataFrame, truth: pd.Series, spec: dict[str, list[str]]
) -> dict[str, pd.Series]:
    """
    Embedding clustering: FAISS nearest-neighbour graph, HDBSCAN and agglomerative.

    Runs every encoder listed in the config, so both model2vec and
    sentence-transformers are exercised.

    Parameters
    ----------
    records, truth, spec
        See ``check_graph_clustering``.

    Returns
    -------
    dict of str to pd.Series
        Per encoder: ``faiss_cc``, ``hdbscan`` and ``agglomerative`` clusterings.

    Notes
    -----
    Processing, for each encoder:
    1. Embed the joined record text.
    2. FAISS exact inner-product search for the ``top_k`` neighbours, then
       connected components above the configured threshold.
    3. HDBSCAN on the embeddings; its noise points become singletons.
    4. Agglomerative clustering with a cosine distance threshold.
    """
    import faiss
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering

    texts = record_texts(records, spec["fields"])
    ids = texts.index
    hdbscan = SETTINGS["embedding_clustering"]["hdbscan"]
    agglomerative = SETTINGS["embedding_clustering"]["agglomerative"]
    model_names = json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))["embedding_clustering"][
        "hdbscan"
    ]["model_name"]

    results = {}
    for model_name in model_names:
        short = model_name.split("/")[-1]
        vectors = encode(texts, model_name, hdbscan["batch_size"])

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        k = min(SETTINGS["blocking"]["tfidf_knn"]["top_k"] + 1, len(ids))
        similarity, neighbours = index.search(vectors, k)
        left = np.repeat(np.arange(len(ids)), k)
        right = neighbours.ravel()
        keep = (left != right) & (right >= 0)
        keep &= (
            similarity.ravel() >= SETTINGS["graph_clustering"]["connected_components"]["threshold"]
        )
        results[f"{short}/faiss_cc"] = clusters_from_pairs(ids, left[keep], right[keep])

        labels = HDBSCAN(
            metric="cosine",
            min_cluster_size=hdbscan["min_cluster_size"],
            min_samples=hdbscan["min_samples"],
            cluster_selection_method=hdbscan["cluster_selection_method"],
            cluster_selection_epsilon=hdbscan["cluster_selection_epsilon"],
            copy=False,
        ).fit_predict(vectors)
        noise = labels == -1
        labels[noise] = labels.max() + 1 + np.arange(noise.sum())
        results[f"{short}/hdbscan"] = pd.Series(labels, index=ids)

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage=agglomerative["linkage"],
            distance_threshold=agglomerative["distance_threshold"],
        )
        results[f"{short}/agglomerative"] = pd.Series(clusterer.fit_predict(vectors), index=ids)
    return results


def check_recordlinkage(
    records: pd.DataFrame, truth: pd.Series, spec: dict[str, list[str]]
) -> dict[str, pd.Series]:
    """
    recordlinkage: blocking, Jaro-Winkler comparisons, ECM and a random forest.

    The random forest is the supervised ceiling's smoke test: it is trained on
    a random half of the candidate pairs with their true labels.

    Parameters
    ----------
    records, truth, spec
        See ``check_graph_clustering``.

    Returns
    -------
    dict of str to pd.Series
        ``ecm`` and ``random_forest`` clusterings (connected components of the
        predicted matches).

    Notes
    -----
    Processing:
    1. Candidate pairs: the union of exact blocks on each blocking column.
    2. Jaro-Winkler similarity of every field (missing values score 0).
    3. Fit the ECM classifier on the comparison vectors and predict matches.
    4. Fit a random forest on half the pairs with true labels; predict all
       pairs at the configured probability threshold.
    """
    import recordlinkage
    from sklearn.ensemble import RandomForestClassifier

    frame = (
        records.set_index("record_id")[spec["fields"]]
        .astype(object)
        .where(records.set_index("record_id")[spec["fields"]].notna(), None)
    )
    indexer = recordlinkage.Index()
    for column in spec["blocking"]:
        indexer.block(column)
    candidates = indexer.index(frame)

    compare = recordlinkage.Compare()
    for field in spec["fields"]:
        compare.string(field, field, method="jarowinkler", missing_value=0, label=field)
    features = compare.compute(candidates, frame)

    position = pd.Series(np.arange(len(frame)), index=frame.index)

    def clusters(matches: pd.MultiIndex) -> pd.Series:
        left = position[matches.get_level_values(0)].to_numpy()
        right = position[matches.get_level_values(1)].to_numpy()
        return clusters_from_pairs(frame.index, left, right)

    ecm_params = dict(SETTINGS["probabilistic"]["recordlinkage_ecm"])
    ecm_params.pop("match_weight_threshold")  # the benchmark's cut, not an ECM argument
    ecm = recordlinkage.ECMClassifier(**ecm_params)
    ecm_matches = ecm.fit_predict(features)

    labels = (
        truth[features.index.get_level_values(0)].to_numpy()
        == truth[features.index.get_level_values(1)].to_numpy()
    ).astype(int)
    train = np.random.default_rng(SEED).random(len(features)) < 0.5
    forest_params = dict(SETTINGS["supervised"]["random_forest"])
    threshold = forest_params.pop("threshold")
    forest = RandomForestClassifier(random_state=SEED, **forest_params).fit(
        features[train], labels[train]
    )
    probability = forest.predict_proba(features)[:, 1]
    forest_matches = features.index[probability >= threshold]
    return {"ecm": clusters(ecm_matches), "random_forest": clusters(forest_matches)}


def check_splink(
    records: pd.DataFrame, truth: pd.Series, spec: dict[str, list[str]]
) -> dict[str, pd.Series]:
    """
    Splink 5: Fellegi-Sunter trained without labels, on DuckDB.

    Parameters
    ----------
    records, truth, spec
        See ``check_graph_clustering``.

    Returns
    -------
    dict of str to pd.Series
        ``splink`` clustering at the configured match-probability threshold.

    Notes
    -----
    Processing:
    1. Register the records with DuckDB.
    2. One Jaro-Winkler comparison per field; prediction blocking on each
       blocking column.
    3. Estimate u by random sampling, then m by EM once per blocking column.
    4. Predict, cluster at the threshold, and map clusters back to records.
    """
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    import splink.comparison_library as cl

    params = SETTINGS["probabilistic"]["splink"]
    frame = records[["record_id", *spec["fields"]]].astype(object)
    frame = frame.where(frame.notna(), None)

    db_api = DuckDBAPI()
    table = db_api.register(frame)
    settings = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="record_id",
        comparisons=[cl.JaroWinklerAtThresholds(field, [0.9, 0.7]) for field in spec["fields"]],
        blocking_rules_to_generate_predictions=[block_on(column) for column in spec["blocking"]],
        max_iterations=params["max_iterations"],
        em_convergence=params["em_convergence"],
    )
    linker = Linker(table, settings, log_level=logging.WARNING)
    linker.training.estimate_u_using_random_sampling(max_pairs=params["u_max_pairs"], seed=SEED)
    for column in spec["blocking"]:
        linker.training.estimate_parameters_using_expectation_maximisation(block_on(column))
    predictions = linker.inference.predict()
    clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_weight=params["match_weight_threshold"]
    ).as_pandas_dataframe()
    labels = clustered.set_index("record_id")["cluster_id"]
    return {"splink": labels.reindex(records["record_id"].to_numpy())}


def check_pyjedai(
    records: pd.DataFrame, truth: pd.Series, spec: dict[str, list[str]]
) -> dict[str, pd.Series]:
    """
    pyJedAI: token blocking and embedding blocking, matching, connected components.

    Parameters
    ----------
    records, truth, spec
        See ``check_graph_clustering``.

    Returns
    -------
    dict of str to pd.Series
        ``pyjedai/token_blocking`` and ``pyjedai/embedding_blocking`` clusterings.

    Notes
    -----
    Processing:
    1. Wrap the records in pyJedAI's ``Data``.
    2. Token blocking, block purging and filtering; or nearest-neighbour
       blocking on MiniLM embeddings (exercises transformers).
    3. Cosine matching of the candidate pairs, then connected components at
       the configured threshold; pyJedAI's node numbers map back to records.
    """
    from pyjedai.block_building import StandardBlocking
    from pyjedai.block_cleaning import BlockFiltering, BlockPurging
    from pyjedai.clustering import ConnectedComponentsClustering
    from pyjedai.datamodel import Data
    from pyjedai.matching import EntityMatching
    from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding

    frame = (
        records[["record_id", *spec["fields"]]]
        .astype(object)
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )
    ids = pd.Index(frame["record_id"])
    threshold = SETTINGS["graph_clustering"]["connected_components"]["threshold"]

    def cluster(blocks: dict, data: Data) -> pd.Series:
        graph = EntityMatching(metric="cosine").predict(blocks, data, tqdm_disable=True)
        groups = ConnectedComponentsClustering().process(
            graph, data, similarity_threshold=threshold
        )
        return clusters_from_groups(ids, [ids[sorted(group)] for group in groups])

    data = Data(dataset_1=frame.copy(), id_column_name_1="record_id", attributes_1=spec["fields"])
    blocks = StandardBlocking().build_blocks(data, tqdm_disable=True)
    blocks = BlockPurging().process(blocks, data, tqdm_disable=True)
    blocks = BlockFiltering().process(blocks, data, tqdm_disable=True)
    token_clusters = cluster(blocks, data)

    data = Data(dataset_1=frame.copy(), id_column_name_1="record_id", attributes_1=spec["fields"])
    built = EmbeddingsNNBlockBuilding(
        vectorizer="sminilm", similarity_search="faiss"
    ).build_blocks(
        data,
        top_k=SETTINGS["blocking"]["tfidf_knn"]["top_k"],
        tqdm_disable=True,
        save_embeddings=False,
    )
    blocks = built[0] if isinstance(built, tuple) else built
    return {
        "pyjedai/token_blocking": token_clusters,
        "pyjedai/embedding_blocking": cluster(blocks, data),
    }


CHECKS: dict[str, Check] = {
    "graph": check_graph_clustering,
    "embeddings": check_embeddings,
    "recordlinkage": check_recordlinkage,
    "splink": check_splink,
    "pyjedai": check_pyjedai,
}


def run_check(name: str, check: Check, dataset: str) -> list[dict[str, Any]]:
    """
    Run one check on one dataset and score every clustering it returns.

    Parameters
    ----------
    name : str
        Key of ``CHECKS``.
    check : Check
        The check function.
    dataset : str
        Key of ``SMOKE_DATASETS``.

    Returns
    -------
    list of dict
        One row per clustering: dataset, method, seconds, clusters, precision,
        recall, f1, fingerprint, status. A failing check gives one row with
        the error message as status.

    Notes
    -----
    Processing:
    1. Load the prepared records and truth, and seed Python and NumPy.
    2. Time the check; catch any exception so other checks still run.
    3. Score each clustering with ``pairwise_scores`` and fingerprint it.
    """
    records, truth = load_prepared(dataset)
    seed_everything(SEED)

    start = time.perf_counter()
    try:
        clusterings = check(records, truth, SMOKE_DATASETS[dataset])
    except Exception as exc:  # noqa: BLE001 - any library error is reported, the rest still run
        logger.exception(f"{dataset} / {name} failed")
        return [
            {
                "dataset": dataset,
                "method": name,
                "status": f"FAILED: {type(exc).__name__}: {exc}"[:120],
            }
        ]
    seconds = time.perf_counter() - start

    rows = []
    for method, clusters in clusterings.items():
        clusters = clusters.astype(str)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seconds": round(seconds, 1),
                "clusters": clusters.nunique(),
                **{
                    key: round(value, 3) for key, value in pairwise_scores(clusters, truth).items()
                },
                "fingerprint": fingerprint(clusters),
                "status": "ok",
            }
        )
    return rows


# Marks the line on which a ``check`` subprocess reports its results.
RESULT_MARKER = "SMOKE_RESULT "


def run_isolated(dataset: str, name: str, timeout: float) -> list[dict[str, Any]]:
    """
    Run one check in a fresh Python process and track its peak memory.

    Parameters
    ----------
    dataset : str
        Key of ``SMOKE_DATASETS``.
    name : str
        Key of ``CHECKS``.
    timeout : float
        Seconds after which the process (and its children) is killed.

    Returns
    -------
    list of dict
        The rows of ``run_check``, each with ``peak_mb``: the highest combined
        resident memory of the process and its children (pyJedAI's Ray
        workers, for example), sampled twice a second. If the process crashes,
        is killed or times out, one row reports that as its status.

    Notes
    -----
    Processing:
    1. Start ``python -m dedup_bench.smoke check <dataset> <name>`` with the
       current environment (so ``PYTHONHASHSEED`` carries over); its log goes
       to the console, its stdout is captured.
    2. Poll the memory of the process tree until it exits or times out.
    3. Parse the ``RESULT_MARKER`` line, or report the exit code.
    """
    import psutil

    command = [sys.executable, "-m", "dedup_bench.smoke", "check", dataset, name]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    tree = psutil.Process(process.pid)
    peak = 0
    start = time.perf_counter()
    status = None
    while process.poll() is None:
        try:
            members = [tree, *tree.children(recursive=True)]
            peak = max(peak, sum(p.memory_info().rss for p in members if p.is_running()))
        except psutil.Error:
            pass
        if time.perf_counter() - start > timeout:
            for member in [*tree.children(recursive=True), tree]:
                member.kill()
            status = f"FAILED: timed out after {timeout:.0f} s"
            break
        time.sleep(0.5)
    stdout, _ = process.communicate()
    peak_mb = round(peak / 2**20)

    results = [line for line in stdout.splitlines() if line.startswith(RESULT_MARKER)]
    if status is None and results:
        rows = json.loads(results[-1][len(RESULT_MARKER) :])
        return [{**row, "peak_mb": peak_mb} for row in rows]
    if status is None:
        status = f"FAILED: process exited with code {process.returncode} (crashed or killed)"
    return [{"dataset": dataset, "method": name, "peak_mb": peak_mb, "status": status}]


@app.command()
def check(
    dataset: Annotated[str, typer.Argument(help="Dataset to run on.")],
    name: Annotated[str, typer.Argument(help="Check to run.")],
) -> None:
    """
    Run one check in this process and print its results (used by ``run``).

    \f
    Parameters
    ----------
    dataset : str
        Key of ``SMOKE_DATASETS``.
    name : str
        Key of ``CHECKS``.

    Returns
    -------
    None
        Prints the rows of ``run_check`` as JSON after ``RESULT_MARKER``.

    Notes
    -----
    Processing:
    1. Run ``run_check`` and print its rows on one marked line.
    """
    rows = run_check(name, CHECKS[name], dataset)
    print(RESULT_MARKER + json.dumps(rows, default=str), flush=True)


@app.command()
def run(
    datasets: Annotated[
        list[str] | None, typer.Argument(help="Datasets (default: febrl3 restaurants).")
    ] = None,
    only: Annotated[
        list[str] | None, typer.Option(help="Run only these checks (repeatable).")
    ] = None,
    timeout: Annotated[float, typer.Option(help="Seconds allowed per check.")] = 1800,
) -> None:
    """
    Run the smoke checks, each in its own process, and print a results table.

    \f
    Parameters
    ----------
    datasets : list of str or None
        Keys of ``SMOKE_DATASETS``; None means all.
    only : list of str or None
        Keys of ``CHECKS`` to run; None means all.
    timeout : float, default 1800
        Seconds allowed per check before it is killed.

    Returns
    -------
    None
        Prints one row per dataset and clustering.

    Raises
    ------
    typer.BadParameter
        If a dataset or check name is unknown.

    Notes
    -----
    Processing:
    1. Validate the names.
    2. Run every selected check on every selected dataset with ``run_isolated``.
    3. Print the table in run order.
    """
    datasets = datasets or list(SMOKE_DATASETS)
    checks = only or list(CHECKS)
    unknown = [d for d in datasets if d not in SMOKE_DATASETS] + [
        c for c in checks if c not in CHECKS
    ]
    if unknown:
        raise typer.BadParameter(f"unknown name(s) {unknown}")

    rows = []
    for dataset in datasets:
        for name in checks:
            logger.info(f"{dataset}: {name}")
            rows.extend(run_isolated(dataset, name, timeout))
    table = pd.DataFrame(rows)
    with pd.option_context(
        "display.width", 200, "display.max_columns", 20, "display.max_colwidth", 120
    ):
        print(table.to_string(index=False))


if __name__ == "__main__":
    app()
