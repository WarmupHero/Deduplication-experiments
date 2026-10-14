"""
Build the frozen candidate sets that every method of the benchmark compares.

Input: normalised prepared datasets (``make data``), their text fields from
``configs/datasets.json``, the datasets, seeds and split shares from
``configs/experiment.json``, and the ``blocking.tfidf_knn`` settings from
``configs/models.json``.

Output: for every dataset and seed, in ``data/interim/<dataset>/seed<seed>/``:

- ``candidates_<split>.parquet`` for train, validation and test:
  ``left_id``, ``right_id`` and ``similarity`` of each candidate pair. Pairs
  never cross splits, because each split is blocked on its own.
- ``blocking.json``: the chosen ``top_k`` and the blocking scores of every
  split, plus the validation scores of every ``top_k`` tried.

``top_k`` is not tuned on the methods' objective. It is the smallest value
whose reachable pair completeness on validation (see
``evaluation.metrics.blocking_scores``) reaches
``target_reachable_pair_completeness``, or the largest value if none does.
"""

import json
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import pandas as pd
import typer

from dedup_bench.blocking.tfidf_knn import nearest_neighbours, neighbour_pairs
from dedup_bench.config import INTERIM_DATA_DIR, MODELS_CONFIG
from dedup_bench.dataset import load_datasets_config, load_prepared, record_texts
from dedup_bench.evaluation.metrics import blocking_scores
from dedup_bench.splits import SPLITS, entity_split, load_experiment
from dedup_bench.tuning import candidates, parse_space

app = typer.Typer()


def candidates_dir(dataset: str, seed: int) -> Path:
    """
    Folder of one dataset's candidate sets for one seed.

    Parameters
    ----------
    dataset : str
        Dataset name.
    seed : int
        Split seed.

    Returns
    -------
    Path
        ``data/interim/<dataset>/seed<seed>``.

    Notes
    -----
    Processing:
    1. Join the interim data folder, dataset name and seed.
    """
    return INTERIM_DATA_DIR / dataset / f"seed{seed}"


def build_candidates(dataset: str, seed: int) -> dict[str, Any]:
    """
    Split one dataset, choose ``top_k`` on validation and save every split's candidates.

    Parameters
    ----------
    dataset : str
        A key of ``configs/datasets.json``.
    seed : int
        Seed of the entity split.

    Returns
    -------
    dict
        The contents written to ``blocking.json``.

    Notes
    -----
    Processing:
    1. Load the normalised records, join their text fields and split them by
       entity.
    2. Find every record's nearest neighbours within its own split, once, for
       the largest ``top_k``.
    3. Score each ``top_k`` on validation, from smallest to largest, and keep
       the first that reaches the target reachable pair completeness.
    4. Write each split's pairs at that ``top_k`` with record IDs, and the
       scores.
    """
    fields = load_datasets_config()[dataset]["text_fields"]
    # top_k is in tuning.json's not_tuned (load_space refuses it): it is chosen here.
    models = json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))
    space = parse_space(models["blocking"]["tfidf_knn"])
    settings = {name: candidates(spec)[0] for name, spec in space.items() if name != "top_k"}
    top_ks = sorted(candidates(space["top_k"]))
    target = settings.pop("target_reachable_pair_completeness")

    records, truth = load_prepared(dataset, normalised=True)
    texts = record_texts(records, fields)
    split = entity_split(truth, seed, load_experiment()["split"])

    neighbours = {}
    for name in SPLITS:
        split_texts = texts[split == name]
        found, similarity = nearest_neighbours(split_texts, max(top_ks), **settings)
        neighbours[name] = (split_texts.index, found, similarity)

    def scores(name: str, top_k: int) -> tuple[pd.DataFrame, dict[str, float]]:
        ids, found, similarity = neighbours[name]
        pairs = neighbour_pairs(found, similarity, top_k)
        pairs = pd.DataFrame(
            {
                "left_id": ids[pairs["left"]],
                "right_id": ids[pairs["right"]],
                "similarity": pairs["similarity"],
            }
        )
        return pairs, blocking_scores(pairs["left_id"], pairs["right_id"], truth[ids])

    tried = {top_k: scores("validation", top_k)[1] for top_k in top_ks}
    reaching = [k for k in top_ks if tried[k]["reachable_pair_completeness"] >= target]
    top_k = reaching[0] if reaching else top_ks[-1]
    if not reaching:
        logger.warning(
            f"{dataset} seed {seed}: no top_k reaches reachable pair completeness {target} "
            f"on validation; using {top_k}"
        )

    out_dir = candidates_dir(dataset, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"dataset": dataset, "seed": seed, "top_k": top_k, "splits": {}}
    for name in SPLITS:
        pairs, split_scores = scores(name, top_k)
        pairs.to_parquet(out_dir / f"candidates_{name}.parquet", index=False)
        report["splits"][name] = {"n_records": int((split == name).sum()), **split_scores}
    report["validation_by_top_k"] = {str(k): v for k, v in tried.items()}
    (out_dir / "blocking.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@app.command()
def build(
    datasets: Annotated[
        list[str] | None, typer.Argument(help="Datasets (default: configs/experiment.json).")
    ] = None,
    seeds: Annotated[
        list[int] | None, typer.Option("--seed", help="Seeds (repeatable; default: all).")
    ] = None,
) -> None:
    """
    Build and save the candidate sets of the given datasets and seeds.

    \f
    Parameters
    ----------
    datasets : list of str or None
        Datasets to build; None means those in ``configs/experiment.json``.
    seeds : list of int or None
        Seeds to build; None means those in ``configs/experiment.json``.

    Returns
    -------
    None
        Writes the files and logs each split's size and blocking scores.

    Notes
    -----
    Processing:
    1. Run ``build_candidates`` for every dataset and seed.
    """
    experiment = load_experiment()
    for dataset in datasets or experiment["datasets"]:
        for seed in seeds or experiment["seeds"]:
            report = build_candidates(dataset, seed)
            for name, s in report["splits"].items():
                logger.info(
                    f"{dataset} seed {seed} top_k {report['top_k']} {name}: "
                    f"{s['n_records']:,} records, {s['n_candidates']:,} pairs, "
                    f"PC {s['pair_completeness']:.3f}, "
                    f"reachable PC {s['reachable_pair_completeness']:.3f}, RR {s['reduction_ratio']:.4f}, "
                    f"PQ {s['pair_quality']:.3f}"
                )


if __name__ == "__main__":
    app()
