"""
Summarise every run into results tables.

Input: the ``result.json`` files under ``reports/results/`` written by
``dedup_bench.experiment``.

Output: in ``reports/tables/``:

- ``runs.csv``: one row per run (dataset, method, seed) with its test metrics,
  runtime, peak memory and provenance
- ``<metric>.md``: one table per metric of ``TABLE_METRICS``, methods by
  datasets, each cell the mean ± standard deviation over seeds
- ``health.md``: per method and dataset, the number of seeds, and whether
  every run used the same clean commit; then the limitations, the methods
  ``configs/experiment.json`` excludes per dataset and why. Their cells in the
  metric tables read ``excluded: <reason>``.
"""

import json
from pathlib import Path

import pandas as pd
import typer

from dedup_bench.config import REPORTS_DIR, RESULTS_DIR
from dedup_bench.splits import load_experiment

TABLES_DIR = REPORTS_DIR / "tables"

# Metrics given a table, with the number of decimals they are shown with.
TABLE_METRICS = {
    "b_cubed_f1": 3,
    "pairwise_f1": 3,
    "cluster_f1": 3,
    "ari": 3,
    "seconds_total": 0,
    "peak_mb": 0,
}

app = typer.Typer()


def load_runs(results_dir: Path) -> pd.DataFrame:
    """
    Every run's result as one row.

    Parameters
    ----------
    results_dir : Path
        Folder holding ``<dataset>/<method>/[<fixed>/]seed<seed>/result.json``.

    Returns
    -------
    pd.DataFrame, shape (n_runs, n_columns)
        ``dataset``, ``method`` (with any fixed setting appended, e.g.
        ``active_learning.random_forest label_budget=50``), ``seed``, the test
        metrics, ``seconds_total``, ``peak_mb``, ``git_commit`` and ``git_dirty``.

    Notes
    -----
    Processing:
    1. Read every ``result.json`` and flatten its test metrics.
    2. Sort by method (fixed settings by value, so budgets run 10, 25, ...),
       dataset and seed.
    """
    rows = []
    for path in sorted(results_dir.rglob("result.json")):
        result = json.loads(path.read_text("utf-8"))
        fixed = " ".join(f"{name}={value}" for name, value in result.get("fixed", {}).items())
        rows.append(
            {
                "dataset": result["dataset"],
                "method": f"{result['method']} {fixed}".strip(),
                "seed": result["seed"],
                **result["test"],
                "seconds_total": result["seconds_total"],
                "peak_mb": result["peak_mb"],
                "git_commit": result["git_commit"],
                "git_dirty": result["git_dirty"],
                "_order": (result["method"], *result.get("fixed", {}).values()),
            }
        )
    runs = pd.DataFrame(rows).sort_values(["_order", "dataset", "seed"], ignore_index=True)
    return runs.drop(columns="_order")


def markdown_table(table: pd.DataFrame) -> str:
    """
    A DataFrame as a GitHub Markdown table, index as the first column.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_rows, n_columns)
        Cells already formatted as strings.

    Returns
    -------
    str
        The table, one line per row.

    Notes
    -----
    Processing:
    1. Write the header, the separator and one line per row.
    """
    header = [table.index.name or "", *map(str, table.columns)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| " + " | ".join([str(i), *row]) + " |" for i, row in zip(table.index, table.values)
    ]
    return "\n".join(lines) + "\n"


def metric_table(runs: pd.DataFrame, metric: str, decimals: int) -> pd.DataFrame:
    """
    Mean ± standard deviation of a metric over seeds, methods by datasets.

    Parameters
    ----------
    runs : pd.DataFrame
        As returned by ``load_runs``.
    metric : str
        Column to summarise.
    decimals : int
        Decimals shown.

    Returns
    -------
    pd.DataFrame, shape (n_methods, n_datasets)
        Cells such as ``0.850 ± 0.012``; empty where a method has no run.

    Notes
    -----
    Processing:
    1. Group by method and dataset, and take the mean and sample standard
       deviation (0 for a single seed).
    2. Format each cell and pivot datasets into columns.
    """
    stats = runs.groupby(["method", "dataset"])[metric].agg(["mean", "std"]).fillna({"std": 0.0})
    cells = stats.apply(lambda r: f"{r['mean']:.{decimals}f} ± {r['std']:.{decimals}f}", axis=1)
    return cells.unstack("dataset").reindex(runs["method"].unique()).fillna("")


def health_table(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Per method and dataset: how many seeds ran, and on how many commits.

    Parameters
    ----------
    runs : pd.DataFrame
        As returned by ``load_runs``.

    Returns
    -------
    pd.DataFrame, shape (n_methods * n_datasets, 4)
        ``seeds``, ``commits`` (distinct ``git_commit``), ``dirty`` (runs on
        uncommitted code) and ``max_peak_mb``, indexed by ``method / dataset``.

    Notes
    -----
    Processing:
    1. Count seeds, distinct commits and dirty runs per group.
    """
    grouped = runs.groupby(["method", "dataset"], sort=False)
    table = pd.DataFrame(
        {
            "seeds": grouped["seed"].nunique(),
            "commits": grouped["git_commit"].nunique(),
            "dirty": grouped["git_dirty"].sum(),
            "max_peak_mb": grouped["peak_mb"].max(),
        }
    )
    table.index = [f"{method} / {dataset}" for method, dataset in table.index]
    table.index.name = "method / dataset"
    return table.astype(str)


def mark_excluded(table: pd.DataFrame, excluded: dict[str, dict[str, str]]) -> pd.DataFrame:
    """
    Mark the cells of methods a dataset excludes, instead of leaving them empty.

    Parameters
    ----------
    table : pd.DataFrame, shape (n_methods, n_datasets)
        A metric table from ``metric_table``.
    excluded : dict
        ``dataset -> method -> reason``, from ``configs/experiment.json``.

    Returns
    -------
    pd.DataFrame
        The table with ``excluded: <reason before the first colon>`` in those
        cells; rows or columns are added if the method or dataset has no run.

    Notes
    -----
    Processing:
    1. Write the short reason into each excluded cell.
    """
    table = table.copy()
    for dataset, methods in excluded.items():
        for method, reason in methods.items():
            table.loc[method, dataset] = f"excluded: {reason.split(':')[0]}"
    return table.fillna("")


def limitations_table(excluded: dict[str, dict[str, str]]) -> pd.DataFrame:
    """
    The excluded methods and why, one row each.

    Parameters
    ----------
    excluded : dict
        ``dataset -> method -> reason``, from ``configs/experiment.json``.

    Returns
    -------
    pd.DataFrame, shape (n_excluded, 1)
        ``reason``, indexed by ``method / dataset``.

    Notes
    -----
    Processing:
    1. Flatten the mapping.
    """
    rows = {f"{m} / {d}": r for d, methods in excluded.items() for m, r in methods.items()}
    return pd.DataFrame({"reason": rows.values()}, index=pd.Index(rows, name="method / dataset"))


@app.command()
def main(
    results_dir: Path = RESULTS_DIR,
    tables_dir: Path = TABLES_DIR,
) -> None:
    """
    Write the run list, one table per metric, and the health table.

    \f
    Parameters
    ----------
    results_dir : Path
        Where the runs' ``result.json`` files are.
    tables_dir : Path
        Where the tables go.

    Returns
    -------
    None
        Writes ``runs.csv``, ``<metric>.md`` and ``health.md``.

    Raises
    ------
    FileNotFoundError
        If there are no results.

    Notes
    -----
    Processing:
    1. Load every run and the exclusions of ``configs/experiment.json``.
    2. Write the run list, the metric tables (excluded cells marked) and the
       health table followed by the limitations.
    """
    runs = load_runs(results_dir)
    if runs.empty:
        raise FileNotFoundError(f"no result.json under {results_dir}")
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(tables_dir / "runs.csv", index=False)
    excluded = load_experiment().get("excluded", {})
    for metric, decimals in TABLE_METRICS.items():
        table = mark_excluded(metric_table(runs, metric, decimals), excluded)
        table.index.name = "method"
        title = f"# Test {metric}, mean ± standard deviation over seeds\n\n"
        (tables_dir / f"{metric}.md").write_text(title + markdown_table(table), "utf-8")
    title = "# Run health: seeds, commits, uncommitted-code runs and peak memory\n\n"
    health = title + markdown_table(health_table(runs))
    if excluded:
        health += "\n## Limitations: methods not run\n\n" + markdown_table(
            limitations_table(excluded)
        )
    (tables_dir / "health.md").write_text(health, "utf-8")


if __name__ == "__main__":
    app()
