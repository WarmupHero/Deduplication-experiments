import json
from pathlib import Path

import pandas as pd
import pytest

from dedup_bench import report


def write_result(root: Path, method: str, seed: int, f1: float, fixed: dict | None = None) -> None:
    folder = root.joinpath("d1", method, *[f"{k}{v}" for k, v in (fixed or {}).items()])
    folder = folder / f"seed{seed}"
    folder.mkdir(parents=True)
    result = {
        "dataset": "d1",
        "method": method,
        "seed": seed,
        "fixed": fixed or {},
        "test": {"b_cubed_f1": f1},
        "seconds_total": 1.0,
        "peak_mb": 100 + seed,
        "git_commit": "abc",
        "git_dirty": seed == 1,
    }
    (folder / "result.json").write_text(json.dumps(result), "utf-8")


def test_metric_table_mean_and_sample_std(tmp_path: Path) -> None:
    # 0.8 and 0.9: mean 0.85, sample std sqrt(0.005) = 0.0707.
    write_result(tmp_path, "m", 0, 0.8)
    write_result(tmp_path, "m", 1, 0.9)
    write_result(tmp_path, "al", 0, 0.5, {"label_budget": 10})
    write_result(tmp_path, "al", 0, 0.6, {"label_budget": 25})
    runs = report.load_runs(tmp_path)
    assert runs["method"].tolist() == ["al label_budget=10", "al label_budget=25", "m", "m"]
    table = report.metric_table(runs, "b_cubed_f1", 3)
    assert table.loc["m", "d1"] == "0.850 ± 0.071"
    assert table.loc["al label_budget=10", "d1"] == "0.500 ± 0.000"
    health = report.health_table(runs)
    assert health.loc["m / d1"].tolist() == ["2", "1", "1", "101"]


def test_markdown_table_layout() -> None:
    table = pd.DataFrame({"d1": ["x"]}, index=pd.Index(["m"], name="method"))
    assert report.markdown_table(table) == "| method | d1 |\n|---|---|\n| m | x |\n"


def test_excluded_methods_are_marked_and_listed_as_limitations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_result(tmp_path / "results", "m", 0, 0.8)
    excluded = {"d1": {"big": "out of memory: about 14 GB"}}
    monkeypatch.setattr(report, "load_experiment", lambda: {"excluded": excluded})
    monkeypatch.setattr(report, "TABLE_METRICS", {"b_cubed_f1": 3})
    report.main(tmp_path / "results", tmp_path / "tables")
    b_cubed = (tmp_path / "tables" / "b_cubed_f1.md").read_text("utf-8")
    assert "| m | 0.800 ± 0.000 |" in b_cubed
    assert "| big | excluded: out of memory |" in b_cubed
    health = (tmp_path / "tables" / "health.md").read_text("utf-8")
    assert "## Limitations: methods not run" in health
    assert "| big / d1 | out of memory: about 14 GB |" in health
