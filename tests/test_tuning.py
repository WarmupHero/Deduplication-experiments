import pytest

from dedup_bench import tuning
from dedup_bench.tuning import Choice, Range


def test_range_points_are_exact_and_inclusive() -> None:
    assert Range(0.5, 0.7, 0.05).points() == [0.5, 0.55, 0.6, 0.65, 0.7]
    assert Range(100, 300, 100).points() == [100, 200, 300]
    assert Range(100, 300, 100).is_int and not Range(0.5, 0.7, 0.05).is_int


def test_parse_space_reads_lists_and_ranges_and_rejects_the_rest() -> None:
    space = tuning.parse_space({"a": [1, 2], "b": {"low": 0.1, "high": 0.3, "step": 0.1}})
    assert space == {"a": Choice((1, 2)), "b": Range(0.1, 0.3, 0.1)}
    for bad in ([], {"low": 0.5, "high": 0.1, "step": 0.1}, {"low": 0, "high": 1}, 3):
        with pytest.raises(ValueError):
            tuning.parse_space({"x": bad})


def test_grid_trials_cover_every_combination_and_respect_the_cap() -> None:
    space = {"a": Choice(("x", "y")), "b": Range(1, 3, 1)}
    trials = tuning.grid_trials(space, max_combinations=10)
    assert len(trials) == 6
    assert {"a": "y", "b": 3} in trials
    with pytest.raises(ValueError, match="combinations"):
        tuning.grid_trials(space, max_combinations=5)


def test_random_trials_are_distinct_reproducible_and_bounded_by_the_space() -> None:
    space = {"a": Choice(("x", "y")), "b": Range(1, 3, 1)}
    first = tuning.random_trials(space, n_trials=4, seed=7)
    assert first == tuning.random_trials(space, n_trials=4, seed=7)
    assert len({tuple(sorted(t.items())) for t in first}) == 4
    assert len(tuning.random_trials(space, n_trials=100, seed=7)) == 6


def settings(strategy: str, n_trials: int = 30) -> dict:
    return {
        "strategy": strategy,
        "seed": 0,
        "max_seconds_per_model": 60,
        "grid": {"max_combinations": 1000},
        "random": {"n_trials": n_trials},
        "optuna": {"n_trials": n_trials, "sampler": "tpe", "n_startup_trials": 5},
    }


@pytest.mark.parametrize("strategy", ["grid", "random", "optuna"])
def test_every_strategy_finds_the_peak_of_a_simple_objective(strategy: str) -> None:
    # Peak at threshold 0.7 with the "average" linkage; [2, 4] checks that
    # list-valued choices pass through Optuna unchanged.
    space = {
        "threshold": Range(0.5, 0.9, 0.05),
        "linkage": Choice(("average", "complete")),
        "ngram_range": Choice(([2, 4],)),
    }

    def objective(params: dict) -> float:
        assert params["ngram_range"] == [2, 4]
        return -abs(params["threshold"] - 0.7) + (0.5 if params["linkage"] == "average" else 0)

    result = tuning.tune(space, objective, settings(strategy))
    assert result.strategy == strategy
    assert result.best_params["linkage"] == "average"
    assert result.best_params["threshold"] == pytest.approx(0.7)
    assert result.best_score == max(t["score"] for t in result.trials)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        tuning.tune({"a": Choice((1,))}, lambda p: 0.0, settings("bayes"))


def test_load_space_requires_values_for_settings_that_are_not_tuned() -> None:
    with pytest.raises(ValueError, match="not tuned"):
        tuning.load_space("active_learning", "random_forest")
    space = tuning.load_space("active_learning", "random_forest", fixed={"label_budget": 50})
    assert space["label_budget"] == Choice((50,))
