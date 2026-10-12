"""
Hyperparameter search shared by every model: grid, random or Optuna.

Input: a model's search space from ``configs/models.json`` (each hyperparameter
a list of choices or a ``{"low", "high", "step"}`` range), the search settings
from ``configs/tuning.json``, and an objective: a function from a setting to
its score on the dev split (higher is better).

Output: a ``TuningResult`` with the best setting, its dev score and every
trial (setting, score, seconds), so the search itself can be inspected.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import itertools
import json
import random
import time
from typing import Any

from dedup_bench.config import CONFIGS_DIR, MODELS_CONFIG

TUNING_CONFIG = CONFIGS_DIR / "tuning.json"

# Optuna's categorical suggestions accept only these value types.
OPTUNA_CATEGORICAL_TYPES = (type(None), bool, int, float, str)


@dataclass(frozen=True)
class Choice:
    """
    A hyperparameter with a fixed list of candidate values.

    Attributes
    ----------
    values : tuple
        The candidates, in config order.
    """

    values: tuple[Any, ...]


@dataclass(frozen=True)
class Range:
    """
    A hyperparameter that takes evenly spaced values from ``low`` to ``high``.

    Attributes
    ----------
    low, high : float
        Inclusive bounds.
    step : float
        Spacing between candidate values.
    """

    low: float
    high: float
    step: float

    @property
    def is_int(self) -> bool:
        """Whether every candidate is a whole number (low, high and step are integers)."""
        return all(
            isinstance(v, int) and not isinstance(v, bool)
            for v in (self.low, self.high, self.step)
        )

    def points(self) -> list[float] | list[int]:
        """
        All candidate values of the range.

        Returns
        -------
        list of int or float
            ``low, low + step, ...`` up to and including ``high`` (within
            rounding); floats are rounded to 10 decimals so that 0.1 + 0.2
            style drift does not create values like 0.30000000000000004.

        Notes
        -----
        Processing:
        1. Count the steps that fit between ``low`` and ``high``.
        2. Generate the values, as integers or rounded floats.
        """
        n_steps = round((self.high - self.low) / self.step)
        values = [self.low + i * self.step for i in range(n_steps + 1)]
        if self.is_int:
            return [int(v) for v in values]
        return [round(v, 10) for v in values]


Spec = Choice | Range


@dataclass
class TuningResult:
    """
    Outcome of a hyperparameter search.

    Attributes
    ----------
    best_params : dict
        The setting with the highest objective.
    best_score : float
        Its objective value on the dev split.
    trials : list of dict
        Every trial in run order: ``params``, ``score`` and ``seconds``.
    strategy : str
        ``grid``, ``random`` or ``optuna``.
    """

    best_params: dict[str, Any]
    best_score: float
    trials: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = ""


def parse_space(params: dict[str, Any]) -> dict[str, Spec]:
    """
    Turn one model's entry in ``configs/models.json`` into a search space.

    Parameters
    ----------
    params : dict
        Hyperparameter name to a list of choices or a range dict.

    Returns
    -------
    dict of str to Choice or Range
        The parsed space, in config order.

    Raises
    ------
    ValueError
        If a value is neither a non-empty list nor a valid range
        (``low <= high`` and ``step > 0``).

    Notes
    -----
    Processing:
    1. Lists become ``Choice``; dicts with low, high and step become ``Range``.
    2. Validate each entry.
    """
    space: dict[str, Spec] = {}
    for name, value in params.items():
        if isinstance(value, list) and value:
            space[name] = Choice(tuple(value))
        elif isinstance(value, dict) and set(value) == {"low", "high", "step"}:
            spec = Range(value["low"], value["high"], value["step"])
            if not (spec.low <= spec.high and spec.step > 0):
                raise ValueError(f"{name}: need low <= high and step > 0, got {value}")
            space[name] = spec
        else:
            raise ValueError(
                f"{name}: expected a non-empty list or a low/high/step range, got {value!r}"
            )
    return space


def load_space(section: str, model: str, fixed: dict[str, Any] | None = None) -> dict[str, Spec]:
    """
    Read one model's search space and pin the settings that are not tuned.

    Parameters
    ----------
    section : str
        Top-level section of ``configs/models.json``, e.g. ``graph_clustering``.
    model : str
        Model inside the section, e.g. ``leiden_cpm``.
    fixed : dict or None, default None
        Values for the model's ``not_tuned`` settings (e.g. the label budget
        being run, or the frozen ``top_k``). Every ``not_tuned`` setting of the
        model must be given here.

    Returns
    -------
    dict of str to Choice or Range
        The search space with ``not_tuned`` settings as single-value choices.

    Raises
    ------
    ValueError
        If a ``not_tuned`` setting of the model has no value in ``fixed``.

    Notes
    -----
    Processing:
    1. Parse the model's entry with ``parse_space``.
    2. Replace each ``not_tuned`` setting with the value given in ``fixed``.
    """
    models = json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))
    space = parse_space(models[section][model])
    not_tuned = json.loads(TUNING_CONFIG.read_text(encoding="utf-8"))["not_tuned"]
    fixed = fixed or {}
    for path in not_tuned:
        path_section, path_model, name = path.split(".")
        if (path_section, path_model) != (section, model):
            continue
        if name not in fixed:
            raise ValueError(f"{path} is not tuned; pass its value in fixed")
        space[name] = Choice((fixed[name],))
    return space


def candidates(spec: Spec) -> list[Any]:
    """
    All candidate values of one hyperparameter.

    Parameters
    ----------
    spec : Choice or Range
        The hyperparameter's specification.

    Returns
    -------
    list
        The choices, or the range's points.

    Notes
    -----
    Processing:
    1. Return the choices or ``Range.points()``.
    """
    return list(spec.values) if isinstance(spec, Choice) else spec.points()


def grid_trials(space: dict[str, Spec], max_combinations: int) -> list[dict[str, Any]]:
    """
    Every combination of candidate values (grid search).

    Parameters
    ----------
    space : dict of str to Choice or Range
        The search space.
    max_combinations : int
        Refuse grids larger than this.

    Returns
    -------
    list of dict
        One setting per combination, in a fixed order.

    Raises
    ------
    ValueError
        If the grid has more than ``max_combinations`` settings; use random
        search or Optuna, or coarsen the ranges' steps.

    Notes
    -----
    Processing:
    1. List every hyperparameter's candidates and count the combinations.
    2. Build the Cartesian product.
    """
    names = list(space)
    values = [candidates(space[name]) for name in names]
    size = 1
    for options in values:
        size *= len(options)
    if size > max_combinations:
        raise ValueError(
            f"grid has {size} combinations, more than max_combinations={max_combinations}"
        )
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def random_trials(space: dict[str, Spec], n_trials: int, seed: int) -> list[dict[str, Any]]:
    """
    Settings drawn at random from the search space (random search).

    Each hyperparameter is drawn uniformly from its candidates; a range is
    sampled on its step grid. Duplicate settings are skipped, so a small space
    yields fewer than ``n_trials`` settings rather than repeats.

    Parameters
    ----------
    space : dict of str to Choice or Range
        The search space.
    n_trials : int
        Maximum number of distinct settings.
    seed : int
        Seed of the generator, so the same settings are drawn every run.

    Returns
    -------
    list of dict
        Distinct settings, in draw order.

    Notes
    -----
    Processing:
    1. Draw settings with a seeded ``random.Random`` until ``n_trials``
       distinct ones are found or the space is exhausted.
    """
    rng = random.Random(seed)
    options = {name: candidates(spec) for name, spec in space.items()}
    space_size = 1
    for values in options.values():
        space_size *= len(values)

    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(trials) < min(n_trials, space_size):
        setting = {name: rng.choice(values) for name, values in options.items()}
        key = json.dumps(setting, sort_keys=True)
        if key not in seen:
            seen.add(key)
            trials.append(setting)
    return trials


def suggest(trial: Any, space: dict[str, Spec]) -> dict[str, Any]:
    """
    Ask an Optuna trial for a value of every hyperparameter.

    Parameters
    ----------
    trial : optuna.Trial
        The current trial.
    space : dict of str to Choice or Range
        The search space.

    Returns
    -------
    dict
        The suggested setting.

    Notes
    -----
    Processing:
    1. Ranges become ``suggest_int`` or ``suggest_float`` with their step.
    2. Choices become ``suggest_categorical``. Values Optuna cannot store
       (lists such as ``[2, 4]``) are suggested by their position in the list.
    """
    setting: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, Range):
            if spec.is_int:
                setting[name] = trial.suggest_int(name, spec.low, spec.high, step=spec.step)
            else:
                value = trial.suggest_float(name, spec.low, spec.high, step=spec.step)
                setting[name] = round(value, 10)
        elif all(isinstance(v, OPTUNA_CATEGORICAL_TYPES) for v in spec.values):
            setting[name] = trial.suggest_categorical(name, list(spec.values))
        else:
            index = trial.suggest_categorical(f"{name}__index", list(range(len(spec.values))))
            setting[name] = spec.values[index]
    return setting


def load_settings() -> dict[str, Any]:
    """
    Read ``configs/tuning.json``.

    Returns
    -------
    dict
        The tuning settings.

    Notes
    -----
    Processing:
    1. Load the JSON file.
    """
    return json.loads(TUNING_CONFIG.read_text(encoding="utf-8"))


def tune(
    space: dict[str, Spec],
    objective: Callable[[dict[str, Any]], float],
    settings: dict[str, Any] | None = None,
) -> TuningResult:
    """
    Search the space for the setting with the highest objective.

    Parameters
    ----------
    space : dict of str to Choice or Range
        The search space, e.g. from ``load_space``.
    objective : callable
        Maps a setting to its dev score (higher is better).
    settings : dict or None, default None
        Search settings; ``configs/tuning.json`` when None.

    Returns
    -------
    TuningResult
        Best setting and score, and every trial.

    Raises
    ------
    ValueError
        If the strategy is unknown, or no trial finished.

    Notes
    -----
    Processing:
    1. Grid or random: build the list of settings, then score them in order
       until the list or ``max_seconds_per_model`` runs out.
    2. Optuna: run a seeded TPE study for ``n_trials`` trials or until the
       time limit, scoring each suggested setting.
    3. Return the best trial and the full trial history.
    """
    settings = settings or load_settings()
    strategy = settings["strategy"]
    seed = settings["seed"]
    deadline = time.perf_counter() + settings["max_seconds_per_model"]
    trials: list[dict[str, Any]] = []

    def evaluate(setting: dict[str, Any]) -> float:
        start = time.perf_counter()
        score = float(objective(setting))
        trials.append({"params": setting, "score": score, "seconds": time.perf_counter() - start})
        return score

    if strategy in ("grid", "random"):
        if strategy == "grid":
            plan = grid_trials(space, settings["grid"]["max_combinations"])
        else:
            plan = random_trials(space, settings["random"]["n_trials"], seed)
        for setting in plan:
            if time.perf_counter() > deadline:
                break
            evaluate(setting)
    elif strategy == "optuna":
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=settings["optuna"]["n_startup_trials"]
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: evaluate(suggest(trial, space)),
            n_trials=settings["optuna"]["n_trials"],
            timeout=max(deadline - time.perf_counter(), 0),
        )
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use grid, random or optuna")

    if not trials:
        raise ValueError("no trial finished within the time limit")
    best = max(trials, key=lambda t: t["score"])
    return TuningResult(best["params"], best["score"], trials, strategy)
