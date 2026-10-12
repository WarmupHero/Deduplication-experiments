import json

import pytest

from dedup_bench import config, tuning

# Hyperparameters whose values are similarities, probabilities, cosine distances or
# fractions, so every candidate must lie in [0, 1].
UNIT_INTERVAL_KEYS = {
    "binarize",
    "cluster_selection_epsilon",
    "distance_threshold",
    "match_probability_threshold",
    "min_similarity",
    "resolution",
    "target_pair_completeness",
    "threshold",
}


def load_models() -> dict:
    with open(config.MODELS_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def iter_models():
    for section, models in load_models().items():
        if section == "description":
            continue
        for model, params in models.items():
            if model != "description":
                yield section, model, params


def test_every_model_has_a_valid_search_space() -> None:
    for section, model, params in iter_models():
        space = tuning.parse_space(params)
        assert space, f"{section}.{model}"


def test_similarity_hyperparameters_lie_in_the_unit_interval() -> None:
    for section, model, params in iter_models():
        for name, spec in tuning.parse_space(params).items():
            if name in UNIT_INTERVAL_KEYS:
                values = tuning.candidates(spec)
                assert all(0 <= v <= 1 for v in values), f"{section}.{model}.{name}"


def test_tuning_settings_are_consistent_with_the_models() -> None:
    settings = tuning.load_settings()
    assert settings["strategy"] in {"grid", "random", "optuna"}
    models = load_models()
    for path in settings["not_tuned"]:
        section, model, name = path.split(".")
        assert name in models[section][model], path


@pytest.mark.parametrize("section, model", [(s, m) for s, m, _ in iter_models()])
def test_every_grid_stays_within_the_cap(section: str, model: str) -> None:
    # Values for the settings that are not tuned; any valid value will do here.
    fixed = {"top_k": 25, "target_pair_completeness": 0.95, "label_budget": 50}
    space = tuning.load_space(section, model, fixed)
    trials = tuning.grid_trials(space, tuning.load_settings()["grid"]["max_combinations"])
    assert trials
