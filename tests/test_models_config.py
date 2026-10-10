import json

from dedup_bench import config

# Hyperparameters whose values are similarities, probabilities, cosine distances or
# fractions, so every candidate must lie in [0, 1].
UNIT_INTERVAL_KEYS = {
    "binarize",
    "cluster_selection_epsilon",
    "distance_threshold",
    "match_probability_threshold",
    "min_similarity",
    "partition_threshold",
    "resolution",
    "target_pair_completeness",
    "threshold",
}


def load_grids() -> dict:
    with open(config.MODELS_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def iter_hyperparameters():
    for section, models in load_grids().items():
        if section == "description":
            continue
        for model, params in models.items():
            if model == "description":
                continue
            for name, values in params.items():
                yield f"{section}.{model}.{name}", name, values


def test_every_hyperparameter_is_a_non_empty_list_of_candidates() -> None:
    for path, _, values in iter_hyperparameters():
        assert isinstance(values, list) and values, path


def test_similarity_hyperparameters_lie_in_the_unit_interval() -> None:
    for path, name, values in iter_hyperparameters():
        if name in UNIT_INTERVAL_KEYS:
            assert all(0 <= value <= 1 for value in values), path
