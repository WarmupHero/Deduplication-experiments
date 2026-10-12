# Deduplication-experiments

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A benchmark of deduplication and entity-resolution methods on Kaggle and academic datasets.

## Setup

Requires Python 3.12. On Windows, the `pyjedai` extra compiles C++ code (`py-stringcompare`),
so install
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first;
the MSVC x64/x86 build tools and a Windows 11 SDK are enough.

```bash
make create_environment   # python -m venv .venv
make requirements         # pip install -e ".[dev,embeddings,pyjedai]"
make test
make data                 # download and prepare the datasets (see below)
make smoke                # run every library once on Febrl 3 and the restaurants
```

## Data

`make data` downloads six datasets into `data/raw/` (with SHA-256 checksums) and prepares
each one in `data/processed/<dataset>/`: `records.parquet` (shuffled records with neutral
IDs) and `truth.parquet` (the true entity of every record). The Kaggle downloads need a
Kaggle API token in the `KAGGLE_API_TOKEN` environment variable.

| Dataset | Domain | Source | Records | Entities | Card |
|---|---|---|---:|---:|---|
| Febrl 3 | synthetic people | `recordlinkage` package | 5,000 | 2,000 | [febrl3](references/febrl3.md) |
| Restaurants | restaurant listings | Kaggle | 864 | 752 | [restaurants](references/restaurants.md) |
| PriceRunner | product offers | Kaggle | 35,311 | 13,233 | [pricerunner](references/pricerunner.md) |
| Skroutz | product offers | Kaggle | 238,170 | 63,995 | [skroutz](references/skroutz.md) |
| Abt-Buy | product offers | Leipzig | 2,173 | 1,076 | [abt_buy](references/abt_buy.md) |
| Cora | citations | HPI | 1,879 | 182 | [cora](references/cora.md) |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── configs            <- Hyperparameter grids for every model (models.json), tuned on the dev split
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Project documentation
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         dedup_bench and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── tests              <- Unit tests, run with `make test`
│
└── dedup_bench        <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes dedup_bench a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

