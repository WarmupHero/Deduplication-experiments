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
make candidates           # split every dataset and build its frozen candidate pairs
make experiment DATASET=cora METHOD=graph_clustering.connected_components
                          # tune on validation, test once, save to reports/results/
make experiments          # every method on every dataset (about 6.5 h), resuming
make report               # results tables in reports/tables/
```

Skroutz is not part of `make experiments`. Run it on its own (about 6-8 h after its candidate
sets, which take about 1.5-2 h):

```bash
python -m dedup_bench.blocking.candidates skroutz
python -m dedup_bench.experiment run-all --dataset skroutz
```

## Limitations

- **Methods not run on Skroutz (out of memory).** The supervised random forest (unlimited
  depth, 500 trees: 2 GB on PriceRunner, about 14 GB on Skroutz), hierarchical clustering and
  embedding agglomerative clustering (a dense distance matrix per connected group of pairs,
  about 13 GB for a 40,000-record group) would exhaust a 16 GB machine. They are listed under
  `excluded` in `configs/experiment.json`, marked `excluded` in the results tables and
  explained in `reports/tables/health.md`.
- **Memory guard.** `run-all` stops any run that leaves less than `min_free_memory_mb`
  (2 GB) of the machine's memory free and reports it, rather than letting the machine run out.

## Data

`make data` downloads six datasets into `data/raw/` (with SHA-256 checksums) and prepares
each one in `data/processed/<dataset>/`: `records.parquet` (shuffled records with neutral
IDs) and `truth.parquet` (the true entity of every record). The Kaggle downloads need a
Kaggle API token in the `KAGGLE_API_TOKEN` environment variable.

`records_normalised.parquet` holds the same records with light, shared normalisation: accents
stripped, case-folded, punctuation and spacing collapsed, and phone numbers reduced to digits.
No stemming, stopword removal or spelling correction, since names, model codes and typos are
what tells entities apart. Every method of the benchmark reads this file.

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
├── configs            <- Search spaces (models.json) and tuning (tuning.json), datasets and
│                         splits (datasets.json, experiment.json); datasets_smoke.json is
│                         used only by make smoke
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

