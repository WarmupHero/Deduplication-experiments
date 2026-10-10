# deduplication

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A benchmark of deduplication and entity-resolution methods on Kaggle and academic datasets.

## Setup

Requires Python 3.12. On Windows, the `dedupe` and `pyjedai` extras compile C/C++ code
(`affinegap`, `dedupe-Levenshtein-search`, `py-stringcompare`), so install
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first;
the MSVC x64/x86 build tools and a Windows 11 SDK are enough.

```bash
make create_environment   # python -m venv .venv
make requirements         # pip install -e ".[dev,dedupe,embeddings,pyjedai]"
make test
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
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

