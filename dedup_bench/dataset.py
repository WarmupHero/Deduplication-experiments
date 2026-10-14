"""
Download the benchmark's six datasets and prepare them in one common format.

Input: three Kaggle files (PriceRunner and Skroutz from
``lakritidis/product-clustering-matching-classification``, the Fodors-Zagat
restaurants from ``adrianevensen/restaurants-entity-resolution``), Abt-Buy from
the Leipzig benchmark site, Cora from HPI, and Febrl 3 from the
``recordlinkage`` package. Kaggle downloads need the ``KAGGLE_API_TOKEN``
environment variable.

Output: for every dataset, the untouched files and a ``manifest.json`` with
their SHA-256 checksums in ``data/raw/<dataset>/``, and two Parquet files in
``data/processed/<dataset>/``:

- ``records.parquet``: ``record_id``, ``source`` and the dataset's text fields
  (missing values are null), in shuffled order with neutral IDs so that
  neither IDs nor row order reveal the entities
- ``records_normalised.parquet``: the same records with every text field
  normalised (``normalise_text``, or ``normalise_digits`` for the fields
  listed in ``configs/datasets.json``); the benchmark's methods read this
- ``truth.parquet``: ``record_id``, ``original_id`` (the ID in the raw file)
  and ``entity_id``, the true entity of every record; methods never read it

Run ``python -m dedup_bench.dataset download`` and then
``python -m dedup_bench.dataset prepare`` (``make data`` runs both). Later
steps read the prepared files with ``load_prepared`` and ``record_texts``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
from typing import Annotated, Any
import urllib.request
import zipfile

from loguru import logger
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import typer

from dedup_bench.config import DATASETS_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Fixed seed for shuffling the records, so prepared files are identical on every run.
SHUFFLE_SEED = 20261010

# HPI refuses requests that carry Python's default user agent.
USER_AGENT = "Mozilla/5.0 (compatible; dedup-bench)"

LAKRITIDIS = "lakritidis/product-clustering-matching-classification"
HPI_CORA = (
    "https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/projekte/repeatability/CORA/"
)


@dataclass(frozen=True)
class RawFile:
    """
    One file to download into a dataset's raw folder.

    Exactly one of ``url``, ``kaggle_dataset`` or ``package_path`` is set.

    Attributes
    ----------
    name : str
        File name in ``data/raw/<dataset>/``. Kaggle may deliver a large file
        zipped, in which case ``.zip`` is appended on disk.
    url : str or None
        Direct download URL.
    kaggle_dataset : str or None
        Kaggle dataset slug (``owner/dataset``); ``kaggle_file`` names the file in it.
    kaggle_file : str or None
        Path of the file inside the Kaggle dataset.
    package_path : str or None
        Path of a data file shipped inside an installed package, relative to
        that package's folder, written as ``package:relative/path``.
    """

    name: str
    url: str | None = None
    kaggle_dataset: str | None = None
    kaggle_file: str | None = None
    package_path: str | None = None


def file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 checksum of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest (64 characters).

    Notes
    -----
    Processing:
    1. Read the file in 1 MiB chunks and feed each chunk to the hash.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_raw_file(raw_dir: Path, name: str) -> Path:
    """
    Find a downloaded file, which Kaggle may have stored zipped.

    Parameters
    ----------
    raw_dir : Path
        The dataset's raw folder.
    name : str
        File name as declared in ``RawFile.name``.

    Returns
    -------
    Path
        ``raw_dir / name`` if it exists, otherwise ``raw_dir / (name + ".zip")``.

    Raises
    ------
    FileNotFoundError
        If neither file exists; run the download first.

    Notes
    -----
    Processing:
    1. Return the plain file if present, else the zipped one, else raise.
    """
    for candidate in (raw_dir / name, raw_dir / f"{name}.zip"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{raw_dir / name} is missing; run the download first.")


def read_raw_bytes(raw_dir: Path, name: str, member: str | None = None) -> bytes:
    """
    Read a raw file's bytes, unzipping it when needed.

    Parameters
    ----------
    raw_dir : Path
        The dataset's raw folder.
    name : str
        File name as declared in ``RawFile.name``.
    member : str or None, default None
        For a zip archive with several files, the member to read. When None,
        a zip must contain exactly one file.

    Returns
    -------
    bytes
        The file's (decompressed) content.

    Raises
    ------
    ValueError
        If ``member`` is None and the zip holds more than one file.

    Notes
    -----
    Processing:
    1. Locate the file with ``find_raw_file``.
    2. If it is a zip, read ``member`` (or its only file) from it; otherwise
       read the file itself.
    """
    path = find_raw_file(raw_dir, name)
    if not zipfile.is_zipfile(path):
        return path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        if member is None:
            members = archive.namelist()
            if len(members) != 1:
                raise ValueError(f"{path} holds {len(members)} files; name the member to read.")
            member = members[0]
        return archive.read(member)


def entities_from_pairs(record_ids: pd.Series, pairs: pd.DataFrame) -> pd.Series:
    """
    Turn a list of matching pairs into an entity ID for every record.

    Records linked directly or through other records form one entity
    (transitive closure); records in no pair are singletons.

    Parameters
    ----------
    record_ids : pd.Series of str, shape (n_records,)
        Every record of the dataset, unique.
    pairs : pd.DataFrame, shape (n_pairs, 2)
        Matching pairs; both columns hold IDs from ``record_ids``.

    Returns
    -------
    pd.Series of str, shape (n_records,)
        Entity ID of each record, aligned with ``record_ids``. An entity is
        named after its smallest record ID, so the result does not depend on
        the order of the input.

    Raises
    ------
    ValueError
        If a pair refers to a record ID that is not in ``record_ids``.

    Notes
    -----
    Processing:
    1. Map record IDs to positions and check every paired ID is known.
    2. Build an undirected sparse graph with one edge per pair and label its
       connected components with SciPy.
    3. Name each component after its smallest record ID.
    """
    positions = pd.Series(np.arange(len(record_ids)), index=record_ids.to_numpy())
    left = pd.Series(pairs.iloc[:, 0].to_numpy())
    right = pd.Series(pairs.iloc[:, 1].to_numpy())
    unknown = set(left[~left.isin(positions.index)]) | set(right[~right.isin(positions.index)])
    if unknown:
        raise ValueError(
            f"{len(unknown)} paired record IDs are not in the records, e.g. {sorted(unknown)[:3]}"
        )

    n = len(record_ids)
    graph = coo_matrix(
        (np.ones(len(pairs)), (positions[left].to_numpy(), positions[right].to_numpy())),
        shape=(n, n),
    )
    _, labels = connected_components(graph, directed=False)

    names = pd.Series(record_ids.to_numpy()).groupby(labels).transform("min")
    return pd.Series(names.to_numpy(), index=record_ids.index, name="entity_id")


def finish(
    records: pd.DataFrame, entity_ids: pd.Series, source: str | pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bring a dataset's records and truth into the common format.

    Parameters
    ----------
    records : pd.DataFrame, shape (n_records, 1 + n_fields)
        A ``record_id`` column followed by the dataset's text fields, as strings.
    entity_ids : pd.Series of str, shape (n_records,)
        True entity of each record, aligned with ``records``.
    source : str or pd.Series of str
        Name of the source table: one name for all records, or one per record.

    Returns
    -------
    records : pd.DataFrame, shape (n_records, 2 + n_fields)
        ``record_id``, ``source`` and the text fields; empty or whitespace-only
        values become null. Rows are shuffled and ``record_id`` is a neutral
        ID (``r0000``, ``r0001``, ...), because the original IDs and file order
        often reveal the entity (Febrl's ``rec-552-dup-3``, files sorted by
        cluster).
    truth : pd.DataFrame, shape (n_records, 3)
        ``record_id``, ``original_id`` (the ID in the raw file) and ``entity_id``.

    Raises
    ------
    ValueError
        If original record IDs are not unique or a record has no entity.

    Notes
    -----
    Processing:
    1. Check that record IDs are unique and every record has an entity.
    2. Strip every field, turn empty strings into nulls and insert ``source``
       after ``record_id``.
    3. Pair each record with its original ID and entity ID.
    4. Shuffle the rows with ``SHUFFLE_SEED`` and number them as new record IDs.
    """
    records = records.reset_index(drop=True).astype("string")
    if records["record_id"].duplicated().any():
        raise ValueError("record IDs must be unique")
    if entity_ids.isna().any() or (entity_ids == "").any():
        raise ValueError("every record needs an entity ID")

    fields = records.columns.drop("record_id")
    records[fields] = records[fields].apply(lambda col: col.str.strip().replace("", pd.NA))
    sources = source if isinstance(source, pd.Series) else pd.Series(source, index=records.index)
    records.insert(1, "source", sources.reset_index(drop=True).astype("string"))

    truth = pd.DataFrame(
        {
            "original_id": records["record_id"],
            "entity_id": entity_ids.reset_index(drop=True),
        }
    ).astype("string")

    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(records))
    records = records.iloc[order].reset_index(drop=True)
    truth = truth.iloc[order].reset_index(drop=True)
    width = len(str(max(len(records) - 1, 0)))
    new_ids = pd.Series([f"r{i:0{width}d}" for i in range(len(records))], dtype="string")
    records["record_id"] = new_ids
    truth.insert(0, "record_id", new_ids)
    return records, truth


def prepare_febrl3(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare Febrl 3, synthetic person records with generated duplicates.

    Parameters
    ----------
    raw_dir : Path
        Folder holding ``dataset3.csv``.

    Returns
    -------
    records, truth : pd.DataFrame
        See ``finish``. Fields: given_name, surname, street_number, address_1,
        address_2, suburb, postcode, state, date_of_birth, soc_sec_id.

    Notes
    -----
    Processing:
    1. Read the CSV as strings; values are separated by a comma and a space.
    2. Take the entity from ``rec_id``: ``rec-N-org`` and ``rec-N-dup-k`` both
       belong to entity ``rec-N``.
    """
    data = read_raw_bytes(raw_dir, "dataset3.csv")
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, skipinitialspace=True)
    entity_ids = df["rec_id"].str.extract(r"^(rec-\d+)-", expand=False)
    return finish(df.rename(columns={"rec_id": "record_id"}), entity_ids, "febrl3")


def prepare_restaurants(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the Fodors-Zagat restaurants, whose ``cluster`` column is the entity.

    Parameters
    ----------
    raw_dir : Path
        Folder holding ``restaurants.csv``.

    Returns
    -------
    records, truth : pd.DataFrame
        See ``finish``. Fields: name, address, city, phone, category.
        Original IDs are the row numbers, zero-padded to four digits.

    Notes
    -----
    Processing:
    1. Read the CSV as strings.
    2. Number the rows as record IDs and use ``cluster`` as the entity.
    """
    data = read_raw_bytes(raw_dir, "restaurants.csv")
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    records = df.drop(columns="cluster")
    records.insert(0, "record_id", [f"{i:04d}" for i in range(len(df))])
    return finish(records, df["cluster"], "restaurants")


# The lakritidis CSVs have no header row.
PRODUCT_COLUMNS = [
    "product_id",
    "title",
    "vendor_id",
    "cluster_id",
    "cluster_label",
    "category_id",
    "category_label",
]


def prepare_product_clusters(
    raw_dir: Path, file_name: str, source: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare a PriceRunner or Skroutz product-offer file.

    Only the offer title and the vendor are kept as fields. The cluster label
    (the canonical product name) and the category are assigned per product,
    so they would reveal the answer.

    Parameters
    ----------
    raw_dir : Path
        Folder holding the file (possibly zipped by Kaggle).
    file_name : str
        ``pricerunner_aggregate.csv`` or ``skroutz_aggregate.csv``.
    source : str
        Value of the ``source`` column.

    Returns
    -------
    records, truth : pd.DataFrame
        See ``finish``. Fields: title, vendor_id. Original IDs are the product
        (offer) IDs and entities the cluster IDs.

    Notes
    -----
    Processing:
    1. Read the headerless CSV as strings with the column names of
       ``PRODUCT_COLUMNS``.
    2. Keep product ID, title and vendor; use the cluster ID as the entity.
    """
    data = read_raw_bytes(raw_dir, file_name)
    df = pd.read_csv(
        io.BytesIO(data), header=None, names=PRODUCT_COLUMNS, dtype=str, keep_default_na=False
    )
    records = df[["product_id", "title", "vendor_id"]].rename(columns={"product_id": "record_id"})
    return finish(records, df["cluster_id"], source)


def prepare_abt_buy(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare Abt-Buy by pooling its two product tables into one.

    Parameters
    ----------
    raw_dir : Path
        Folder holding ``Abt-Buy.zip`` with ``Abt.csv``, ``Buy.csv`` and
        ``abt_buy_perfectMapping.csv``.

    Returns
    -------
    records, truth : pd.DataFrame
        See ``finish``. Fields: name, description, manufacturer (Buy only),
        price. Original IDs are prefixed ``abt-`` or ``buy-`` and ``source`` is
        ``abt`` or ``buy``.

    Notes
    -----
    Processing:
    1. Read both tables as strings; ``Abt.csv`` is Latin-1, the others UTF-8.
    2. Prefix the IDs with their table and stack the tables.
    3. Turn the perfect mapping into entities with ``entities_from_pairs``.
    """

    def read(member: str, encoding: str) -> pd.DataFrame:
        data = read_raw_bytes(raw_dir, "Abt-Buy.zip", member)
        return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding=encoding)

    abt = read("Abt.csv", "latin-1")
    buy = read("Buy.csv", "utf-8")
    mapping = read("abt_buy_perfectMapping.csv", "utf-8")

    abt["id"] = "abt-" + abt["id"]
    buy["id"] = "buy-" + buy["id"]
    records = pd.concat([abt, buy], ignore_index=True).rename(columns={"id": "record_id"})
    records = records[["record_id", "name", "description", "manufacturer", "price"]].fillna("")
    sources = pd.Series(["abt"] * len(abt) + ["buy"] * len(buy))

    pairs = pd.DataFrame({"a": "abt-" + mapping["idAbt"], "b": "buy-" + mapping["idBuy"]})
    entity_ids = entities_from_pairs(records["record_id"], pairs)
    return finish(records, entity_ids, sources)


def prepare_cora(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare HPI's Cora citations, whose duplicates come as a list of pairs.

    Parameters
    ----------
    raw_dir : Path
        Folder holding ``cora.tsv`` and ``cora_DPL.tsv``.

    Returns
    -------
    records, truth : pd.DataFrame
        See ``finish``. Fields: the 16 citation columns of ``cora.tsv``
        (authors, title, booktitle, journal, year, ...). Original IDs are the
        file's ``id`` column.

    Notes
    -----
    Processing:
    1. Read both tab-separated files as strings, without quote handling,
       since citations contain unbalanced quotes.
    2. Turn the duplicate-pair list into entities with ``entities_from_pairs``.
    """

    def read(name: str) -> pd.DataFrame:
        data = read_raw_bytes(raw_dir, name)
        return pd.read_csv(io.BytesIO(data), sep="\t", dtype=str, keep_default_na=False, quoting=3)

    cora = read("cora.tsv")
    pairs = read("cora_DPL.tsv")
    records = cora.rename(columns={"id": "record_id"})
    records = records[["record_id", *records.columns.drop("record_id")]]
    entity_ids = entities_from_pairs(records["record_id"], pairs[["id1", "id2"]])
    return finish(records, entity_ids, "cora")


@dataclass(frozen=True)
class Dataset:
    """
    A benchmark dataset: where its files come from and how to prepare them.

    Attributes
    ----------
    files : tuple of RawFile
        Files to download into ``data/raw/<dataset>/``.
    prepare : callable
        Function from the raw folder to ``(records, truth)``.
    n_records : int
        Expected number of records after preparation.
    n_entities : int
        Expected number of true entities after preparation.
    """

    files: tuple[RawFile, ...]
    prepare: Callable[[Path], tuple[pd.DataFrame, pd.DataFrame]]
    n_records: int
    n_entities: int


DATASETS: dict[str, Dataset] = {
    "febrl3": Dataset(
        files=(RawFile("dataset3.csv", package_path="recordlinkage:datasets/febrl/dataset3.csv"),),
        prepare=prepare_febrl3,
        n_records=5000,
        n_entities=2000,
    ),
    "restaurants": Dataset(
        files=(
            RawFile(
                "restaurants.csv",
                kaggle_dataset="adrianevensen/restaurants-entity-resolution",
                kaggle_file="restaurants.csv",
            ),
        ),
        prepare=prepare_restaurants,
        n_records=864,
        n_entities=752,
    ),
    "pricerunner": Dataset(
        files=(
            RawFile(
                "pricerunner_aggregate.csv",
                kaggle_dataset=LAKRITIDIS,
                kaggle_file="products_csv/pricerunner_aggregate.csv",
            ),
        ),
        prepare=lambda raw_dir: prepare_product_clusters(
            raw_dir, "pricerunner_aggregate.csv", "pricerunner"
        ),
        n_records=35311,
        n_entities=13233,
    ),
    "skroutz": Dataset(
        files=(
            RawFile(
                "skroutz_aggregate.csv",
                kaggle_dataset=LAKRITIDIS,
                kaggle_file="products_csv/skroutz_aggregate.csv",
            ),
        ),
        prepare=lambda raw_dir: prepare_product_clusters(
            raw_dir, "skroutz_aggregate.csv", "skroutz"
        ),
        n_records=238170,
        n_entities=63995,
    ),
    "abt_buy": Dataset(
        files=(
            RawFile("Abt-Buy.zip", url="https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip"),
        ),
        prepare=prepare_abt_buy,
        n_records=2173,
        n_entities=1076,
    ),
    "cora": Dataset(
        files=(
            RawFile("cora.tsv", url=HPI_CORA + "cora.tsv"),
            RawFile("cora_DPL.tsv", url=HPI_CORA + "cora_DPL.tsv"),
        ),
        prepare=prepare_cora,
        n_records=1879,
        n_entities=182,
    ),
}


def fetch(raw_file: RawFile, raw_dir: Path) -> Path:
    """
    Download one raw file into a dataset's raw folder.

    Parameters
    ----------
    raw_file : RawFile
        What to download and where it comes from.
    raw_dir : Path
        The dataset's raw folder; must exist.

    Returns
    -------
    Path
        The file written (with ``.zip`` appended if Kaggle delivered it zipped).

    Raises
    ------
    ValueError
        If ``raw_file`` names no source.

    Notes
    -----
    Processing:
    1. Package file: copy it from the installed package.
    2. URL: download it with a browser-like user agent.
    3. Kaggle: download it with the Kaggle API, which authenticates from the
       ``KAGGLE_API_TOKEN`` environment variable.
    """
    target = raw_dir / raw_file.name
    if raw_file.package_path:
        package, relative = raw_file.package_path.split(":", 1)
        module = importlib.import_module(package)
        shutil.copyfile(Path(module.__file__).parent / relative, target)
        return target
    if raw_file.url:
        request = urllib.request.Request(raw_file.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as f:
            shutil.copyfileobj(response, f)
        return target
    if raw_file.kaggle_dataset:
        # Imported here: importing the Kaggle package needs credentials.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(
            raw_file.kaggle_dataset, raw_file.kaggle_file, path=str(raw_dir), quiet=True
        )
        return find_raw_file(raw_dir, raw_file.name)
    raise ValueError(f"{raw_file.name} has no source")


def download_dataset(name: str, force: bool = False) -> None:
    """
    Download a dataset's raw files and record their checksums.

    Parameters
    ----------
    name : str
        Key of ``DATASETS``.
    force : bool, default False
        Download again even if the files already exist.

    Returns
    -------
    None
        Writes the files and ``manifest.json`` (file name, source, SHA-256,
        download time in UTC) to ``data/raw/<name>/``.

    Notes
    -----
    Processing:
    1. Skip the dataset if its manifest exists and ``force`` is False.
    2. Fetch every file and hash it.
    3. Write the manifest.
    """
    raw_dir = RAW_DATA_DIR / name
    manifest_path = raw_dir / "manifest.json"
    if manifest_path.exists() and not force:
        logger.info(f"{name}: already downloaded")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for raw_file in DATASETS[name].files:
        path = fetch(raw_file, raw_dir)
        source = (
            raw_file.url
            or raw_file.package_path
            or f"kaggle:{raw_file.kaggle_dataset}/{raw_file.kaggle_file}"
        )
        entries.append(
            {
                "file": path.name,
                "source": source,
                "sha256": file_sha256(path),
                "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        logger.info(f"{name}: {path.name} ({path.stat().st_size:,} bytes)")
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def verify_manifest(raw_dir: Path) -> None:
    """
    Check that the raw files still match their recorded checksums.

    Parameters
    ----------
    raw_dir : Path
        A dataset's raw folder with ``manifest.json``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the manifest is missing; run the download first.
    ValueError
        If a file's checksum differs from the manifest.

    Notes
    -----
    Processing:
    1. Read the manifest and recompute every listed file's SHA-256.
    """
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is missing; run the download first.")
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
        if file_sha256(raw_dir / entry["file"]) != entry["sha256"]:
            raise ValueError(f"{raw_dir / entry['file']} changed since it was downloaded")


def prepare_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare one dataset and save it in the common format.

    Parameters
    ----------
    name : str
        Key of ``DATASETS``.

    Returns
    -------
    records, truth : pd.DataFrame
        See ``finish``; also written to ``data/processed/<name>/``.

    Raises
    ------
    ValueError
        If the record or entity count differs from the expected one.

    Notes
    -----
    Processing:
    1. Verify the raw files against the manifest.
    2. Run the dataset's prepare function and check the counts.
    3. Write ``records.parquet`` and ``truth.parquet``.
    """
    dataset = DATASETS[name]
    raw_dir = RAW_DATA_DIR / name
    verify_manifest(raw_dir)

    records, truth = dataset.prepare(raw_dir)
    n_records, n_entities = len(records), truth["entity_id"].nunique()
    if (n_records, n_entities) != (dataset.n_records, dataset.n_entities):
        raise ValueError(
            f"{name}: {n_records} records / {n_entities} entities, "
            f"expected {dataset.n_records} / {dataset.n_entities}"
        )

    digits_fields = load_datasets_config().get(name, {}).get("digits_fields", [])
    out_dir = PROCESSED_DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(out_dir / "records.parquet", index=False)
    normalise_records(records, digits_fields).to_parquet(
        out_dir / "records_normalised.parquet", index=False
    )
    truth.to_parquet(out_dir / "truth.parquet", index=False)
    logger.success(f"{name}: {n_records:,} records, {n_entities:,} entities")
    return records, truth


def load_datasets_config() -> dict[str, Any]:
    """
    Read ``configs/datasets.json``.

    Returns
    -------
    dict
        Per dataset, its text fields and normalisation settings.

    Notes
    -----
    Processing:
    1. Load the JSON file.
    """
    return json.loads(DATASETS_CONFIG.read_text(encoding="utf-8"))


def normalise_text(values: pd.Series) -> pd.Series:
    """
    Light, method-independent text normalisation for entity resolution.

    Removes differences of form that never tell two entities apart (accents,
    case, punctuation, spacing) and keeps everything that might (words,
    digits, model codes). Deliberately no stemming, stopword removal or
    spelling correction: they change names and codes, and typos are signal.

    Parameters
    ----------
    values : pd.Series of str, shape (n,)
        Text values; nulls allowed.

    Returns
    -------
    pd.Series of str, shape (n,)
        Normalised text; null where the input is null or nothing is left.

    Notes
    -----
    Processing:
    1. Decompose with Unicode NFKD (``é`` becomes ``e`` plus an accent, ``ﬁ``
       becomes ``fi``) and drop the combining accents.
    2. Case-fold (a stronger lowercase: ``ß`` becomes ``ss``).
    3. Replace every run of characters that are not letters or digits
       (punctuation, underscores, spaces) with one space, and trim.
    """
    text = values.astype("string").str.normalize("NFKD")
    text = text.str.replace(r"[̀-ͯ]", "", regex=True).str.casefold()
    text = text.str.replace(r"[\W_]+", " ", regex=True).str.strip()
    return text.mask(text == "")


def normalise_digits(values: pd.Series) -> pd.Series:
    """
    Keep only the digits, for fields such as phone numbers.

    Parameters
    ----------
    values : pd.Series of str, shape (n,)
        Values; nulls allowed.

    Returns
    -------
    pd.Series of str, shape (n,)
        The digits of each value; null where there are none.

    Notes
    -----
    Processing:
    1. Delete every character that is not a digit.
    """
    digits = values.astype("string").str.replace(r"\D+", "", regex=True)
    return digits.mask(digits == "")


def normalise_records(records: pd.DataFrame, digits_fields: list[str]) -> pd.DataFrame:
    """
    Normalise every text field of a prepared dataset.

    Parameters
    ----------
    records : pd.DataFrame, shape (n_records, n_columns)
        Prepared records: ``record_id``, ``source`` and text fields.
    digits_fields : list of str
        Fields reduced to their digits; every other field gets
        ``normalise_text``.

    Returns
    -------
    pd.DataFrame, same shape
        The records with normalised fields; ``record_id`` and ``source`` are
        unchanged.

    Notes
    -----
    Processing:
    1. Apply ``normalise_digits`` or ``normalise_text`` to each field.
    """
    out = records.copy()
    for field in out.columns.drop(["record_id", "source"]):
        normalise = normalise_digits if field in digits_fields else normalise_text
        out[field] = normalise(out[field])
    return out


def load_prepared(name: str, normalised: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Read a prepared dataset.

    Parameters
    ----------
    name : str
        Key of ``DATASETS``.
    normalised : bool, default False
        Read ``records_normalised.parquet`` (what the benchmark's methods
        use) instead of the original text in ``records.parquet``.

    Returns
    -------
    records : pd.DataFrame, shape (n_records, n_columns)
        The records as written by ``prepare_dataset``.
    truth : pd.Series of str, shape (n_records,)
        True entity of every record, indexed by record ID.

    Notes
    -----
    Processing:
    1. Read the records and truth files and index the entity IDs by record ID.
    """
    file_name = "records_normalised.parquet" if normalised else "records.parquet"
    records = pd.read_parquet(PROCESSED_DATA_DIR / name / file_name)
    truth = pd.read_parquet(PROCESSED_DATA_DIR / name / "truth.parquet")
    return records, pd.Series(truth["entity_id"].to_numpy(), index=truth["record_id"].to_numpy())


def record_texts(records: pd.DataFrame, fields: list[str]) -> pd.Series:
    """
    Join a record's non-null fields into one string.

    Parameters
    ----------
    records : pd.DataFrame, shape (n_records, n_columns)
        Prepared records with ``record_id`` and the listed fields.
    fields : list of str
        Fields to join, in order.

    Returns
    -------
    pd.Series of str, shape (n_records,)
        Space-joined text, indexed by record ID.

    Notes
    -----
    Processing:
    1. Fill nulls with empty strings, join the fields with spaces and collapse
       repeated spaces.
    """
    text = records[fields].fillna("").astype(str).agg(" ".join, axis=1)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.Series(text.to_numpy(), index=records["record_id"].to_numpy())


def selected(names: list[str] | None) -> list[str]:
    """
    Validate dataset names given on the command line.

    Parameters
    ----------
    names : list of str or None
        Requested datasets; None or empty means all of them.

    Returns
    -------
    list of str
        The datasets to process, in ``DATASETS`` order.

    Raises
    ------
    typer.BadParameter
        If a name is not in ``DATASETS``.

    Notes
    -----
    Processing:
    1. Default to every dataset; reject unknown names.
    """
    if not names:
        return list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise typer.BadParameter(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
    return [n for n in DATASETS if n in names]


@app.command()
def download(
    names: Annotated[
        list[str] | None, typer.Argument(help="Datasets to download (default: all).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Download again even if already present.")] = False,
) -> None:
    """
    Download the raw files of the given datasets (default: all).

    \f
    Parameters
    ----------
    names : list of str or None
        Datasets to download; None or empty means all.
    force : bool, default False
        Download again even if the files already exist.

    Returns
    -------
    None
        Writes ``data/raw/<dataset>/`` for each dataset.

    Notes
    -----
    Processing:
    1. Validate the names with ``selected``.
    2. Call ``download_dataset`` for each.
    """
    for name in selected(names):
        download_dataset(name, force)


@app.command()
def prepare(
    names: Annotated[
        list[str] | None, typer.Argument(help="Datasets to prepare (default: all).")
    ] = None,
) -> None:
    """
    Prepare the given datasets (default: all) in the common format.

    \f
    Parameters
    ----------
    names : list of str or None
        Datasets to prepare; None or empty means all.

    Returns
    -------
    None
        Writes ``data/processed/<dataset>/`` for each dataset.

    Notes
    -----
    Processing:
    1. Validate the names with ``selected``.
    2. Call ``prepare_dataset`` for each.
    """
    for name in selected(names):
        prepare_dataset(name)


if __name__ == "__main__":
    app()
