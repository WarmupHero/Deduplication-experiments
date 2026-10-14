import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest
import typer

from dedup_bench import dataset
from dedup_bench.config import PROCESSED_DATA_DIR


def entity_of(records: pd.DataFrame, truth: pd.DataFrame) -> dict[str, str]:
    """Map each original ID to its entity, independent of the shuffled order."""
    assert (records["record_id"] == truth["record_id"]).all()
    return dict(zip(truth["original_id"], truth["entity_id"]))


def test_entities_from_pairs_follows_transitive_links_and_keeps_singletons() -> None:
    ids = pd.Series(["d", "b", "a", "c", "e"])
    pairs = pd.DataFrame({"x": ["a", "c"], "y": ["b", "b"]})
    entities = dataset.entities_from_pairs(ids, pairs)
    assert entities.tolist() == ["d", "a", "a", "a", "e"]


def test_entities_from_pairs_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="not in the records"):
        dataset.entities_from_pairs(pd.Series(["a", "b"]), pd.DataFrame({"x": ["a"], "y": ["z"]}))


def test_finish_shuffles_renames_and_keeps_the_truth_mapping() -> None:
    records = pd.DataFrame({"record_id": ["x1", "x2", "x3"], "name": [" ann ", "", "bob"]})
    out, truth = dataset.finish(records, pd.Series(["e1", "e1", "e2"]), "demo")

    assert out["record_id"].tolist() == ["r0", "r1", "r2"]
    assert out.columns.tolist() == ["record_id", "source", "name"]
    assert entity_of(out, truth) == {"x1": "e1", "x2": "e1", "x3": "e2"}
    names = dict(zip(truth["original_id"], out["name"]))
    assert names["x1"] == "ann" and pd.isna(names["x2"])

    again, _ = dataset.finish(records, pd.Series(["e1", "e1", "e2"]), "demo")
    assert again.equals(out)


def test_finish_rejects_duplicate_ids_and_missing_entities() -> None:
    with pytest.raises(ValueError, match="unique"):
        dataset.finish(pd.DataFrame({"record_id": ["a", "a"]}), pd.Series(["e", "e"]), "s")
    with pytest.raises(ValueError, match="entity"):
        dataset.finish(pd.DataFrame({"record_id": ["a", "b"]}), pd.Series(["e", None]), "s")


def test_prepare_febrl3_reads_the_entity_from_rec_id(tmp_path: Path) -> None:
    (tmp_path / "dataset3.csv").write_text(
        "rec_id, given_name, surname\n"
        "rec-7-org, ann, lee\n"
        "rec-7-dup-0, anne, lee\n"
        "rec-12-org, bob, ray\n",
        encoding="utf-8",
    )
    records, truth = dataset.prepare_febrl3(tmp_path)
    assert entity_of(records, truth) == {
        "rec-7-org": "rec-7",
        "rec-7-dup-0": "rec-7",
        "rec-12-org": "rec-12",
    }
    assert set(records["given_name"]) == {"ann", "anne", "bob"}


def test_prepare_restaurants_uses_the_cluster_column(tmp_path: Path) -> None:
    (tmp_path / "restaurants.csv").write_text(
        "name,address,city,phone,category,cluster\n"
        "a,1 st,la,1,x,0\n"
        "a.,1 st.,la,1,y,0\n"
        "b,2 st,sf,2,z,1\n",
        encoding="utf-8",
    )
    records, truth = dataset.prepare_restaurants(tmp_path)
    assert entity_of(records, truth) == {"0000": "0", "0001": "0", "0002": "1"}
    assert "cluster" not in records.columns


def test_prepare_product_clusters_reads_zipped_csv_and_drops_label_columns(tmp_path: Path) -> None:
    rows = "1,apple iphone 8,10,5,Apple iPhone 8,77,Phones\n2,iphone 8 64gb,11,5,Apple iPhone 8,77,Phones\n"
    with zipfile.ZipFile(tmp_path / "shop.csv.zip", "w") as archive:
        archive.writestr("shop.csv", rows)
    records, truth = dataset.prepare_product_clusters(tmp_path, "shop.csv", "shop")
    assert records.columns.tolist() == ["record_id", "source", "title", "vendor_id"]
    assert entity_of(records, truth) == {"1": "5", "2": "5"}


def test_prepare_abt_buy_pools_both_tables(tmp_path: Path) -> None:
    abt = '"id","name","description","price"\r\n1,"Caf\xe9 maker","x","$10"\r\n2,"TV","y",\r\n'
    buy = '"id","name","description","manufacturer","price"\r\n9,"Cafe maker","",ACME,\r\n'
    mapping = '"idAbt","idBuy"\r\n1,9\r\n'
    with zipfile.ZipFile(tmp_path / "Abt-Buy.zip", "w") as archive:
        archive.writestr("Abt.csv", abt.encode("latin-1"))
        archive.writestr("Buy.csv", buy.encode("utf-8"))
        archive.writestr("abt_buy_perfectMapping.csv", mapping.encode("utf-8"))

    records, truth = dataset.prepare_abt_buy(tmp_path)
    assert entity_of(records, truth) == {"abt-1": "abt-1", "buy-9": "abt-1", "abt-2": "abt-2"}
    by_id = records.set_index(truth["original_id"].to_numpy())
    assert by_id.loc["abt-1", "name"] == "Café maker"
    assert by_id.loc["buy-9", "source"] == "buy"
    assert pd.isna(by_id.loc["abt-1", "manufacturer"])


def test_prepare_cora_handles_unbalanced_quotes(tmp_path: Path) -> None:
    (tmp_path / "cora.tsv").write_text(
        'authors\tid\ttitle\nA. Smith\t1\t"Learning\nA Smith\t2\tLearning\nB. Jones\t3\tOther\n',
        encoding="utf-8",
    )
    (tmp_path / "cora_DPL.tsv").write_text("id1\tid2\r\n1\t2\r\n", encoding="utf-8")
    records, truth = dataset.prepare_cora(tmp_path)
    assert entity_of(records, truth) == {"1": "1", "2": "1", "3": "3"}
    assert records.columns.tolist() == ["record_id", "source", "authors", "title"]


@pytest.mark.parametrize("name", list(dataset.DATASETS))
def test_prepared_dataset_matches_published_counts(name: str) -> None:
    out_dir = PROCESSED_DATA_DIR / name
    if not (out_dir / "truth.parquet").exists():
        pytest.skip(f"{name} not prepared; run make data")
    records = pd.read_parquet(out_dir / "records.parquet")
    truth = pd.read_parquet(out_dir / "truth.parquet")
    expected = dataset.DATASETS[name]
    assert len(records) == expected.n_records
    assert truth["entity_id"].nunique() == expected.n_entities
    assert (records["record_id"] == truth["record_id"]).all()
    assert not {"entity_id", "original_id", "cluster", "cluster_id"} & set(records.columns)


def test_read_raw_bytes_requires_a_member_for_multi_file_zips(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "two.zip", "w") as archive:
        archive.writestr("a.csv", "x")
        archive.writestr("b.csv", "y")
    with pytest.raises(ValueError, match="name the member"):
        dataset.read_raw_bytes(tmp_path, "two.zip")
    assert dataset.read_raw_bytes(tmp_path, "two.zip", "b.csv") == b"y"
    assert io.BytesIO(dataset.read_raw_bytes(tmp_path, "two.zip", "a.csv")).read() == b"x"


def test_normalise_text_hand_computed() -> None:
    values = pd.Series(
        [
            "Les Célébrités",
            "212/484-5113",
            "Utgoff, P. E., & Clouse, J. A.",
            "TiVo HD Digital Video Recorder - Up to 20 Hours in HD! - TCD652160",
            "Straße_ﬁve",
            "  ...  ",
            None,
        ]
    )
    out = dataset.normalise_text(values)
    assert out.iloc[:5].tolist() == [
        "les celebrites",
        "212 484 5113",
        "utgoff p e clouse j a",
        "tivo hd digital video recorder up to 20 hours in hd tcd652160",
        "strasse five",
    ]
    assert out.iloc[5:].isna().all()


def test_normalise_text_makes_the_sample_duplicates_equal() -> None:
    # Two spellings of the same Cora author list differ only in form.
    out = dataset.normalise_text(
        pd.Series(["Utgoff, P.E. and Clouse J.A.", "UTGOFF P E AND CLOUSE J A"])
    )
    assert out.iloc[0] == out.iloc[1] == "utgoff p e and clouse j a"


def test_normalise_digits() -> None:
    out = dataset.normalise_digits(pd.Series(["212/484-5113", "(212) 484 5113", "n/a", None]))
    assert out.iloc[:2].tolist() == ["2124845113", "2124845113"]
    assert out.iloc[2:].isna().all()


def test_normalise_records_applies_digits_fields_and_keeps_ids() -> None:
    records = pd.DataFrame(
        {"record_id": ["r0"], "source": ["Zagat"], "name": ["Café X"], "phone": ["212/484-5113"]}
    )
    out = dataset.normalise_records(records, digits_fields=["phone"])
    assert out.iloc[0].tolist() == ["r0", "Zagat", "cafe x", "2124845113"]
    assert records.loc[0, "name"] == "Café X"


def test_record_texts_joins_fields_and_skips_nulls() -> None:
    records = pd.DataFrame({"record_id": ["r0", "r1"], "a": ["x", None], "b": [" y ", "z"]})
    texts = dataset.record_texts(records, ["a", "b"])
    assert texts.to_dict() == {"r0": "x y", "r1": "z"}


def test_file_sha256_of_abc(tmp_path: Path) -> None:
    (tmp_path / "abc.txt").write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert dataset.file_sha256(tmp_path / "abc.txt") == expected


def test_find_raw_file_prefers_the_plain_file_then_the_zip(tmp_path: Path) -> None:
    (tmp_path / "a.csv.zip").write_bytes(b"zip")
    assert dataset.find_raw_file(tmp_path, "a.csv").name == "a.csv.zip"
    (tmp_path / "a.csv").write_bytes(b"plain")
    assert dataset.find_raw_file(tmp_path, "a.csv").name == "a.csv"
    with pytest.raises(FileNotFoundError, match="run the download"):
        dataset.find_raw_file(tmp_path, "b.csv")


def test_verify_manifest_detects_changed_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        dataset.verify_manifest(tmp_path)
    (tmp_path / "a.csv").write_bytes(b"abc")
    manifest = [{"file": "a.csv", "sha256": dataset.file_sha256(tmp_path / "a.csv")}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset.verify_manifest(tmp_path)
    (tmp_path / "a.csv").write_bytes(b"abd")
    with pytest.raises(ValueError, match="changed"):
        dataset.verify_manifest(tmp_path)


def test_fetch_copies_package_files_and_downloads_urls(tmp_path: Path) -> None:
    package_file = dataset.RawFile("config.py", package_path="dedup_bench:config.py")
    copied = dataset.fetch(package_file, tmp_path)
    assert copied.read_bytes() == (Path(dataset.__file__).parent / "config.py").read_bytes()

    (tmp_path / "source.txt").write_bytes(b"hello")
    url_file = dataset.RawFile("downloaded.txt", url=(tmp_path / "source.txt").as_uri())
    assert dataset.fetch(url_file, tmp_path).read_bytes() == b"hello"

    with pytest.raises(ValueError, match="no source"):
        dataset.fetch(dataset.RawFile("nothing.txt"), tmp_path)


def test_selected_validates_names() -> None:
    assert dataset.selected(None) == list(dataset.DATASETS)
    assert dataset.selected(["cora", "febrl3"]) == ["febrl3", "cora"]
    with pytest.raises(typer.BadParameter, match="unknown"):
        dataset.selected(["nope"])


def test_prepare_dataset_writes_original_normalised_and_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw" / "demo"
    raw_dir.mkdir(parents=True)
    (raw_dir / "demo.csv").write_text(
        "id,name,phone\n1,Café,212-484\n2,CAFE,212 484\n3,Bar,\n", encoding="utf-8"
    )
    manifest = [{"file": "demo.csv", "sha256": dataset.file_sha256(raw_dir / "demo.csv")}]
    (raw_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = tmp_path / "datasets.json"
    config.write_text(json.dumps({"demo": {"text_fields": ["name"], "digits_fields": ["phone"]}}))

    def prepare_demo(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(raw / "demo.csv", dtype=str).rename(columns={"id": "record_id"})
        return dataset.finish(df, pd.Series(["a", "a", "b"]), "demo")

    demo = dataset.Dataset((dataset.RawFile("demo.csv"),), prepare_demo, 3, 2)
    monkeypatch.setitem(dataset.DATASETS, "demo", demo)
    monkeypatch.setattr(dataset, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(dataset, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(dataset, "DATASETS_CONFIG", config)
    dataset.prepare_dataset("demo")

    original, truth = dataset.load_prepared("demo")
    normalised, _ = dataset.load_prepared("demo", normalised=True)
    assert set(original["name"]) == {"Café", "CAFE", "Bar"}
    assert set(normalised["name"]) == {"cafe", "bar"}
    assert set(normalised["phone"].dropna()) == {"212484"}
    assert normalised["record_id"].tolist() == original["record_id"].tolist()
    assert truth.groupby(truth).size().sort_values().tolist() == [1, 2]

    monkeypatch.setitem(dataset.DATASETS, "demo", dataset.Dataset(demo.files, prepare_demo, 4, 2))
    with pytest.raises(ValueError, match="expected 4"):
        dataset.prepare_dataset("demo")


@pytest.mark.parametrize("name", list(dataset.DATASETS))
def test_prepared_normalised_records_match_the_originals(name: str) -> None:
    out_dir = PROCESSED_DATA_DIR / name
    if not (out_dir / "records_normalised.parquet").exists():
        pytest.skip(f"{name} not prepared; run make data")
    original = pd.read_parquet(out_dir / "records.parquet")
    normalised = pd.read_parquet(out_dir / "records_normalised.parquet")
    assert normalised.columns.tolist() == original.columns.tolist()
    assert normalised["record_id"].equals(original["record_id"])
