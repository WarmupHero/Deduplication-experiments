import io
from pathlib import Path
import zipfile

import pandas as pd
import pytest

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
