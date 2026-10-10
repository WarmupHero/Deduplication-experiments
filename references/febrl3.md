# Febrl 3

| | |
|---|---|
| Domain | Synthetic person records (names, addresses, dates of birth, social security IDs) |
| Source | `dataset3.csv` shipped with the [`recordlinkage`](https://github.com/J535D165/recordlinkage) package, generated with the Febrl data generator (Peter Christen, ANU) |
| Licence | Distributed with `recordlinkage` under BSD-3-Clause |
| Records → entities | 5,000 → 2,000 |
| Entity sizes | 835 singletons; largest entity 6 records |
| Ground truth | Encoded in the original ID: `rec-N-org` and `rec-N-dup-k` belong to entity `rec-N` |
| Fields kept | given_name, surname, street_number, address_1, address_2, suburb, postcode, state, date_of_birth, soc_sec_id |

Notes:

- Values in the raw file are separated by a comma and a space.
- The original IDs reveal the entity, so the prepared records carry neutral IDs; the original ID is kept only in `truth.parquet`.
- The `recordlinkage` documentation misprints the number of originals without duplicates; the counts above are computed from the file.
