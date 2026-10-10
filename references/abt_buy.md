# Abt-Buy

| | |
|---|---|
| Domain | Consumer electronics from the Abt.com and Buy.com shops |
| Source | [`Abt-Buy.zip`](https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip) from the [Leipzig benchmark datasets for entity resolution](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution) (Köpcke, Thor & Rahm, PVLDB 2010) |
| Licence | CC BY 4.0 |
| Records → entities | 2,173 (1,081 Abt + 1,092 Buy) → 1,076 |
| Entity sizes | No singletons; 1,055 pairs and 21 triples |
| Ground truth | `abt_buy_perfectMapping.csv` (1,097 Abt–Buy pairs), closed transitively |
| Fields kept | name, description, manufacturer (Buy only), price |

Notes:

- The two tables are pooled into one, and `source` says which shop a record comes from.
- `Abt.csv` is Latin-1; the other files are UTF-8.
- Known as one of the hard product-matching benchmarks: long, noisy descriptions and model numbers written differently in each shop.
