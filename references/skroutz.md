# Skroutz

| | |
|---|---|
| Domain | Product offers from 652 vendors on the Greek price-comparison site Skroutz, 12 categories |
| Source | Kaggle [`lakritidis/product-clustering-matching-classification`](https://www.kaggle.com/datasets/lakritidis/product-clustering-matching-classification), file `products_csv/skroutz_aggregate.csv` (uploaded by the dataset's author, Leonidas Akritidis) |
| Licence | GPL 2 |
| Records → entities | 238,170 → 63,995 |
| Entity sizes | 27,170 singletons; largest entity 100 records |
| Ground truth | The cluster ID column |
| Fields kept | title, vendor_id |

Notes:

- The benchmark's large dataset, used for the scaling experiments.
- Same layout as PriceRunner: no header, and the cluster label and category are dropped because they would reveal the answer.
- Greek titles are transliterated to Latin letters, lower-cased and stripped of punctuation.
