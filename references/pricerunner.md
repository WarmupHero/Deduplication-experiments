# PriceRunner

| | |
|---|---|
| Domain | Product offers from 306 merchants on the PriceRunner price-comparison site, 10 categories |
| Source | Kaggle [`lakritidis/product-clustering-matching-classification`](https://www.kaggle.com/datasets/lakritidis/product-clustering-matching-classification), file `products_csv/pricerunner_aggregate.csv` (uploaded by the dataset's author, Leonidas Akritidis) |
| Licence | GPL 2 on Kaggle; the same data is on the [UCI repository](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering) under CC BY 4.0 |
| Records → entities | 35,311 → 13,233 |
| Entity sizes | 5,374 singletons; largest entity 27 records |
| Ground truth | The cluster ID column |
| Fields kept | title, vendor_id |

Notes:

- The CSV has no header. Its columns are product ID, title, vendor ID, cluster ID, cluster label, category ID and category label.
- The cluster label (the canonical product name) and the category are assigned per product, so they would reveal the answer; they are dropped.
- Titles are lower-cased with punctuation removed.
