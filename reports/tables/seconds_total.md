# Test seconds_total, mean ± standard deviation over seeds

| method | abt_buy | cora | febrl3 | pricerunner | restaurants |
|---|---|---|---|---|---|
| active_learning.random_forest label_budget=10 | 1 ± 0 | 2 ± 0 | 3 ± 0 | 17 ± 0 | 2 ± 0 |
| active_learning.random_forest label_budget=25 | 2 ± 0 | 2 ± 0 | 3 ± 0 | 17 ± 0 | 2 ± 0 |
| active_learning.random_forest label_budget=50 | 2 ± 0 | 2 ± 0 | 3 ± 0 | 17 ± 0 | 2 ± 0 |
| active_learning.random_forest label_budget=100 | 2 ± 0 | 2 ± 0 | 3 ± 0 | 17 ± 1 | 2 ± 0 |
| active_learning.random_forest label_budget=200 | 2 ± 0 | 2 ± 0 | 3 ± 0 | 16 ± 0 | 2 ± 0 |
| active_learning.random_forest label_budget=500 | 180 ± 1 | 188 ± 6 | 219 ± 17 | 581 ± 8 | 307 ± 4 |
| baselines.exact_key | 1 ± 0 | 1 ± 0 | 1 ± 0 | 8 ± 1 | 1 ± 0 |
| embedding_clustering.agglomerative | 11 ± 5 | 9 ± 5 | 24 ± 7 | 335 ± 43 | 8 ± 5 |
| embedding_clustering.hdbscan | 9 ± 8 | 9 ± 6 | 13 ± 8 | 56 ± 15 | 7 ± 5 |
| graph_clustering.center | 1 ± 0 | 1 ± 0 | 1 ± 0 | 4 ± 1 | 1 ± 0 |
| graph_clustering.connected_components | 1 ± 0 | 1 ± 0 | 1 ± 0 | 3 ± 1 | 1 ± 0 |
| graph_clustering.hierarchical | 5 ± 0 | 2 ± 1 | 19 ± 0 | 405 ± 29 | 1 ± 0 |
| graph_clustering.leiden_cpm | 1 ± 0 | 1 ± 0 | 1 ± 0 | 10 ± 1 | 1 ± 0 |
| graph_clustering.markov | 1 ± 0 | 1 ± 0 | 2 ± 0 | 9 ± 1 | 1 ± 0 |
| graph_clustering.merge_center | 1 ± 0 | 1 ± 0 | 1 ± 0 | 4 ± 1 | 1 ± 0 |
| probabilistic.recordlinkage_ecm | 3 ± 0 | 11 ± 3 | 13 ± 1 | 34 ± 1 | 8 ± 2 |
| probabilistic.splink | 3 ± 0 | 3 ± 0 | 5 ± 0 | 7 ± 1 | 2 ± 1 |
| supervised.hist_gradient_boosting | 23 ± 4 | 28 ± 4 | 34 ± 2 | 122 ± 30 | 29 ± 3 |
| supervised.random_forest | 21 ± 2 | 25 ± 4 | 26 ± 1 | 1147 ± 97 | 26 ± 4 |
