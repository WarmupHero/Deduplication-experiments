# Run health: seeds, commits, uncommitted-code runs and peak memory

| method / dataset | seeds | commits | dirty | max_peak_mb |
|---|---|---|---|---|
| active_learning.random_forest label_budget=10 / abt_buy | 5 | 1 | 0 | 218 |
| active_learning.random_forest label_budget=10 / cora | 5 | 1 | 0 | 234 |
| active_learning.random_forest label_budget=10 / febrl3 | 5 | 1 | 0 | 249 |
| active_learning.random_forest label_budget=10 / pricerunner | 5 | 1 | 0 | 610 |
| active_learning.random_forest label_budget=10 / restaurants | 5 | 1 | 0 | 199 |
| active_learning.random_forest label_budget=10 / skroutz | 5 | 1 | 0 | 1123 |
| active_learning.random_forest label_budget=25 / abt_buy | 5 | 1 | 0 | 218 |
| active_learning.random_forest label_budget=25 / cora | 5 | 1 | 0 | 234 |
| active_learning.random_forest label_budget=25 / febrl3 | 5 | 1 | 0 | 249 |
| active_learning.random_forest label_budget=25 / pricerunner | 5 | 1 | 0 | 603 |
| active_learning.random_forest label_budget=25 / restaurants | 5 | 1 | 0 | 199 |
| active_learning.random_forest label_budget=25 / skroutz | 5 | 1 | 0 | 1081 |
| active_learning.random_forest label_budget=50 / abt_buy | 5 | 1 | 0 | 218 |
| active_learning.random_forest label_budget=50 / cora | 5 | 1 | 0 | 234 |
| active_learning.random_forest label_budget=50 / febrl3 | 5 | 1 | 0 | 249 |
| active_learning.random_forest label_budget=50 / pricerunner | 5 | 1 | 0 | 593 |
| active_learning.random_forest label_budget=50 / restaurants | 5 | 1 | 0 | 199 |
| active_learning.random_forest label_budget=50 / skroutz | 5 | 1 | 0 | 1056 |
| active_learning.random_forest label_budget=100 / abt_buy | 5 | 1 | 0 | 218 |
| active_learning.random_forest label_budget=100 / cora | 5 | 1 | 0 | 233 |
| active_learning.random_forest label_budget=100 / febrl3 | 5 | 1 | 0 | 249 |
| active_learning.random_forest label_budget=100 / pricerunner | 5 | 1 | 0 | 583 |
| active_learning.random_forest label_budget=100 / restaurants | 5 | 1 | 0 | 199 |
| active_learning.random_forest label_budget=100 / skroutz | 5 | 1 | 0 | 1088 |
| active_learning.random_forest label_budget=200 / abt_buy | 5 | 1 | 0 | 217 |
| active_learning.random_forest label_budget=200 / cora | 5 | 1 | 0 | 234 |
| active_learning.random_forest label_budget=200 / febrl3 | 5 | 1 | 0 | 249 |
| active_learning.random_forest label_budget=200 / pricerunner | 5 | 1 | 0 | 579 |
| active_learning.random_forest label_budget=200 / restaurants | 5 | 1 | 0 | 199 |
| active_learning.random_forest label_budget=200 / skroutz | 5 | 1 | 0 | 1089 |
| active_learning.random_forest label_budget=500 / abt_buy | 5 | 1 | 0 | 221 |
| active_learning.random_forest label_budget=500 / cora | 5 | 1 | 0 | 237 |
| active_learning.random_forest label_budget=500 / febrl3 | 5 | 1 | 0 | 250 |
| active_learning.random_forest label_budget=500 / pricerunner | 5 | 1 | 0 | 611 |
| active_learning.random_forest label_budget=500 / restaurants | 5 | 1 | 0 | 199 |
| active_learning.random_forest label_budget=500 / skroutz | 5 | 1 | 0 | 1408 |
| baselines.exact_key / abt_buy | 5 | 1 | 0 | 192 |
| baselines.exact_key / cora | 5 | 1 | 0 | 193 |
| baselines.exact_key / febrl3 | 5 | 1 | 0 | 202 |
| baselines.exact_key / pricerunner | 5 | 1 | 0 | 342 |
| baselines.exact_key / restaurants | 5 | 1 | 0 | 181 |
| baselines.exact_key / skroutz | 5 | 1 | 0 | 740 |
| embedding_clustering.agglomerative / abt_buy | 5 | 1 | 0 | 1059 |
| embedding_clustering.agglomerative / cora | 5 | 1 | 0 | 1230 |
| embedding_clustering.agglomerative / febrl3 | 5 | 1 | 0 | 989 |
| embedding_clustering.agglomerative / pricerunner | 5 | 1 | 0 | 1685 |
| embedding_clustering.agglomerative / restaurants | 5 | 1 | 0 | 1012 |
| embedding_clustering.hdbscan / abt_buy | 5 | 1 | 0 | 1127 |
| embedding_clustering.hdbscan / cora | 5 | 1 | 0 | 1112 |
| embedding_clustering.hdbscan / febrl3 | 5 | 1 | 0 | 1010 |
| embedding_clustering.hdbscan / pricerunner | 5 | 1 | 0 | 1574 |
| embedding_clustering.hdbscan / restaurants | 5 | 1 | 0 | 1100 |
| graph_clustering.center / abt_buy | 5 | 1 | 0 | 195 |
| graph_clustering.center / cora | 5 | 1 | 0 | 194 |
| graph_clustering.center / febrl3 | 5 | 1 | 0 | 203 |
| graph_clustering.center / pricerunner | 5 | 1 | 0 | 360 |
| graph_clustering.center / restaurants | 5 | 1 | 0 | 182 |
| graph_clustering.center / skroutz | 5 | 1 | 0 | 753 |
| graph_clustering.connected_components / abt_buy | 5 | 1 | 0 | 194 |
| graph_clustering.connected_components / cora | 5 | 1 | 0 | 194 |
| graph_clustering.connected_components / febrl3 | 5 | 1 | 0 | 203 |
| graph_clustering.connected_components / pricerunner | 5 | 1 | 0 | 356 |
| graph_clustering.connected_components / restaurants | 5 | 1 | 0 | 183 |
| graph_clustering.connected_components / skroutz | 5 | 1 | 0 | 730 |
| graph_clustering.hierarchical / abt_buy | 5 | 1 | 0 | 195 |
| graph_clustering.hierarchical / cora | 5 | 1 | 0 | 194 |
| graph_clustering.hierarchical / febrl3 | 5 | 1 | 0 | 203 |
| graph_clustering.hierarchical / pricerunner | 5 | 1 | 0 | 1083 |
| graph_clustering.hierarchical / restaurants | 5 | 1 | 0 | 184 |
| graph_clustering.leiden_cpm / abt_buy | 5 | 1 | 0 | 223 |
| graph_clustering.leiden_cpm / cora | 5 | 1 | 0 | 224 |
| graph_clustering.leiden_cpm / febrl3 | 5 | 1 | 0 | 231 |
| graph_clustering.leiden_cpm / pricerunner | 5 | 1 | 0 | 395 |
| graph_clustering.leiden_cpm / restaurants | 5 | 1 | 0 | 211 |
| graph_clustering.leiden_cpm / skroutz | 5 | 1 | 0 | 823 |
| graph_clustering.markov / abt_buy | 5 | 1 | 0 | 195 |
| graph_clustering.markov / cora | 5 | 1 | 0 | 196 |
| graph_clustering.markov / febrl3 | 5 | 1 | 0 | 204 |
| graph_clustering.markov / pricerunner | 5 | 1 | 0 | 411 |
| graph_clustering.markov / restaurants | 5 | 1 | 0 | 183 |
| graph_clustering.markov / skroutz | 5 | 1 | 0 | 1254 |
| graph_clustering.merge_center / abt_buy | 5 | 1 | 0 | 193 |
| graph_clustering.merge_center / cora | 5 | 1 | 0 | 194 |
| graph_clustering.merge_center / febrl3 | 5 | 1 | 0 | 202 |
| graph_clustering.merge_center / pricerunner | 5 | 1 | 0 | 353 |
| graph_clustering.merge_center / restaurants | 5 | 1 | 0 | 183 |
| graph_clustering.merge_center / skroutz | 5 | 1 | 0 | 759 |
| probabilistic.recordlinkage_ecm / abt_buy | 5 | 1 | 0 | 201 |
| probabilistic.recordlinkage_ecm / cora | 5 | 1 | 0 | 214 |
| probabilistic.recordlinkage_ecm / febrl3 | 5 | 1 | 0 | 246 |
| probabilistic.recordlinkage_ecm / pricerunner | 5 | 1 | 0 | 467 |
| probabilistic.recordlinkage_ecm / restaurants | 5 | 1 | 0 | 189 |
| probabilistic.recordlinkage_ecm / skroutz | 5 | 1 | 0 | 1133 |
| probabilistic.splink / abt_buy | 5 | 1 | 0 | 315 |
| probabilistic.splink / cora | 5 | 1 | 0 | 438 |
| probabilistic.splink / febrl3 | 5 | 1 | 0 | 427 |
| probabilistic.splink / pricerunner | 5 | 1 | 0 | 615 |
| probabilistic.splink / restaurants | 5 | 1 | 0 | 348 |
| probabilistic.splink / skroutz | 5 | 1 | 0 | 1186 |
| supervised.hist_gradient_boosting / abt_buy | 5 | 1 | 0 | 212 |
| supervised.hist_gradient_boosting / cora | 5 | 1 | 0 | 235 |
| supervised.hist_gradient_boosting / febrl3 | 5 | 1 | 0 | 253 |
| supervised.hist_gradient_boosting / pricerunner | 5 | 1 | 0 | 728 |
| supervised.hist_gradient_boosting / restaurants | 5 | 1 | 0 | 196 |
| supervised.hist_gradient_boosting / skroutz | 5 | 1 | 0 | 1764 |
| supervised.random_forest / abt_buy | 5 | 1 | 0 | 246 |
| supervised.random_forest / cora | 5 | 1 | 0 | 241 |
| supervised.random_forest / febrl3 | 5 | 1 | 0 | 251 |
| supervised.random_forest / pricerunner | 5 | 1 | 0 | 2023 |
| supervised.random_forest / restaurants | 5 | 1 | 0 | 196 |

## Limitations: methods not run

| method / dataset | reason |
|---|---|
| supervised.random_forest / skroutz | out of memory: an unlimited-depth 500-tree forest took 2 GB on PriceRunner and would need about 14 GB here |
| graph_clustering.hierarchical / skroutz | out of memory: a dense distance matrix per connected group, about 13 GB for a 40,000-record group at low thresholds |
| embedding_clustering.agglomerative / skroutz | out of memory: the same dense distance matrix per connected group as hierarchical |
