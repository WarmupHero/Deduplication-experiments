# Test peak_mb, mean ± standard deviation over seeds

| method | abt_buy | cora | febrl3 | pricerunner | restaurants |
|---|---|---|---|---|---|
| active_learning.random_forest label_budget=10 | 213 ± 5 | 223 ± 9 | 235 ± 12 | 500 ± 80 | 196 ± 3 |
| active_learning.random_forest label_budget=25 | 212 ± 5 | 223 ± 9 | 235 ± 12 | 495 ± 79 | 196 ± 3 |
| active_learning.random_forest label_budget=50 | 212 ± 6 | 222 ± 9 | 234 ± 12 | 487 ± 80 | 196 ± 3 |
| active_learning.random_forest label_budget=100 | 211 ± 7 | 222 ± 9 | 233 ± 13 | 476 ± 81 | 196 ± 3 |
| active_learning.random_forest label_budget=200 | 211 ± 7 | 221 ± 10 | 233 ± 14 | 466 ± 88 | 196 ± 3 |
| active_learning.random_forest label_budget=500 | 213 ± 9 | 223 ± 12 | 233 ± 17 | 482 ± 115 | 195 ± 4 |
| baselines.exact_key | 189 ± 4 | 190 ± 4 | 198 ± 6 | 320 ± 31 | 180 ± 2 |
| embedding_clustering.agglomerative | 906 ± 96 | 1076 ± 166 | 873 ± 85 | 1584 ± 90 | 872 ± 122 |
| embedding_clustering.hdbscan | 1022 ± 113 | 988 ± 97 | 882 ± 85 | 1319 ± 179 | 924 ± 160 |
| graph_clustering.center | 190 ± 5 | 191 ± 4 | 198 ± 6 | 328 ± 38 | 181 ± 2 |
| graph_clustering.connected_components | 191 ± 5 | 190 ± 4 | 198 ± 6 | 324 ± 39 | 182 ± 2 |
| graph_clustering.hierarchical | 191 ± 5 | 191 ± 4 | 199 ± 6 | 971 ± 77 | 182 ± 2 |
| graph_clustering.leiden_cpm | 219 ± 5 | 220 ± 5 | 227 ± 6 | 368 ± 36 | 210 ± 2 |
| graph_clustering.markov | 191 ± 4 | 193 ± 4 | 199 ± 6 | 388 ± 38 | 182 ± 2 |
| graph_clustering.merge_center | 190 ± 4 | 191 ± 4 | 198 ± 6 | 328 ± 37 | 181 ± 2 |
| probabilistic.recordlinkage_ecm | 196 ± 5 | 206 ± 7 | 234 ± 12 | 407 ± 60 | 187 ± 2 |
| probabilistic.splink | 281 ± 29 | 354 ± 60 | 355 ± 64 | 541 ± 84 | 300 ± 40 |
| supervised.hist_gradient_boosting | 205 ± 7 | 219 ± 13 | 235 ± 16 | 567 ± 138 | 192 ± 4 |
| supervised.random_forest | 233 ± 12 | 225 ± 12 | 233 ± 16 | 1850 ± 172 | 193 ± 3 |
