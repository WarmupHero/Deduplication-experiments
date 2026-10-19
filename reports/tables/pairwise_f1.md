# Test pairwise_f1, mean ± standard deviation over seeds

| method | abt_buy | cora | febrl3 | pricerunner | restaurants |
|---|---|---|---|---|---|
| active_learning.random_forest label_budget=10 | 0.115 ± 0.178 | 0.798 ± 0.394 | 0.799 ± 0.447 | 0.123 ± 0.159 | 0.374 ± 0.513 |
| active_learning.random_forest label_budget=25 | 0.348 ± 0.235 | 0.945 ± 0.050 | 1.000 ± 0.001 | 0.183 ± 0.096 | 0.967 ± 0.032 |
| active_learning.random_forest label_budget=50 | 0.452 ± 0.102 | 0.979 ± 0.017 | 1.000 ± 0.000 | 0.144 ± 0.056 | 0.954 ± 0.053 |
| active_learning.random_forest label_budget=100 | 0.436 ± 0.151 | 0.979 ± 0.023 | 1.000 ± 0.000 | 0.290 ± 0.101 | 0.966 ± 0.032 |
| active_learning.random_forest label_budget=200 | 0.587 ± 0.054 | 0.980 ± 0.021 | 1.000 ± 0.000 | 0.315 ± 0.092 | 0.949 ± 0.049 |
| active_learning.random_forest label_budget=500 | 0.607 ± 0.049 | 0.979 ± 0.019 | 1.000 ± 0.000 | 0.332 ± 0.104 | 0.972 ± 0.030 |
| baselines.exact_key | 0.019 ± 0.013 | 0.130 ± 0.069 | 0.013 ± 0.005 | 0.092 ± 0.004 | 0.097 ± 0.066 |
| embedding_clustering.agglomerative | 0.594 ± 0.050 | 0.642 ± 0.230 | 0.995 ± 0.003 | 0.310 ± 0.025 | 0.912 ± 0.070 |
| embedding_clustering.hdbscan | 0.672 ± 0.037 | 0.926 ± 0.073 | 0.926 ± 0.018 | 0.266 ± 0.005 | 0.345 ± 0.070 |
| graph_clustering.center | 0.720 ± 0.041 | 0.227 ± 0.114 | 0.981 ± 0.008 | 0.401 ± 0.009 | 0.950 ± 0.029 |
| graph_clustering.connected_components | 0.548 ± 0.047 | 0.943 ± 0.055 | 1.000 ± 0.000 | 0.198 ± 0.038 | 0.960 ± 0.028 |
| graph_clustering.hierarchical | 0.692 ± 0.059 | 0.658 ± 0.263 | 1.000 ± 0.000 | 0.474 ± 0.006 | 0.957 ± 0.020 |
| graph_clustering.leiden_cpm | 0.659 ± 0.053 | 0.708 ± 0.267 | 1.000 ± 0.000 | 0.481 ± 0.006 | 0.962 ± 0.021 |
| graph_clustering.markov | 0.616 ± 0.098 | 0.580 ± 0.272 | 1.000 ± 0.000 | 0.441 ± 0.018 | 0.932 ± 0.010 |
| graph_clustering.merge_center | 0.550 ± 0.049 | 0.944 ± 0.052 | 1.000 ± 0.000 | 0.207 ± 0.022 | 0.950 ± 0.029 |
| probabilistic.recordlinkage_ecm | 0.318 ± 0.057 | 0.863 ± 0.102 | 1.000 ± 0.000 | 0.092 ± 0.003 | 0.964 ± 0.049 |
| probabilistic.splink | 0.189 ± 0.012 | 0.969 ± 0.052 | 1.000 ± 0.000 | 0.089 ± 0.003 | 0.914 ± 0.014 |
| supervised.hist_gradient_boosting | 0.560 ± 0.080 | 0.990 ± 0.013 | 1.000 ± 0.000 | 0.182 ± 0.041 | 0.971 ± 0.025 |
| supervised.random_forest | 0.594 ± 0.031 | 0.981 ± 0.024 | 1.000 ± 0.000 | 0.187 ± 0.046 | 0.955 ± 0.053 |
