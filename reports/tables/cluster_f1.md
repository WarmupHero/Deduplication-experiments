# Test cluster_f1, mean ± standard deviation over seeds

| method | abt_buy | cora | febrl3 | pricerunner | restaurants |
|---|---|---|---|---|---|
| active_learning.random_forest label_budget=10 | 0.097 ± 0.173 | 0.653 ± 0.369 | 0.848 ± 0.329 | 0.274 ± 0.054 | 0.865 ± 0.107 |
| active_learning.random_forest label_budget=25 | 0.217 ± 0.171 | 0.862 ± 0.118 | 0.999 ± 0.002 | 0.330 ± 0.042 | 0.990 ± 0.010 |
| active_learning.random_forest label_budget=50 | 0.257 ± 0.104 | 0.858 ± 0.096 | 1.000 ± 0.000 | 0.319 ± 0.040 | 0.989 ± 0.010 |
| active_learning.random_forest label_budget=100 | 0.242 ± 0.111 | 0.919 ± 0.051 | 1.000 ± 0.000 | 0.331 ± 0.045 | 0.990 ± 0.010 |
| active_learning.random_forest label_budget=200 | 0.405 ± 0.077 | 0.899 ± 0.050 | 1.000 ± 0.000 | 0.321 ± 0.038 | 0.987 ± 0.008 |
| active_learning.random_forest label_budget=500 | 0.434 ± 0.023 | 0.913 ± 0.024 | 1.000 ± 0.000 | 0.333 ± 0.043 | 0.992 ± 0.008 |
| baselines.exact_key | 0.007 ± 0.005 | 0.167 ± 0.026 | 0.246 ± 0.013 | 0.251 ± 0.008 | 0.801 ± 0.026 |
| embedding_clustering.agglomerative | 0.552 ± 0.029 | 0.464 ± 0.179 | 0.980 ± 0.012 | 0.240 ± 0.007 | 0.970 ± 0.020 |
| embedding_clustering.hdbscan | 0.652 ± 0.036 | 0.644 ± 0.089 | 0.785 ± 0.024 | 0.225 ± 0.008 | 0.607 ± 0.078 |
| graph_clustering.center | 0.727 ± 0.040 | 0.396 ± 0.038 | 0.974 ± 0.008 | 0.326 ± 0.002 | 0.985 ± 0.011 |
| graph_clustering.connected_components | 0.486 ± 0.038 | 0.816 ± 0.138 | 1.000 ± 0.000 | 0.342 ± 0.013 | 0.989 ± 0.007 |
| graph_clustering.hierarchical | 0.658 ± 0.072 | 0.708 ± 0.103 | 1.000 ± 0.000 | 0.352 ± 0.002 | 0.986 ± 0.005 |
| graph_clustering.leiden_cpm | 0.643 ± 0.060 | 0.770 ± 0.147 | 1.000 ± 0.000 | 0.353 ± 0.001 | 0.988 ± 0.004 |
| graph_clustering.markov | 0.568 ± 0.037 | 0.724 ± 0.114 | 1.000 ± 0.000 | 0.357 ± 0.005 | 0.976 ± 0.009 |
| graph_clustering.merge_center | 0.515 ± 0.056 | 0.828 ± 0.145 | 1.000 ± 0.000 | 0.338 ± 0.006 | 0.985 ± 0.011 |
| probabilistic.recordlinkage_ecm | 0.249 ± 0.029 | 0.547 ± 0.207 | 1.000 ± 0.000 | 0.218 ± 0.007 | 0.991 ± 0.012 |
| probabilistic.splink | 0.087 ± 0.006 | 0.881 ± 0.082 | 1.000 ± 0.000 | 0.251 ± 0.008 | 0.972 ± 0.007 |
| supervised.hist_gradient_boosting | 0.464 ± 0.097 | 0.937 ± 0.032 | 1.000 ± 0.000 | 0.420 ± 0.004 | 0.992 ± 0.004 |
| supervised.random_forest | 0.477 ± 0.061 | 0.926 ± 0.029 | 1.000 ± 0.000 | 0.417 ± 0.002 | 0.989 ± 0.010 |
