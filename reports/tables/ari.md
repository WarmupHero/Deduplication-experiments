# Test ari, mean ± standard deviation over seeds

| method | abt_buy | cora | febrl3 | pricerunner | restaurants | skroutz |
|---|---|---|---|---|---|---|
| active_learning.random_forest label_budget=10 | 0.114 ± 0.177 | 0.777 ± 0.435 | 0.799 ± 0.447 | 0.122 ± 0.159 | 0.374 ± 0.513 | 0.262 ± 0.214 |
| active_learning.random_forest label_budget=25 | 0.348 ± 0.235 | 0.939 ± 0.055 | 1.000 ± 0.001 | 0.183 ± 0.096 | 0.967 ± 0.032 | 0.189 ± 0.172 |
| active_learning.random_forest label_budget=50 | 0.451 ± 0.102 | 0.977 ± 0.020 | 1.000 ± 0.000 | 0.143 ± 0.057 | 0.954 ± 0.053 | 0.148 ± 0.087 |
| active_learning.random_forest label_budget=100 | 0.435 ± 0.150 | 0.976 ± 0.026 | 1.000 ± 0.000 | 0.289 ± 0.101 | 0.966 ± 0.032 | 0.240 ± 0.154 |
| active_learning.random_forest label_budget=200 | 0.586 ± 0.054 | 0.979 ± 0.023 | 1.000 ± 0.000 | 0.315 ± 0.092 | 0.949 ± 0.049 | 0.253 ± 0.156 |
| active_learning.random_forest label_budget=500 | 0.606 ± 0.049 | 0.977 ± 0.021 | 1.000 ± 0.000 | 0.332 ± 0.104 | 0.972 ± 0.030 | 0.319 ± 0.206 |
| baselines.exact_key | 0.019 ± 0.013 | 0.121 ± 0.069 | 0.013 ± 0.005 | 0.092 ± 0.004 | 0.097 ± 0.066 | 0.110 ± 0.004 |
| embedding_clustering.agglomerative | 0.592 ± 0.050 | 0.617 ± 0.250 | 0.995 ± 0.003 | 0.310 ± 0.025 | 0.911 ± 0.071 | excluded: out of memory |
| embedding_clustering.hdbscan | 0.671 ± 0.037 | 0.917 ± 0.084 | 0.926 ± 0.018 | 0.266 ± 0.005 | 0.344 ± 0.070 |  |
| graph_clustering.center | 0.719 ± 0.041 | 0.213 ± 0.116 | 0.981 ± 0.008 | 0.401 ± 0.009 | 0.950 ± 0.029 | 0.298 ± 0.009 |
| graph_clustering.connected_components | 0.547 ± 0.047 | 0.936 ± 0.061 | 1.000 ± 0.000 | 0.198 ± 0.038 | 0.959 ± 0.028 | 0.313 ± 0.028 |
| graph_clustering.hierarchical | 0.691 ± 0.059 | 0.636 ± 0.283 | 1.000 ± 0.000 | 0.473 ± 0.006 | 0.957 ± 0.020 | excluded: out of memory |
| graph_clustering.leiden_cpm | 0.658 ± 0.053 | 0.686 ± 0.289 | 1.000 ± 0.000 | 0.480 ± 0.006 | 0.962 ± 0.021 | 0.498 ± 0.010 |
| graph_clustering.markov | 0.615 ± 0.098 | 0.492 ± 0.335 | 1.000 ± 0.000 | 0.441 ± 0.018 | 0.932 ± 0.010 | 0.541 ± 0.033 |
| graph_clustering.merge_center | 0.549 ± 0.049 | 0.938 ± 0.058 | 1.000 ± 0.000 | 0.206 ± 0.022 | 0.950 ± 0.029 | 0.137 ± 0.011 |
| probabilistic.recordlinkage_ecm | 0.316 ± 0.057 | 0.852 ± 0.103 | 1.000 ± 0.000 | 0.092 ± 0.003 | 0.964 ± 0.049 | 0.104 ± 0.004 |
| probabilistic.splink | 0.188 ± 0.012 | 0.966 ± 0.056 | 1.000 ± 0.000 | 0.089 ± 0.003 | 0.914 ± 0.014 | 0.094 ± 0.003 |
| supervised.hist_gradient_boosting | 0.558 ± 0.080 | 0.990 ± 0.014 | 1.000 ± 0.000 | 0.181 ± 0.041 | 0.971 ± 0.025 | 0.427 ± 0.024 |
| supervised.random_forest | 0.593 ± 0.031 | 0.980 ± 0.025 | 1.000 ± 0.000 | 0.186 ± 0.047 | 0.955 ± 0.053 | excluded: out of memory |
