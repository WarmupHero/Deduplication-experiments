# Test b_cubed_f1, mean ± standard deviation over seeds

| method | abt_buy | cora | febrl3 | pricerunner | restaurants | skroutz |
|---|---|---|---|---|---|---|
| active_learning.random_forest label_budget=10 | 0.705 ± 0.067 | 0.800 ± 0.392 | 0.915 ± 0.187 | 0.614 ± 0.068 | 0.955 ± 0.036 | 0.663 ± 0.069 |
| active_learning.random_forest label_budget=25 | 0.763 ± 0.074 | 0.955 ± 0.050 | 1.000 ± 0.000 | 0.671 ± 0.043 | 0.997 ± 0.003 | 0.699 ± 0.055 |
| active_learning.random_forest label_budget=50 | 0.790 ± 0.040 | 0.970 ± 0.028 | 1.000 ± 0.000 | 0.674 ± 0.047 | 0.996 ± 0.004 | 0.717 ± 0.053 |
| active_learning.random_forest label_budget=100 | 0.785 ± 0.049 | 0.978 ± 0.025 | 1.000 ± 0.000 | 0.695 ± 0.048 | 0.997 ± 0.003 | 0.726 ± 0.066 |
| active_learning.random_forest label_budget=200 | 0.845 ± 0.024 | 0.982 ± 0.010 | 1.000 ± 0.000 | 0.693 ± 0.047 | 0.995 ± 0.004 | 0.749 ± 0.036 |
| active_learning.random_forest label_budget=500 | 0.854 ± 0.009 | 0.984 ± 0.009 | 1.000 ± 0.000 | 0.699 ± 0.051 | 0.997 ± 0.003 | 0.760 ± 0.029 |
| baselines.exact_key | 0.667 ± 0.003 | 0.420 ± 0.074 | 0.577 ± 0.008 | 0.587 ± 0.009 | 0.934 ± 0.009 | 0.501 ± 0.004 |
| embedding_clustering.agglomerative | 0.844 ± 0.021 | 0.817 ± 0.077 | 0.996 ± 0.002 | 0.619 ± 0.007 | 0.990 ± 0.008 | excluded: out of memory |
| embedding_clustering.hdbscan | 0.870 ± 0.014 | 0.924 ± 0.078 | 0.956 ± 0.007 | 0.626 ± 0.004 | 0.840 ± 0.032 |  |
| graph_clustering.center | 0.885 ± 0.017 | 0.540 ± 0.088 | 0.991 ± 0.003 | 0.693 ± 0.005 | 0.995 ± 0.003 | 0.623 ± 0.004 |
| graph_clustering.connected_components | 0.850 ± 0.012 | 0.945 ± 0.055 | 1.000 ± 0.000 | 0.695 ± 0.004 | 0.996 ± 0.003 | 0.702 ± 0.003 |
| graph_clustering.hierarchical | 0.884 ± 0.021 | 0.848 ± 0.097 | 1.000 ± 0.000 | 0.711 ± 0.004 | 0.995 ± 0.002 | excluded: out of memory |
| graph_clustering.leiden_cpm | 0.878 ± 0.018 | 0.875 ± 0.109 | 1.000 ± 0.000 | 0.714 ± 0.004 | 0.996 ± 0.002 | 0.685 ± 0.004 |
| graph_clustering.markov | 0.868 ± 0.023 | 0.753 ± 0.163 | 1.000 ± 0.000 | 0.715 ± 0.004 | 0.992 ± 0.002 | 0.718 ± 0.002 |
| graph_clustering.merge_center | 0.852 ± 0.010 | 0.948 ± 0.053 | 1.000 ± 0.000 | 0.697 ± 0.005 | 0.995 ± 0.003 | 0.699 ± 0.003 |
| probabilistic.recordlinkage_ecm | 0.752 ± 0.011 | 0.864 ± 0.078 | 1.000 ± 0.000 | 0.570 ± 0.008 | 0.997 ± 0.005 | 0.476 ± 0.003 |
| probabilistic.splink | 0.696 ± 0.003 | 0.972 ± 0.031 | 1.000 ± 0.000 | 0.586 ± 0.009 | 0.991 ± 0.001 | 0.494 ± 0.004 |
| supervised.hist_gradient_boosting | 0.849 ± 0.022 | 0.991 ± 0.008 | 1.000 ± 0.000 | 0.742 ± 0.005 | 0.997 ± 0.002 | 0.791 ± 0.003 |
| supervised.random_forest | 0.858 ± 0.008 | 0.984 ± 0.012 | 1.000 ± 0.000 | 0.744 ± 0.005 | 0.996 ± 0.004 | excluded: out of memory |
