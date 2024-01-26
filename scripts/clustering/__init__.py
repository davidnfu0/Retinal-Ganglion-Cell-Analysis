"""
Author: David Felipe
Contact: https://github.com/davidnfu0
Last Modification: 2024-01-19

clustering: A package providing various utilities for clustering analysis.
"""

from .clustering_hyperparams_search import (
    plot_elbow_method,
    plot_knee_method,
    plot_silhouette_scores,
    gmm_bic_score,
    find_best_param,
    params_table,
    plot_dendrogram,
)
from .clustering_evaluation import (
    plot_clusters,
    proximity_matrix,
    calculate_clustering_metrics,
    plot_clustering_metrics,
)
from .clustering_utils import clustering_dict_table
from .clustering_analysis import (
    plot_contrast_cluster,
    plot_elipses,
    calculate_cluster_statistics,
    plot_columns_cluster_histograms,
    plot_cluster_stim_bins,
)

__all__ = [
    "plot_elbow_method",
    "plot_knee_method",
    "plot_silhouette_scores",
    "gmm_bic_score",
    "find_best_param",
    "params_table",
    "plot_dendrogram",
    "plot_clusters",
    "proximity_matrix",
    "calculate_clustering_metrics",
    "plot_clustering_metrics",
    "clustering_dict_table",
    "plot_contrast_cluster",
    "plot_elipses",
    "calculate_cluster_statistics",
    "plot_columns_cluster_histograms",
    "plot_cluster_stim_bins",
]
