"""
File Name: clustering_hyperparams_search.py

Author: David Felipe
Contact: https://github.com/davidnfu0
Last Modification: 2024-01-19

Description:
This module, 'clustering_hyperparams_search', is designed as a comprehensive tool for exploring and optimizing clustering algorithms. 
Its primary objective is to simplify the process of identifying the most effective hyperparameters for these models.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Dict, Any, Union
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_elbow_method(
    title: str,
    axs: plt.Axes,
    X: np.ndarray,
    cluster_range: List[int],
    subplot_coords: Tuple[int, int],
    vertical_lines: List[int],
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plots the elbow method curve for determining the optimal number of clusters in KMeans clustering.

    Args:
    - title (str): The title of the plot.
    - axs (matplotlib.pyplot.Axes): The Axes object where the plot will be drawn.
    - X (numpy.ndarray): The dataset used for clustering.
    - cluster_range (List[int]): A list of cluster counts to evaluate.
    - subplot_coords (Tuple[int, int]): Row and column indices for the subplot.
    - vertical_lines (List[int]): Positions on the x-axis to draw vertical lines for reference.
    - **kwargs (Dict[str, Any]): Additional keyword arguments for matplotlib plot customization.

    Returns:
    - None
    """
    inertias = [
        KMeans(n_clusters=k, random_state=0).fit(X).inertia_ for k in cluster_range
    ]

    subplot_row, subplot_column = subplot_coords
    try:
        ax = axs[subplot_row, subplot_column]
    except IndexError:
        ax = axs[subplot_column]
    except TypeError:
        ax = axs
    ax.plot(cluster_range, inertias, "-o", **kwargs)
    for xline in vertical_lines:
        ax.axvline(x=xline, color="r", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_xticks(cluster_range)


def plot_knee_method(
    title: str,
    axs: plt.Axes,
    X: np.ndarray,
    neighbors: int,
    subplot_coords: Tuple[int, int],
    y_line: float,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plots the knee method for cluster analysis with additional customization options.

    Args:
    - title (str): Title of the plot.
    - axs (matplotlib.pyplot.Axes): Axes object to plot on.
    - X (numpy.ndarray): Dataset for clustering.
    - neighbors (int): Number of neighbors to use for kneighbors queries.
    - subplot_coords (Tuple[int, int]): Coordinates (row, column) of the subplot.
    - y_line (float): Y-axis value where a horizontal line should be drawn.
    - **kwargs (Dict[str, Any]): Additional keyword arguments for plot customization.

    Returns:
    - None
    """
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)[:, neighbors - 1]

    subplot_row, subplot_column = subplot_coords
    try:
        ax = axs[subplot_row, subplot_column]
    except IndexError:
        ax = axs[subplot_column]
    except TypeError:
        ax = axs
    ax.plot(distances, **kwargs)
    ax.axhline(y=y_line, color="r", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Ordered Data Points")
    ax.set_ylabel("Distance to Nearest Neighbor")


def plot_silhouette_scores(
    title: str,
    axs: plt.Axes,
    X: np.ndarray,
    cluster_range: List[int],
    subplot_coords: Tuple[int, int],
    vertical_line: int,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plots silhouette scores for a range of cluster numbers with additional customization options.

    Args:
    - title (str): Title of the plot.
    - X (np.ndarray): The dataset for clustering.
    - cluster_range (List[int]): A range of cluster numbers to evaluate.
    - subplot_coords (Tuple[int, int]): Coordinates (row, column) of the subplot.
    - vertical_line (int): x-axis position to draw a vertical line.
    - **kwargs (Dict[str, Any]): Additional keyword arguments for plot customization.

    Returns:
    - None
    """
    silhouette_scores = [
        silhouette_score(X, KMeans(n_clusters=k, random_state=42).fit_predict(X))
        for k in cluster_range
    ]

    subplot_row, subplot_column = subplot_coords
    try:
        ax = axs[subplot_row, subplot_column]
    except IndexError:
        ax = axs[subplot_column]
    except TypeError:
        ax = axs
    ax.plot(cluster_range, silhouette_scores, marker="o", **kwargs)
    ax.axvline(x=vertical_line, c="r", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")


def gmm_bic_score(
    estimator: GaussianMixture, X: Union[np.ndarray, pd.DataFrame]
) -> float:
    """
    Calculates the Bayesian Information Criterion (BIC) score for a given estimator.

    Args:
    - estimator (GaussianMixture): The Gaussian Mixture Model (GMM) estimator.
    - X (np.ndarray or pd.DataFrame): The dataset for which the BIC score is to be computed.

    Returns:
    - float: The negative BIC score of the estimator on the dataset.
    """
    return -estimator.bic(X)


def find_best_param(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    param_grid: Dict[str, Any],
    scoring: str,
    **kwargs: Dict[str, Any],
) -> GridSearchCV:
    """
    Performs grid search to find the best parameters for a given model with additional options.

    Args:
    -  model: The machine learning model for which the best parameters are to be found.
    - X (np.ndarray or pd.DataFrame): The dataset used for fitting the model.
    - param_grid (dict): The parameter grid to search over.
    - scoring (str): The scoring method to use.
    - **kwargs (Dict[str, Any]): Additional keyword arguments for the GridSearchCV.

    Returns:
    - GridSearchCV: The GridSearchCV object after fitting it to the data.
    """
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, **kwargs)
    grid_search.fit(X)
    return grid_search


def params_table(grid_search: GridSearchCV) -> pd.DataFrame:
    """
    Creates a DataFrame from the results of a grid search.

    Args:
    -grid_search (GridSearchCV): The GridSearchCV object containing the results of the grid search.

    Returns:
    -DataFrame: A DataFrame containing the parameters and corresponding mean test scores.
    """
    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df.sort_values(by="mean_test_score", inplace=True)
    return df


def plot_dendrogram(
    title: str,
    axs: plt.Axes,
    X: np.ndarray,
    subplot_coords: Tuple[int, int],
    vertical_line: int,
    mode: str = "ward",
    metric: str = "euclidean",
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plots a dendrogram for hierarchical clustering with additional customization options.

    Args:
    - title (str): Title of the plot.
    - axs (matplotlib.pyplot.Axes): Axes object to plot on.
    - X (np.ndarray): The dataset used for clustering.
    - subplot_coords (Tuple[int, int]): Coordinates (row, column) of the subplot.
    - vertical_line (int): x-axis position to draw a vertical line.
    - mode (str): Linkage criterion for the hierarchical clustering.
    - metric (str): Metric used to compute the linkage.
    - **kwargs (Dict[str, Any]): Additional keyword arguments for plot customization.

    Returns:
    - None
    """
    subplot_row, subplot_column = subplot_coords
    try:
        ax = axs[subplot_row, subplot_column]
    except IndexError:
        ax = axs[subplot_column]
    except TypeError:
        ax = axs
    Z = linkage(X, method=mode, metric=metric)
    dendrogram(Z, ax=ax, orientation="right", **kwargs)
    ax.axvline(x=vertical_line, c="r", linestyle="--")
    ax.set_title(title)
