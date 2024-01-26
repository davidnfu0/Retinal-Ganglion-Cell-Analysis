"""
File Name: clustering_analysis.py

Author: David Felipe
Contact: https://github.com/davidnfu0
Last Modification: 2024-01-25

Description:
This module provides a set of tools for the analysis of the results of clustering algorithms.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from typing import List, Dict, Set, Tuple, Any


def _get_subplot(axs: plt.Axes, subplot_coords: Tuple[int, int]) -> plt.Axes:
    """
    Retrieves the appropriate subplot axis based on the provided coordinates.

    Args:
    - axs (matplotlib.pyplot.Axes): Axes object to plot on.
    - subplot_coords (Tuple[int, int]): Coordinates (row, column) of the subplot.

    Returns:
    - plt.Axes: The subplot axis.
    """
    subplot_row, subplot_column = subplot_coords
    try:
        return axs[subplot_row, subplot_column]
    except (IndexError, TypeError):
        return axs[subplot_column if isinstance(axs, np.ndarray) else subplot_row]


def plot_contrast_cluster(
    title: str,
    axs: plt.Axes,
    contrast_dict: Dict[Any, List[float]],
    model_labels: np.ndarray,
    X: pd.DataFrame,
    time: List[float],
    subplot_coords: Tuple[int, int],
    **kwargs: Any,
) -> None:
    """
    Plots the contrast values over time for each cluster.

    Args:
    - title (str): Title of the plot.
    - axs (matplotlib.pyplot.Axes): The Axes object or array of Axes objects for plotting.
    - contrast_dict (Dict[Any, List[float]]): Dictionary containing contrast values for each template.
    - model_labels (np.ndarray): Array of cluster labels corresponding to each data point in 'X'.
    - X (pd.DataFrame): DataFrame containing the 'template' column used to link data points to contrast values.
    - time (List[float]): List of time points at which contrast values are recorded.
    - subplot_coords (Tuple[int, int]): Tuple specifying the subplot coordinates on the Axes object.
    - **kwargs: Additional keyword arguments for matplotlib plot function.

    Returns:
    - None
    """
    ax = _get_subplot(axs, subplot_coords)
    colors = plt.cm.Set1(np.arange(len(np.unique(model_labels))))
    for i, label in enumerate(np.unique(model_labels)):
        if label >= 0:
            templates = X["template"][model_labels == label]
            for temp in templates:
                if temp in contrast_dict:
                    ax.plot(time, contrast_dict[temp], color=colors[i], **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Time to spike")
    ax.set_ylabel("Contrast")


def plot_elipses(
    title: str,
    axs: plt.Axes,
    rfs: Dict[str, Dict[str, float]],
    X: pd.DataFrame,
    model_labels: np.ndarray,
    subplot_coords: Tuple[int, int],
    xmax: int = 40,
    xmin: int = 0,
    ymin: int = 0,
    ymax: int = 40,
    thrsh: float = 0.9,
) -> None:
    """
    Plots ellipses representing receptive fields for different clusters.

    Args:
    - title (str): Title of the plot.
    - axs (matplotlib.pyplot.Axes): The Axes object or array of Axes objects for plotting.
    - rfs (Dict[str, Dict[str, float]]): Dictionary containing receptive field data for each template.
    - X (pd.DataFrame): DataFrame containing the 'template' column to link data points to receptive fields.
    - model_labels (np.ndarray): Array of cluster labels corresponding to each data point in 'X'.
    - subplot_coords (Tuple[int, int]): Tuple specifying the subplot coordinates on the Axes object.
    - xmax, xmin, ymin, ymax (int, optional): Limits for the x and y axes of the plot.
    - thrsh (float, optional): Threshold value for filtering data based on the 'exc' field in receptive field data.

    Returns:
    - None
    """
    ax = _get_subplot(axs, subplot_coords)
    colors = plt.cm.Set1(np.arange(len(np.unique(model_labels))))
    for i, label in enumerate(np.unique(model_labels)):
        templates = X["template"][model_labels == label]
        for temp in templates:
            rf = rfs.get(temp)
            if rf and _is_rf_in_range(rf, xmax, xmin, ymax, ymin, thrsh):
                ellipse = Ellipse(
                    xy=(rf["x"], rf["y"]),
                    width=rf["w"],
                    height=rf["h"],
                    angle=rf["a"],
                    color=colors[i],
                    fill=False,
                    alpha=0.6,
                    label=temp.strip("temp_"),
                    linewidth=2,
                )
                ax.add_patch(ellipse)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _is_rf_in_range(
    rf: Dict[str, float], xmax: int, xmin: int, ymax: int, ymin: int, thrsh: float
) -> bool:
    """
    Checks if the receptive field data is within specified range and threshold.
    """
    return (rf["exc"] < thrsh) and (xmin < rf["x"] < xmax) and (ymin < rf["y"] < ymax)


def calculate_cluster_statistics(
    df: pd.DataFrame,
    df_cols: pd.DataFrame,
    model_labels: np.ndarray,
    columns_list: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Calculates and returns statistical metrics (mean, median, standard deviation) for each cluster.

    Args:
    - df (pd.DataFrame): DataFrame containing the 'template' column used for clustering.
    - df_cols (pd.DataFrame): DataFrame containing the columns for which statistics will be calculated.
    - model_labels (np.ndarray): Array of cluster labels corresponding to each data point in 'df'.
    - columns_list (List[str]): List of column names in 'df_cols' for which statistics are to be calculated.

    Returns:
    - Dict[str, pd.DataFrame]: A dictionary with keys 'Mean', 'Median', and 'Std', each mapping to a DataFrame
                              containing the corresponding statistical metric for each cluster and column.
    """
    unique_labels = set(model_labels)
    stats_dict = {
        metric: pd.DataFrame(index=list(unique_labels), columns=columns_list)
        for metric in ["Mean", "Median", "Std"]
    }
    data_dict = {col: {label: [] for label in unique_labels} for col in columns_list}

    for i, label in enumerate(model_labels):
        temp = df["template"][i]
        for col in columns_list:
            data = df_cols[df_cols["template"] == temp][col]
            if data.size > 0:
                data_dict[col][label].append(float(data))

    for label in unique_labels:
        for col in columns_list:
            data = data_dict[col][label]
            if len(data) > 0:
                stats_dict["Mean"].loc[label, col] = round(float(np.mean(data)), 5)
                stats_dict["Median"].loc[label, col] = round(float(np.median(data)), 5)
                stats_dict["Std"].loc[label, col] = round(float(np.std(data)), 5)

    for metric in stats_dict:
        stats_dict[metric] = stats_dict[metric].astype(float)

    return stats_dict


def plot_cluster_stim_bins(
    title: str,
    df: pd.DataFrame,
    columns_bin_list: List[str],
    stim_df: pd.DataFrame,
    model_labels: List[int],
    **kwargs: Any,
) -> None:
    """
    Plot bar and line graphs for clustered data.

    Args:
    - title (str): Title of the plot.
    - df (DataFrame): DataFrame containing the main data.
    - columns_bin_list (list of str): List of column names for binning.
    - stim_df (DataFrame): DataFrame with stimulation data.
    - model_labels (list of int): List of labels for the model clusters.
    - **kwargs: Additional keyword arguments for bar plot customization.

    Returns:
    - None
    """
    unique_labels = set(model_labels)
    data_dict, mean_dict = _prepare_plotting_data(
        df, columns_bin_list, stim_df, model_labels, unique_labels
    )

    rows = 2
    columns = max(len(unique_labels), 2)
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(columns * 4, rows * 4))
    colors = plt.cm.Set1(np.arange(len(unique_labels)))

    for j, label in enumerate(unique_labels):
        if label != -1:
            _plot_cluster_data(
                axs, j, label, data_dict, mean_dict, columns_bin_list, colors, **kwargs
            )

    fig.suptitle(title)
    fig.tight_layout()
    fig.show()


def plot_columns_cluster_histograms(
    title: str,
    df: pd.DataFrame,
    df_cols: pd.DataFrame,
    model_labels: np.ndarray,
    columns_list: List[str],
    bins: int = 10,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plots histograms for each cluster and specified column.

    Args:
    - title (str): Title of the plot.
    - df (pd.DataFrame): DataFrame containing the 'template' column used for clustering.
    - df_cols (pd.DataFrame): DataFrame containing the columns to be plotted in histograms.
    - model_labels (np.ndarray): Array of cluster labels corresponding to each data point in 'df'.
    - columns_list (List[str]): List of column names in 'df_cols' to be plotted.
    - bins (int, optional): Number of bins for the histograms.
    - **kwargs: Additional keyword arguments for matplotlib histogram function.

    Returns:
    - None
    """
    rows = 0
    columns = len(columns_list)
    data_dict = dict()
    for col in columns_list:
        data_dict[col] = dict()
    for label in set(model_labels):
        for col in columns_list:
            data_dict[col][label] = []
        rows += 1
    for i, label in enumerate(model_labels):
        temp = df["template"][i]
        for col in columns_list:
            data_dict[col][label].append(
                float(df_cols[df_cols["template"] == temp][col])
            )
    colors = plt.cm.Set1(range(len(set(model_labels))))
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(columns * 2, rows * 2))
    for i, col in enumerate(columns_list):
        for j, label in enumerate(set(model_labels)):
            if label != -1:
                axs[j, i].hist(
                    data_dict[col][label], bins=bins, color=colors[label], **kwargs
                )
                if i == 0:
                    axs[j, i].set_title(f"cluster {label}")
                    axs[j, i].set_ylabel("Count")
                if j == rows - 1:
                    axs[j, i].set_xlabel(col)
    fig.suptitle(f"{title} histograms")
    fig.tight_layout()
    fig.show()


def _prepare_plotting_data(
    df: pd.DataFrame,
    columns_bin_list: List[str],
    stim_df: pd.DataFrame,
    model_labels: List[int],
    unique_labels: Set[int],
) -> Tuple[Dict[int, Dict[str, List[float]]], Dict[int, Dict[str, float]]]:
    """
    Prepares data for plotting.

    Args:
    - df (DataFrame): DataFrame containing the main data.
    - columns_bin_list (list of str): List of column names for binning.
    - stim_df (DataFrame): DataFrame with stimulation data.
    - model_labels (list of int): List of labels for the model clusters.
    - unique_labels (set of int): Set of unique labels.

    Returns:
    - Tuple of two dictionaries containing data for plotting.
    """
    data_dict = {
        label: {bin: [] for bin in columns_bin_list} for label in unique_labels
    }
    mean_dict = {label: {} for label in unique_labels}

    for i, label in enumerate(model_labels):
        temp = df["template"][i]
        for bin in columns_bin_list:
            data = float(stim_df[stim_df["template"] == temp][bin])
            data_dict[label][bin].append(data)

    for label in unique_labels:
        for bin in columns_bin_list:
            mean_dict[label][bin] = np.mean(data_dict[label][bin])

    return data_dict, mean_dict


def _plot_cluster_data(
    axs: np.ndarray,
    j: int,
    label: int,
    data_dict: Dict[int, Dict[str, List[float]]],
    mean_dict: Dict[int, Dict[str, float]],
    columns_bin_list: List[str],
    colors: np.ndarray,
    **kwargs: Any,
) -> None:
    """
    Plots bar and line graphs for a single cluster.

    Args:
    - axs (np.ndarray): Array of subplot axes.
    - j (int): Index of the current subplot.
    - label (int): Label of the current cluster.
    - data_dict (dict): Dictionary containing the data for each cluster.
    - mean_dict (dict): Dictionary containing the mean values for each cluster.
    - columns_bin_list (list of str): List of column names for binning.
    - colors (np.ndarray): Array of colors for different clusters.
    - **kwargs: Additional keyword arguments for bar plot customization.

    Returns:
    - None
    """
    axs[0, j].bar(
        range(len(mean_dict[label])),
        height=mean_dict[label].values(),
        color=colors[label],
        **kwargs,
    )
    for i in range(len(data_dict[label][columns_bin_list[0]])):
        spikes = [data_dict[label][bin][i] for bin in columns_bin_list]
        axs[1, j].plot(
            range(len(mean_dict[label])), spikes, color=colors[label], alpha=0.7
        )

    axs[1, j].set_ylabel("Spikes")
    axs[1, j].set_xlabel("Bins")
    axs[0, j].set_title(f"Cluster {label}")
    axs[0, j].set_ylabel("Mean spikes")
