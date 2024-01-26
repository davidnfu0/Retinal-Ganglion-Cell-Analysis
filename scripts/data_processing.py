"""
File Name: data_processing.py

Author: David Felipe
Contact: https://github.com/davidnfu0
Last Modification: 2024-01-22

Description:
This module provides a set of tools for data processing
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from sklearn.decomposition import PCA


def pca_information_loss(
    title: str, dataframe: pd.DataFrame, max_components=Optional[None]
) -> None:
    """
    This function takes a DataFrame and performs PCA on it with a varying number of components.
    It plots the cumulative explained variance to show the information loss.

    Parameters:
    - title (str): Title of the plot
    - dataframe (pd.DataFrame): pandas DataFrame with numerical data
    - max_components (int, Optional): the maximum number of components to consider. If None, it's set to the number of features.

    Returns:
    - None
    """
    max_components = max_components or dataframe.shape[1]

    data = dataframe - dataframe.mean()

    cumulative_variance = []

    plt.figure(figsize=(10, 6))

    for n_components in range(1, max_components + 1):
        pca = PCA(n_components=n_components)
        pca.fit(data)
        cumulative_variance.append(sum(pca.explained_variance_ratio_))
        if (
            len(cumulative_variance) >= 2
            and cumulative_variance[-2] <= 0.99
            and cumulative_variance[-1] > 0.99
        ):
            plt.axvline(x=n_components, color="r", linestyle="--")
        if (
            len(cumulative_variance) >= 2
            and cumulative_variance[-2] <= 0.95
            and cumulative_variance[-1] > 0.95
        ):
            plt.axvline(x=n_components, color="g", linestyle="--")
        if (
            len(cumulative_variance) >= 2
            and cumulative_variance[-2] <= 0.9
            and cumulative_variance[-1] > 0.9
        ):
            plt.axvline(x=n_components, color="orange", linestyle="--")

    plt.axhline(y=0.99, color="r", linestyle="--")
    plt.axhline(y=0.95, color="g", linestyle="--")
    plt.axhline(y=0.9, color="orange", linestyle="--")
    plt.plot(range(1, max_components + 1), cumulative_variance, marker=".", color="b")
    plt.xlabel("Number of Components")
    plt.xticks(range(0, max_components + 1, 10), rotation=90)
    plt.tick_params(axis="x", labelsize=6)
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Information Loss {title}")
    plt.show()
