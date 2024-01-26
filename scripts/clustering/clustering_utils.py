"""
File Name: clustering_utils.py

Author: David Felipe
Contact: https://github.com/davidnfu0
Last Modification: 2024-01-25

Description:
This file contains a set of functions for visualizing clustering results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, Any, Union


def clustering_dict_table(
    title: str,
    data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
    show_annotations: bool = True,
    show_xlabels: bool = True,
    cell_size: float = 1,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Creates a set of heatmaps for each column in the provided data, useful for visualizing clustering results.

    This function can take a dictionary of dictionaries or a pandas DataFrame as input. Each column in the data
    represents a different cluster or category, with rows representing different items or entities. The function
    plots a heatmap for each column, side by side.

    Args:
    - title (str): The title for the entire set of heatmaps.
    - data (Union[Dict[str, Dict[str, float]], pd.DataFrame]): The clustering data to be visualized. This can be either
        a dictionary of dictionaries, where the outer dictionary keys are column names and inner dictionaries map row names
        to values, or a pandas DataFrame.
    - show_annotations (bool, optional): Whether to display the data values in the heatmaps. Defaults to True.
    - show_xlabels (bool, optional): Whether to display the x-axis labels. Defaults to True.
    - cell_size (float, optional): The size of each cell in the heatmaps. Defaults to 1.
    - **kwargs: Additional keyword arguments to pass to seaborn's heatmap function.

    Returns:
    - None: The function directly creates and displays the plot.
    """
    if isinstance(data, dict):
        data = pd.DataFrame.from_dict(data)
    num_columns = len(data.columns)
    num_rows = len(data.index)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=num_columns,
        figsize=(num_columns * cell_size, num_rows * cell_size),
    )

    fig.subplots_adjust(
        wspace=0, hspace=0
    )  # Adjust subplot parameters to reduce space between heatmaps

    for i, column in enumerate(data.columns):
        column_data = data[[column]]
        sns.heatmap(
            column_data,
            ax=axs[i],
            cmap="coolwarm",
            cbar=False,
            annot=show_annotations,
            **kwargs,
        )
        axs[i].tick_params(axis="y", which="both", length=0)  # Hide y-axis ticks
        if show_xlabels:
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)
        else:
            axs[i].set_xticklabels([])
        axs[i].set_yticklabels([]) if i > 0 else axs[i].set_yticklabels(
            axs[i].get_yticklabels(), rotation=0
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.show()
