"""
File Name: plots.py

Author: David Felipe
Contact: https://github.com/davidnfu0
Last Modification: 2024-01-25

Description:
This module provides a set of tools for plotting
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_hist(
    title: str,
    X: np.ndarray,
    xlab: str = "",
    ylab: str = "",
    row: int = 1,
    column: int = 1,
    plot_number: int = 1,
    bins: int = 30,
) -> None:
    """
    Plot a histogram.

    Args:
    - title (str): Title of the histogram.
    - X (np.ndarray): Data array for the histogram.
    - xlab (str, optional): Label for the x-axis. Default is an empty string.
    - ylab (str, optional): Label for the y-axis. Default is an empty string.
    - row, column, plot_number (int, optional): Subplot positioning parameters. Default is 1.
    - bins (int, optional): Number of bins in the histogram. Default is 30.

    Returns:
    - None
    """
    plt.subplot(row, column, plot_number)
    plt.hist(X, bins=bins)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)


def plot_hist2d(
    title: str,
    X: np.ndarray,
    Y: np.ndarray,
    xlab: str = "",
    ylab: str = "",
    row: int = 1,
    column: int = 1,
    plot_number: int = 1,
    bins: int = 30,
) -> None:
    """
    Plot a 2D histogram.

    Args:
    - title (str): Title of the histogram.
    - X, Y (np.ndarray): Data arrays for the 2D histogram.
    - xlab, ylab (str, optional): Labels for the x and y axes. Default is an empty string.
    - row, column, plot_number (int, optional): Subplot positioning parameters. Default is 1.
    - bins (int, optional): Number of bins in each dimension. Default is 30.

    Returns:
    - None
    """
    plt.subplot(row, column, plot_number)
    plt.hist2d(X, Y, bins=bins)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.colorbar()


def plot_scatter(
    title: str,
    X: np.ndarray,
    Y: np.ndarray,
    xlab: str = "",
    ylab: str = "",
    color="b",
    cmap: str = "",
    row: int = 1,
    column: int = 1,
    plot_number: int = 1,
) -> None:
    """
    Plot a scatter plot.

    Args:
    - title (str): Title of the scatter plot.
    - X, Y (np.ndarray): Data arrays for the scatter plot.
    - xlab, ylab (str, optional): Labels for the x and y axes. Default is an empty string.
    - color (optional): Color of the points. Default is "b" (blue).
    - cmap (str, optional): Colormap for the points. Default is an empty string.
    - row, column, plot_number (int, optional): Subplot positioning parameters. Default is 1.

    Returns:
    - None
    """
    plt.subplot(row, column, plot_number)
    plt.scatter(x=X, y=Y, c=color, cmap=cmap)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    if cmap != "":
        plt.colorbar()


def plot_line(
    title: str,
    X: np.ndarray,
    xlab: str = "",
    ylab: str = "",
    row: int = 1,
    column: int = 1,
    plot_number: int = 1,
) -> None:
    """
    Plot a line graph.

    Args:
    - title (str): Title of the line graph.
    - X (np.ndarray): Data array for the line graph.
    - xlab, ylab (str, optional): Labels for the x and y axes. Default is an empty string.
    - row, column, plot_number (int, optional): Subplot positioning parameters. Default is 1.

    Returns:
    - None
    """
    plt.subplot(row, column, plot_number)
    plt.plot(X)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
