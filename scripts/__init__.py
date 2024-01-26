"""
Author: David Felipe
Contact: https://github.com/davidnfu0
Last Modification: 2024-01-21

scripts: package providing versatile tools for process automation
"""
from .utils import (
    load_yaml_config,
    save_yaml,
    hide_warnings,
    safe_delete,
    truncate_float,
)
from .data_processing import pca_information_loss
from .plots import plot_hist, plot_hist2d, plot_line, plot_scatter

__all__ = [
    "pca_information_loss",
    "load_yaml_config",
    "save_yaml",
    "hide_warnings",
    "safe_delete",
    "truncate_float",
    "plot_hist",
    "plot_hist2d",
    "plot_line",
    "plot_scatter",
]
