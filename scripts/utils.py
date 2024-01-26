"""
File Name: utils.py

Author: David Felipe
Contact: https://github.com/davidnfu0
Last Modification: 2024-01-25

Description:
This module provides a set of general utilities
"""

import yaml
import warnings


def load_yaml_config(config_path: str):
    """
    Load a configuration from a YAML file.

    Args:
    - config_path (str): The file path to the YAML configuration file.

    Returns:
    - dict: The configuration data loaded from the YAML file.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(data, file_path):
    """
    Save a Python dictionary to a YAML file.

    Args:
    - data (dict): The dictionary to save.
    - file_path (str): The path to the file where the YAML content will be saved.

    Returns:
    - None: The function doesn't return anything but saves data to a file.
    """
    try:
        with open(file_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
    except Exception as e:
        print(f"Error writing to the YAML file: {e}")


def hide_warnings() -> None:
    """
    Hide warning messages in the output.

    Args:
    - None

    Returns:
    - None
    """
    warnings.filterwarnings("ignore")
    warnings.warn("DelftStack")
    warnings.warn("Do not show this message")


def safe_delete(dictionary: dict, key) -> None:
    """
    Safely delete a key from a dictionary.

    Args:
    - dictionary (dict): The dictionary from which the key should be removed.
    - key: The key to be removed.

    Returns:
    - None
    """
    try:
        del dictionary[key]
    except KeyError:
        pass


def truncate_float(float_number, decimal_places):
    """
    Truncate a float to a specified number of decimal places.

    Args:
    - float_number (float): The float number to be truncated.
    - decimal_places (int): The number of decimal places to retain.

    Returns:
    - float: The truncated float number.
    """
    multiplier = 10**decimal_places
    return int(float_number * multiplier) / multiplier
