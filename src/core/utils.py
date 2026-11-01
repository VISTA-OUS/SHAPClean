"""Configuration loading and reproducibility utilities shared across the project."""

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load the central YAML configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration mapping.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML root is not a mapping.
    """
    resolved_path = Path(config_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must hold a mapping.")
    return config


def set_seed(seed: int) -> None:
    """Set deterministic seeds for supported libraries.

    Does not touch `PYTHONHASHSEED`: that variable only affects Python's string
    hash randomization if set in the environment *before* the interpreter
    starts, so assigning it here mid-process has no effect on the running
    process and would be misleading.

    Args:
        seed: Global pseudo-random seed.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)