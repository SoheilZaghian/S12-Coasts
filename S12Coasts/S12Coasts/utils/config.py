"""
s12coasts.utils.config
======================
Configuration loading utilities with defaults for both pipeline stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

DEFAULT_PATCH_CONFIG: dict = {
    "region_name": "MyRegion",
    "n_points": 300,
    "patch_size_m": 5120,
    "patch_size_px": 512,
    "resolution_m": 10,
    "cloud_cover_threshold": 30,
    "max_temporal_gap_days": 365,
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "collection_sen2": "COPERNICUS/S2_SR_HARMONIZED",
    "collection_sen1": "COPERNICUS/S1_GRD",
    "directory_path": "data/shapefiles/extracted/",
    "zip_files_directory": "data/shapefiles/zipped/",
    "output_folder": "outputs/patches/",
    "drive_folder_prefix": "S12Coasts_Patches",
}

DEFAULT_EOBC_CONFIG: dict = {
    "input_dir": "data/patches/",
    "output_dir": "outputs/masks/",
    "region_name": "MyRegion",
    "feature_bands": [13, 14, 15, 16, 17, 18, 19],
    "n_pca_components": 5,
    "patch_size_px": 512,
    "visualise": False,
    "metadata": {
        "climate_zone": None,
        "agro_ecological_zone": None,
        "country_name": None,
        "region_name": None,
    },
}


def load_config(
    config_path: Optional[str] = None,
    config_type: str = "eobc",
) -> dict:
    """
    Load a YAML configuration file, merging with stage-appropriate defaults.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML configuration file. If None, returns defaults.
    config_type : str
        Which default config to use as base: ``"eobc"`` or ``"patch"``.

    Returns
    -------
    dict
        Merged configuration dictionary.

    Examples
    --------
    >>> from s12coasts.utils.config import load_config
    >>> config = load_config("configs/eobc_config.yaml", config_type="eobc")
    >>> config = load_config("configs/patch_config.yaml", config_type="patch")
    """
    base = (
        DEFAULT_EOBC_CONFIG.copy()
        if config_type == "eobc"
        else DEFAULT_PATCH_CONFIG.copy()
    )

    if config_path is None:
        logger.info(f"No config path provided. Using default '{config_type}' config.")
        return base

    path = Path(config_path)
    if not path.exists():
        logger.warning(
            f"Config file not found: {path}. Using default '{config_type}' config."
        )
        return base

    with open(path) as f:
        user_config = yaml.safe_load(f) or {}

    base.update(user_config)
    logger.info(f"Configuration loaded from: {path}")
    return base
