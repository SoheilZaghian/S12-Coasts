"""
s12coasts
=========
Automated Global Multimodal Benchmark for Water Body and Coastline Segmentation
via Ensemble Object-Based Clustering (EOBC).

Paper:
    S12-Coasts: Automated Generation of a Global Multimodal Benchmark for
    Water Body and Coastline Segmentation via Ensemble Object-Based Clustering.
    ISPRS Journal of Photogrammetry and Remote Sensing, 2025.

Authors:
    Soheil Zaghian, Seyed Ali Ahmadi, Ali Mohammadzadeh, Sadegh Jamali

Quick Start
-----------
Stage 1 — Patch Generation:
    from s12coasts.patch_generation import PatchProcessor, ShapefileProcessor
    from s12coasts.utils.config import load_config

    config = load_config("configs/patch_config.yaml")

Stage 2 — EOBC Water Mask Generation:
    from s12coasts.eobc import EOBCPipeline
    from s12coasts.utils.config import load_config

    config = load_config("configs/eobc_config.yaml")
    pipeline = EOBCPipeline(config)
    pipeline.run()

CLI Entry Points:
    s12coasts-patch --config configs/patch_config.yaml
    s12coasts-eobc  --config configs/eobc_config.yaml
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("s12coasts")
except PackageNotFoundError:
    __version__ = "1.0.0"

__author__ = (
    "Soheil Zaghian, Seyed Ali Ahmadi, "
    "Ali Mohammadzadeh, Sadegh Jamali"
)
__license__ = "MIT"
__paper__ = (
    "ISPRS Journal of Photogrammetry and Remote Sensing, 2025. "
    "doi:10.XXXXX/isprs.2025.XXXXX"
)

# ---------------------------------------------------------------------------
# Public API — top-level imports
# ---------------------------------------------------------------------------

# Core EOBC pipeline
from s12coasts.eobc.pipeline import EOBCPipeline
from s12coasts.eobc.features import FeatureExtractor
from s12coasts.eobc.optimiser import ClusteringOptimiser
from s12coasts.eobc.clusterer import EOBCClusterer
from s12coasts.eobc.algorithms import build_algorithm_registry, PARAM_GRIDS

# Patch generation
from s12coasts.patch_generation.sampling import (
    ShapefileProcessor,
    PointGenerator,
)
from s12coasts.patch_generation.boundary import PatchBoundaryCreator
from s12coasts.patch_generation.processor import PatchProcessor

# I/O utilities
from s12coasts.io.geotiff import GeoTIFFExporter
from s12coasts.io.metadata import MetadataManager

# General utilities
from s12coasts.utils.config import load_config, DEFAULT_PATCH_CONFIG, DEFAULT_EOBC_CONFIG
from s12coasts.utils.visualise import visualise_patch

__all__ = [
    # EOBC
    "EOBCPipeline",
    "FeatureExtractor",
    "ClusteringOptimiser",
    "EOBCClusterer",
    "build_algorithm_registry",
    "PARAM_GRIDS",
    # Patch generation
    "ShapefileProcessor",
    "PointGenerator",
    "PatchBoundaryCreator",
    "PatchProcessor",
    # I/O
    "GeoTIFFExporter",
    "MetadataManager",
    # Utils
    "load_config",
    "DEFAULT_PATCH_CONFIG",
    "DEFAULT_EOBC_CONFIG",
    "visualise_patch",
]
