"""s12coasts.patch_generation — Stage 1: patch sampling and pre-processing."""

from s12coasts.patch_generation.sampling import ShapefileProcessor, PointGenerator
from s12coasts.patch_generation.boundary import PatchBoundaryCreator
from s12coasts.patch_generation.processor import PatchProcessor
from s12coasts.patch_generation.filters import (
    SPECTRAL_INDICES,
    SENTINEL2_BANDS,
    scale_reflectance,
    add_spectral_index,
    add_all_indices,
    add_date_diff,
)

__all__ = [
    "ShapefileProcessor",
    "PointGenerator",
    "PatchBoundaryCreator",
    "PatchProcessor",
    "SPECTRAL_INDICES",
    "SENTINEL2_BANDS",
    "scale_reflectance",
    "add_spectral_index",
    "add_all_indices",
    "add_date_diff",
]
