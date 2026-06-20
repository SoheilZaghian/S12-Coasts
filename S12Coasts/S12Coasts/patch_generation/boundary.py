"""
s12coasts.patch_generation.boundary
=====================================
Patch boundary creation — generates square GEE polygon features centred on
each sampled centroid point for Sentinel-1/2 image patch extraction.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PatchBoundaryCreator:
    """
    Creates square patch boundary polygons (GEE Features) around centroid
    points for use in the Google Earth Engine batch export pipeline.

    Parameters
    ----------
    patch_size_m : int
        Side length of the square patch in metres.
        Default: 5120 m → 512 px at 10 m GSD.

    Examples
    --------
    >>> from s12coasts.patch_generation.boundary import PatchBoundaryCreator
    >>> creator = PatchBoundaryCreator(patch_size_m=5120)
    >>> buffered_polygons = centroid_ee_fc.map(creator.create_patch_boundary)
    """

    def __init__(self, patch_size_m: int = 5120) -> None:
        self.patch_size_m = patch_size_m
        self.half_size    = patch_size_m / 2
        logger.debug(
            f"PatchBoundaryCreator: patch_size={patch_size_m} m "
            f"({patch_size_m // 10} px at 10 m GSD)."
        )

    def create_patch_boundary(self, feature) -> "ee.Feature":
        """
        Generate a square bounding box centred on a GEE point feature.

        Parameters
        ----------
        feature : ee.Feature
            A GEE Feature with a Point geometry.

        Returns
        -------
        ee.Feature
            A square polygon feature covering the patch extent,
            with all properties from the input feature copied.

        Notes
        -----
        This method is designed to be mapped over a GEE FeatureCollection:
            ``buffered = centroids_fc.map(creator.create_patch_boundary)``
        """
        try:
            import ee
        except ImportError as exc:
            raise ImportError(
                "Google Earth Engine is required for PatchBoundaryCreator.\n"
                "Install with: pip install earthengine-api geemap"
            ) from exc

        point  = feature.geometry()
        buffer = point.buffer(self.half_size)
        bounds = buffer.bounds()
        return ee.Feature(bounds).copyProperties(feature)
