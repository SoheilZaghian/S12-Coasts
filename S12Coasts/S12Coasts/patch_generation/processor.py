"""
s12coasts.patch_generation.processor
======================================
Core patch processor — retrieves paired Sentinel-1/Sentinel-2 imagery for
each patch polygon and submits GEE batch export tasks to Google Drive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from s12coasts.patch_generation.filters import (
    SENTINEL2_BANDS,
    add_all_indices,
    add_date_diff,
    scale_reflectance,
)

logger = logging.getLogger(__name__)


class PatchProcessor:
    """
    Retrieves paired Sentinel-1/Sentinel-2 imagery for each patch boundary
    polygon and submits GEE batch export tasks to Google Drive.

    Parameters
    ----------
    buffered_polygons : ee.FeatureCollection
        GEE FeatureCollection of square patch boundary polygons.
    boundary_layer : geopandas.GeoDataFrame
        Water boundary GeoDataFrame (OSM / NOAA / HydroLakes).
    config : dict
        Pipeline configuration dictionary (from ``load_config()``).

    Examples
    --------
    >>> from s12coasts.patch_generation.processor import PatchProcessor
    >>> processor = PatchProcessor(buffered_polygons, coast_gdf, config)
    >>> metadata_df = processor.process_all_patches()
    """

    def __init__(
        self,
        buffered_polygons,
        boundary_layer,
        config: dict,
    ) -> None:
        try:
            import ee
            import geemap
            self._ee     = ee
            self._geemap = geemap
        except ImportError as exc:
            raise ImportError(
                "Google Earth Engine and geemap are required for PatchProcessor.\n"
                "Install with: pip install earthengine-api geemap\n"
                "Authenticate with: earthengine authenticate"
            ) from exc

        self.buffered_polygons = buffered_polygons
        self.boundary_layer    = boundary_layer
        self.config            = config
        self.region_name       = config["region_name"]
        self.output_folder     = Path(config["output_folder"])
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.n_patches = buffered_polygons.size().getInfo()
        logger.info(
            f"PatchProcessor: '{self.region_name}' — "
            f"{self.n_patches} patch(es) to process."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_boundary_image(self, ee_polygon, patch_gdf):
        """
        Rasterise the water boundary vector clipped to a patch polygon.

        The boundary is painted onto a zero image:
            background = 0, boundary pixels = 255.
        """
        patch_gdf   = patch_gdf.to_crs(self.boundary_layer.crs)
        clipped     = self.boundary_layer.clip(patch_gdf.geometry)
        boundary_ee = self._geemap.gdf_to_ee(clipped)
        zero_image  = self._ee.Image.constant(0)
        painted     = zero_image.paint(boundary_ee, 255)
        return painted.clip(ee_polygon)

    def _build_s2_collection(self, ee_polygon):
        """
        Filter and prepare the Sentinel-2 collection for a patch polygon.

        Returns a tuple of (collection_with_indices, best_s2_date).
        """
        collection = (
            self._ee.ImageCollection(self.config["collection_sen2"])
            .filterDate(self.config["start_date"], self.config["end_date"])
            .filter(self._ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                self.config["cloud_cover_threshold"],
            ))
            .filterBounds(ee_polygon)
            .select(SENTINEL2_BANDS)
            .map(scale_reflectance)
            .map(add_all_indices)
        )
        best_s2      = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
        s2_date      = self._ee.Date(best_s2.get("system:time_start"))
        return collection, s2_date

    def _build_s1_image(self, ee_polygon, s2_date):
        """
        Select the Sentinel-1 acquisition temporally closest to the S2 date.

        Search window: ±``max_temporal_gap_days`` around the S2 acquisition.
        """
        gap = self.config["max_temporal_gap_days"]
        collection = (
            self._ee.ImageCollection(self.config["collection_sen1"])
            .filterDate(
                s2_date.advance(-gap, "day"),
                s2_date.advance(gap, "day"),
            )
            .filter(self._ee.Filter.listContains("system:band_names", "VV"))
            .filter(self._ee.Filter.listContains("system:band_names", "VH"))
            .filterBounds(ee_polygon)
            .map(lambda img: add_date_diff(img, s2_date))
        )
        return collection.sort("date_diff").first().clip(ee_polygon)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_patch(self, patch_index: int) -> dict:
        """
        Process a single patch: retrieve S1/S2 imagery, stack bands with the
        water boundary image, and submit a GEE export task to Google Drive.

        Parameters
        ----------
        patch_index : int
            Zero-based index of the patch in ``buffered_polygons``.

        Returns
        -------
        dict
            Metadata record for this patch.
        """
        record: dict = {
            "patch_index": patch_index,
            "patch_id": f"Patch_{self.region_name}_{patch_index}",
            "s1_image_id": None,
            "s2_image_id": None,
            "s1_date": None,
            "s1_time": None,
            "s2_date": None,
            "s2_time": None,
            "export_status": "failed",
        }

        try:
            ee = self._ee
            feature    = ee.Feature(
                self.buffered_polygons
                .toList(self.buffered_polygons.size())
                .get(patch_index)
            )
            ee_polygon = feature.geometry()
            patch_gdf  = self._geemap.ee_to_gdf(ee.FeatureCollection([feature]))

            # Build image components
            boundary_image = self._create_boundary_image(ee_polygon, patch_gdf)
            s2_collection, s2_date = self._build_s2_collection(ee_polygon)
            s2_image = s2_collection.sort("CLOUDY_PIXEL_PERCENTAGE").first().clip(ee_polygon)
            s1_image = self._build_s1_image(ee_polygon, s2_date)

            # Stack: S2 (bands + indices) | S1 (VV, VH) | water boundary
            final_image = (
                s2_image
                .addBands(s1_image)
                .addBands(boundary_image.rename("Water_boundary"))
                .toFloat()
            )

            # Submit GEE export task
            drive_folder = (
                f"{self.config['drive_folder_prefix']}_{self.region_name}"
            )
            task = ee.batch.Export.image.toDrive(
                image=final_image,
                description=record["patch_id"],
                scale=self.config["resolution_m"],
                region=ee_polygon,
                fileFormat="GeoTIFF",
                folder=drive_folder,
            )
            task.start()

            # Collect metadata
            s2_dt = s2_date.format().getInfo()
            s1_dt = ee.Date(s1_image.get("system:time_start")).format().getInfo()

            record.update({
                "s1_image_id":   s1_image.get("system:index").getInfo(),
                "s2_image_id":   s2_image.get("system:index").getInfo(),
                "s1_date":       s1_dt.split("T")[0],
                "s1_time":       s1_dt.split("T")[1],
                "s2_date":       s2_dt.split("T")[0],
                "s2_time":       s2_dt.split("T")[1],
                "export_status": "submitted",
                "drive_folder":  drive_folder,
            })
            logger.debug(f"[{patch_index}] Submitted → {drive_folder}")

        except Exception as exc:
            logger.error(f"[{patch_index}] Failed: {exc}")

        return record

    def process_all_patches(self) -> pd.DataFrame:
        """
        Process all patches and return a metadata DataFrame.

        The DataFrame is also saved to ``output_folder/<region_name>_metadata.csv``.

        Returns
        -------
        pd.DataFrame
            Metadata for all processed patches.
        """
        logger.info(
            f"Processing {self.n_patches} patch(es) for '{self.region_name}'..."
        )
        records = []
        for i in tqdm(range(self.n_patches), desc=f"[{self.region_name}]"):
            records.append(self.process_patch(i))

        df      = pd.DataFrame(records)
        csv_out = self.output_folder / f"{self.region_name}_metadata.csv"
        df.to_csv(csv_out, index=False)

        n_ok   = (df["export_status"] == "submitted").sum()
        n_fail = (df["export_status"] == "failed").sum()
        logger.info(
            f"Completed: {n_ok} submitted · {n_fail} failed "
            f"out of {self.n_patches} patches.\n"
            f"Metadata saved → {csv_out}\n"
            "Monitor exports: https://code.earthengine.google.com/tasks"
        )
        return df
