"""
s12coasts.patch_generation.sampling
=====================================
Shapefile loading and stratified random point sampling along water boundary
vectors for the S12-Coasts patch generation pipeline.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


class ShapefileProcessor:
    """
    Extracts and loads regional water boundary shapefiles.

    Parameters
    ----------
    directory_path : str or Path
        Directory where extracted shapefiles are stored (auto-created if absent).
    zip_files_directory : str or Path
        Directory containing zipped regional shapefiles (.zip).

    Examples
    --------
    >>> from s12coasts.patch_generation.sampling import ShapefileProcessor
    >>> processor = ShapefileProcessor(
    ...     directory_path="data/shapefiles/extracted/",
    ...     zip_files_directory="data/shapefiles/zipped/",
    ... )
    >>> coast_gdf = processor.process_region("Madagascar_line")
    """

    def __init__(
        self,
        directory_path: str | Path,
        zip_files_directory: str | Path,
    ) -> None:
        self.directory_path    = Path(directory_path)
        self.zip_files_directory = Path(zip_files_directory)
        self.coast_lines: list[gpd.GeoDataFrame] = []

        if not self.zip_files_directory.exists():
            raise FileNotFoundError(
                f"Zip files directory not found: {self.zip_files_directory}\n"
                "Place regional shapefiles (.zip) in this directory before running."
            )

        self.regions: list[str] = [
            f.stem for f in self.zip_files_directory.glob("*.zip")
        ]
        logger.info(
            f"ShapefileProcessor: found {len(self.regions)} region(s) — "
            f"{self.regions}"
        )

    # ------------------------------------------------------------------

    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """
        Extract a zip archive to a target directory.

        Returns
        -------
        bool
            True on success, False on failure.
        """
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
            logger.info(f"Extracted: {zip_path.name} → {extract_to}")
            return True
        except zipfile.BadZipFile:
            logger.error(f"Corrupt zip file: {zip_path}")
        except Exception as exc:
            logger.error(f"Extraction failed for {zip_path}: {exc}")
        return False

    def load_shapefile(self, file_path: Path) -> Optional[gpd.GeoDataFrame]:
        """
        Load a shapefile into a GeoDataFrame.

        Returns
        -------
        GeoDataFrame or None
        """
        if not file_path.exists():
            logger.error(f"Shapefile not found: {file_path}")
            return None
        try:
            gdf = gpd.read_file(file_path)
            logger.info(
                f"Loaded: {file_path.name} — "
                f"{len(gdf)} feature(s), CRS: {gdf.crs}"
            )
            return gdf
        except Exception as exc:
            logger.error(f"Failed to load {file_path}: {exc}")
            return None

    def process_region(self, region_name: str) -> Optional[gpd.GeoDataFrame]:
        """
        Extract (if needed) and load the shapefile for a given region.

        Parameters
        ----------
        region_name : str
            Name of the region — must match a ``.zip`` filename (without extension)
            in ``zip_files_directory``.

        Returns
        -------
        GeoDataFrame or None
            Loaded coastline GeoDataFrame, or None on failure.
        """
        region_dir     = self.directory_path / region_name
        shapefile_path = region_dir / f"{region_name}.shp"

        if not shapefile_path.exists():
            zip_path = self.zip_files_directory / f"{region_name}.zip"
            if not zip_path.exists():
                logger.error(
                    f"Zip file not found: {zip_path}\n"
                    f"Available regions: {self.regions}"
                )
                return None
            if not self.extract_zip(zip_path, region_dir):
                return None

        gdf = self.load_shapefile(shapefile_path)
        if gdf is not None:
            self.coast_lines.append(gdf)
        return gdf


# ---------------------------------------------------------------------------


class PointGenerator:
    """
    Generates evenly spaced centroid points along a LineString geometry
    for stratified random patch sampling.

    Examples
    --------
    >>> from s12coasts.patch_generation.sampling import PointGenerator
    >>> from shapely.ops import unary_union
    >>> line = unary_union(coast_gdf.geometry)
    >>> points = PointGenerator.generate_points(line, num_points=300)
    >>> len(points)
    300
    """

    @staticmethod
    def generate_points(line, num_points: int) -> list[Point]:
        """
        Interpolate ``num_points`` equally spaced points along ``line``.

        Parameters
        ----------
        line : shapely geometry
            A LineString or MultiLineString coastline geometry.
        num_points : int
            Number of centroid points to generate.

        Returns
        -------
        list of shapely.geometry.Point
        """
        distances = np.linspace(0, line.length, num_points)
        points    = [line.interpolate(d) for d in distances]
        logger.debug(
            f"Generated {len(points)} centroid point(s) along coastline "
            f"(total length: {line.length:.1f} map units)."
        )
        return points

    @staticmethod
    def to_geodataframe(
        points: list[Point],
        crs: str = "EPSG:4269",
    ) -> gpd.GeoDataFrame:
        """
        Convert a list of points to an exploded GeoDataFrame.

        Parameters
        ----------
        points : list of Point
        crs : str
            Coordinate reference system (default: NAD83 geographic).

        Returns
        -------
        gpd.GeoDataFrame
        """
        multipoint = unary_union(points)
        gdf = (
            gpd.GeoDataFrame(geometry=[multipoint], crs=crs)
            .explode(index_parts=True)
            .reset_index(drop=True)
        )
        logger.debug(f"Converted {len(gdf)} point(s) to GeoDataFrame (CRS: {crs}).")
        return gdf
