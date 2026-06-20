"""
S12-Coasts: Patch Generation Pipeline
======================================
Stage 1 of the EOBC framework — stratified random sampling along water boundary
vectors and extraction of paired Sentinel-1/Sentinel-2 image patches via Google
Earth Engine (GEE).

Paper:
    S12-Coasts: Automated Generation of a Global Multimodal Benchmark for
    Water Body and Coastline Segmentation via Ensemble Object-Based Clustering.
    ISPRS Journal of Photogrammetry and Remote Sensing, 2025.

Authors:
    Soheil Zaghian, Seyed Ali Ahmadi, Ali Mohammadzadeh, Sadegh Jamali

Usage:
    python patch_generation.py --config configs/patch_config.yaml
    python patch_generation.py --region Madagascar_line --n_points 300

Requirements:
    pip install earthengine-api geemap geopandas shapely tqdm pandas pyyaml

Notes:
    - Authenticate GEE before first use: earthengine authenticate
    - Outputs are exported to Google Drive via the GEE batch export API.
    - Each patch is 512×512 pixels at 10 m resolution (5,120 m × 5,120 m).
"""

import os
import zipfile
import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from tqdm import tqdm

# GEE imports — initialised in main() after authentication
try:
    import ee
    import geemap
except ImportError as e:
    raise ImportError(
        "Google Earth Engine and geemap are required.\n"
        "Install with: pip install earthengine-api geemap\n"
        "Then authenticate: earthengine authenticate"
    ) from e


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("S12Coasts.PatchGeneration")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "region_name": "Madagascar_line",
    "n_points": 300,
    "patch_size_m": 5120,          # metres  → 512 px at 10 m resolution
    "patch_size_px": 512,          # pixels
    "resolution_m": 10,            # GSD in metres
    "cloud_cover_threshold": 30,   # maximum Sentinel-2 cloud cover (%)
    "max_temporal_gap_days": 365,  # search window around S2 date (±days)
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "collection_sen2": "COPERNICUS/S2_SR_HARMONIZED",
    "collection_sen1": "COPERNICUS/S1_GRD",
    "directory_path": "data/shapefiles/extracted/",
    "zip_files_directory": "data/shapefiles/zipped/",
    "output_folder": "outputs/patches/",
    "drive_folder_prefix": "S12Coasts_Patches",
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        config.update(user_config)
        logger.info(f"Configuration loaded from {config_path}")
    else:
        logger.info("Using default configuration.")
    return config


# ---------------------------------------------------------------------------
# Shapefile Processing
# ---------------------------------------------------------------------------

class ShapefileProcessor:
    """
    Extracts and loads regional water boundary shapefiles.

    Parameters
    ----------
    directory_path : str
        Directory where extracted shapefiles are stored.
    zip_files_directory : str
        Directory containing zipped shapefiles.
    """

    def __init__(self, directory_path: str, zip_files_directory: str):
        self.directory_path = Path(directory_path)
        self.zip_files_directory = Path(zip_files_directory)
        self.coast_lines: list[gpd.GeoDataFrame] = []

        if not self.zip_files_directory.exists():
            raise FileNotFoundError(
                f"Zip files directory not found: {self.zip_files_directory}\n"
                "Please place your regional shapefiles (.zip) in this directory."
            )

        self.regions = [
            f.stem for f in self.zip_files_directory.glob("*.zip")
        ]
        logger.info(f"Found {len(self.regions)} region(s): {self.regions}")

    def extract_zip(self, zip_file_path: Path, extract_to_path: Path) -> bool:
        """Extract a zip archive. Returns True on success."""
        try:
            extract_to_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, "r") as zf:
                zf.extractall(extract_to_path)
            logger.info(f"Extracted: {zip_file_path.name} → {extract_to_path}")
            return True
        except zipfile.BadZipFile:
            logger.error(f"Corrupt zip file: {zip_file_path}")
        except Exception as e:
            logger.error(f"Extraction failed for {zip_file_path}: {e}")
        return False

    def load_shapefile(self, file_path: Path) -> Optional[gpd.GeoDataFrame]:
        """Load a shapefile into a GeoDataFrame."""
        if not file_path.exists():
            logger.error(f"Shapefile not found: {file_path}")
            return None
        try:
            gdf = gpd.read_file(file_path)
            logger.info(
                f"Loaded shapefile: {file_path.name} "
                f"({len(gdf)} features, CRS: {gdf.crs})"
            )
            return gdf
        except Exception as e:
            logger.error(f"Failed to load shapefile {file_path}: {e}")
            return None

    def process_region(self, region_name: str) -> Optional[gpd.GeoDataFrame]:
        """
        Extract (if needed) and load the shapefile for a given region.

        Parameters
        ----------
        region_name : str
            Name of the region (must match a .zip filename without extension).

        Returns
        -------
        GeoDataFrame or None
        """
        region_dir = self.directory_path / region_name
        shapefile_path = region_dir / f"{region_name}.shp"

        if not shapefile_path.exists():
            zip_path = self.zip_files_directory / f"{region_name}.zip"
            if not zip_path.exists():
                logger.error(
                    f"Zip file not found: {zip_path}\n"
                    f"Available regions: {self.regions}"
                )
                return None
            success = self.extract_zip(zip_path, region_dir)
            if not success:
                return None

        gdf = self.load_shapefile(shapefile_path)
        if gdf is not None:
            self.coast_lines.append(gdf)
        return gdf


# ---------------------------------------------------------------------------
# Point Generation
# ---------------------------------------------------------------------------

class PointGenerator:
    """Generates evenly spaced points along a LineString geometry."""

    @staticmethod
    def generate_points(line, num_points: int) -> list:
        """
        Interpolate `num_points` equally spaced points along `line`.

        Parameters
        ----------
        line : shapely geometry
            A LineString or MultiLineString geometry.
        num_points : int
            Number of centroid points to generate.

        Returns
        -------
        list of shapely.geometry.Point
        """
        distances = np.linspace(0, line.length, num_points)
        return [line.interpolate(d) for d in distances]


# ---------------------------------------------------------------------------
# Buffer / Patch Boundary Creator
# ---------------------------------------------------------------------------

class PatchBoundaryCreator:
    """
    Creates square patch boundaries (GEE Features) around centroid points.

    Parameters
    ----------
    patch_size_m : int
        Side length of the square patch in metres (default: 5120 → 512 px @ 10 m).
    """

    def __init__(self, patch_size_m: int = 5120):
        self.patch_size_m = patch_size_m
        self.half_size = patch_size_m / 2

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
            A square polygon feature covering the patch extent.
        """
        point = feature.geometry()
        buffer = point.buffer(self.half_size)
        bounds = buffer.bounds()
        return ee.Feature(bounds).copyProperties(feature)


# ---------------------------------------------------------------------------
# Spectral Index Calculations
# ---------------------------------------------------------------------------

# Band formulae for Sentinel-2 surface reflectance (SR) products.
# All inputs are divided by 10000 to convert to [0, 1] reflectance.

SPECTRAL_INDICES = [
    # MNDWI — Modified Normalised Difference Water Index
    (
        lambda img: img.select("B3").subtract(img.select("B11"))
                       .divide(img.select("B3").add(img.select("B11"))),
        "MNDWI",
    ),
    # NDVI — Normalised Difference Vegetation Index
    (
        lambda img: img.select("B8").subtract(img.select("B4"))
                       .divide(img.select("B8").add(img.select("B4"))),
        "NDVI",
    ),
    # AWEI — Automated Water Extraction Index
    (
        lambda img: img.select("B2")
                       .add(img.select("B3").multiply(2.5))
                       .subtract(img.select("B8").add(img.select("B11")).multiply(1.5))
                       .subtract(img.select("B12").multiply(0.25)),
        "AWEI",
    ),
    # WRI — Water Ratio Index
    (
        lambda img: img.select("B3").add(img.select("B4"))
                       .divide(img.select("B8").add(img.select("B11"))),
        "WRI",
    ),
    # NDPI — Normalised Difference Pond Index
    (
        lambda img: img.select("B8").subtract(img.select("B3"))
                       .divide(img.select("B8").add(img.select("B3"))),
        "NDPI",
    ),
]


def add_spectral_index(image, formula, name) -> "ee.Image":
    """Compute a spectral index and append it as a new band."""
    return image.addBands(formula(image).rename(name))


def scale_reflectance(image) -> "ee.Image":
    """Convert Sentinel-2 DN values to surface reflectance [0, 1]."""
    return image.divide(10000)


# ---------------------------------------------------------------------------
# Water Boundary and Patch Processor
# ---------------------------------------------------------------------------

class PatchProcessor:
    """
    Core processor: generates paired Sentinel-1/Sentinel-2 patches and exports
    them to Google Drive via the GEE batch export API.

    Parameters
    ----------
    buffered_polygons : ee.FeatureCollection
        GEE FeatureCollection of square patch boundary polygons.
    boundary_layer : gpd.GeoDataFrame
        Water boundary GeoDataFrame (OSM / NOAA / HydroLakes).
    config : dict
        Processing configuration dictionary.
    """

    def __init__(
        self,
        buffered_polygons,
        boundary_layer: gpd.GeoDataFrame,
        config: dict,
    ):
        self.buffered_polygons = buffered_polygons
        self.boundary_layer = boundary_layer
        self.config = config
        self.region_name = config["region_name"]
        self.output_folder = Path(config["output_folder"])
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.n_patches = buffered_polygons.size().getInfo()
        logger.info(
            f"Initialised PatchProcessor for '{self.region_name}' "
            f"with {self.n_patches} patch(es)."
        )

        # Metadata accumulator
        self._records: list[dict] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_water_boundary_image(
        self,
        ee_polygon,
        patch_gdf: gpd.GeoDataFrame,
    ) -> "ee.Image":
        """
        Rasterise the water boundary vector within a patch extent.

        The boundary is painted onto a zero image (background = 0,
        boundary pixels = 255) and clipped to the patch polygon.
        """
        patch_gdf = patch_gdf.to_crs(self.boundary_layer.crs)
        clipped = self.boundary_layer.clip(patch_gdf.geometry)
        boundary_ee = geemap.gdf_to_ee(clipped)
        zero_image = ee.Image.constant(0)
        painted = zero_image.paint(boundary_ee, 255)
        return painted.clip(ee_polygon)

    @staticmethod
    def _add_date_diff(image, s2_date) -> "ee.Image":
        """Compute absolute day difference between an image and a reference date."""
        s1_date = ee.Date(image.get("system:time_start"))
        diff = s1_date.difference(s2_date, "day").abs()
        return image.set("date_diff", diff)

    def _build_sentinel2_collection(self, ee_polygon) -> tuple:
        """
        Filter and prepare the Sentinel-2 collection for a patch polygon.

        Returns
        -------
        tuple: (ee.ImageCollection with indices, ee.Date of best S2 image)
        """
        s2_bands = [
            "B1", "B2", "B3", "B4", "B5", "B6",
            "B7", "B8", "B8A", "B9", "B11", "B12", "QA60",
        ]

        collection = (
            ee.ImageCollection(self.config["collection_sen2"])
            .filterDate(self.config["start_date"], self.config["end_date"])
            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    self.config["cloud_cover_threshold"],
                )
            )
            .filterBounds(ee_polygon)
            .select(s2_bands)
            .map(scale_reflectance)
        )

        # Select the least cloudy image to anchor the S1 search date
        best_s2 = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
        s2_date = ee.Date(best_s2.get("system:time_start"))

        # Add spectral indices to the full collection
        for formula, name in SPECTRAL_INDICES:
            collection = collection.map(
                lambda img, f=formula, n=name: add_spectral_index(img, f, n)
            )

        return collection, s2_date

    def _build_sentinel1_image(self, ee_polygon, s2_date) -> "ee.Image":
        """
        Select the Sentinel-1 image temporally closest to the S2 acquisition date.

        The search window is ±`max_temporal_gap_days` around the S2 date.
        """
        gap = self.config["max_temporal_gap_days"]
        collection = (
            ee.ImageCollection(self.config["collection_sen1"])
            .filterDate(
                s2_date.advance(-gap, "day"),
                s2_date.advance(gap, "day"),
            )
            .filter(ee.Filter.listContains("system:band_names", "VV"))
            .filter(ee.Filter.listContains("system:band_names", "VH"))
            .filterBounds(ee_polygon)
            .map(lambda img: PatchProcessor._add_date_diff(img, s2_date))
        )
        return collection.sort("date_diff").first().clip(ee_polygon)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_patch(self, patch_index: int) -> dict:
        """
        Process a single patch: build water boundary image, retrieve S1/S2
        imagery, stack bands, and submit a GEE export task.

        Parameters
        ----------
        patch_index : int
            Zero-based index of the patch in `buffered_polygons`.

        Returns
        -------
        dict
            Metadata record for this patch (IDs, dates, export status).
        """
        record = {
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
            # Retrieve the patch feature and polygon geometry
            feature = ee.Feature(
                self.buffered_polygons
                .toList(self.buffered_polygons.size())
                .get(patch_index)
            )
            ee_polygon = feature.geometry()
            patch_gdf = geemap.ee_to_gdf(ee.FeatureCollection([feature]))

            # Build water boundary image
            boundary_image = self._create_water_boundary_image(
                ee_polygon, patch_gdf
            )

            # Build Sentinel-2 collection + best acquisition date
            s2_collection, s2_date = self._build_sentinel2_collection(ee_polygon)
            s2_image = (
                s2_collection.sort("CLOUDY_PIXEL_PERCENTAGE")
                .first()
                .clip(ee_polygon)
            )

            # Build best-matching Sentinel-1 image
            s1_image = self._build_sentinel1_image(ee_polygon, s2_date)

            # Stack: S2 bands + indices | S1 VV/VH | water boundary
            final_image = (
                s2_image
                .addBands(s1_image)
                .addBands(boundary_image.rename("Water_boundary"))
                .toFloat()
            )

            # Submit GEE export task to Google Drive
            drive_folder = (
                f"{self.config['drive_folder_prefix']}_{self.region_name}"
            )
            export_params = {
                "image": final_image,
                "description": record["patch_id"],
                "scale": self.config["resolution_m"],
                "region": ee_polygon,
                "fileFormat": "GeoTIFF",
                "folder": drive_folder,
            }
            task = ee.batch.Export.image.toDrive(**export_params)
            task.start()

            # Collect metadata
            s2_datetime = s2_date.format().getInfo()
            s1_datetime = (
                ee.Date(s1_image.get("system:time_start")).format().getInfo()
            )

            record.update({
                "s1_image_id": s1_image.get("system:index").getInfo(),
                "s2_image_id": s2_image.get("system:index").getInfo(),
                "s1_date": s1_datetime.split("T")[0],
                "s1_time": s1_datetime.split("T")[1],
                "s2_date": s2_datetime.split("T")[0],
                "s2_time": s2_datetime.split("T")[1],
                "export_status": "submitted",
                "drive_folder": drive_folder,
            })

            logger.debug(
                f"[{patch_index}] Export submitted → Drive folder: {drive_folder}"
            )

        except Exception as e:
            logger.error(f"[{patch_index}] Processing failed: {e}")

        return record

    def process_all_patches(self) -> pd.DataFrame:
        """
        Process all patches in the buffered polygon collection.

        Returns
        -------
        pd.DataFrame
            Metadata DataFrame saved to CSV in `output_folder`.
        """
        logger.info(
            f"Starting patch generation for '{self.region_name}' "
            f"({self.n_patches} patches)..."
        )

        records = []
        for i in tqdm(range(self.n_patches), desc=f"[{self.region_name}]"):
            record = self.process_patch(i)
            records.append(record)

        df = pd.DataFrame(records)
        csv_path = self.output_folder / f"{self.region_name}_metadata.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Metadata saved → {csv_path}")

        n_ok = (df["export_status"] == "submitted").sum()
        n_fail = (df["export_status"] == "failed").sum()
        logger.info(
            f"Completed: {n_ok} submitted, {n_fail} failed "
            f"out of {self.n_patches} patches."
        )
        return df


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S12-Coasts Stage 1: Patch Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (overrides defaults).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region name (overrides config file value).",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=None,
        help="Number of centroid points to sample (overrides config).",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for imagery search (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for imagery search (YYYY-MM-DD).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Command-line overrides
    if args.region:
        config["region_name"] = args.region
    if args.n_points:
        config["n_points"] = args.n_points
    if args.start_date:
        config["start_date"] = args.start_date
    if args.end_date:
        config["end_date"] = args.end_date

    # Authenticate and initialise GEE
    try:
        ee.Initialize()
        logger.info("Google Earth Engine initialised successfully.")
    except Exception:
        logger.info("GEE not authenticated. Running earthengine authenticate...")
        ee.Authenticate()
        ee.Initialize()

    region_name = config["region_name"]
    logger.info(f"Processing region: {region_name}")

    # Stage 1a: Load water boundary shapefile
    processor = ShapefileProcessor(
        directory_path=config["directory_path"],
        zip_files_directory=config["zip_files_directory"],
    )
    coast_gdf = processor.process_region(region_name)
    if coast_gdf is None:
        logger.error(f"Could not load shapefile for region '{region_name}'. Exiting.")
        return

    # Stage 1b: Generate evenly spaced centroid points along the coastline
    line_union = unary_union(coast_gdf.geometry)
    points = PointGenerator.generate_points(line_union, config["n_points"])
    multipoint = unary_union(points)

    centroid_gdf = (
        gpd.GeoDataFrame(geometry=[multipoint], crs="EPSG:4269")
        .explode(index_parts=True)
        .reset_index(drop=True)
    )
    centroid_ee = geemap.gdf_to_ee(centroid_gdf)

    # Stage 1c: Create square patch boundary polygons around each centroid
    patch_creator = PatchBoundaryCreator(patch_size_m=config["patch_size_m"])
    buffered_polygons = centroid_ee.map(patch_creator.create_patch_boundary)
    logger.info(
        f"Generated {buffered_polygons.size().getInfo()} patch boundary polygon(s)."
    )

    # Stage 1d: Process all patches and submit GEE export tasks
    patch_processor = PatchProcessor(
        buffered_polygons=buffered_polygons,
        boundary_layer=coast_gdf,
        config=config,
    )
    metadata_df = patch_processor.process_all_patches()

    logger.info(
        f"Patch generation complete. "
        f"{(metadata_df['export_status'] == 'submitted').sum()} export task(s) "
        f"submitted to Google Drive."
    )
    logger.info(
        "Monitor export tasks at: https://code.earthengine.google.com/tasks"
    )


if __name__ == "__main__":
    main()
