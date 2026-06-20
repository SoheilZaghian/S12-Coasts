"""
s12coasts.patch_generation.cli
================================
Command-line interface entry point for Stage 1 — Patch Generation.

Usage
-----
    # Via installed entry point
    s12coasts-patch --config configs/patch_config.yaml

    # Via Python module
    python -m s12coasts.patch_generation.cli --config configs/patch_config.yaml

    # Full CLI options
    s12coasts-patch \\
        --region     Madagascar_line \\
        --n_points   300 \\
        --start_date 2018-01-01 \\
        --end_date   2024-12-31
"""

from __future__ import annotations

import argparse
import logging

from shapely.ops import unary_union

from s12coasts.patch_generation.boundary import PatchBoundaryCreator
from s12coasts.patch_generation.processor import PatchProcessor
from s12coasts.patch_generation.sampling import PointGenerator, ShapefileProcessor
from s12coasts.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("s12coasts.patch_generation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="s12coasts-patch",
        description=(
            "S12-Coasts Stage 1: Patch Generation Pipeline.\n\n"
            "Performs stratified random sampling along water boundary vectors, "
            "retrieves paired Sentinel-1/Sentinel-2 imagery via Google Earth Engine, "
            "and submits patch export tasks to Google Drive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  s12coasts-patch --config configs/patch_config.yaml\n"
            "  s12coasts-patch --region Madagascar_line --n_points 300\n"
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="Path to YAML configuration file (overrides defaults).",
    )
    parser.add_argument(
        "--region", type=str, default=None, metavar="NAME",
        help="Region name (must match a .zip shapefile in zip_files_directory).",
    )
    parser.add_argument(
        "--n_points", type=int, default=None, metavar="N",
        help="Number of centroid points to sample along the coastline.",
    )
    parser.add_argument(
        "--start_date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Start date for imagery search.",
    )
    parser.add_argument(
        "--end_date", type=str, default=None, metavar="YYYY-MM-DD",
        help="End date for imagery search.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config, config_type="patch")

    # CLI overrides
    if args.region:     config["region_name"]  = args.region
    if args.n_points:   config["n_points"]      = args.n_points
    if args.start_date: config["start_date"]    = args.start_date
    if args.end_date:   config["end_date"]      = args.end_date

    region_name = config["region_name"]

    logger.info(
        f"S12-Coasts Patch Generation Pipeline\n"
        f"  Region     : {region_name}\n"
        f"  N points   : {config['n_points']}\n"
        f"  Date range : {config['start_date']} → {config['end_date']}\n"
        f"  Patch size : {config['patch_size_m']} m "
        f"({config['patch_size_px']} px @ {config['resolution_m']} m GSD)\n"
        f"  Cloud limit: {config['cloud_cover_threshold']}%"
    )

    # Authenticate and initialise GEE
    try:
        import ee
        try:
            ee.Initialize()
            logger.info("Google Earth Engine initialised.")
        except Exception:
            logger.info("GEE not authenticated — running earthengine authenticate...")
            ee.Authenticate()
            ee.Initialize()
    except ImportError as exc:
        raise ImportError(
            "Google Earth Engine is required for patch generation.\n"
            "Install with: pip install earthengine-api geemap"
        ) from exc

    import geemap

    # Stage 1a — Load water boundary shapefile
    shp_processor = ShapefileProcessor(
        directory_path=config["directory_path"],
        zip_files_directory=config["zip_files_directory"],
    )
    coast_gdf = shp_processor.process_region(region_name)
    if coast_gdf is None:
        logger.error(f"Could not load shapefile for '{region_name}'. Exiting.")
        return

    # Stage 1b — Generate centroid points along the coastline
    line_union    = unary_union(coast_gdf.geometry)
    points        = PointGenerator.generate_points(line_union, config["n_points"])
    centroid_gdf  = PointGenerator.to_geodataframe(points)
    centroid_ee   = geemap.gdf_to_ee(centroid_gdf)

    # Stage 1c — Create square patch boundaries
    patch_creator    = PatchBoundaryCreator(patch_size_m=config["patch_size_m"])
    buffered_polygons = centroid_ee.map(patch_creator.create_patch_boundary)
    logger.info(
        f"Created {buffered_polygons.size().getInfo()} patch boundary polygon(s)."
    )

    # Stage 1d — Process all patches and submit GEE exports
    processor = PatchProcessor(buffered_polygons, coast_gdf, config)
    processor.process_all_patches()


if __name__ == "__main__":
    main()
