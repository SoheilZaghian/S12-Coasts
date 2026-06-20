"""
s12coasts.eobc.cli
==================
Command-line interface entry point for Stage 2 — EOBC Water Mask Generation.

Usage
-----
    # Via installed entry point
    s12coasts-eobc --config configs/eobc_config.yaml

    # Via Python module
    python -m s12coasts.eobc.cli --config configs/eobc_config.yaml

    # Full CLI options
    s12coasts-eobc \\
        --input_dir  data/patches/Huron_Lake \\
        --output_dir outputs/masks/Huron_Lake \\
        --region     Huron_Lake \\
        --visualise
"""

from __future__ import annotations

import argparse
import logging

from s12coasts.eobc.pipeline import EOBCPipeline
from s12coasts.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("s12coasts.eobc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="s12coasts-eobc",
        description=(
            "S12-Coasts Stage 2: Ensemble Object-Based Clustering (EOBC) "
            "Water Mask Generation.\n\n"
            "Processes paired Sentinel-1/Sentinel-2 GeoTIFF patches and "
            "generates binary water mask annotations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  s12coasts-eobc --config configs/eobc_config.yaml\n"
            "  s12coasts-eobc --input_dir data/patches/Huron_Lake "
            "--output_dir outputs/masks/Huron_Lake --region Huron_Lake\n"
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="Path to YAML configuration file (overrides defaults).",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None, metavar="PATH",
        help="Directory containing input GeoTIFF patches.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, metavar="PATH",
        help="Root output directory — S1/, S2/, and gt/ are created here.",
    )
    parser.add_argument(
        "--region", type=str, default=None, metavar="NAME",
        help="Region name (used in logging and metadata CSV).",
    )
    parser.add_argument(
        "--n_pca", type=int, default=None, metavar="N",
        help="Number of PCA components for feature extraction (default: 5).",
    )
    parser.add_argument(
        "--visualise", action="store_true",
        help="Display patch visualisations during processing.",
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

    config = load_config(args.config, config_type="eobc")

    # CLI overrides
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.region:
        config["region_name"] = args.region
        config.setdefault("metadata", {})["region_name"] = args.region
    if args.n_pca:
        config["n_pca_components"] = args.n_pca
    if args.visualise:
        config["visualise"] = True

    logger.info(
        f"S12-Coasts EOBC Pipeline\n"
        f"  Region     : {config['region_name']}\n"
        f"  Input dir  : {config['input_dir']}\n"
        f"  Output dir : {config['output_dir']}\n"
        f"  PCA comps  : {config['n_pca_components']}\n"
        f"  Visualise  : {config['visualise']}"
    )

    pipeline = EOBCPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
