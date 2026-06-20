"""
s12coasts.eobc.pipeline
=======================
Top-level EOBCPipeline orchestrator — iterates over all GeoTIFF patches,
runs the full EOBC processing chain, and writes paired S1/S2/gt outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from osgeo import gdal, gdalconst
from skimage.measure import label
from skimage.morphology import disk, dilation
from tqdm import tqdm

from s12coasts.eobc.algorithms import build_algorithm_registry
from s12coasts.eobc.clusterer import EOBCClusterer
from s12coasts.eobc.features import FeatureExtractor
from s12coasts.eobc.optimiser import ClusteringOptimiser
from s12coasts.io.geotiff import GeoTIFFExporter
from s12coasts.io.metadata import MetadataManager
from s12coasts.utils.config import load_config

logger = logging.getLogger(__name__)

# Band indices (0-based) in the stacked GeoTIFF
BAND_MNDWI    = 13
BAND_VV       = 18
BAND_VH       = 19
BAND_BOUNDARY = 20

# Sentinel-2 raw bands (1-based for GDAL)
S2_BANDS = list(range(1, 19))   # bands 1–18 (13 raw + 5 indices)
S1_BANDS = [19, 20]             # VV, VH
GT_BAND  = [21]                 # water boundary / ground truth


class EOBCPipeline:
    """
    Top-level orchestrator for the EOBC water mask generation pipeline.

    Iterates over all GeoTIFF patches in ``input_dir``, applies the full
    EOBC processing chain, and writes paired S1/S2/gt GeoTIFF outputs.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary (from ``load_config()``).

    Examples
    --------
    >>> from s12coasts.eobc.pipeline import EOBCPipeline
    >>> from s12coasts.utils.config import load_config
    >>> config = load_config("configs/eobc_config.yaml")
    >>> pipeline = EOBCPipeline(config)
    >>> pipeline.run()
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.input_dir = Path(config["input_dir"])
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_extractor = FeatureExtractor(
            feature_bands=config["feature_bands"],
            n_pca_components=config["n_pca_components"],
        )
        self.metadata = MetadataManager(
            output_dir=self.output_dir,
            region_metadata=config.get("metadata", {}),
        )
        self._patch_size = config["patch_size_px"]

        self.patch_files = sorted(
            list(self.input_dir.glob("*.tif")),
            key=lambda p: (
                int(p.stem.split("_")[-1])
                if p.stem.split("_")[-1].isdigit()
                else 0
            ),
        )
        logger.info(
            f"EOBCPipeline initialised — {len(self.patch_files)} patch(es) "
            f"in '{self.input_dir}'."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _crop_window(self, H: int, W: int) -> tuple[int, int, int, int]:
        """Compute centre-crop window for a ``patch_size_px`` square."""
        half = self._patch_size // 2
        cx, cy = H // 2, W // 2
        return cx - half, cx + half, cy - half, cy + half

    def _polyline_path(
        self,
        image_array: np.ndarray,
        patch_index: int,
    ) -> tuple[np.ndarray, np.ndarray, EOBCClusterer]:
        """
        Full EOBC chain for polyline-based boundary patches (OSM / NOAA).

        Steps:
            1. Connected component analysis → labelled regions
            2. Object-based feature extraction + PCA
            3. Grid-search hyperparameter optimisation
            4. Ensemble Object-Based Clustering
            5. Water class identification via max MNDWI  [Eq. 1]
            6. Region-based majority voting → final mask  [Eq. 2]
        """
        boundary  = image_array[BAND_BOUNDARY]
        mndwi     = image_array[BAND_MNDWI]

        # Step 1 — Connected component analysis
        dilated     = dilation(boundary, disk(1))
        label_image = label(~dilated.astype(bool), connectivity=1)

        # Step 2 — Object-based feature extraction + PCA
        pca_features = self.feature_extractor.extract(image_array, label_image)

        # Step 3 — Grid-search optimisation
        optimiser = ClusteringOptimiser(pca_features, patch_id=patch_index)
        _, best_params_df = optimiser.run()

        # Steps 4–6 — Ensemble clustering + water identification + majority vote
        clusterer = EOBCClusterer(
            algorithms=build_algorithm_registry(),
            best_params_df=best_params_df,
            pca_data=pca_features,
        )
        clusterer.cluster()
        water_mask = clusterer.majority_vote(label_image, mndwi)

        return water_mask, label_image, clusterer

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full EOBC pipeline over all patches in the input directory.

        Outputs per patch:
            - ``S2/<patch_id>.tif``  — 18-band Sentinel-2 GeoTIFF
            - ``S1/<patch_id>.tif``  — 2-band Sentinel-1 GeoTIFF (VV, VH)
            - ``gt/<patch_id>.tif``  — binary water mask GeoTIFF
            - ``data_information.csv`` — updated patch metadata
        """
        logger.info(
            f"Starting EOBC pipeline — {len(self.patch_files)} patch(es) · "
            f"region: '{self.config['region_name']}'"
        )
        n_ok = n_skip = n_fail = 0

        for i, patch_path in enumerate(
            tqdm(self.patch_files, desc=f"EOBC [{self.config['region_name']}]")
        ):
            try:
                ds = gdal.Open(str(patch_path), gdalconst.GA_ReadOnly)
                if ds is None:
                    logger.warning(f"Cannot open: {patch_path.name}. Skipping.")
                    n_skip += 1
                    continue

                image_array = ds.ReadAsArray()
                ds = None

                if image_array.ndim != 3 or image_array.shape[0] < 21:
                    logger.warning(
                        f"Unexpected shape {image_array.shape} in "
                        f"'{patch_path.name}'. Expected ≥21 bands. Skipping."
                    )
                    n_skip += 1
                    continue

                _, H, W = image_array.shape
                x0, x1, y0, y1 = self._crop_window(H, W)

                # Determine pathway: polygon (HydroLakes) or polyline (OSM/NOAA)
                boundary = image_array[BAND_BOUNDARY]
                is_polygon = np.any(boundary > 0) and not np.any(boundary == 255)

                if is_polygon:
                    # Pathway A — rasterise polygon directly
                    water_mask = (boundary > 0).astype(np.uint8)
                    label_image = None
                    clusterer = None
                else:
                    # Pathway B — full EOBC chain
                    water_mask, label_image, clusterer = self._polyline_path(
                        image_array, patch_index=i
                    )

                # Export S1 / S2 / gt GeoTIFFs
                patch_id = self.metadata.next_id()
                exporter = GeoTIFFExporter(
                    input_file=patch_path,
                    output_dir=self.output_dir,
                    crop_window=(x0, x1, y0, y1),
                )
                exporter.export_bands(f"S2/S2_{patch_id}.tif", S2_BANDS,
                                      dtype=gdalconst.GDT_UInt32, scale=10000.0)
                exporter.export_bands(f"S1/S1_{patch_id}.tif", S1_BANDS,
                                      dtype=gdalconst.GDT_Int32)
                exporter.export_mask(f"gt/gt_{patch_id}.tif", water_mask)

                self.metadata.append(patch_id)

                # Optional visualisation
                if self.config.get("visualise") and label_image is not None:
                    from s12coasts.utils.visualise import visualise_patch
                    per_algo_masks = [
                        clusterer.build_region_mask(
                            label_image, lbl, image_array[BAND_MNDWI]
                        )
                        for lbl in clusterer._per_algorithm_labels
                    ] + [water_mask]
                    visualise_patch(
                        i, image_array, label_image, per_algo_masks,
                        list(build_algorithm_registry().keys()),
                        crop_window=(x0, x1, y0, y1),
                    )

                n_ok += 1

            except Exception as exc:
                logger.error(f"[{patch_path.name}] Unhandled error: {exc}")
                n_fail += 1

        self.metadata.save()
        logger.info(
            f"Pipeline complete — {n_ok} succeeded · {n_skip} skipped · "
            f"{n_fail} failed out of {len(self.patch_files)} patches."
        )
