"""
S12-Coasts: Ensemble Object-Based Clustering (EOBC) Framework
==============================================================
Stage 2 of the S12-Coasts pipeline — automated water mask generation from
paired Sentinel-1/Sentinel-2 image patches using object-based ensemble
clustering.

Given a directory of GeoTIFF patches (each containing stacked S1, S2, and
water boundary bands), this module:
    1. Creates labelled region images via connected component analysis.
    2. Extracts object-based spectral and SAR features per labelled region.
    3. Applies an ensemble of five clustering algorithms with grid-search
       hyperparameter optimisation.
    4. Identifies the water class via maximum mean MNDWI (Eq. 1).
    5. Generates the final binary water mask via region-based majority
       voting (Eq. 2).
    6. Exports paired S1 / S2 / ground-truth mask GeoTIFF files.

Paper:
    S12-Coasts: Automated Generation of a Global Multimodal Benchmark for
    Water Body and Coastline Segmentation via Ensemble Object-Based Clustering.
    ISPRS Journal of Photogrammetry and Remote Sensing, 2025.

Authors:
    Soheil Zaghian, Seyed Ali Ahmadi, Ali Mohammadzadeh, Sadegh Jamali

Usage:
    python eobc.py --config configs/eobc_config.yaml
    python eobc.py --input_dir data/patches/Huron_Lake --output_dir outputs/masks

Requirements:
    pip install numpy gdal scikit-learn scikit-fuzzy scikit-image
                scipy matplotlib seaborn pandas tqdm pyyaml

Band Layout (expected GeoTIFF stack):
    Bands  1–13  : Sentinel-2 surface reflectance (B1–B12, QA60)
    Bands 14–18  : Sentinel-2 spectral indices (MNDWI, NDVI, AWEI, WRI, NDPI)
    Bands 19–20  : Sentinel-1 SAR (VV, VH)
    Band  21     : Water boundary raster (0 = background, 255 = boundary)
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from osgeo import gdal, gdalconst
from scipy import stats
from skimage.measure import label
from skimage.morphology import disk, dilation
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

try:
    from skfuzzy import cmeans
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    warnings.warn(
        "scikit-fuzzy not installed. Fuzzy C-Means will be skipped.\n"
        "Install with: pip install scikit-fuzzy",
        stacklevel=2,
    )

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("S12Coasts.EOBC")


# ---------------------------------------------------------------------------
# Band indices (0-based) within the stacked GeoTIFF
# ---------------------------------------------------------------------------

# Sentinel-2 spectral indices (appended after raw bands in Stage 1)
BAND_MNDWI = 13   # Modified Normalised Difference Water Index
BAND_NDVI  = 14   # Normalised Difference Vegetation Index
BAND_AWEI  = 15   # Automated Water Extraction Index
BAND_WRI   = 16   # Water Ratio Index
BAND_NDPI  = 17   # Normalised Difference Pond Index

# Sentinel-1 SAR polarisation bands
BAND_VV    = 18
BAND_VH    = 19

# Water boundary raster (0 = background, 255 = boundary pixel)
BAND_BOUNDARY = 20

# Feature bands used for object-based clustering
# Order: MNDWI, NDVI, AWEI, WRI, NDPI, VV, VH
DEFAULT_FEATURE_BANDS = [
    BAND_MNDWI, BAND_NDVI, BAND_AWEI,
    BAND_WRI, BAND_NDPI, BAND_VV, BAND_VH,
]


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    # I/O
    "input_dir": "data/patches/",
    "output_dir": "outputs/masks/",
    "region_name": "MyRegion",

    # Feature extraction
    "feature_bands": DEFAULT_FEATURE_BANDS,
    "n_pca_components": 5,

    # Patch cropping (centre crop to 512×512)
    "patch_size_px": 512,

    # Metadata
    "metadata": {
        "climate_zone": None,
        "agro_ecological_zone": None,
        "country_name": None,
        "region_name": None,
    },

    # Visualisation
    "visualise": False,      # Set True to display patch plots during processing
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML, merging with defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config.update(yaml.safe_load(f))
        logger.info(f"Configuration loaded from {config_path}")
    return config


# ---------------------------------------------------------------------------
# Clustering algorithm registry and hyperparameter grids
# ---------------------------------------------------------------------------

def build_algorithm_registry(fuzzy_available: bool = FUZZY_AVAILABLE) -> dict:
    """Return the dictionary of active clustering algorithms."""
    registry = {
        "KMeans": KMeans(random_state=42),
        "AgglomerativeClustering": AgglomerativeClustering(),
        "BayesianGMM": BayesianGaussianMixture(random_state=42),
        "AffinityPropagation": AffinityPropagation(),
        "MeanShift": MeanShift(),
    }
    if fuzzy_available:
        registry["FuzzyCMeans"] = None   # handled separately
    return registry


PARAM_GRIDS: dict = {
    "KMeans": {
        "n_clusters": [2],
        "algorithm": ["lloyd", "elkan"],
    },
    "AgglomerativeClustering": {
        "n_clusters": [2],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean", "manhattan", "cosine"],
    },
    "BayesianGMM": {
        "n_components": [2],
        "covariance_type": ["full", "tied", "diag", "spherical"],
    },
    "AffinityPropagation": {
        "damping": [0.5, 0.75, 0.9],
    },
    "MeanShift": {
        "bandwidth": [None, 0.5, 1.0, 1.5, 2.0, 3.0],
    },
    "FuzzyCMeans": {
        "c": [2],
        "m": [1.5, 2.0, 2.5, 3.5],
        "error": [1e-4, 1e-6],
        "maxiter": [int(1e5)],
    },
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

_METRIC_SENTINEL = {
    "Silhouette Score": -999.0,
    "Davies-Bouldin Index": 999.0,
    "Calinski-Harabasz Index": -999.0,
}


def _evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute unsupervised clustering quality metrics.

    Returns default sentinel values when fewer than 2 clusters are produced.
    """
    n_labels = len(np.unique(labels))
    if 1 < n_labels < len(X):
        return {
            "Silhouette Score": silhouette_score(X, labels),
            "Davies-Bouldin Index": davies_bouldin_score(X, labels),
            "Calinski-Harabasz Index": calinski_harabasz_score(X, labels),
        }
    return _METRIC_SENTINEL.copy()


# ---------------------------------------------------------------------------
# ClusteringOptimiser — grid search over hyperparameters
# ---------------------------------------------------------------------------

class ClusteringOptimiser:
    """
    Grid-search hyperparameter optimisation for the ensemble of clustering
    algorithms used in the EOBC framework.

    Optimisation criterion: maximum Silhouette Score (no ground truth required).

    Parameters
    ----------
    pca_data : np.ndarray
        Object-based feature matrix after PCA, shape (n_regions, n_components).
    patch_id : int or str
        Identifier for the current patch (used in logging only).
    """

    def __init__(self, pca_data: np.ndarray, patch_id: int | str = 0):
        self.X = pca_data
        self.patch_id = patch_id
        self.algorithms = build_algorithm_registry()
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.best_params_df: pd.DataFrame = pd.DataFrame()

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run grid search for all active clustering algorithms.

        Returns
        -------
        results_df : pd.DataFrame
            Full grid-search results (all parameter combinations × algorithms).
        best_params_df : pd.DataFrame
            Best parameter configuration per algorithm (highest Silhouette Score).
        """
        records = []

        for name, algorithm in self.algorithms.items():
            if name not in PARAM_GRIDS:
                logger.warning(f"No parameter grid for '{name}'. Skipping.")
                continue

            for params in ParameterGrid(PARAM_GRIDS[name]):
                labels, metrics = self._fit_predict(name, algorithm, params)
                records.append({
                    "Algorithm": name,
                    "Parameters": str(params),
                    **metrics,
                })

        self.results_df = pd.DataFrame(records)
        self.best_params_df = self._select_best(self.results_df)
        return self.results_df, self.best_params_df

    def _fit_predict(
        self,
        name: str,
        algorithm,
        params: dict,
    ) -> tuple[np.ndarray, dict]:
        """Fit a single algorithm with given params and return labels + metrics."""
        try:
            if name == "FuzzyCMeans":
                _, u, *_ = cmeans(self.X.T, **params)
                labels = np.argmax(u, axis=0)
            else:
                alg = algorithm.set_params(**params)
                labels = alg.fit_predict(self.X)

            metrics = _evaluate_clustering(self.X, labels)

        except Exception as exc:
            logger.debug(
                f"[patch {self.patch_id}] {name} | params={params} failed: {exc}"
            )
            labels = np.zeros(len(self.X), dtype=int)
            metrics = _METRIC_SENTINEL.copy()

        return labels, metrics

    @staticmethod
    def _select_best(results_df: pd.DataFrame) -> pd.DataFrame:
        """Select the best-performing parameter set per algorithm."""
        idx = results_df.groupby("Algorithm")["Silhouette Score"].idxmax()
        return results_df.loc[idx].reset_index(drop=True)

    def plot_results(self) -> None:
        """Visualise grid-search results across evaluation metrics."""
        if self.results_df.empty:
            raise RuntimeError(
                "No results available. Call run() before plot_results()."
            )
        metrics = [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(24, 4))
        for ax, metric in zip(axes, metrics):
            ascending = metric == "Davies-Bouldin Index"
            df_sorted = self.results_df.sort_values(metric, ascending=ascending)
            sns.barplot(
                data=df_sorted, x=metric, y="Algorithm", ax=ax, dodge=False
            )
            ax.set_title(metric)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# EOBCClusterer — applies best params and generates water masks
# ---------------------------------------------------------------------------

class EOBCClusterer:
    """
    Applies the best clustering configurations identified by ClusteringOptimiser
    to generate per-algorithm binary label maps, then fuses them via
    region-based majority voting.

    Parameters
    ----------
    algorithms : dict
        Algorithm registry from build_algorithm_registry().
    best_params_df : pd.DataFrame
        Best parameter DataFrame from ClusteringOptimiser.
    pca_data : np.ndarray
        Object-based PCA feature matrix, shape (n_regions, n_components).
    """

    def __init__(
        self,
        algorithms: dict,
        best_params_df: pd.DataFrame,
        pca_data: np.ndarray,
    ):
        self.algorithms = algorithms
        self.best_params_df = best_params_df
        self.X = pca_data
        self._per_algorithm_labels: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Step 4: Ensemble Object-Based Clustering
    # ------------------------------------------------------------------

    def cluster(self) -> list[np.ndarray]:
        """
        Apply each algorithm with its best parameters to the PCA feature matrix.

        Returns
        -------
        list of np.ndarray
            Per-algorithm cluster label arrays, shape (n_regions,) each.
        """
        self._per_algorithm_labels = []

        for name, algorithm in self.algorithms.items():
            row = self.best_params_df[self.best_params_df["Algorithm"] == name]
            if row.empty:
                logger.debug(f"No best params found for '{name}'. Skipping.")
                continue

            params = eval(row["Parameters"].values[0])   # noqa: S307

            try:
                if name == "FuzzyCMeans":
                    _, u, *_ = cmeans(self.X.T, **params)
                    labels = np.argmax(u, axis=0)
                else:
                    alg = algorithm.set_params(**params)
                    labels = alg.fit_predict(self.X)

                self._per_algorithm_labels.append(labels)

            except Exception as exc:
                logger.warning(f"Clustering with '{name}' failed: {exc}")

        return self._per_algorithm_labels

    # ------------------------------------------------------------------
    # Step 5: Water class identification via max mean MNDWI  [Eq. 1]
    # ------------------------------------------------------------------

    @staticmethod
    def identify_water_class(
        cluster_labels: np.ndarray,
        mndwi_values: np.ndarray,
    ) -> np.ndarray:
        """
        Assign water (1) to the cluster with the highest mean MNDWI value.

        Implements Eq. 1:
            c^m(C_j^m) = 1  if j = argmax_j μ_MNDWI(C_j^m)
                        = 0  otherwise

        Parameters
        ----------
        cluster_labels : np.ndarray
            Per-region cluster assignments, shape (n_regions,).
        mndwi_values : np.ndarray
            Mean MNDWI value per labelled region, shape (n_regions,).

        Returns
        -------
        np.ndarray
            Binary water class array, shape (n_regions,): 1 = water, 0 = non-water.
        """
        unique_clusters = np.unique(cluster_labels)
        if len(unique_clusters) < 2:
            return np.zeros_like(cluster_labels)

        cluster_mndwi = {
            c: np.nanmean(mndwi_values[cluster_labels == c])
            for c in unique_clusters
        }
        water_cluster = max(cluster_mndwi, key=cluster_mndwi.get)

        water_class = np.zeros_like(cluster_labels)
        water_class[cluster_labels == water_cluster] = 1
        return water_class

    # ------------------------------------------------------------------
    # Map region-level labels back to pixel space
    # ------------------------------------------------------------------

    def build_region_mask(
        self,
        label_image: np.ndarray,
        region_labels: np.ndarray,
        mndwi_image: np.ndarray,
    ) -> np.ndarray:
        """
        Map per-region binary water class assignments back to a pixel-level mask.

        Parameters
        ----------
        label_image : np.ndarray
            Connected component label image, shape (H, W).
        region_labels : np.ndarray
            Per-algorithm cluster labels, shape (n_regions,).
        mndwi_image : np.ndarray
            MNDWI image, shape (H, W), used for water class identification.

        Returns
        -------
        np.ndarray
            Binary pixel-level water mask, shape (H, W).
        """
        unique_regions = np.unique(label_image)
        region_mndwi = np.array([
            np.nanmean(mndwi_image[label_image == r])
            for r in unique_regions
        ])

        water_classes = self.identify_water_class(region_labels, region_mndwi)

        mask = np.zeros_like(label_image, dtype=np.uint8)
        for region, water_flag in zip(unique_regions, water_classes):
            mask[label_image == region] = water_flag

        return mask

    # ------------------------------------------------------------------
    # Step 6: Region-based majority voting  [Eq. 2]
    # ------------------------------------------------------------------

    def majority_vote(
        self,
        label_image: np.ndarray,
        mndwi_image: np.ndarray,
    ) -> np.ndarray:
        """
        Fuse per-algorithm masks via region-based majority voting.

        Implements Eq. 2:
            ĉ(R_l) = 1  if Σ_m c^m(R_l) > M/2
                   = 0  otherwise

        Parameters
        ----------
        label_image : np.ndarray
            Connected component label image, shape (H, W).
        mndwi_image : np.ndarray
            MNDWI image, shape (H, W).

        Returns
        -------
        np.ndarray
            Final binary water mask, shape (H, W): 1 = water, 0 = non-water.
        """
        per_algorithm_masks = [
            self.build_region_mask(label_image, labels, mndwi_image)
            for labels in self._per_algorithm_labels
        ]

        if not per_algorithm_masks:
            logger.warning("No algorithm masks available for majority voting.")
            return np.zeros_like(label_image, dtype=np.uint8)

        # Region-based majority vote (mode across algorithm stack)
        stack = np.stack(per_algorithm_masks, axis=0)
        final_mask = stats.mode(stack, axis=0, keepdims=False).mode.astype(np.uint8)
        return final_mask


# ---------------------------------------------------------------------------
# FeatureExtractor — object-based feature extraction + PCA
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts object-based spectral and SAR features for each labelled region,
    then applies min-max normalisation and PCA dimensionality reduction.

    Parameters
    ----------
    feature_bands : list of int
        Zero-based band indices in the image stack to use as features.
    n_pca_components : int
        Number of PCA components to retain (default: 5).
    """

    def __init__(
        self,
        feature_bands: list[int] = DEFAULT_FEATURE_BANDS,
        n_pca_components: int = 5,
    ):
        self.feature_bands = feature_bands
        self.n_pca_components = n_pca_components

    def extract(
        self,
        image_array: np.ndarray,
        label_image: np.ndarray,
    ) -> np.ndarray:
        """
        Build a per-region feature matrix and apply normalisation + PCA.

        For each labelled region R_l the following statistics are computed
        over each feature band: median, variance, mean.

        Parameters
        ----------
        image_array : np.ndarray
            Full image array, shape (n_bands, H, W).
        label_image : np.ndarray
            Connected component label image, shape (H, W).

        Returns
        -------
        np.ndarray
            PCA-transformed feature matrix, shape (n_regions, n_pca_components).
        """
        feature_stack = image_array[self.feature_bands, :, :]
        unique_regions = np.unique(label_image)
        n_regions = len(unique_regions)
        n_features = len(self.feature_bands) * 3   # median, variance, mean

        raw_features = np.zeros((n_regions, n_features))

        for i, region in enumerate(unique_regions):
            mask = label_image == region
            region_pixels = feature_stack[:, mask]   # shape: (n_bands, n_pixels)

            raw_features[i, : len(self.feature_bands)] = np.nanmedian(region_pixels, axis=1)
            raw_features[i, len(self.feature_bands) : 2 * len(self.feature_bands)] = (
                np.nanvar(region_pixels, axis=1)
            )
            raw_features[i, 2 * len(self.feature_bands) :] = np.nanmean(region_pixels, axis=1)

        raw_features = np.nan_to_num(raw_features, nan=0.0)

        # Normalise to [0, 1]
        normalised = MinMaxScaler().fit_transform(raw_features)

        # PCA dimensionality reduction
        n_components = min(self.n_pca_components, n_regions, normalised.shape[1])
        pca_result = PCA(n_components=n_components).fit_transform(normalised)

        return pca_result


# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------

class GeoTIFFExporter:
    """
    Handles cropping and exporting of GeoTIFF patches for the S12-Coasts
    dataset structure (S1 / S2 / gt subdirectories).

    Parameters
    ----------
    input_file : str or Path
        Path to the source stacked GeoTIFF.
    output_dir : str or Path
        Root output directory (S1/, S2/, and gt/ subdirectories are created here).
    crop_window : tuple of int
        (start_x, end_x, start_y, end_y) pixel coordinates for centre crop.
    """

    def __init__(
        self,
        input_file: str | Path,
        output_dir: str | Path,
        crop_window: tuple[int, int, int, int],
    ):
        self.input_file = str(input_file)
        self.output_dir = Path(output_dir)
        self.start_x, self.end_x, self.start_y, self.end_y = crop_window
        self.width = self.end_x - self.start_x
        self.height = self.end_y - self.start_y

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("S1", "S2", "gt"):
            (self.output_dir / sub).mkdir(exist_ok=True)

    def _open_source(self) -> gdal.Dataset:
        ds = gdal.Open(self.input_file, gdalconst.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"Cannot open: {self.input_file}")
        return ds

    def export_bands(
        self,
        output_path: str | Path,
        band_indices: list[int],
        dtype: int = gdalconst.GDT_UInt32,
        scale: float = 1.0,
    ) -> None:
        """
        Crop selected bands from the source raster and save as a GeoTIFF.

        Parameters
        ----------
        output_path : str or Path
            Output file path (relative to output_dir or absolute).
        band_indices : list of int
            1-based band indices to export.
        dtype : int
            GDAL data type for the output file.
        scale : float
            Multiplicative scale factor applied to band data before writing.
        """
        output_path = self.output_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        src = self._open_source()
        n_bands = src.RasterCount
        if max(band_indices) > n_bands or min(band_indices) < 1:
            raise ValueError(
                f"Band indices {band_indices} out of range "
                f"(source has {n_bands} bands)."
            )

        # Adjust geotransform to the crop window origin
        gt = src.GetGeoTransform()
        new_gt = (
            gt[0] + self.start_x * gt[1] + self.start_y * gt[2],
            gt[1], gt[2],
            gt[3] + self.start_x * gt[4] + self.start_y * gt[5],
            gt[4], gt[5],
        )

        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(str(output_path), self.width, self.height, len(band_indices), dtype)
        dst.SetGeoTransform(new_gt)
        dst.SetProjection(src.GetProjection())

        for out_idx, src_idx in enumerate(band_indices, start=1):
            data = (
                src.GetRasterBand(src_idx)
                .ReadAsArray(self.start_x, self.start_y, self.width, self.height)
                .astype(np.float32)
                * scale
            )
            dst.GetRasterBand(out_idx).WriteArray(data)

        dst.FlushCache()
        dst = None
        src = None
        logger.debug(f"Exported: {output_path}")

    def export_mask(
        self,
        output_path: str | Path,
        mask_array: np.ndarray,
        dtype: int = gdalconst.GDT_Byte,
    ) -> None:
        """
        Save a NumPy binary mask array as a single-band GeoTIFF.

        Parameters
        ----------
        output_path : str or Path
            Output file path (relative to output_dir).
        mask_array : np.ndarray
            2-D binary array (full image extent — cropped internally).
        dtype : int
            GDAL data type (default: Byte).
        """
        output_path = self.output_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        src = self._open_source()
        gt = src.GetGeoTransform()
        projection = src.GetProjection()
        src = None

        cropped = mask_array[self.start_y:self.end_y, self.start_x:self.end_x]

        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(str(output_path), self.width, self.height, 1, dtype)
        dst.SetGeoTransform(gt)
        dst.SetProjection(projection)
        dst.GetRasterBand(1).WriteArray(cropped)
        dst.FlushCache()
        dst = None
        logger.debug(f"Mask exported: {output_path}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_patch(
    patch_index: int,
    image_array: np.ndarray,
    label_image: np.ndarray,
    masks: list[np.ndarray],
    algorithm_names: list[str],
    crop_window: tuple[int, int, int, int],
) -> None:
    """Display S2 RGB, VV, water boundary, label image, and clustering masks."""
    x0, x1, y0, y1 = crop_window

    titles = (
        ["RGB", "VV", "Water Boundary", "Label Image"]
        + list(algorithm_names)
        + ["Majority Vote"]
    )
    rgb = np.clip(
        image_array[[3, 2, 1], x0:x1, y0:y1].transpose(1, 2, 0) * 5.0 / 10000,
        0, 1,
    )
    images = (
        [
            rgb,
            image_array[BAND_VV, x0:x1, y0:y1],
            image_array[BAND_BOUNDARY, x0:x1, y0:y1],
            label_image[x0:x1, y0:y1],
        ]
        + [m[x0:x1, y0:y1] for m in masks]
    )
    cmaps = ["viridis", "gray", "gray", "jet"] + ["Blues"] * len(masks)

    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    fig.suptitle(f"Patch {patch_index + 1}", fontsize=14, fontweight="bold")
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# MetadataManager
# ---------------------------------------------------------------------------

class MetadataManager:
    """
    Manages the dataset metadata CSV (patch IDs, region info, quality flags).

    Parameters
    ----------
    output_dir : Path
        Root output directory where `data_information.csv` is written.
    region_metadata : dict
        Region-level metadata fields (climate zone, country, etc.).
    """

    COLUMNS = [
        "id", "region_name", "climate_zone", "agro_ecological_zone",
        "country_name", "S1", "S2", "gt", "quality_label", "date",
    ]

    def __init__(self, output_dir: Path, region_metadata: dict):
        self.output_dir = output_dir
        self.region_metadata = region_metadata
        self.csv_path = output_dir / "data_information.csv"
        self._df = self._load_or_create()

    def _load_or_create(self) -> pd.DataFrame:
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=self.COLUMNS)

    def next_id(self) -> str:
        """Return zero-padded patch ID."""
        return str(len(self._df) + 1).zfill(5)

    def append(self, patch_id: str) -> None:
        """Append a metadata row for a new patch."""
        row = {
            "id": patch_id,
            "region_name": self.region_metadata.get("region_name"),
            "climate_zone": self.region_metadata.get("climate_zone"),
            "agro_ecological_zone": self.region_metadata.get("agro_ecological_zone"),
            "country_name": self.region_metadata.get("country_name"),
            "S1": "Yes",
            "S2": "Yes",
            "gt": "Yes",
            "quality_label": None,   # assigned during manual quality control
            "date": None,
        }
        self._df = pd.concat(
            [self._df, pd.DataFrame([row])], ignore_index=True
        )

    def save(self) -> None:
        """Persist the metadata DataFrame to CSV."""
        self._df.to_csv(self.csv_path, index=False)
        logger.info(f"Metadata saved → {self.csv_path}")


# ---------------------------------------------------------------------------
# EOBCPipeline — top-level orchestrator
# ---------------------------------------------------------------------------

class EOBCPipeline:
    """
    Top-level orchestrator for the EOBC water mask generation pipeline.

    Iterates over all GeoTIFF patches in `input_dir`, applies the full
    EOBC processing chain, and writes paired S1 / S2 / gt GeoTIFF outputs.

    Parameters
    ----------
    config : dict
        Pipeline configuration (from load_config()).
    """

    def __init__(self, config: dict):
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

        # Collect and sort patch files
        self.patch_files = sorted(
            [f for f in self.input_dir.glob("*.tif")],
            key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
        )
        logger.info(
            f"EOBC pipeline initialised — {len(self.patch_files)} patch(es) "
            f"in '{self.input_dir}'."
        )

    def _compute_crop_window(self, row: int, col: int) -> tuple[int, int, int, int]:
        """Compute centre-crop pixel coordinates for a 512×512 window."""
        half = self._patch_size // 2
        cx, cy = row // 2, col // 2
        return cx - half, cx + half, cy - half, cy + half

    def _process_polyline_patch(
        self,
        image_array: np.ndarray,
        patch_index: int,
    ) -> np.ndarray:
        """
        Full EOBC processing chain for a polyline-based boundary patch.

        Steps:
            1. Connected component analysis → labelled regions
            2. Object-based feature extraction + PCA
            3. Grid-search hyperparameter optimisation
            4. Ensemble Object-Based Clustering
            5. Water class identification via max MNDWI  [Eq. 1]
            6. Region-based majority voting → final mask  [Eq. 2]

        Parameters
        ----------
        image_array : np.ndarray
            Full stacked image, shape (n_bands, H, W).
        patch_index : int
            Patch index (for logging).

        Returns
        -------
        np.ndarray
            Final binary water mask, shape (H, W).
        """
        boundary = image_array[BAND_BOUNDARY]
        mndwi = image_array[BAND_MNDWI]

        # Step 1 — Connected component analysis
        dilated = dilation(boundary, disk(1))
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

    def run(self) -> None:
        """
        Execute the EOBC pipeline over all patches in the input directory.
        """
        logger.info(f"Starting EOBC pipeline for {len(self.patch_files)} patch(es)...")

        n_success, n_skip, n_fail = 0, 0, 0

        for i, patch_path in enumerate(
            tqdm(self.patch_files, desc=f"EOBC [{self.config['region_name']}]")
        ):
            try:
                # Load patch
                ds = gdal.Open(str(patch_path), gdalconst.GA_ReadOnly)
                if ds is None:
                    logger.warning(f"Cannot open: {patch_path}. Skipping.")
                    n_skip += 1
                    continue

                image_array = ds.ReadAsArray()
                ds = None

                if image_array.ndim != 3 or image_array.shape[0] < 21:
                    logger.warning(
                        f"Unexpected shape {image_array.shape} in {patch_path.name}. "
                        "Expected (≥21 bands, H, W). Skipping."
                    )
                    n_skip += 1
                    continue

                _, H, W = image_array.shape
                x0, x1, y0, y1 = self._compute_crop_window(H, W)
                crop_window = (x0, x1, y0, y1)

                # --- Polygon-based path (HydroLakes) ---
                boundary = image_array[BAND_BOUNDARY]
                is_polygon = np.any(boundary > 0) and not np.any(boundary == 255)

                if is_polygon:
                    # Boundary already encodes a filled polygon → rasterise directly
                    water_mask = (boundary > 0).astype(np.uint8)
                    label_image = None
                    clusterer = None
                else:
                    # --- Polyline-based path (OSM / NOAA) --- full EOBC chain
                    water_mask, label_image, clusterer = self._process_polyline_patch(
                        image_array, patch_index=i
                    )

                # Export paired S1 / S2 / gt GeoTIFFs
                patch_id = self.metadata.next_id()
                exporter = GeoTIFFExporter(
                    input_file=patch_path,
                    output_dir=self.output_dir,
                    crop_window=(x0, x1, y0, y1),
                )

                # Sentinel-2: bands 1–13 (raw) + 14–18 (indices)
                exporter.export_bands(
                    f"S2/S2_{patch_id}.tif",
                    band_indices=list(range(1, 19)),
                    dtype=gdalconst.GDT_UInt32,
                    scale=10000.0,
                )
                # Sentinel-1: VV (band 19) + VH (band 20)
                exporter.export_bands(
                    f"S1/S1_{patch_id}.tif",
                    band_indices=[19, 20],
                    dtype=gdalconst.GDT_Int32,
                )
                # Ground truth water mask
                exporter.export_mask(
                    f"gt/gt_{patch_id}.tif",
                    mask_array=water_mask,
                )

                self.metadata.append(patch_id)

                # Optional visualisation
                if self.config.get("visualise") and label_image is not None:
                    masks = (
                        [
                            clusterer.build_region_mask(
                                label_image, labels, image_array[BAND_MNDWI]
                            )
                            for labels in clusterer._per_algorithm_labels
                        ]
                        + [water_mask]
                    )
                    visualise_patch(
                        i, image_array, label_image, masks,
                        list(build_algorithm_registry().keys()),
                        crop_window=(x0, x1, y0, y1),
                    )

                n_success += 1

            except Exception as exc:
                logger.error(f"[{patch_path.name}] Unhandled error: {exc}")
                n_fail += 1

        self.metadata.save()
        logger.info(
            f"EOBC pipeline complete — "
            f"{n_success} succeeded · {n_skip} skipped · {n_fail} failed "
            f"out of {len(self.patch_files)} patches."
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S12-Coasts Stage 2: EOBC Water Mask Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing input GeoTIFF patches.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root output directory for S1 / S2 / gt exports.",
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help="Region name (used in logging and metadata).",
    )
    parser.add_argument(
        "--visualise", action="store_true",
        help="Display patch visualisations during processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.region:
        config["region_name"] = args.region
        config.setdefault("metadata", {})["region_name"] = args.region
    if args.visualise:
        config["visualise"] = True

    pipeline = EOBCPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
