"""
s12coasts.eobc.features
=======================
Object-based feature extraction and PCA dimensionality reduction.

For each labelled region produced by connected component analysis, three
statistics are computed per feature band (median, variance, mean), then
normalised with MinMaxScaler and reduced via PCA.

Feature bands (0-based indices in the stacked GeoTIFF):
    13 — MNDWI   Modified Normalised Difference Water Index
    14 — NDVI    Normalised Difference Vegetation Index
    15 — AWEI    Automated Water Extraction Index
    16 — WRI     Water Ratio Index
    17 — NDPI    Normalised Difference Pond Index
    18 — VV      Sentinel-1 VV polarisation
    19 — VH      Sentinel-1 VH polarisation
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Default feature band indices (0-based)
DEFAULT_FEATURE_BANDS: list[int] = [13, 14, 15, 16, 17, 18, 19]


class FeatureExtractor:
    """
    Extracts object-based spectral and SAR features for each labelled region,
    then applies min-max normalisation followed by PCA.

    Parameters
    ----------
    feature_bands : list of int
        Zero-based band indices in the image stack to use as features.
        Defaults to [MNDWI, NDVI, AWEI, WRI, NDPI, VV, VH].
    n_pca_components : int
        Number of PCA components to retain. Automatically capped at
        min(n_regions, n_features). Default: 5.

    Examples
    --------
    >>> from s12coasts.eobc.features import FeatureExtractor
    >>> extractor = FeatureExtractor(n_pca_components=5)
    >>> pca_features = extractor.extract(image_array, label_image)
    >>> pca_features.shape
    (n_regions, 5)
    """

    def __init__(
        self,
        feature_bands: list[int] = DEFAULT_FEATURE_BANDS,
        n_pca_components: int = 5,
    ) -> None:
        self.feature_bands = feature_bands
        self.n_pca_components = n_pca_components

    def extract(
        self,
        image_array: np.ndarray,
        label_image: np.ndarray,
    ) -> np.ndarray:
        """
        Build a per-region feature matrix and apply normalisation + PCA.

        For each labelled region R_l, the following statistics are computed
        over each feature band: median, variance, mean — yielding a feature
        vector of length ``3 × len(feature_bands)`` per region.

        Parameters
        ----------
        image_array : np.ndarray
            Full image array, shape (n_bands, H, W).
        label_image : np.ndarray
            Connected component label image, shape (H, W). Each unique
            integer value identifies a spatially connected region.

        Returns
        -------
        np.ndarray
            PCA-transformed feature matrix, shape (n_regions, n_components).

        Raises
        ------
        ValueError
            If any feature band index exceeds the number of available bands.
        """
        n_bands = image_array.shape[0]
        if max(self.feature_bands) >= n_bands:
            raise ValueError(
                f"Feature band index {max(self.feature_bands)} exceeds "
                f"available bands (n_bands={n_bands})."
            )

        feature_stack = image_array[self.feature_bands]   # (n_feat, H, W)
        unique_regions = np.unique(label_image)
        n_regions = len(unique_regions)
        n_feat = len(self.feature_bands)
        n_features = n_feat * 3   # median · variance · mean

        raw = np.zeros((n_regions, n_features), dtype=np.float32)

        for i, region in enumerate(unique_regions):
            pixels = feature_stack[:, label_image == region]   # (n_feat, n_px)
            raw[i, :n_feat]        = np.nanmedian(pixels, axis=1)
            raw[i, n_feat:2*n_feat] = np.nanvar(pixels, axis=1)
            raw[i, 2*n_feat:]      = np.nanmean(pixels, axis=1)

        raw = np.nan_to_num(raw, nan=0.0)

        # Min-max normalisation to [0, 1]
        normalised = MinMaxScaler().fit_transform(raw)

        # PCA — cap components at available dimensions
        n_components = min(self.n_pca_components, n_regions, n_features)
        pca_result = PCA(n_components=n_components).fit_transform(normalised)

        logger.debug(
            f"FeatureExtractor: {n_regions} regions · "
            f"{n_features} raw features → {n_components} PCA components."
        )
        return pca_result
