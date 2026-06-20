"""
s12coasts.eobc.clusterer
========================
Ensemble Object-Based Clustering and region-based majority voting.

Implements the two core formulae of the EOBC framework:

Eq. 1 — Water Class Identification:
    c^m(C_j^m) = 1  if j = argmax_j μ_MNDWI(C_j^m)
               = 0  otherwise

Eq. 2 — Region-Based Majority Voting:
    ĉ(R_l) = 1  if Σ_m c^m(R_l) > M/2
           = 0  otherwise
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from s12coasts.eobc.algorithms import FUZZY_AVAILABLE, build_algorithm_registry

logger = logging.getLogger(__name__)


class EOBCClusterer:
    """
    Applies the best clustering configurations found by ClusteringOptimiser,
    identifies the water class per algorithm via maximum mean MNDWI (Eq. 1),
    and fuses results via region-based majority voting (Eq. 2).

    Parameters
    ----------
    algorithms : dict
        Algorithm registry from ``build_algorithm_registry()``.
    best_params_df : pd.DataFrame
        Best parameter DataFrame from ``ClusteringOptimiser``.
    pca_data : np.ndarray
        PCA feature matrix, shape (n_regions, n_components).

    Examples
    --------
    >>> from s12coasts.eobc.clusterer import EOBCClusterer
    >>> from s12coasts.eobc.algorithms import build_algorithm_registry
    >>> clusterer = EOBCClusterer(build_algorithm_registry(), best_params_df, pca_features)
    >>> clusterer.cluster()
    >>> water_mask = clusterer.majority_vote(label_image, mndwi_image)
    """

    def __init__(
        self,
        algorithms: dict,
        best_params_df: pd.DataFrame,
        pca_data: np.ndarray,
    ) -> None:
        self.algorithms = algorithms
        self.best_params_df = best_params_df
        self.X = pca_data
        self._per_algorithm_labels: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Step 4: Ensemble Object-Based Clustering
    # ------------------------------------------------------------------

    def cluster(self) -> list[np.ndarray]:
        """
        Apply each algorithm with its optimised parameters to the PCA features.

        Returns
        -------
        list of np.ndarray
            Per-algorithm region label arrays, shape (n_regions,) each.
        """
        self._per_algorithm_labels = []

        for name, algorithm in self.algorithms.items():
            row = self.best_params_df[self.best_params_df["Algorithm"] == name]
            if row.empty:
                logger.debug(f"No best params for '{name}'. Skipping.")
                continue

            params = eval(row["Parameters"].values[0])  # noqa: S307

            try:
                if name == "FuzzyCMeans" and FUZZY_AVAILABLE:
                    from skfuzzy import cmeans
                    _, u, *_ = cmeans(self.X.T, **params)
                    labels = np.argmax(u, axis=0)
                else:
                    alg = algorithm.set_params(**params)
                    labels = alg.fit_predict(self.X)

                self._per_algorithm_labels.append(labels)
                logger.debug(f"Clustered with '{name}': {len(np.unique(labels))} cluster(s).")

            except Exception as exc:
                logger.warning(f"Clustering with '{name}' failed: {exc}")

        return self._per_algorithm_labels

    # ------------------------------------------------------------------
    # Step 5: Water class identification via max mean MNDWI  [Eq. 1]
    # ------------------------------------------------------------------

    @staticmethod
    def identify_water_class(
        cluster_labels: np.ndarray,
        mndwi_per_region: np.ndarray,
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
        mndwi_per_region : np.ndarray
            Mean MNDWI value per labelled region, shape (n_regions,).

        Returns
        -------
        np.ndarray
            Binary water class array, shape (n_regions,): 1=water, 0=non-water.
        """
        unique_clusters = np.unique(cluster_labels)
        if len(unique_clusters) < 2:
            logger.debug("Only one cluster produced — assigning all regions as non-water.")
            return np.zeros_like(cluster_labels, dtype=np.uint8)

        cluster_mndwi = {
            c: float(np.nanmean(mndwi_per_region[cluster_labels == c]))
            for c in unique_clusters
        }
        water_cluster = max(cluster_mndwi, key=cluster_mndwi.get)

        water_class = np.zeros_like(cluster_labels, dtype=np.uint8)
        water_class[cluster_labels == water_cluster] = 1
        return water_class

    # ------------------------------------------------------------------
    # Pixel-level mask reconstruction
    # ------------------------------------------------------------------

    def build_region_mask(
        self,
        label_image: np.ndarray,
        region_labels: np.ndarray,
        mndwi_image: np.ndarray,
    ) -> np.ndarray:
        """
        Map per-region binary water class assignments back to pixel space.

        Parameters
        ----------
        label_image : np.ndarray
            Connected component label image, shape (H, W).
        region_labels : np.ndarray
            Per-algorithm cluster labels for each region, shape (n_regions,).
        mndwi_image : np.ndarray
            MNDWI image, shape (H, W).

        Returns
        -------
        np.ndarray
            Binary pixel-level water mask, shape (H, W).
        """
        unique_regions = np.unique(label_image)
        mndwi_per_region = np.array([
            float(np.nanmean(mndwi_image[label_image == r]))
            for r in unique_regions
        ])

        water_classes = self.identify_water_class(region_labels, mndwi_per_region)

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
            Final binary water mask, shape (H, W): 1=water, 0=non-water.
        """
        if not self._per_algorithm_labels:
            logger.warning(
                "No per-algorithm labels found. "
                "Call cluster() before majority_vote()."
            )
            return np.zeros_like(label_image, dtype=np.uint8)

        per_algorithm_masks = [
            self.build_region_mask(label_image, labels, mndwi_image)
            for labels in self._per_algorithm_labels
        ]

        # Region-based majority vote — mode across the algorithm stack
        stack = np.stack(per_algorithm_masks, axis=0)   # (M, H, W)
        final_mask = stats.mode(stack, axis=0, keepdims=False).mode.astype(np.uint8)

        water_px = int(np.sum(final_mask))
        total_px = final_mask.size
        logger.debug(
            f"Majority vote complete — water: {water_px} px "
            f"({100 * water_px / total_px:.1f}%) of {total_px} total."
        )
        return final_mask
