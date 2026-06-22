"""
Tests for s12coasts.eobc.clusterer — EOBCClusterer and majority voting.
"""

import numpy as np
import pandas as pd
import pytest

from s12coasts.eobc.clusterer import EOBCClusterer
from s12coasts.eobc.algorithms import build_algorithm_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pca_features():
    """Synthetic PCA feature matrix — 10 regions, 5 components."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 5)).astype(np.float32)


@pytest.fixture
def best_params_df():
    """Minimal best_params_df covering KMeans only."""
    return pd.DataFrame([
        {
            "Algorithm":              "KMeans",
            "Parameters":            "{'n_clusters': 2, 'algorithm': 'lloyd'}",
            "Silhouette Score":       0.5,
            "Davies-Bouldin Index":   0.8,
            "Calinski-Harabasz Index": 20.0,
        }
    ])


@pytest.fixture
def label_image():
    """Simple 64×64 label image with 10 labelled regions."""
    img = np.zeros((64, 64), dtype=np.int32)
    for region_id in range(10):
        row = (region_id // 5) * 32
        col = (region_id % 5)  * 12
        img[row:row+32, col:col+12] = region_id + 1
    return img


@pytest.fixture
def mndwi_image(label_image):
    """
    Synthetic MNDWI image — regions 1–5 are water (high MNDWI),
    regions 6–10 are land (low MNDWI).
    """
    mndwi = np.full((64, 64), -0.5, dtype=np.float32)
    for r in range(1, 6):
        mndwi[label_image == r] = 0.5
    return mndwi


# ---------------------------------------------------------------------------
# Tests: identify_water_class
# ---------------------------------------------------------------------------

class TestIdentifyWaterClass:

    def test_water_class_is_high_mndwi_cluster(self):
        """Cluster with higher mean MNDWI should be assigned water (1)."""
        cluster_labels    = np.array([0, 0, 1, 1, 0])
        mndwi_per_region  = np.array([0.5, 0.5, -0.3, -0.3, 0.4])
        result = EOBCClusterer.identify_water_class(cluster_labels, mndwi_per_region)
        # Cluster 0 has mean MNDWI = (0.5+0.5+0.4)/3 ≈ 0.47 → water
        assert result[0] == 1
        assert result[2] == 0

    def test_single_cluster_returns_zeros(self):
        """Single cluster — all regions should be non-water."""
        cluster_labels   = np.zeros(5, dtype=int)
        mndwi_per_region = np.array([0.2, 0.3, 0.1, 0.4, 0.2])
        result = EOBCClusterer.identify_water_class(cluster_labels, mndwi_per_region)
        assert np.all(result == 0)

    def test_output_is_binary(self):
        """Output should only contain values 0 and 1."""
        rng = np.random.default_rng(1)
        cluster_labels   = rng.integers(0, 2, size=20)
        mndwi_per_region = rng.uniform(-1, 1, size=20)
        result = EOBCClusterer.identify_water_class(cluster_labels, mndwi_per_region)
        assert set(np.unique(result)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests: cluster + majority_vote
# ---------------------------------------------------------------------------

class TestEOBCClusterer:

    def test_cluster_returns_list(self, pca_features, best_params_df):
        """cluster() should return a list of label arrays."""
        algorithms = {"KMeans": build_algorithm_registry()["KMeans"]}
        clusterer  = EOBCClusterer(algorithms, best_params_df, pca_features)
        labels     = clusterer.cluster()
        assert isinstance(labels, list)
        assert len(labels) >= 1
        assert labels[0].shape == (10,)

    def test_majority_vote_output_shape(
        self, pca_features, best_params_df, label_image, mndwi_image
    ):
        """majority_vote() should return a mask with same shape as label_image."""
        algorithms = {"KMeans": build_algorithm_registry()["KMeans"]}
        clusterer  = EOBCClusterer(algorithms, best_params_df, pca_features)
        clusterer.cluster()
        mask = clusterer.majority_vote(label_image, mndwi_image)
        assert mask.shape == label_image.shape

    def test_majority_vote_is_binary(
        self, pca_features, best_params_df, label_image, mndwi_image
    ):
        """Majority vote output should contain only 0 and 1."""
        algorithms = {"KMeans": build_algorithm_registry()["KMeans"]}
        clusterer  = EOBCClusterer(algorithms, best_params_df, pca_features)
        clusterer.cluster()
        mask = clusterer.majority_vote(label_image, mndwi_image)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_majority_vote_without_cluster_raises_warning(
        self, pca_features, best_params_df, label_image, mndwi_image, caplog
    ):
        """Calling majority_vote() before cluster() should log a warning."""
        import logging
        algorithms = {"KMeans": build_algorithm_registry()["KMeans"]}
        clusterer  = EOBCClusterer(algorithms, best_params_df, pca_features)
        # Do NOT call cluster() first
        with caplog.at_level(logging.WARNING):
            mask = clusterer.majority_vote(label_image, mndwi_image)
        assert np.all(mask == 0)
        assert any("cluster()" in msg for msg in caplog.messages)

    def test_water_regions_identified_correctly(
        self, pca_features, best_params_df, label_image, mndwi_image
    ):
        """
        Water regions (high MNDWI) should be predominantly classified as water.
        Not a strict test — clustering is stochastic — but water pixels should
        dominate in known water regions.
        """
        algorithms = {"KMeans": build_algorithm_registry()["KMeans"]}
        clusterer  = EOBCClusterer(algorithms, best_params_df, pca_features)
        clusterer.cluster()
        mask = clusterer.majority_vote(label_image, mndwi_image)
        # Regions 1–5 are water in the fixture
        water_region_pixels = mask[label_image <= 5]
        # At least some water pixels should be detected
        assert np.sum(water_region_pixels) > 0
