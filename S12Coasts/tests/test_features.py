"""
Tests for s12coasts.eobc.features — FeatureExtractor.
"""

import numpy as np
import pytest

from s12coasts.eobc.features import FeatureExtractor, DEFAULT_FEATURE_BANDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image():
    """Create a synthetic 21-band image array (n_bands, H, W)."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, size=(21, 64, 64)).astype(np.float32)


@pytest.fixture
def synthetic_label_image():
    """Create a simple label image with 4 labelled regions."""
    label_img = np.zeros((64, 64), dtype=np.int32)
    label_img[:32, :32] = 1
    label_img[:32, 32:] = 2
    label_img[32:, :32] = 3
    label_img[32:, 32:] = 4
    return label_img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:

    def test_output_shape(self, synthetic_image, synthetic_label_image):
        """PCA output should have shape (n_regions, n_pca_components)."""
        extractor = FeatureExtractor(
            feature_bands=DEFAULT_FEATURE_BANDS,
            n_pca_components=5,
        )
        result = extractor.extract(synthetic_image, synthetic_label_image)
        n_regions = len(np.unique(synthetic_label_image))
        assert result.shape == (n_regions, 5), (
            f"Expected ({n_regions}, 5), got {result.shape}"
        )

    def test_no_nan_in_output(self, synthetic_image, synthetic_label_image):
        """PCA output should contain no NaN values."""
        extractor = FeatureExtractor(n_pca_components=3)
        result = extractor.extract(synthetic_image, synthetic_label_image)
        assert not np.any(np.isnan(result)), "NaN values found in PCA output."

    def test_pca_components_capped(self, synthetic_image):
        """PCA components should be capped at min(n_regions, n_features)."""
        # Only 2 regions
        label_img = np.zeros((64, 64), dtype=np.int32)
        label_img[:32, :] = 1
        label_img[32:, :] = 2

        extractor = FeatureExtractor(n_pca_components=100)
        result = extractor.extract(synthetic_image, label_img)
        # n_components capped at min(2 regions, n_features)
        assert result.shape[0] == 2
        assert result.shape[1] <= 2

    def test_invalid_band_index_raises(self, synthetic_image, synthetic_label_image):
        """Should raise ValueError for out-of-range band indices."""
        extractor = FeatureExtractor(feature_bands=[99, 100])
        with pytest.raises(ValueError, match="Band index"):
            extractor.extract(synthetic_image, synthetic_label_image)

    def test_single_region(self, synthetic_image):
        """Should handle a label image with a single region."""
        label_img = np.ones((64, 64), dtype=np.int32)
        extractor = FeatureExtractor(n_pca_components=3)
        result = extractor.extract(synthetic_image, label_img)
        # Only 1 region — PCA will produce 1 sample
        assert result.shape[0] == 1
