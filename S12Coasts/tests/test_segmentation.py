"""
Tests for s12coasts.segmentation — model, metrics, and training utilities.

These tests use synthetic data to avoid requiring real satellite imagery,
trained model weights, or GPU resources.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Skip entire module if deep learning dependencies are not available
# ---------------------------------------------------------------------------

try:
    from transformers import SegformerForSemanticSegmentation
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DL_AVAILABLE,
    reason="transformers not installed — skipping segmentation tests.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def segformer_model(device):
    """Load a minimal SegFormer (MiT-B0) for fast CPU testing."""
    from s12coasts.segmentation.model import load_segformer
    return load_segformer(backbone="nvidia/mit-b0").to(device)


@pytest.fixture
def synthetic_batch():
    """Synthetic batch mimicking a TorchGeo DataLoader output."""
    rng = torch.Generator().manual_seed(42)
    return {
        "image": torch.rand(4, 20, 64, 64, generator=rng),   # (B, C, H, W)
        "mask":  torch.randint(0, 2, (4, 1, 64, 64)),         # (B, 1, H, W)
    }


@pytest.fixture
def perfect_pred():
    """Perfect binary prediction — all pixels correct."""
    labels = np.array([[0, 1, 1, 0], [1, 1, 0, 0]])
    return labels, labels.copy()


@pytest.fixture
def random_pred():
    """Random binary prediction."""
    rng = np.random.default_rng(0)
    true  = rng.integers(0, 2, size=(64, 64))
    pred  = rng.integers(0, 2, size=(64, 64))
    return pred, true


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_perfect_prediction_gives_ones(self, perfect_pred):
        """All metrics should equal 1.0 for a perfect prediction."""
        from s12coasts.segmentation.model import compute_metrics
        pred, true = perfect_pred
        metrics = compute_metrics(pred, true)
        for key, val in metrics.items():
            assert abs(val - 1.0) < 1e-6, (
                f"Expected {key}=1.0 for perfect prediction, got {val:.6f}"
            )

    def test_all_zeros_prediction(self):
        """Predicting all non-water should give Recall=0 for the water class."""
        from s12coasts.segmentation.model import compute_metrics
        true = np.array([1, 1, 0, 0])
        pred = np.zeros_like(true)
        metrics = compute_metrics(pred, true)
        assert metrics["Recall"] == 0.0

    def test_output_keys(self, random_pred):
        """compute_metrics should return all five expected metric keys."""
        from s12coasts.segmentation.model import compute_metrics
        pred, true = random_pred
        metrics = compute_metrics(pred, true)
        expected_keys = {"OA", "IoU", "F_score", "Precision", "Recall"}
        assert set(metrics.keys()) == expected_keys

    def test_metric_values_in_range(self, random_pred):
        """All metric values should lie in [0, 1]."""
        from s12coasts.segmentation.model import compute_metrics
        pred, true = random_pred
        metrics = compute_metrics(pred, true)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, (
                f"{key}={val:.4f} is outside [0, 1]."
            )

    def test_flat_and_2d_inputs_equivalent(self):
        """Metrics should be identical for 1-D and 2-D inputs."""
        from s12coasts.segmentation.model import compute_metrics
        rng = np.random.default_rng(1)
        true_2d = rng.integers(0, 2, (32, 32))
        pred_2d = rng.integers(0, 2, (32, 32))
        m2d = compute_metrics(pred_2d,        true_2d)
        m1d = compute_metrics(pred_2d.ravel(), true_2d.ravel())
        for key in m2d:
            assert abs(m2d[key] - m1d[key]) < 1e-9


# ---------------------------------------------------------------------------
# Tests: load_segformer
# ---------------------------------------------------------------------------

class TestLoadSegformer:

    def test_model_loads(self, segformer_model):
        """SegFormer should load without errors."""
        assert segformer_model is not None

    def test_model_has_parameters(self, segformer_model):
        """Model should have at least 1M trainable parameters."""
        n_params = sum(p.numel() for p in segformer_model.parameters()) / 1e6
        assert n_params >= 1.0, f"Too few parameters: {n_params:.1f}M"

    def test_forward_pass(self, segformer_model, device):
        """Forward pass should produce logits of shape (B, num_labels, H, W)."""
        pixel_values = torch.rand(2, 3, 64, 64).to(device)
        with torch.no_grad():
            outputs = segformer_model(pixel_values=pixel_values)
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 2      # batch size
        assert outputs.logits.shape[1] == 2      # num_labels (water / non-water)

    def test_num_labels(self, segformer_model):
        """Model should be configured for 2 output classes."""
        assert segformer_model.config.num_labels == 2

    def test_label_mapping(self, segformer_model):
        """Model should have correct water / non-water label mapping."""
        assert segformer_model.config.id2label[0] == "non-water"
        assert segformer_model.config.id2label[1] == "water"


# ---------------------------------------------------------------------------
# Tests: filter_invalid_samples
# ---------------------------------------------------------------------------

class TestFilterInvalidSamples:

    def test_valid_batch_returns_all_indices(self, synthetic_batch):
        """All indices should be returned for a clean batch."""
        from s12coasts.segmentation.model import filter_invalid_samples
        valid = filter_invalid_samples(synthetic_batch)
        assert valid == list(range(synthetic_batch["image"].shape[0]))

    def test_nan_image_excluded(self, synthetic_batch):
        """Samples with NaN image values should be excluded."""
        from s12coasts.segmentation.model import filter_invalid_samples
        batch = {k: v.clone() for k, v in synthetic_batch.items()}
        batch["image"][0, 0, 0, 0] = float("nan")   # corrupt sample 0
        valid = filter_invalid_samples(batch)
        assert 0 not in valid

    def test_invalid_mask_excluded(self, synthetic_batch):
        """Samples with mask values outside {0, 1} should be excluded."""
        from s12coasts.segmentation.model import filter_invalid_samples
        batch = {k: v.clone() for k, v in synthetic_batch.items()}
        batch["mask"][1, 0, 0, 0] = 255   # corrupt sample 1
        valid = filter_invalid_samples(batch)
        assert 1 not in valid

    def test_all_invalid_returns_empty(self):
        """Empty batch or all-NaN batch should return an empty list."""
        from s12coasts.segmentation.model import filter_invalid_samples
        batch = {
            "image": torch.full((2, 20, 64, 64), float("nan")),
            "mask":  torch.zeros(2, 1, 64, 64),
        }
        valid = filter_invalid_samples(batch)
        assert valid == []


# ---------------------------------------------------------------------------
# Tests: evaluate (single-batch smoke test)
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_evaluate_returns_dict(self, segformer_model, synthetic_batch, device):
        """evaluate() should return a dict with the five expected metric keys."""
        from torch.utils.data import DataLoader
        from s12coasts.segmentation.train import evaluate

        # Wrap synthetic batch in a simple iterable DataLoader substitute
        class _FakeLoader:
            def __iter__(self):
                yield synthetic_batch
            def __len__(self):
                return 1

        metrics = evaluate(
            model=segformer_model,
            loader=_FakeLoader(),
            device=device,
            num_bands=3,    # MiT-B0 expects 3-channel input in default config
        )
        expected_keys = {"OA", "IoU", "F_score", "Precision", "Recall"}
        assert set(metrics.keys()) == expected_keys

    def test_evaluate_metrics_in_range(self, segformer_model, synthetic_batch, device):
        """All evaluate() metrics should lie in [0, 1]."""
        from s12coasts.segmentation.train import evaluate

        class _FakeLoader:
            def __iter__(self):
                yield {
                    "image": synthetic_batch["image"][:, :3, :, :],
                    "mask":  synthetic_batch["mask"],
                }
            def __len__(self):
                return 1

        metrics = evaluate(
            model=segformer_model,
            loader=_FakeLoader(),
            device=device,
            num_bands=3,
        )
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key}={val:.4f} outside [0, 1]."


# ---------------------------------------------------------------------------
# Tests: visualise utilities
# ---------------------------------------------------------------------------

class TestVisualise:

    def test_plot_rgb_shape(self):
        """plot_rgb() should return an (H, W, 3) array clipped to [0, 1]."""
        from s12coasts.segmentation.visualise import plot_rgb
        image = torch.rand(20, 64, 64)
        rgb   = plot_rgb(image, brightness=3.0)
        assert rgb.shape == (64, 64, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_plot_triplet_runs_without_error(self, tmp_path):
        """plot_triplet() should save a PNG file without raising errors."""
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for tests
        from s12coasts.segmentation.visualise import plot_triplet

        rng = np.random.default_rng(0)
        rgb = np.clip(rng.random((64, 64, 3)), 0, 1)
        gt  = rng.integers(0, 2, (64, 64))
        pred = rng.integers(0, 2, (64, 64))

        save_path = tmp_path / "test_triplet.png"
        plot_triplet(rgb, gt, pred, title="Test", save_path=save_path)
        assert save_path.exists(), "plot_triplet() did not save the figure."
