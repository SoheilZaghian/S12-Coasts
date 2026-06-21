"""
s12coasts.segmentation.visualise
==================================
Visualisation utilities for SegFormer segmentation results.

Provides functions for:
    - Plotting RGB / ground truth / prediction triplets
    - Batch-level visualisation during inference
    - Saving high-resolution SVG/PNG figures for publication
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchgeo.datasets.utils import unbind_samples

logger = logging.getLogger(__name__)

# Default RGB band indices (0-based) in the S2 image tensor
# Bands 3=Red, 2=Green, 1=Blue for standard Sentinel-2 RGB
DEFAULT_RGB_BANDS: List[int] = [3, 2, 1]


def plot_rgb(image: torch.Tensor, brightness: float = 3.0) -> np.ndarray:
    """
    Convert a Sentinel-2 image tensor to a display-ready RGB array.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor, shape (C, H, W).
    brightness : float
        Multiplicative brightness factor. Default: 3.0.

    Returns
    -------
    np.ndarray
        RGB array, shape (H, W, 3), clipped to [0, 1].
    """
    rgb = image[DEFAULT_RGB_BANDS, :, :].permute(1, 2, 0).numpy()
    return np.clip(brightness * rgb, 0.0, 1.0)


def plot_triplet(
    rgb: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "",
    save_path: Optional[str | Path] = None,
    brightness: float = 2.5,
    dpi: int = 300,
    fmt: str = "png",
) -> None:
    """
    Plot an RGB image alongside its ground truth mask and model prediction.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image array, shape (H, W, 3) or torch.Tensor (C, H, W).
    ground_truth : np.ndarray
        Binary ground truth mask, shape (H, W).
    prediction : np.ndarray
        Binary predicted mask, shape (H, W).
    title : str
        Figure title.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    brightness : float
        Brightness factor for RGB display. Default: 2.5.
    dpi : int
        Figure DPI for saved files. Default: 300.
    fmt : str
        Output format: ``"png"`` or ``"svg"``. Default: ``"png"``.

    Examples
    --------
    >>> from s12coasts.segmentation.visualise import plot_triplet
    >>> plot_triplet(rgb, gt, pred, title="Ontario Lake — patch 001",
    ...              save_path="figures/ontario_001.png")
    """
    if isinstance(rgb, torch.Tensor):
        rgb = np.clip(brightness * rgb[DEFAULT_RGB_BANDS].permute(1, 2, 0).numpy(), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(ground_truth.squeeze(), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(prediction.squeeze(), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("Prediction", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format=fmt)
        logger.info(f"Figure saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_batch(
    batch: dict,
    brightness: float = 3.0,
    cols: int = 4,
    width: int = 5,
    rgb_bands: List[int] = DEFAULT_RGB_BANDS,
) -> None:
    """
    Plot a full batch of images and masks in a grid layout.

    Parameters
    ----------
    batch : dict
        Batch dictionary with keys ``"image"`` and ``"mask"``.
    brightness : float
        Brightness factor for RGB display. Default: 3.0.
    cols : int
        Number of columns in the grid (each column = image + mask). Default: 4.
    width : int
        Width of each subplot in inches. Default: 5.
    rgb_bands : list of int
        Band indices for RGB display. Default: [3, 2, 1].
    """
    samples = unbind_samples(batch.copy())
    n       = 2 * len(samples) if ("image" in batch and "mask" in batch) else len(samples)
    rows    = n // cols + (1 if n % cols != 0 else 0)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if "image" in batch and "mask" in batch:
        for ax, sample in zip(axs.reshape(-1)[::2], samples):
            rgb = torch.clamp(brightness * sample["image"], 0, 1)
            ax.imshow(rgb.permute(1, 2, 0).numpy()[:, :, rgb_bands])
            ax.axis("off")

        for ax, sample in zip(axs.reshape(-1)[1::2], samples):
            ax.imshow(sample["mask"].squeeze().numpy(), cmap="Blues")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def save_inference_figures(
    predicted_masks: list,
    true_masks: list,
    rgb_images: list,
    save_dir: str | Path,
    train_region: str,
    test_region: str,
    brightness: float = 2.5,
    max_figures: int = 30,
    fmt: str = "svg",
    dpi: int = 600,
) -> None:
    """
    Save RGB / ground truth / prediction triplet figures for all inference batches.

    Parameters
    ----------
    predicted_masks : list of np.ndarray
        List of predicted mask arrays (one per batch).
    true_masks : list of np.ndarray
        List of ground truth mask arrays (one per batch).
    rgb_images : list of torch.Tensor
        List of RGB image tensors (one per batch).
    save_dir : str or Path
        Directory where figures are saved.
    train_region : str
        Name of the training region (used in filenames).
    test_region : str
        Name of the test region (used in filenames).
    brightness : float
        Brightness factor for RGB display. Default: 2.5.
    max_figures : int
        Maximum number of batches to save. Default: 30.
    fmt : str
        Output format: ``"svg"`` or ``"png"``. Default: ``"svg"``.
    dpi : int
        Figure DPI. Default: 600.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    counter = 1
    for batch_idx, (preds, gts, rgbs) in enumerate(
        zip(predicted_masks, true_masks, rgb_images)
    ):
        if batch_idx >= max_figures:
            break

        for pred, gt, rgb in zip(preds, gts, rgbs):
            rgb_display = np.clip(
                brightness * np.transpose(rgb.numpy(), (1, 2, 0)),
                0.0,
                1.0,
            )
            save_path = save_dir / (
                f"{test_region}_{train_region}_{counter}.{fmt}"
            )
            plot_triplet(
                rgb=rgb_display,
                ground_truth=gt[0] if gt.ndim == 3 else gt,
                prediction=pred,
                title=f"Train: {train_region} | Test: {test_region} | #{counter}",
                save_path=save_path,
                dpi=dpi,
                fmt=fmt,
            )
            counter += 1

    logger.info(
        f"Saved {counter - 1} inference figure(s) → {save_dir}"
    )
