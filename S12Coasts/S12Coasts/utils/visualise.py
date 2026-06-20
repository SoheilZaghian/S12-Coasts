"""
s12coasts.utils.visualise
=========================
Patch visualisation utilities for quality control and debugging.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Band indices (0-based) in the stacked GeoTIFF
_BAND_MNDWI = 13
_BAND_VV = 18
_BAND_BOUNDARY = 20


def visualise_patch(
    patch_index: int,
    image_array: np.ndarray,
    label_image: np.ndarray,
    masks: list[np.ndarray],
    algorithm_names: list[str],
    crop_window: tuple[int, int, int, int],
    save_path: str | None = None,
) -> None:
    """
    Display a multi-panel visualisation of a single patch showing the S2 RGB
    composite, Sentinel-1 VV band, water boundary, connected component label
    image, per-algorithm clustering masks, and the final majority-vote mask.

    Parameters
    ----------
    patch_index : int
        Zero-based patch index (used in the figure title).
    image_array : np.ndarray
        Full stacked image array, shape (n_bands, H, W).
    label_image : np.ndarray
        Connected component label image, shape (H, W).
    masks : list of np.ndarray
        Per-algorithm binary masks plus the final majority-vote mask.
    algorithm_names : list of str
        Names of the clustering algorithms (for subplot titles).
    crop_window : tuple of int
        (x_start, x_end, y_start, y_end) pixel coordinates of the centre crop.
    save_path : str, optional
        If provided, saves the figure to this path instead of displaying it.

    Examples
    --------
    >>> from s12coasts.utils.visualise import visualise_patch
    >>> visualise_patch(
    ...     patch_index=0,
    ...     image_array=image_array,
    ...     label_image=label_image,
    ...     masks=water_masks,
    ...     algorithm_names=["KMeans", "AgglomerativeClustering", ...],
    ...     crop_window=(0, 512, 0, 512),
    ... )
    """
    x0, x1, y0, y1 = crop_window

    titles = (
        ["RGB", "VV", "Water Boundary", "Label Image"]
        + list(algorithm_names)
        + ["Majority Vote"]
    )

    # S2 RGB composite — bands 3 (Red), 2 (Green), 1 (Blue), scaled to [0, 1]
    rgb = np.clip(
        image_array[[3, 2, 1], x0:x1, y0:y1].transpose(1, 2, 0) * 5.0 / 10000,
        0.0,
        1.0,
    )

    images = (
        [
            rgb,
            image_array[_BAND_VV, x0:x1, y0:y1],
            image_array[_BAND_BOUNDARY, x0:x1, y0:y1],
            label_image[x0:x1, y0:y1],
        ]
        + [m[x0:x1, y0:y1] for m in masks]
    )
    cmaps = ["viridis", "gray", "gray", "jet"] + ["Blues"] * len(masks)

    n_panels = len(images)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.suptitle(f"Patch {patch_index + 1}", fontsize=14, fontweight="bold")

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Patch visualisation saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)
