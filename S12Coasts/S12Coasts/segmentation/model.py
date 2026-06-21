"""
s12coasts.segmentation.model
==============================
SegFormer baseline model initialisation and metric computation.

Model
-----
SegFormer with MiT-B1 backbone (~14M parameters) — selected to balance
efficiency and segmentation accuracy for binary water body segmentation.
Reference: Xie et al. (2021), SegFormer: Simple and Efficient Design for
           Semantic Segmentation with Transformers.

Metrics
-------
OA        : Overall Accuracy
IoU       : Intersection over Union  (Jaccard Score)
F-score   : Harmonic mean of Precision and Recall
Precision : Positive predictive value
Recall    : True positive rate (Sensitivity)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from transformers import SegformerForSemanticSegmentation

logger = logging.getLogger(__name__)

# Default SegFormer backbone — MiT-B1 (~14M parameters)
DEFAULT_BACKBONE: str = "nvidia/mit-b1"

# Binary segmentation: water (1) / non-water (0)
ID2LABEL: dict = {0: "non-water", 1: "water"}
LABEL2ID: dict = {"non-water": 0, "water": 1}


def load_segformer(
    backbone: str = DEFAULT_BACKBONE,
    num_labels: int = 2,
    pretrained: bool = True,
) -> SegformerForSemanticSegmentation:
    """
    Load a SegFormer model for binary water body segmentation.

    Parameters
    ----------
    backbone : str
        HuggingFace model ID for the MiT encoder backbone.
        Options: ``"nvidia/mit-b0"`` through ``"nvidia/mit-b5"``.
        Default: ``"nvidia/mit-b1"`` (~14M parameters).
    num_labels : int
        Number of segmentation classes. Default: 2 (water / non-water).
    pretrained : bool
        Whether to load ImageNet-pretrained encoder weights. Default: True.

    Returns
    -------
    SegformerForSemanticSegmentation
        Initialised model ready for fine-tuning.

    Examples
    --------
    >>> from s12coasts.segmentation.model import load_segformer
    >>> model = load_segformer(backbone="nvidia/mit-b1")
    >>> print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    """
    logger.info(f"Loading SegFormer backbone: {backbone}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"SegFormer loaded — {n_params:.1f}M parameters.")
    return model


def compute_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute five segmentation evaluation metrics for binary water/non-water
    classification.

    Parameters
    ----------
    pred_labels : np.ndarray
        Predicted label array (any shape — flattened internally).
    true_labels : np.ndarray
        Ground-truth label array (same shape as pred_labels).

    Returns
    -------
    dict with keys: ``OA``, ``IoU``, ``F_score``, ``Precision``, ``Recall``.

    Notes
    -----
    All metrics target the water class (label = 1).
    Zero-division is handled by returning 0.0.

    Examples
    --------
    >>> from s12coasts.segmentation.model import compute_metrics
    >>> metrics = compute_metrics(pred, true)
    >>> print(f"OA={metrics['OA']:.4f}  IoU={metrics['IoU']:.4f}")
    """
    pred_flat = pred_labels.flatten()
    true_flat = true_labels.flatten()

    return {
        "OA":        accuracy_score(true_flat, pred_flat),
        "IoU":       jaccard_score(true_flat, pred_flat, zero_division=0),
        "F_score":   f1_score(true_flat, pred_flat, zero_division=0),
        "Precision": precision_score(true_flat, pred_flat, zero_division=0),
        "Recall":    recall_score(true_flat, pred_flat, zero_division=0),
    }


def filter_invalid_samples(transformed_batch: dict) -> list[int]:
    """
    Identify valid sample indices — excluding batches with NaN image values
    or masks containing values other than 0 and 1.

    Parameters
    ----------
    transformed_batch : dict
        Batch dictionary with keys ``"image"`` and ``"mask"``.

    Returns
    -------
    list of int
        Indices of valid samples within the batch.
    """
    n = transformed_batch["image"].shape[0]
    all_ids = set(range(n))

    # Find samples with NaN in image
    nan_ids = set(
        torch.unique(
            torch.where(torch.isnan(transformed_batch["image"]))[0]
        ).tolist()
    )

    # Find samples with invalid mask values (not in {0, 1})
    invalid_mask_ids = set()
    for i, mask in enumerate(transformed_batch["mask"]):
        unique_vals = torch.unique(mask)
        if (unique_vals[(unique_vals != 0) & (unique_vals != 1)]).numel() > 0:
            invalid_mask_ids.add(i)

    valid_ids = sorted(all_ids - nan_ids - invalid_mask_ids)

    if not valid_ids:
        logger.debug("All samples in batch are invalid (NaN or bad mask).")

    return valid_ids
