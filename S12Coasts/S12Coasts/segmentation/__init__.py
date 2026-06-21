"""
s12coasts.segmentation
=======================
Stage 3 — SegFormer baseline model training and LORI-CV evaluation.

Components
----------
dataset    : S12CoastsDataset — georeferenced DataLoader pipeline
model      : load_segformer(), compute_metrics()
train      : train_region(), evaluate()
evaluate   : LORICVEvaluator — full LORI-CV pipeline
visualise  : plot_triplet(), plot_batch(), save_inference_figures()
cli        : s12coasts-segment entry point
"""

from s12coasts.segmentation.dataset import S12CoastsDataset
from s12coasts.segmentation.model import (
    load_segformer,
    compute_metrics,
    filter_invalid_samples,
    DEFAULT_BACKBONE,
    ID2LABEL,
    LABEL2ID,
)
from s12coasts.segmentation.train import train_region, evaluate
from s12coasts.segmentation.evaluate import LORICVEvaluator
from s12coasts.segmentation.visualise import (
    plot_triplet,
    plot_batch,
    save_inference_figures,
)

__all__ = [
    # Dataset
    "S12CoastsDataset",
    # Model
    "load_segformer",
    "compute_metrics",
    "filter_invalid_samples",
    "DEFAULT_BACKBONE",
    "ID2LABEL",
    "LABEL2ID",
    # Training
    "train_region",
    "evaluate",
    # LORI-CV
    "LORICVEvaluator",
    # Visualisation
    "plot_triplet",
    "plot_batch",
    "save_inference_figures",
]
