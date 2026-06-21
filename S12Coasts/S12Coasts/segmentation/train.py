"""
s12coasts.segmentation.train
==============================
SegFormer training loop for the S12-Coasts baseline evaluation.

Training follows a standard supervised fine-tuning procedure:
    - AdamW optimiser with cosine annealing learning rate schedule
    - Cross-entropy loss on upsampled logits
    - Per-epoch validation with OA, IoU, F-score, Precision, Recall
    - Model checkpoint saved at best validation IoU

Usage
-----
    from s12coasts.segmentation.train import train_region
    from s12coasts.segmentation.model import load_segformer

    model = load_segformer()
    history = train_region(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        region_name="Ontario_Lake",
        save_dir="outputs/models/",
        n_epochs=50,
        lr=6e-5,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from s12coasts.segmentation.model import compute_metrics, filter_invalid_samples

logger = logging.getLogger(__name__)


def train_region(
    model: SegformerForSemanticSegmentation,
    train_loader: DataLoader,
    val_loader: DataLoader,
    region_name: str,
    save_dir: str | Path = "outputs/models/",
    n_epochs: int = 50,
    lr: float = 6e-5,
    device: torch.device | None = None,
    num_bands: int = 20,
) -> dict:
    """
    Fine-tune a SegFormer model on a single S12-Coasts region.

    Parameters
    ----------
    model : SegformerForSemanticSegmentation
        Initialised SegFormer model (from ``load_segformer()``).
    train_loader : DataLoader
        Training DataLoader from ``S12CoastsDataset``.
    val_loader : DataLoader
        Validation DataLoader from ``S12CoastsDataset``.
    region_name : str
        Name of the training region — used for model checkpoint filename.
    save_dir : str or Path
        Directory to save the best model checkpoint. Default: ``outputs/models/``.
    n_epochs : int
        Number of training epochs. Default: 50.
    lr : float
        AdamW learning rate. Default: 6e-5.
    device : torch.device, optional
        Compute device. Defaults to CUDA if available, otherwise CPU.
    num_bands : int
        Number of input image bands to use (S2 + S1). Default: 20.

    Returns
    -------
    dict
        Training history with keys:
        ``train_loss``, ``val_OA``, ``val_IoU``, ``val_F_score``,
        ``val_Precision``, ``val_Recall``, ``best_epoch``, ``best_iou``.

    Examples
    --------
    >>> from s12coasts.segmentation.train import train_region
    >>> from s12coasts.segmentation.model import load_segformer
    >>> model = load_segformer()
    >>> history = train_region(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     region_name="Ontario_Lake",
    ...     n_epochs=50,
    ... )
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{region_name}.pth"

    model.to(device)
    optimiser = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs)

    history: dict = {
        "train_loss": [],
        "val_OA":        [],
        "val_IoU":       [],
        "val_F_score":   [],
        "val_Precision": [],
        "val_Recall":    [],
        "best_epoch":    0,
        "best_iou":      0.0,
    }

    logger.info(
        f"Training on region: '{region_name}' | "
        f"device: {device} | epochs: {n_epochs} | lr: {lr}"
    )

    for epoch in range(n_epochs):
        # ── Training ──────────────────────────────────────────────────
        model.train()
        epoch_losses: list[float] = []

        for batch in tqdm(
            train_loader,
            desc=f"[{region_name}] Epoch {epoch + 1}/{n_epochs} — train",
            leave=False,
        ):
            valid_ids = filter_invalid_samples(batch)
            if not valid_ids:
                continue

            pixel_values = (
                batch["image"][valid_ids, :num_bands, :, :].to(device)
            )
            labels = batch["mask"][valid_ids, 0, :, :].to(device).long()

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history["train_loss"].append(mean_loss)

        # ── Validation ────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device=device, num_bands=num_bands)
        history["val_OA"].append(val_metrics["OA"])
        history["val_IoU"].append(val_metrics["IoU"])
        history["val_F_score"].append(val_metrics["F_score"])
        history["val_Precision"].append(val_metrics["Precision"])
        history["val_Recall"].append(val_metrics["Recall"])

        logger.info(
            f"[{region_name}] Epoch {epoch + 1}/{n_epochs} — "
            f"loss: {mean_loss:.4f} | "
            f"OA: {val_metrics['OA']:.4f} | "
            f"IoU: {val_metrics['IoU']:.4f} | "
            f"F-score: {val_metrics['F_score']:.4f}"
        )

        # Save best checkpoint
        if val_metrics["IoU"] > history["best_iou"]:
            history["best_iou"]   = val_metrics["IoU"]
            history["best_epoch"] = epoch + 1
            torch.save(model.state_dict(), model_path)
            logger.info(
                f"  ✓ New best IoU: {val_metrics['IoU']:.4f} "
                f"— checkpoint saved → {model_path}"
            )

    logger.info(
        f"Training complete — best IoU: {history['best_iou']:.4f} "
        f"at epoch {history['best_epoch']}."
    )
    return history


def evaluate(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    device: torch.device | None = None,
    num_bands: int = 20,
) -> dict[str, float]:
    """
    Evaluate a SegFormer model on a DataLoader and return mean metrics.

    Parameters
    ----------
    model : SegformerForSemanticSegmentation
        Trained SegFormer model.
    loader : DataLoader
        DataLoader to evaluate on.
    device : torch.device, optional
        Compute device (defaults to CUDA if available).
    num_bands : int
        Number of input image bands. Default: 20.

    Returns
    -------
    dict
        Mean metrics over all batches:
        ``OA``, ``IoU``, ``F_score``, ``Precision``, ``Recall``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_metrics: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            valid_ids = filter_invalid_samples(batch)
            if not valid_ids:
                continue

            pixel_values = batch["image"][valid_ids, :num_bands, :, :].to(device)
            true_masks   = batch["mask"][valid_ids, 0, :, :].to(device).long()

            outputs  = model(pixel_values=pixel_values)
            logits   = F.interpolate(
                outputs.logits,
                size=true_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            predicted = logits.argmax(dim=1)

            pred_np = predicted.detach().cpu().numpy()
            true_np = true_masks.detach().cpu().numpy()

            for pred, true in zip(pred_np, true_np):
                all_metrics.append(compute_metrics(pred, true))

    if not all_metrics:
        logger.warning("No valid batches found during evaluation.")
        return {"OA": 0.0, "IoU": 0.0, "F_score": 0.0, "Precision": 0.0, "Recall": 0.0}

    return {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0]
    }
