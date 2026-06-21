"""
s12coasts.segmentation.evaluate
=================================
Leave-One-Region-In Cross-Validation (LORI-CV) evaluation framework.

In LORI-CV, the SegFormer baseline is:
    1. Trained on patches from a single region–source combination.
    2. Evaluated on patches from all remaining region–source combinations.
    3. This procedure is repeated for all 33 region–source combinations.

The resulting performance matrix (33×33) forms the cross-regional heatmap
reported in the S12-Coasts paper (Figure 9).

Usage
-----
    from s12coasts.segmentation.evaluate import LORICVEvaluator

    evaluator = LORICVEvaluator(
        dataset_root="outputs/masks/",
        model_dir="outputs/models/",
        results_dir="outputs/results/",
        regions=["Ontario_Lake", "Baltic_Sea", ...],
    )
    evaluator.run()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from s12coasts.segmentation.dataset import S12CoastsDataset
from s12coasts.segmentation.model import (
    compute_metrics,
    filter_invalid_samples,
    load_segformer,
)
from s12coasts.segmentation.train import train_region, evaluate

logger = logging.getLogger(__name__)


class LORICVEvaluator:
    """
    Leave-One-Region-In Cross-Validation (LORI-CV) evaluator.

    For each region in ``regions``, trains a SegFormer model on that region
    and evaluates it on all other regions. Results are saved as a CSV matrix
    and a summary DataFrame.

    Parameters
    ----------
    dataset_root : str or Path
        Root directory of the S12-Coasts dataset (contains region subdirs).
    model_dir : str or Path
        Directory where trained model checkpoints are saved.
    results_dir : str or Path
        Directory where evaluation results are saved.
    regions : list of str
        List of region names to include in LORI-CV.
        These should match subdirectory names under ``dataset_root``.
    quality_labels : list of str
        Quality labels to include. Default: ``["H"]``.
    n_epochs : int
        Number of training epochs per region. Default: 50.
    lr : float
        Learning rate for AdamW optimiser. Default: 6e-5.
    batch_size : int
        DataLoader batch size. Default: 8.
    patch_size : int
        Patch sampling size in pixels. Default: 512.
    num_bands : int
        Number of input bands (S1 + S2). Default: 20.
    device : torch.device, optional
        Compute device (defaults to CUDA if available).

    Examples
    --------
    >>> from s12coasts.segmentation.evaluate import LORICVEvaluator
    >>> evaluator = LORICVEvaluator(
    ...     dataset_root="outputs/masks/",
    ...     model_dir="outputs/models/",
    ...     results_dir="outputs/results/",
    ...     regions=["Ontario_Lake", "Baltic_Sea", "Great_Lakes"],
    ...     n_epochs=50,
    ... )
    >>> evaluator.run()
    """

    def __init__(
        self,
        dataset_root: str | Path,
        model_dir: str | Path,
        results_dir: str | Path,
        regions: list[str],
        quality_labels: list[str] = None,
        n_epochs: int = 50,
        lr: float = 6e-5,
        batch_size: int = 8,
        patch_size: int = 512,
        num_bands: int = 20,
        device: torch.device | None = None,
    ) -> None:
        self.dataset_root  = Path(dataset_root)
        self.model_dir     = Path(model_dir)
        self.results_dir   = Path(results_dir)
        self.regions       = regions
        self.quality_labels = quality_labels or ["H"]
        self.n_epochs      = n_epochs
        self.lr            = lr
        self.batch_size    = batch_size
        self.patch_size    = patch_size
        self.num_bands     = num_bands
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"LORICVEvaluator initialised — {len(regions)} region(s) · "
            f"device: {self.device}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_dataset(self, region: str) -> S12CoastsDataset:
        """Build an S12CoastsDataset for a single region."""
        return S12CoastsDataset(
            root=self.dataset_root,
            region_name=region,
            quality_labels=self.quality_labels,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
        )

    def _train_on_region(
        self,
        model: SegformerForSemanticSegmentation,
        train_region_name: str,
    ) -> None:
        """Train the model on a single training region."""
        ds = self._build_dataset(train_region_name)
        ds.train_val_split(split_ratio=0.8)
        ds.create_samplers()
        ds.create_dataloaders()
        train_loader, val_loader = ds.get_dataloaders()

        train_region(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            region_name=train_region_name,
            save_dir=self.model_dir,
            n_epochs=self.n_epochs,
            lr=self.lr,
            device=self.device,
            num_bands=self.num_bands,
        )

    def _evaluate_on_region(
        self,
        model: SegformerForSemanticSegmentation,
        test_region_name: str,
    ) -> dict[str, float]:
        """Evaluate the model on a single test region."""
        ds = self._build_dataset(test_region_name)
        inference_loader = ds.get_inference_loader()
        return evaluate(
            model=model,
            loader=inference_loader,
            device=self.device,
            num_bands=self.num_bands,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full LORI-CV evaluation across all regions.

        For each training region:
            1. Initialise a fresh SegFormer model.
            2. Train on the training region.
            3. Load the best checkpoint.
            4. Evaluate on all other regions.
            5. Save per-region results to CSV.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with columns:
            ``Train_Region``, ``Test_Region``, ``OA``, ``IoU``,
            ``F_score``, ``Precision``, ``Recall``.
        """
        all_records: list[dict] = []

        for train_region_name in tqdm(
            self.regions, desc="LORI-CV — Training regions"
        ):
            logger.info(
                f"\n{'='*60}\n"
                f"Training region: {train_region_name}\n"
                f"{'='*60}"
            )

            # Initialise a fresh model for each training region
            model = load_segformer()
            model.to(self.device)

            # Train
            self._train_on_region(model, train_region_name)

            # Load best checkpoint
            ckpt_path = self.model_dir / f"{train_region_name}.pth"
            if ckpt_path.exists():
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=self.device)
                )
                logger.info(f"Loaded best checkpoint: {ckpt_path}")
            else:
                logger.warning(
                    f"No checkpoint found for '{train_region_name}'. "
                    "Using last epoch weights."
                )

            # Evaluate on all regions (including the training region)
            region_records: list[dict] = []
            for test_region_name in tqdm(
                self.regions,
                desc=f"  Evaluating [{train_region_name}]",
                leave=False,
            ):
                logger.info(
                    f"  Test region: {test_region_name}"
                )
                try:
                    metrics = self._evaluate_on_region(model, test_region_name)
                except Exception as exc:
                    logger.error(
                        f"  Failed to evaluate on '{test_region_name}': {exc}"
                    )
                    metrics = {
                        "OA": None, "IoU": None, "F_score": None,
                        "Precision": None, "Recall": None,
                    }

                record = {
                    "Train_Region": train_region_name,
                    "Test_Region":  test_region_name,
                    **metrics,
                }
                region_records.append(record)
                all_records.append(record)

                logger.info(
                    f"    OA={metrics.get('OA', 'N/A'):.4f} | "
                    f"IoU={metrics.get('IoU', 'N/A'):.4f} | "
                    f"F-score={metrics.get('F_score', 'N/A'):.4f}"
                    if metrics.get("OA") is not None else
                    f"    [{test_region_name}] — evaluation failed."
                )

            # Save per-training-region results
            region_csv = self.results_dir / f"{train_region_name}.csv"
            pd.DataFrame(region_records).to_csv(region_csv, index=False)
            logger.info(f"  Results saved → {region_csv}")

        # Compile full summary
        summary_df = pd.DataFrame(all_records)
        summary_path = self.results_dir / "lori_cv_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(
            f"\nLORI-CV complete — full summary saved → {summary_path}\n"
            f"  {len(self.regions)} training regions · "
            f"{len(self.regions)} test regions · "
            f"{len(summary_df)} total evaluations."
        )
        return summary_df

    def load_summary(self) -> pd.DataFrame:
        """
        Load a previously saved LORI-CV summary CSV.

        Returns
        -------
        pd.DataFrame
        """
        summary_path = self.results_dir / "lori_cv_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No LORI-CV summary found at {summary_path}.\n"
                "Run evaluator.run() first."
            )
        return pd.read_csv(summary_path)

    def pivot_metric(self, metric: str = "OA") -> pd.DataFrame:
        """
        Pivot the LORI-CV summary into a train × test matrix for a given metric.

        Parameters
        ----------
        metric : str
            Metric to pivot. One of ``"OA"``, ``"IoU"``, ``"F_score"``,
            ``"Precision"``, ``"Recall"``. Default: ``"OA"``.

        Returns
        -------
        pd.DataFrame
            Matrix of shape (n_regions, n_regions) with training regions as
            rows and test regions as columns.

        Examples
        --------
        >>> matrix = evaluator.pivot_metric("IoU")
        >>> matrix.to_csv("results/iou_matrix.csv")
        """
        df = self.load_summary()
        return df.pivot(index="Train_Region", columns="Test_Region", values=metric)
