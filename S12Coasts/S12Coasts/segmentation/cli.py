"""
s12coasts.segmentation.cli
============================
Command-line interface entry point for Stage 3 — SegFormer Baseline Training
and LORI-CV Evaluation.

Usage
-----
    # Train on a single region
    s12coasts-segment --mode train \\
        --dataset_root outputs/masks/ \\
        --region Ontario_Lake \\
        --model_dir outputs/models/ \\
        --n_epochs 50

    # Run full LORI-CV evaluation across all regions
    s12coasts-segment --mode lori_cv \\
        --dataset_root outputs/masks/ \\
        --model_dir outputs/models/ \\
        --results_dir outputs/results/ \\
        --regions Ontario_Lake Baltic_Sea Great_Lakes

    # Evaluate a trained model on a specific test region
    s12coasts-segment --mode evaluate \\
        --dataset_root outputs/masks/ \\
        --model_dir outputs/models/ \\
        --train_region Ontario_Lake \\
        --test_region Baltic_Sea
"""

from __future__ import annotations

import argparse
import logging

import torch

from s12coasts.segmentation.model import load_segformer
from s12coasts.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("s12coasts.segmentation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="s12coasts-segment",
        description=(
            "S12-Coasts Stage 3: SegFormer Baseline Training and LORI-CV Evaluation.\n\n"
            "Modes:\n"
            "  train     — fine-tune SegFormer on a single region\n"
            "  lori_cv   — run full LORI-CV across all regions\n"
            "  evaluate  — evaluate a trained model on a test region\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "lori_cv", "evaluate"],
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset_root", type=str, default="outputs/masks/", metavar="PATH",
        help="Root directory of the S12-Coasts dataset.",
    )
    parser.add_argument(
        "--model_dir", type=str, default="outputs/models/", metavar="PATH",
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--results_dir", type=str, default="outputs/results/", metavar="PATH",
        help="Directory for evaluation results (LORI-CV mode).",
    )
    parser.add_argument(
        "--figures_dir", type=str, default="outputs/figures/", metavar="PATH",
        help="Directory for saved inference figures.",
    )
    parser.add_argument(
        "--region", type=str, default=None, metavar="NAME",
        help="Training region name (train mode).",
    )
    parser.add_argument(
        "--train_region", type=str, default=None, metavar="NAME",
        help="Training region name (evaluate mode).",
    )
    parser.add_argument(
        "--test_region", type=str, default=None, metavar="NAME",
        help="Test region name (evaluate mode).",
    )
    parser.add_argument(
        "--regions", type=str, nargs="+", default=None, metavar="NAME",
        help="List of region names for LORI-CV (space-separated).",
    )
    parser.add_argument(
        "--quality_labels", type=str, nargs="+", default=["H"],
        help="Quality labels to include. Default: H",
    )
    parser.add_argument(
        "--backbone", type=str, default="nvidia/mit-b1",
        help="SegFormer backbone (default: nvidia/mit-b1).",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--lr", type=float, default=6e-5,
        help="AdamW learning rate (default: 6e-5).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="DataLoader batch size (default: 8).",
    )
    parser.add_argument(
        "--num_bands", type=int, default=20,
        help="Number of input image bands (default: 20).",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.7,
        help="Train/val split ratio (default: 0.7).",
    )
    parser.add_argument(
        "--save_figures", action="store_true",
        help="Save inference figures during evaluation.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Mode: train ───────────────────────────────────────────────────────
    if args.mode == "train":
        if not args.region:
            raise ValueError("--region is required for train mode.")

        from s12coasts.segmentation.dataset import S12CoastsDataset
        from s12coasts.segmentation.train import train_region

        logger.info(
            f"Training on region: '{args.region}' | "
            f"backbone: {args.backbone} | epochs: {args.n_epochs}"
        )

        ds = S12CoastsDataset(
            root=args.dataset_root,
            region_name=args.region,
            quality_labels=args.quality_labels,
            batch_size=args.batch_size,
        )
        ds.train_val_split(split_ratio=args.split_ratio)
        ds.create_samplers()
        ds.create_dataloaders()
        train_loader, val_loader = ds.get_dataloaders()

        model = load_segformer(backbone=args.backbone)
        history = train_region(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            region_name=args.region,
            save_dir=args.model_dir,
            n_epochs=args.n_epochs,
            lr=args.lr,
            device=device,
            num_bands=args.num_bands,
        )
        logger.info(
            f"Training complete — best IoU: {history['best_iou']:.4f} "
            f"at epoch {history['best_epoch']}."
        )

    # ── Mode: lori_cv ─────────────────────────────────────────────────────
    elif args.mode == "lori_cv":
        if not args.regions:
            raise ValueError(
                "--regions is required for lori_cv mode.\n"
                "Example: s12coasts-segment --mode lori_cv "
                "--regions Ontario_Lake Baltic_Sea Great_Lakes"
            )

        from s12coasts.segmentation.evaluate import LORICVEvaluator

        logger.info(
            f"LORI-CV evaluation | {len(args.regions)} region(s) | "
            f"epochs: {args.n_epochs}"
        )

        evaluator = LORICVEvaluator(
            dataset_root=args.dataset_root,
            model_dir=args.model_dir,
            results_dir=args.results_dir,
            regions=args.regions,
            quality_labels=args.quality_labels,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_bands=args.num_bands,
            device=device,
        )
        summary_df = evaluator.run()

        logger.info(
            f"\nLORI-CV Summary (mean across all test regions):\n"
            f"  OA        : {summary_df['OA'].mean():.4f}\n"
            f"  IoU       : {summary_df['IoU'].mean():.4f}\n"
            f"  F-score   : {summary_df['F_score'].mean():.4f}\n"
            f"  Precision : {summary_df['Precision'].mean():.4f}\n"
            f"  Recall    : {summary_df['Recall'].mean():.4f}"
        )

    # ── Mode: evaluate ────────────────────────────────────────────────────
    elif args.mode == "evaluate":
        if not args.train_region or not args.test_region:
            raise ValueError(
                "--train_region and --test_region are required for evaluate mode."
            )

        import torch
        from s12coasts.segmentation.dataset import S12CoastsDataset
        from s12coasts.segmentation.train import evaluate

        logger.info(
            f"Evaluating: train='{args.train_region}' → "
            f"test='{args.test_region}'"
        )

        # Load trained model
        ckpt_path = f"{args.model_dir}/{args.train_region}.pth"
        model = load_segformer(backbone=args.backbone)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )
        model.to(device)
        logger.info(f"Loaded checkpoint: {ckpt_path}")

        # Build test dataset
        test_ds = S12CoastsDataset(
            root=args.dataset_root,
            region_name=args.test_region,
            quality_labels=args.quality_labels,
            batch_size=args.batch_size,
        )
        test_loader = test_ds.get_inference_loader()

        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            num_bands=args.num_bands,
        )

        logger.info(
            f"\nResults — train: '{args.train_region}' | "
            f"test: '{args.test_region}'\n"
            f"  OA        : {metrics['OA']:.4f}\n"
            f"  IoU       : {metrics['IoU']:.4f}\n"
            f"  F-score   : {metrics['F_score']:.4f}\n"
            f"  Precision : {metrics['Precision']:.4f}\n"
            f"  Recall    : {metrics['Recall']:.4f}"
        )


if __name__ == "__main__":
    main()
