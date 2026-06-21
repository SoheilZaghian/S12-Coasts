"""
s12coasts.segmentation.dataset
================================
Dataset loading and DataLoader creation for the S12-Coasts SegFormer baseline.

Loads paired Sentinel-1/Sentinel-2 image patches and binary water masks from
the S12-Coasts dataset structure (S1/ S2/ gt/ subdirectories) using TorchGeo
RasterDataset for georeferenced patch sampling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units

logger = logging.getLogger(__name__)


class S12CoastsDataset:
    """
    Dataset pipeline for the S12-Coasts benchmark.

    Loads paired S1/S2/gt GeoTIFF patches for a single region, filters by
    quality label, and creates train/validation DataLoaders.

    Parameters
    ----------
    root : str or Path
        Root directory of the S12-Coasts dataset.
        Expected structure:
            root/
                data_information.csv
                <region_name>/
                    S1/  S1_00001.tif ...
                    S2/  S2_00001.tif ...
                    gt/  gt_00001.tif ...
    region_name : str
        Name of the region to load (must match a subdirectory under root).
    quality_labels : list of str
        Quality labels to include. Default: ``["H"]``.
        Use ``["H", "B1"]`` to include Boundary Error Level 1 patches.
    patch_size : int
        Patch size in pixels for georeferenced sampling. Default: 512.
    batch_size : int
        DataLoader batch size. Default: 8.
    crs : str
        Coordinate reference system for the dataset. Default: ``"EPSG:3395"``.
    res : int
        Spatial resolution in metres. Default: 10.
    metadata_filename : str
        Name of the metadata CSV file. Default: ``"data_information.csv"``.

    Examples
    --------
    >>> from s12coasts.segmentation.dataset import S12CoastsDataset
    >>> ds = S12CoastsDataset(
    ...     root="outputs/masks/",
    ...     region_name="Ontario_Lake",
    ...     quality_labels=["H"],
    ...     batch_size=8,
    ... )
    >>> ds.train_val_split(split_ratio=0.7)
    >>> ds.create_samplers()
    >>> ds.create_dataloaders()
    >>> train_loader, val_loader = ds.get_dataloaders()
    """

    def __init__(
        self,
        root: str | Path,
        region_name: str,
        quality_labels: list[str] = None,
        patch_size: int = 512,
        batch_size: int = 8,
        crs: str = "EPSG:3395",
        res: int = 10,
        metadata_filename: str = "data_information.csv",
    ) -> None:
        self.root         = Path(root)
        self.region_name  = region_name
        self.quality_labels = quality_labels or ["H"]
        self.patch_size   = patch_size
        self.batch_size   = batch_size
        self.crs          = crs
        self.res          = res

        # Load metadata and filter by region + quality
        self._metadata      = self._load_metadata(metadata_filename)
        self._patch_ids     = self._get_patch_ids()

        logger.info(
            f"S12CoastsDataset: region='{region_name}' — "
            f"{len(self._patch_ids)} patch(es) with quality {self.quality_labels}."
        )

        # Build GeoTIFF path lists
        self._s2_paths = self._build_paths("S2")
        self._s1_paths = self._build_paths("S1")
        self._gt_paths = self._build_paths("gt")

        # Create TorchGeo RasterDatasets
        self._s2_ds = self._make_raster_dataset(self._s2_paths, is_image=True)
        self._s1_ds = self._make_raster_dataset(self._s1_paths, is_image=True)
        self._gt_ds = self._make_raster_dataset(self._gt_paths, is_image=False)

        # Intersect all three datasets geographically
        self.full_dataset = self._s2_ds & self._s1_ds & self._gt_ds

        # Initialised by train_val_split / create_samplers / create_dataloaders
        self.train_size: int  = 0
        self.val_size: int    = 0
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader   | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metadata(self, filename: str) -> pd.DataFrame:
        """Load patch metadata CSV from the dataset root."""
        csv_path = self.root / filename
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {csv_path}\n"
                "Run the EOBC pipeline first to generate this file."
            )
        return pd.read_csv(csv_path)

    def _get_patch_ids(self) -> list[int]:
        """Filter metadata by region name and quality label."""
        region_rows = self._metadata[
            self._metadata["region_name"] == self.region_name
        ]
        quality_rows = region_rows[
            region_rows["quality_label"].isin(self.quality_labels)
        ]
        ids = quality_rows["id"].tolist()
        if not ids:
            logger.warning(
                f"No patches found for region='{self.region_name}' "
                f"with quality {self.quality_labels}. "
                "Check region_name and quality_labels in your config."
            )
        return ids

    def _build_paths(self, image_type: str) -> list[str]:
        """Build sorted file path list for a given image type (S1/S2/gt)."""
        region_dir = self.root / self.region_name / image_type
        return sorted([
            str(region_dir / f"{image_type}_{str(pid).zfill(5)}.tif")
            for pid in self._patch_ids
        ])

    @staticmethod
    def _scale_image(item: dict) -> dict:
        """Min-max normalise image bands to [0, 1]."""
        img     = item["image"]
        min_val = img.min()
        max_val = img.max()
        denom   = (max_val - min_val).clamp(min=1e-8)
        item["image"] = (img - min_val) / denom
        return item

    @staticmethod
    def _scale_mask(item: dict) -> dict:
        """Normalise mask to binary integer {0, 1}."""
        mask    = item["mask"]
        min_val = mask.min()
        max_val = mask.max()
        denom   = (max_val - min_val).clamp(min=1e-8)
        item["mask"] = ((mask - min_val) / denom).to(torch.int64)
        return item

    def _make_raster_dataset(
        self,
        paths: list[str],
        is_image: bool = True,
    ) -> RasterDataset:
        """Create a TorchGeo RasterDataset from a list of file paths."""
        transform = self._scale_image if is_image else self._scale_mask
        ds = RasterDataset(
            paths=paths,
            transforms=transform,
            crs=self.crs,
            res=self.res,
        )
        ds.is_image = is_image
        return ds

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_val_split(self, split_ratio: float = 0.7) -> None:
        """
        Compute train/validation split sizes.

        Parameters
        ----------
        split_ratio : float
            Fraction of patches to use for training. Default: 0.7.
        """
        n = len(self.full_dataset)
        self.train_size = int(split_ratio * n)
        self.val_size   = n - self.train_size
        logger.info(
            f"Split: {self.train_size} train / {self.val_size} val "
            f"(ratio={split_ratio})."
        )

    def create_samplers(self) -> None:
        """Create georeferenced random samplers for training and validation."""
        self._train_sampler = RandomGeoSampler(
            self.full_dataset,
            size=self.patch_size,
            length=self.train_size,
            units=Units.PIXELS,
        )
        self._val_sampler = RandomGeoSampler(
            self.full_dataset,
            size=self.patch_size,
            length=self.val_size,
            units=Units.PIXELS,
        )

    def create_dataloaders(self, num_workers: int = 0) -> None:
        """
        Create train and validation DataLoaders.

        Parameters
        ----------
        num_workers : int
            Number of DataLoader worker processes. Default: 0 (main process).
        """
        self.train_loader = DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            sampler=self._train_sampler,
            collate_fn=stack_samples,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            sampler=self._val_sampler,
            collate_fn=stack_samples,
            num_workers=num_workers,
        )

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Return the train and validation DataLoaders.

        Returns
        -------
        tuple of (DataLoader, DataLoader)

        Raises
        ------
        RuntimeError
            If ``create_dataloaders()`` has not been called yet.
        """
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError(
                "DataLoaders not created yet. "
                "Call train_val_split() → create_samplers() → "
                "create_dataloaders() first."
            )
        return self.train_loader, self.val_loader

    def get_inference_loader(self, num_workers: int = 0) -> DataLoader:
        """
        Create and return a DataLoader over the full dataset (no split).

        Used for LORI-CV inference on test regions.

        Parameters
        ----------
        num_workers : int
            Number of DataLoader worker processes.

        Returns
        -------
        DataLoader
        """
        n = len(self.full_dataset)
        sampler = RandomGeoSampler(
            self.full_dataset,
            size=self.patch_size,
            length=n,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=stack_samples,
            num_workers=num_workers,
        )
