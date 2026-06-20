"""
s12coasts.io.geotiff
=====================
GeoTIFF export utilities — crops selected bands from stacked input GeoTIFFs
and writes paired S1/S2/gt outputs in the S12-Coasts dataset structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from osgeo import gdal, gdalconst

logger = logging.getLogger(__name__)


class GeoTIFFExporter:
    """
    Crops and exports GeoTIFF patches in the S12-Coasts dataset structure.

    Creates three subdirectories under ``output_dir``:
        - ``S1/``  — Sentinel-1 SAR bands (VV, VH)
        - ``S2/``  — Sentinel-2 multispectral bands + spectral indices
        - ``gt/``  — Binary water mask (ground truth)

    Parameters
    ----------
    input_file : str or Path
        Path to the source stacked GeoTIFF.
    output_dir : str or Path
        Root output directory.
    crop_window : tuple of int
        ``(x_start, x_end, y_start, y_end)`` pixel coordinates
        defining the centre-crop window.

    Examples
    --------
    >>> from s12coasts.io.geotiff import GeoTIFFExporter
    >>> exporter = GeoTIFFExporter(
    ...     input_file="data/patches/patch_001.tif",
    ...     output_dir="outputs/masks/Huron_Lake",
    ...     crop_window=(0, 512, 0, 512),
    ... )
    >>> exporter.export_bands("S2/S2_00001.tif", band_indices=list(range(1, 19)))
    >>> exporter.export_bands("S1/S1_00001.tif", band_indices=[19, 20])
    >>> exporter.export_mask("gt/gt_00001.tif", mask_array=water_mask)
    """

    def __init__(
        self,
        input_file: str | Path,
        output_dir: str | Path,
        crop_window: tuple[int, int, int, int],
    ) -> None:
        self.input_file = str(input_file)
        self.output_dir = Path(output_dir)
        self.x0, self.x1, self.y0, self.y1 = crop_window
        self.width  = self.x1 - self.x0
        self.height = self.y1 - self.y0

        # Create output subdirectories
        for sub in ("S1", "S2", "gt"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _open_source(self) -> gdal.Dataset:
        ds = gdal.Open(self.input_file, gdalconst.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"Cannot open source file: {self.input_file}")
        return ds

    def _adjusted_geotransform(self, gt: tuple) -> tuple:
        """Adjust a geotransform to the crop window origin."""
        return (
            gt[0] + self.x0 * gt[1] + self.y0 * gt[2],
            gt[1], gt[2],
            gt[3] + self.x0 * gt[4] + self.y0 * gt[5],
            gt[4], gt[5],
        )

    # ------------------------------------------------------------------

    def export_bands(
        self,
        output_path: str | Path,
        band_indices: list[int],
        dtype: int = gdalconst.GDT_UInt32,
        scale: float = 1.0,
    ) -> None:
        """
        Crop selected bands from the source raster and save as a GeoTIFF.

        Parameters
        ----------
        output_path : str or Path
            Output path relative to ``output_dir``.
        band_indices : list of int
            1-based GDAL band indices to export.
        dtype : int
            GDAL data type for the output file (default: UInt32).
        scale : float
            Multiplicative scale factor applied before writing (default: 1.0).
            Use ``scale=10000.0`` for Sentinel-2 SR → DN conversion.

        Raises
        ------
        ValueError
            If any band index is out of range.
        FileNotFoundError
            If the source file cannot be opened.
        """
        out_path = self.output_dir / output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        src = self._open_source()
        n_bands = src.RasterCount

        if max(band_indices) > n_bands or min(band_indices) < 1:
            raise ValueError(
                f"Band indices {band_indices} out of range "
                f"(source has {n_bands} band(s))."
            )

        new_gt = self._adjusted_geotransform(src.GetGeoTransform())
        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(
            str(out_path), self.width, self.height, len(band_indices), dtype
        )
        dst.SetGeoTransform(new_gt)
        dst.SetProjection(src.GetProjection())

        for out_idx, src_idx in enumerate(band_indices, start=1):
            data = (
                src.GetRasterBand(src_idx)
                   .ReadAsArray(self.x0, self.y0, self.width, self.height)
                   .astype(np.float32)
                * scale
            )
            dst.GetRasterBand(out_idx).WriteArray(data)

        dst.FlushCache()
        dst = src = None
        logger.debug(f"Exported {len(band_indices)} band(s) → {out_path}")

    def export_mask(
        self,
        output_path: str | Path,
        mask_array: np.ndarray,
        dtype: int = gdalconst.GDT_Byte,
    ) -> None:
        """
        Save a 2-D binary NumPy array as a single-band GeoTIFF water mask.

        Parameters
        ----------
        output_path : str or Path
            Output path relative to ``output_dir``.
        mask_array : np.ndarray
            Full-resolution binary mask, shape (H_full, W_full).
            Cropped internally to the ``crop_window``.
        dtype : int
            GDAL data type (default: Byte — values 0 or 1).
        """
        out_path = self.output_dir / output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        src = self._open_source()
        gt  = src.GetGeoTransform()
        prj = src.GetProjection()
        src = None

        cropped = mask_array[self.y0:self.y1, self.x0:self.x1]

        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(str(out_path), self.width, self.height, 1, dtype)
        dst.SetGeoTransform(gt)
        dst.SetProjection(prj)
        dst.GetRasterBand(1).WriteArray(cropped)
        dst.FlushCache()
        dst = None
        logger.debug(f"Mask exported → {out_path}")
