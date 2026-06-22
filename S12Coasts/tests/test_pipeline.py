"""
Tests for s12coasts.eobc.pipeline — EOBCPipeline (integration tests).

These tests use synthetic GeoTIFF files to avoid requiring real satellite data.
"""

import numpy as np
import pytest
from pathlib import Path

try:
    from osgeo import gdal, gdalconst
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_geotiff(
    path: Path,
    n_bands: int = 21,
    height: int = 64,
    width: int = 64,
    boundary_value: int = 255,
) -> None:
    """
    Write a synthetic stacked GeoTIFF to ``path`` for testing.

    Band layout mirrors the S12-Coasts stacked format:
        Bands  1–13  : Sentinel-2 SR (random uniform [0, 1])
        Bands 14–18  : Spectral indices (random uniform [-1, 1])
        Bands 19–20  : Sentinel-1 VV/VH (random uniform [-20, 0])
        Band  21     : Water boundary (0 background, boundary_value on edges)
    """
    rng = np.random.default_rng(42)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), width, height, n_bands, gdalconst.GDT_Float32)

    # Set a simple identity geotransform and WGS84 projection
    ds.SetGeoTransform((0.0, 10.0, 0.0, 0.0, 0.0, -10.0))
    from osgeo import osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    for b in range(1, n_bands + 1):
        if b <= 13:
            data = rng.uniform(0, 1, (height, width)).astype(np.float32)
        elif b <= 18:
            data = rng.uniform(-1, 1, (height, width)).astype(np.float32)
        elif b <= 20:
            data = rng.uniform(-20, 0, (height, width)).astype(np.float32)
        else:
            # Water boundary band — draw a simple border line
            data = np.zeros((height, width), dtype=np.float32)
            data[height // 2, :] = boundary_value  # horizontal line

        ds.GetRasterBand(b).WriteArray(data)

    ds.FlushCache()
    ds = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_patch_dir(tmp_path):
    """Create a temporary directory with two synthetic GeoTIFF patches."""
    if not GDAL_AVAILABLE:
        pytest.skip("GDAL not installed — skipping GeoTIFF-dependent tests.")

    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()

    for i in range(1, 3):
        _create_synthetic_geotiff(patch_dir / f"patch_{i}.tif")

    return patch_dir


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for EOBC exports."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture
def eobc_config(tmp_patch_dir, tmp_output_dir):
    """Minimal EOBC config for testing with synthetic patches."""
    return {
        "input_dir":        str(tmp_patch_dir),
        "output_dir":       str(tmp_output_dir),
        "region_name":      "TestRegion",
        "feature_bands":    [13, 14, 15, 16, 17, 18, 19],
        "n_pca_components": 3,
        "patch_size_px":    32,    # small crop for speed
        "visualise":        False,
        "metadata": {
            "region_name":  "TestRegion",
            "climate_zone": "Temperate",
            "country_name": "Test",
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GDAL_AVAILABLE, reason="GDAL not installed")
class TestEOBCPipeline:

    def test_pipeline_runs_without_error(self, eobc_config):
        """EOBCPipeline.run() should complete without raising exceptions."""
        from s12coasts.eobc.pipeline import EOBCPipeline
        pipeline = EOBCPipeline(eobc_config)
        pipeline.run()   # should not raise

    def test_output_directories_created(self, eobc_config, tmp_output_dir):
        """S1/, S2/, and gt/ output directories should be created."""
        from s12coasts.eobc.pipeline import EOBCPipeline
        pipeline = EOBCPipeline(eobc_config)
        pipeline.run()

        for sub in ("S1", "S2", "gt"):
            assert (tmp_output_dir / sub).exists(), (
                f"Expected output subdirectory '{sub}' not found."
            )

    def test_metadata_csv_created(self, eobc_config, tmp_output_dir):
        """data_information.csv should be created in the output directory."""
        from s12coasts.eobc.pipeline import EOBCPipeline
        pipeline = EOBCPipeline(eobc_config)
        pipeline.run()

        csv_path = tmp_output_dir / "data_information.csv"
        assert csv_path.exists(), "Metadata CSV not found."

    def test_metadata_has_correct_row_count(self, eobc_config, tmp_output_dir):
        """Metadata CSV should have one row per successfully processed patch."""
        import pandas as pd
        from s12coasts.eobc.pipeline import EOBCPipeline

        pipeline = EOBCPipeline(eobc_config)
        pipeline.run()

        df = pd.read_csv(tmp_output_dir / "data_information.csv")
        # 2 synthetic patches — both should succeed
        assert len(df) == 2, f"Expected 2 metadata rows, got {len(df)}."

    def test_gt_masks_are_binary(self, eobc_config, tmp_output_dir):
        """All ground-truth mask files should contain only 0 and 1."""
        from s12coasts.eobc.pipeline import EOBCPipeline

        pipeline = EOBCPipeline(eobc_config)
        pipeline.run()

        gt_dir = tmp_output_dir / "gt"
        gt_files = list(gt_dir.glob("*.tif"))
        assert len(gt_files) > 0, "No ground-truth mask files found."

        for gt_path in gt_files:
            ds   = gdal.Open(str(gt_path))
            data = ds.GetRasterBand(1).ReadAsArray()
            ds   = None
            unique_vals = set(np.unique(data))
            assert unique_vals.issubset({0, 1}), (
                f"Non-binary values {unique_vals} found in {gt_path.name}."
            )

    def test_patch_ids_are_zero_padded(self, eobc_config, tmp_output_dir):
        """Exported filenames should use zero-padded 5-digit patch IDs."""
        from s12coasts.eobc.pipeline import EOBCPipeline

        pipeline = EOBCPipeline(eobc_config)
        pipeline.run()

        s2_files = sorted((tmp_output_dir / "S2").glob("*.tif"))
        for f in s2_files:
            stem = f.stem   # e.g. "S2_00001"
            pid  = stem.split("_")[-1]
            assert len(pid) == 5 and pid.isdigit(), (
                f"Patch ID '{pid}' in '{f.name}' is not a 5-digit zero-padded integer."
            )


# ---------------------------------------------------------------------------
# Tests: GeoTIFFExporter
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GDAL_AVAILABLE, reason="GDAL not installed")
class TestGeoTIFFExporter:

    def test_export_bands_creates_file(self, tmp_patch_dir, tmp_output_dir):
        """export_bands() should create a GeoTIFF file at the specified path."""
        from s12coasts.io.geotiff import GeoTIFFExporter

        patch_path = list(tmp_patch_dir.glob("*.tif"))[0]
        exporter   = GeoTIFFExporter(
            input_file=patch_path,
            output_dir=tmp_output_dir,
            crop_window=(0, 32, 0, 32),
        )
        exporter.export_bands("S2/S2_test.tif", band_indices=list(range(1, 14)))
        assert (tmp_output_dir / "S2" / "S2_test.tif").exists()

    def test_export_mask_creates_file(self, tmp_patch_dir, tmp_output_dir):
        """export_mask() should create a GeoTIFF binary mask file."""
        from s12coasts.io.geotiff import GeoTIFFExporter

        patch_path = list(tmp_patch_dir.glob("*.tif"))[0]
        mask       = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 1

        exporter = GeoTIFFExporter(
            input_file=patch_path,
            output_dir=tmp_output_dir,
            crop_window=(0, 32, 0, 32),
        )
        exporter.export_mask("gt/gt_test.tif", mask_array=mask)
        assert (tmp_output_dir / "gt" / "gt_test.tif").exists()

    def test_invalid_band_index_raises(self, tmp_patch_dir, tmp_output_dir):
        """export_bands() should raise ValueError for out-of-range band indices."""
        from s12coasts.io.geotiff import GeoTIFFExporter

        patch_path = list(tmp_patch_dir.glob("*.tif"))[0]
        exporter   = GeoTIFFExporter(
            input_file=patch_path,
            output_dir=tmp_output_dir,
            crop_window=(0, 32, 0, 32),
        )
        with pytest.raises(ValueError, match="out of range"):
            exporter.export_bands("S2/bad.tif", band_indices=[99, 100])


# ---------------------------------------------------------------------------
# Tests: MetadataManager
# ---------------------------------------------------------------------------

class TestMetadataManager:

    def test_next_id_increments(self, tmp_output_dir):
        """next_id() should return incrementing zero-padded IDs."""
        from s12coasts.io.metadata import MetadataManager
        meta = MetadataManager(tmp_output_dir, {"region_name": "Test"})
        assert meta.next_id() == "00001"
        meta.append("00001")
        assert meta.next_id() == "00002"

    def test_save_and_reload(self, tmp_output_dir):
        """Saved metadata should be reloadable with the same records."""
        from s12coasts.io.metadata import MetadataManager
        meta = MetadataManager(tmp_output_dir, {"region_name": "Test"})
        meta.append(meta.next_id())
        meta.append(meta.next_id())
        meta.save()

        # Reload
        meta2 = MetadataManager(tmp_output_dir, {"region_name": "Test"})
        assert len(meta2.dataframe) == 2

    def test_update_quality_label(self, tmp_output_dir):
        """update_quality_label() should update the correct row."""
        from s12coasts.io.metadata import MetadataManager
        meta = MetadataManager(tmp_output_dir, {"region_name": "Test"})
        meta.append("00001")
        meta.update_quality_label("00001", "H")
        assert meta.dataframe.loc[
            meta.dataframe["id"] == "00001", "quality_label"
        ].values[0] == "H"

    def test_invalid_quality_label_raises(self, tmp_output_dir):
        """update_quality_label() should raise ValueError for invalid labels."""
        from s12coasts.io.metadata import MetadataManager
        meta = MetadataManager(tmp_output_dir, {"region_name": "Test"})
        meta.append("00001")
        with pytest.raises(ValueError, match="Invalid quality label"):
            meta.update_quality_label("00001", "X")

    def test_summary_returns_dict(self, tmp_output_dir):
        """summary() should return a dict with label counts and percentages."""
        from s12coasts.io.metadata import MetadataManager
        meta = MetadataManager(tmp_output_dir, {"region_name": "Test"})
        meta.append("00001")
        meta.append("00002")
        meta.update_quality_label("00001", "H")
        meta.update_quality_label("00002", "B1")
        summary = meta.summary()
        assert "H" in summary
        assert "B1" in summary
        assert summary["H"]["count"] == 1
        assert summary["H"]["percentage"] == 50.0
