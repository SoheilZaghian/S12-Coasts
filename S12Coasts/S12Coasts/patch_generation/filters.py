"""
s12coasts.patch_generation.filters
=====================================
Spectral index computation and filtering utilities for Sentinel-2 image
collections within the GEE patch generation pipeline.

Spectral Indices
----------------
MNDWI : Modified Normalised Difference Water Index
    (B3 - B11) / (B3 + B11)
NDVI  : Normalised Difference Vegetation Index
    (B8 - B4) / (B8 + B4)
AWEI  : Automated Water Extraction Index
    B2 + 2.5·B3 - 1.5·(B8 + B11) - 0.25·B12
WRI   : Water Ratio Index
    (B3 + B4) / (B8 + B11)
NDPI  : Normalised Difference Pond Index
    (B8 - B3) / (B8 + B3)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spectral index definitions
# Each entry: (formula_lambda, output_band_name)
# ---------------------------------------------------------------------------

SPECTRAL_INDICES: list[tuple] = [
    (
        lambda img: img.select("B3").subtract(img.select("B11"))
                       .divide(img.select("B3").add(img.select("B11"))),
        "MNDWI",
    ),
    (
        lambda img: img.select("B8").subtract(img.select("B4"))
                       .divide(img.select("B8").add(img.select("B4"))),
        "NDVI",
    ),
    (
        lambda img: (
            img.select("B2")
               .add(img.select("B3").multiply(2.5))
               .subtract(img.select("B8").add(img.select("B11")).multiply(1.5))
               .subtract(img.select("B12").multiply(0.25))
        ),
        "AWEI",
    ),
    (
        lambda img: img.select("B3").add(img.select("B4"))
                       .divide(img.select("B8").add(img.select("B11"))),
        "WRI",
    ),
    (
        lambda img: img.select("B8").subtract(img.select("B3"))
                       .divide(img.select("B8").add(img.select("B3"))),
        "NDPI",
    ),
]

# Sentinel-2 bands to retrieve from GEE
SENTINEL2_BANDS: list[str] = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12", "QA60",
]


def scale_reflectance(image) -> "ee.Image":
    """Convert Sentinel-2 DN values to surface reflectance [0, 1]."""
    return image.divide(10000)


def add_spectral_index(image, formula, band_name: str) -> "ee.Image":
    """Compute a spectral index and append it as a new named band."""
    return image.addBands(formula(image).rename(band_name))


def add_all_indices(image) -> "ee.Image":
    """Add all five spectral indices to a Sentinel-2 image."""
    for formula, name in SPECTRAL_INDICES:
        image = add_spectral_index(image, formula, name)
    return image


def add_date_diff(image, reference_date) -> "ee.Image":
    """
    Compute the absolute day difference between an image and a reference date,
    and store it as a metadata property ``date_diff``.

    Used to identify the Sentinel-1 acquisition closest to a Sentinel-2 date.
    """
    try:
        import ee
    except ImportError as exc:
        raise ImportError(
            "Google Earth Engine is required.\n"
            "Install with: pip install earthengine-api geemap"
        ) from exc

    s1_date = ee.Date(image.get("system:time_start"))
    diff    = s1_date.difference(reference_date, "day").abs()
    return image.set("date_diff", diff)
