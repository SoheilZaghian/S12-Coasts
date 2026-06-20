"""
s12coasts.io.metadata
======================
Patch metadata management — tracks patch IDs, region information, and quality
labels in a persistent CSV file (``data_information.csv``).

Quality Labels
--------------
H  — High Quality: fine detail, strong spatio-temporal alignment.
B1 — Boundary Error L1: primary water only; fine features omitted.
B2 — Boundary Error L2: artefacts, disconnected boundaries, or misalignment.
M  — Model Error: framework failure; excluded from final dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Manages the S12-Coasts dataset metadata CSV file.

    Tracks patch IDs, region metadata, sensor availability flags, quality
    labels assigned during visual inspection, and acquisition dates.

    Parameters
    ----------
    output_dir : Path or str
        Root output directory where ``data_information.csv`` is written.
    region_metadata : dict
        Region-level metadata fields:
        ``{climate_zone, agro_ecological_zone, country_name, region_name}``.

    Examples
    --------
    >>> from s12coasts.io.metadata import MetadataManager
    >>> meta = MetadataManager(
    ...     output_dir=Path("outputs/masks/Huron_Lake"),
    ...     region_metadata={"region_name": "Huron_Lake", "country_name": "Canada"},
    ... )
    >>> patch_id = meta.next_id()      # "00001"
    >>> meta.append(patch_id)
    >>> meta.save()
    """

    COLUMNS: list[str] = [
        "id",
        "region_name",
        "climate_zone",
        "agro_ecological_zone",
        "country_name",
        "S1",
        "S2",
        "gt",
        "quality_label",   # assigned during expert visual inspection
        "date",
    ]

    QUALITY_LABELS: tuple[str, ...] = ("H", "B1", "B2", "M")

    def __init__(
        self,
        output_dir: Path | str,
        region_metadata: dict,
    ) -> None:
        self.output_dir      = Path(output_dir)
        self.region_metadata = region_metadata
        self.csv_path        = self.output_dir / "data_information.csv"
        self._df             = self._load_or_create()

    def _load_or_create(self) -> pd.DataFrame:
        """Load an existing metadata CSV or create an empty one."""
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            logger.debug(
                f"Loaded existing metadata: {len(df)} record(s) from {self.csv_path}"
            )
            return df
        logger.debug(f"Creating new metadata file: {self.csv_path}")
        return pd.DataFrame(columns=self.COLUMNS)

    def next_id(self) -> str:
        """Return the next zero-padded five-digit patch ID."""
        return str(len(self._df) + 1).zfill(5)

    def append(self, patch_id: str) -> None:
        """
        Append a metadata row for a newly exported patch.

        Parameters
        ----------
        patch_id : str
            Zero-padded patch ID (e.g. ``"00001"``).
        """
        row = {
            "id":                   patch_id,
            "region_name":          self.region_metadata.get("region_name"),
            "climate_zone":         self.region_metadata.get("climate_zone"),
            "agro_ecological_zone": self.region_metadata.get("agro_ecological_zone"),
            "country_name":         self.region_metadata.get("country_name"),
            "S1":                   "Yes",
            "S2":                   "Yes",
            "gt":                   "Yes",
            "quality_label":        None,   # assigned during visual inspection
            "date":                 None,
        }
        self._df = pd.concat(
            [self._df, pd.DataFrame([row])], ignore_index=True
        )

    def update_quality_label(self, patch_id: str, label: str) -> None:
        """
        Update the quality label for an existing patch record.

        Parameters
        ----------
        patch_id : str
            Patch ID to update.
        label : str
            Quality label — one of ``{"H", "B1", "B2", "M"}``.

        Raises
        ------
        ValueError
            If ``label`` is not a valid quality label.
        KeyError
            If ``patch_id`` is not found in the metadata.
        """
        if label not in self.QUALITY_LABELS:
            raise ValueError(
                f"Invalid quality label '{label}'. "
                f"Must be one of {self.QUALITY_LABELS}."
            )
        mask = self._df["id"] == patch_id
        if not mask.any():
            raise KeyError(f"Patch ID '{patch_id}' not found in metadata.")

        self._df.loc[mask, "quality_label"] = label
        logger.debug(f"Quality label updated: patch {patch_id} → {label}")

    def save(self) -> None:
        """Persist the metadata DataFrame to ``data_information.csv``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.csv_path, index=False)
        logger.info(
            f"Metadata saved → {self.csv_path} "
            f"({len(self._df)} record(s))"
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a copy of the current metadata DataFrame."""
        return self._df.copy()

    def summary(self) -> dict:
        """
        Return a summary of quality label distribution.

        Returns
        -------
        dict
            Counts and percentages per quality label.
        """
        total = len(self._df)
        if total == 0:
            return {}
        counts = self._df["quality_label"].value_counts(dropna=False).to_dict()
        return {
            label: {
                "count":      counts.get(label, 0),
                "percentage": round(100 * counts.get(label, 0) / total, 1),
            }
            for label in list(self.QUALITY_LABELS) + [None]
            if label in counts or label is None
        }
