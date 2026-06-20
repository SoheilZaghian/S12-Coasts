"""
s12coasts.eobc.optimiser
========================
Grid-search hyperparameter optimisation for the EOBC ensemble of clustering
algorithms. Optimisation is performed without ground truth using three
unsupervised quality metrics:

    - Silhouette Score         : higher is better
    - Davies-Bouldin Index     : lower is better
    - Calinski-Harabasz Index  : higher is better

The best configuration per algorithm is selected by maximum Silhouette Score.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import ParameterGrid

from s12coasts.eobc.algorithms import PARAM_GRIDS, FUZZY_AVAILABLE, build_algorithm_registry

logger = logging.getLogger(__name__)

# Sentinel values for invalid / failed clustering configurations
_METRIC_SENTINEL = {
    "Silhouette Score": -999.0,
    "Davies-Bouldin Index": 999.0,
    "Calinski-Harabasz Index": -999.0,
}


def _evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute unsupervised clustering quality metrics.

    Returns sentinel values when fewer than 2 distinct clusters are found.
    """
    n_unique = len(np.unique(labels))
    if 1 < n_unique < len(X):
        return {
            "Silhouette Score": silhouette_score(X, labels),
            "Davies-Bouldin Index": davies_bouldin_score(X, labels),
            "Calinski-Harabasz Index": calinski_harabasz_score(X, labels),
        }
    return _METRIC_SENTINEL.copy()


class ClusteringOptimiser:
    """
    Grid-search hyperparameter optimisation for the EOBC clustering ensemble.

    Parameters
    ----------
    pca_data : np.ndarray
        Object-based feature matrix after PCA, shape (n_regions, n_components).
    patch_id : int or str
        Identifier for the current patch (used in logging only).

    Attributes
    ----------
    results_df : pd.DataFrame
        Full grid-search results across all algorithms and parameter combinations.
    best_params_df : pd.DataFrame
        Best parameter configuration per algorithm.

    Examples
    --------
    >>> from s12coasts.eobc.optimiser import ClusteringOptimiser
    >>> optimiser = ClusteringOptimiser(pca_features, patch_id=0)
    >>> results_df, best_params_df = optimiser.run()
    """

    def __init__(
        self,
        pca_data: np.ndarray,
        patch_id: Union[int, str] = 0,
    ) -> None:
        self.X = pca_data
        self.patch_id = patch_id
        self.algorithms = build_algorithm_registry()
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.best_params_df: pd.DataFrame = pd.DataFrame()

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full grid search over all active clustering algorithms.

        Returns
        -------
        results_df : pd.DataFrame
            All parameter combinations and their metric scores.
        best_params_df : pd.DataFrame
            Best configuration per algorithm (highest Silhouette Score).
        """
        records = []

        for name, algorithm in self.algorithms.items():
            if name not in PARAM_GRIDS:
                logger.warning(f"No parameter grid defined for '{name}'. Skipping.")
                continue

            for params in ParameterGrid(PARAM_GRIDS[name]):
                labels, metrics = self._fit_predict(name, algorithm, params)
                records.append({
                    "Algorithm": name,
                    "Parameters": str(params),
                    **metrics,
                })

        self.results_df = pd.DataFrame(records)
        self.best_params_df = self._select_best(self.results_df)

        logger.debug(
            f"[patch {self.patch_id}] Optimisation complete — "
            f"{len(self.results_df)} configurations evaluated across "
            f"{len(self.algorithms)} algorithms."
        )
        return self.results_df, self.best_params_df

    def _fit_predict(
        self,
        name: str,
        algorithm,
        params: dict,
    ) -> tuple[np.ndarray, dict]:
        """Fit a single algorithm with given params and return labels + metrics."""
        try:
            if name == "FuzzyCMeans" and FUZZY_AVAILABLE:
                from skfuzzy import cmeans
                _, u, *_ = cmeans(self.X.T, **params)
                labels = np.argmax(u, axis=0)
            else:
                alg = algorithm.set_params(**params)
                labels = alg.fit_predict(self.X)

            metrics = _evaluate(self.X, labels)

        except Exception as exc:
            logger.debug(
                f"[patch {self.patch_id}] {name} | {params} → failed: {exc}"
            )
            labels = np.zeros(len(self.X), dtype=int)
            metrics = _METRIC_SENTINEL.copy()

        return labels, metrics

    @staticmethod
    def _select_best(results_df: pd.DataFrame) -> pd.DataFrame:
        """Select the best-performing parameter set per algorithm."""
        if results_df.empty:
            return pd.DataFrame()
        idx = results_df.groupby("Algorithm")["Silhouette Score"].idxmax()
        return results_df.loc[idx].reset_index(drop=True)

    def plot_results(self, save_path: str | None = None) -> None:
        """
        Visualise grid-search results across all three evaluation metrics.

        Parameters
        ----------
        save_path : str, optional
            If provided, saves the figure instead of displaying it.
        """
        if self.results_df.empty:
            raise RuntimeError(
                "No results to plot. Call run() before plot_results()."
            )

        metrics = [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(24, 4))
        fig.suptitle(f"Clustering Optimisation — Patch {self.patch_id}", fontsize=13)

        for ax, metric in zip(axes, metrics):
            ascending = metric == "Davies-Bouldin Index"
            df_sorted = self.results_df.sort_values(metric, ascending=ascending)
            sns.barplot(data=df_sorted, x=metric, y="Algorithm", ax=ax, dodge=False)
            ax.set_title(metric, fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Optimisation plot saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)
