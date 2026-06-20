"""
s12coasts.eobc.algorithms
=========================
Clustering algorithm registry and hyperparameter grids for the EOBC framework.

The five algorithms selected represent diverse clustering principles:
    - K-Means              : partitional
    - Agglomerative        : hierarchical
    - Bayesian GMM         : model-based / probabilistic
    - Affinity Propagation : graph-based / exemplar
    - Fuzzy C-Means        : soft / fuzzy partitional

Adding a new algorithm
----------------------
1. Add it to ``build_algorithm_registry()``.
2. Add its parameter grid to ``PARAM_GRIDS``.
3. If it requires special fit/predict handling (e.g. Fuzzy C-Means), add a
   branch in ``ClusteringOptimiser._fit_predict()`` and
   ``EOBCClusterer.cluster()``.
"""

from __future__ import annotations

import logging
import warnings

from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans, MeanShift
from sklearn.mixture import BayesianGaussianMixture

logger = logging.getLogger(__name__)

try:
    from skfuzzy import cmeans as _fuzzy_cmeans  # noqa: F401
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    warnings.warn(
        "scikit-fuzzy is not installed. Fuzzy C-Means will be excluded from "
        "the ensemble.\nInstall with: pip install scikit-fuzzy",
        stacklevel=2,
    )


def build_algorithm_registry(include_fuzzy: bool = FUZZY_AVAILABLE) -> dict:
    """
    Build and return the active clustering algorithm registry.

    Parameters
    ----------
    include_fuzzy : bool
        Whether to include Fuzzy C-Means (requires scikit-fuzzy). Defaults to
        True if scikit-fuzzy is installed, False otherwise.

    Returns
    -------
    dict
        Mapping of algorithm name → sklearn-compatible estimator instance.
        Fuzzy C-Means is stored as None (handled separately in fit/predict).

    Examples
    --------
    >>> from s12coasts.eobc.algorithms import build_algorithm_registry
    >>> registry = build_algorithm_registry()
    >>> list(registry.keys())
    ['KMeans', 'AgglomerativeClustering', 'BayesianGMM',
     'AffinityPropagation', 'MeanShift', 'FuzzyCMeans']
    """
    registry: dict = {
        "KMeans": KMeans(random_state=42),
        "AgglomerativeClustering": AgglomerativeClustering(),
        "BayesianGMM": BayesianGaussianMixture(random_state=42),
        "AffinityPropagation": AffinityPropagation(),
        "MeanShift": MeanShift(),
    }
    if include_fuzzy:
        registry["FuzzyCMeans"] = None  # fit/predict handled in clusterer

    n = len(registry)
    logger.debug(f"Algorithm registry built with {n} algorithm(s): {list(registry)}")
    return registry


# ---------------------------------------------------------------------------
# Hyperparameter grids for grid-search optimisation
# ---------------------------------------------------------------------------

PARAM_GRIDS: dict = {
    "KMeans": {
        "n_clusters": [2],
        "algorithm": ["lloyd", "elkan"],
    },
    "AgglomerativeClustering": {
        "n_clusters": [2],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean", "manhattan", "cosine"],
    },
    "BayesianGMM": {
        "n_components": [2],
        "covariance_type": ["full", "tied", "diag", "spherical"],
    },
    "AffinityPropagation": {
        "damping": [0.5, 0.75, 0.9],
    },
    "MeanShift": {
        "bandwidth": [None, 0.5, 1.0, 1.5, 2.0, 3.0],
    },
    "FuzzyCMeans": {
        "c": [2],
        "m": [1.5, 2.0, 2.5, 3.5],
        "error": [1e-4, 1e-6],
        "maxiter": [int(1e5)],
    },
}
