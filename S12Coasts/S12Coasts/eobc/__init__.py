"""s12coasts.eobc — Ensemble Object-Based Clustering framework."""

from s12coasts.eobc.algorithms import build_algorithm_registry, PARAM_GRIDS, FUZZY_AVAILABLE
from s12coasts.eobc.features import FeatureExtractor, DEFAULT_FEATURE_BANDS
from s12coasts.eobc.optimiser import ClusteringOptimiser
from s12coasts.eobc.clusterer import EOBCClusterer
from s12coasts.eobc.pipeline import EOBCPipeline

__all__ = [
    "EOBCPipeline",
    "EOBCClusterer",
    "ClusteringOptimiser",
    "FeatureExtractor",
    "build_algorithm_registry",
    "PARAM_GRIDS",
    "FUZZY_AVAILABLE",
    "DEFAULT_FEATURE_BANDS",
]
