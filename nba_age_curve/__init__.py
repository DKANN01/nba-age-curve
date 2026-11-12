"""PCA + nearest-neighbor career projections for NBA players."""

from nba_age_curve.config import (
    DEFAULT_EPSILON,
    DEFAULT_METRIC,
    DEFAULT_N_NEIGHBORS,
    FEATURES,
    N_COMPONENTS,
    PC_COLS,
    PROJECTABLE_METRICS,
)
from nba_age_curve.data import load_player_seasons
from nba_age_curve.pca import SeasonPCA, fit_season_pca
from nba_age_curve.projection import Projection, project_metric
from nba_age_curve.similarity import (
    NeighborMatch,
    match_centroid_knn,
    match_epsilon,
    match_standard_knn,
)

__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_METRIC",
    "DEFAULT_N_NEIGHBORS",
    "FEATURES",
    "N_COMPONENTS",
    "PC_COLS",
    "PROJECTABLE_METRICS",
    "NeighborMatch",
    "Projection",
    "SeasonPCA",
    "fit_season_pca",
    "load_player_seasons",
    "match_centroid_knn",
    "match_epsilon",
    "match_standard_knn",
    "project_metric",
]
