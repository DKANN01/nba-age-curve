"""Three ways to find a player's closest stylistic comparisons.

All three answer the same question -- who else looked like this player at the
same stage of their career? -- but they build the comparison space differently,
so they will not agree, and are not meant to:

``match_standard_knn``
    Averages the raw features over each player's first *n* seasons, then fits a
    *fresh* PCA on just those player averages. The component space therefore
    depends on the query: it describes how the eligible peer group varies.

``match_centroid_knn``
    Uses the global season-level PCA (fit once over all 20k player-seasons),
    and takes each player's centroid -- the mean of their season coordinates --
    over their first *n* seasons. The space is fixed across queries.

``match_epsilon``
    The same global centroids, but returns every player inside a fixed radius
    rather than a fixed count. A player with no close peers gets few or no
    matches, which is the honest answer; k-NN would return five distant ones
    and present them as neighbours anyway.

In all three, "first *n* seasons" means career years up to *n* among players who
reached career year *n*. Coverage of that window need not be complete: a player
whose record starts partway through their career contributes the recorded part,
and ``coverage`` on each match reports how many seasons that was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from nba_age_curve.config import (
    DEFAULT_EPSILON,
    DEFAULT_N_NEIGHBORS,
    FEATURES,
    N_COMPONENTS,
    PC_COLS,
)
from nba_age_curve.data import aggregate_careers, first_n_seasons
from nba_age_curve.pca import fit_season_pca

STANDARD = "Standard k-NN"
CENTROID = "Centroid k-NN"
EPSILON = "ε-neighbourhood"


@dataclass
class NeighborMatch:
    """The comparison set for one player, under one method."""

    method: str
    target_id: int
    n_seasons: int
    neighbors: pd.DataFrame
    pool_size: int
    warnings: list[str] = field(default_factory=list)

    @property
    def ids(self) -> list[int]:
        return self.neighbors["PLAYER_ID"].tolist()

    @property
    def distances(self) -> np.ndarray:
        return self.neighbors["distance"].to_numpy()

    def __len__(self) -> int:
        return len(self.neighbors)


class TargetNotComparable(ValueError):
    """The target has no observed season inside the matching window.

    Raised for players who debuted before the dataset begins by more than the
    window length -- Kareem Abdul-Jabbar's record starts at career year 11, so
    at a cutoff of 5 there is nothing of his to match on. A censored player
    whose record *does* reach into the window is matched on the part of it that
    was recorded.
    """


def _require_in_pool(frame: pd.DataFrame, target_id: int, n_seasons: int) -> None:
    if target_id not in set(frame["PLAYER_ID"]):
        raise TargetNotComparable(
            f"This player's record starts after career year {n_seasons}, so "
            f"none of the first {n_seasons} years are in the data. Raise the "
            f"cutoff to reach the seasons that were recorded."
        )


def _target_n_seasons(df: pd.DataFrame, target_id: int) -> int:
    seasons = df.loc[df["PLAYER_ID"] == target_id, "Season_No"]
    if seasons.empty:
        raise KeyError(f"PLAYER_ID {target_id} not found")
    return int(seasons.max())


def _decorate(
    neighbors: pd.DataFrame, lookup: pd.DataFrame, df: pd.DataFrame
) -> pd.DataFrame:
    """Attach display fields (label, total seasons, censoring flag, coverage)."""
    careers = df.groupby("PLAYER_ID").agg(
        label=("label", "first"),
        PLAYER_NAME=("PLAYER_NAME", "first"),
        total_seasons=("Season_No", "max"),
        left_censored=("left_censored", "max"),
        first_season=("SEASON", "min"),
        last_season=("SEASON", "max"),
    )
    out = neighbors.merge(careers, left_on="PLAYER_ID", right_index=True, how="left")
    if "coverage" in lookup.columns:
        out = out.merge(
            lookup[["PLAYER_ID", "coverage"]], on="PLAYER_ID", how="left"
        )
    return out.sort_values("distance")


def _knn_query(
    frame: pd.DataFrame, cols: list[str], target_id: int, k: int
) -> pd.DataFrame:
    """Nearest ``k`` rows to the target, excluding the target itself."""
    # k + 1 because the target is always its own nearest neighbour at distance 0.
    n_neighbors = min(k + 1, len(frame))
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(frame[cols])
    target_row = frame.index[frame["PLAYER_ID"] == target_id]
    distances, indices = knn.kneighbors(frame.loc[target_row, cols])

    out = pd.DataFrame(
        {
            "PLAYER_ID": frame.iloc[indices[0]]["PLAYER_ID"].to_numpy(),
            "distance": distances[0],
        }
    )
    return out[out["PLAYER_ID"] != target_id].head(k)


def match_standard_knn(
    df: pd.DataFrame,
    target_id: int,
    n_seasons: int | None = None,
    k: int = DEFAULT_N_NEIGHBORS,
) -> NeighborMatch:
    """Match on career-average features, with PCA refit on the peer group."""
    n_seasons = n_seasons or _target_n_seasons(df, target_id)
    subset = first_n_seasons(df, n_seasons)
    careers = aggregate_careers(subset, FEATURES).reset_index(drop=True)
    _require_in_pool(careers, target_id, n_seasons)

    warnings: list[str] = []
    if len(careers) <= N_COMPONENTS:
        return NeighborMatch(
            STANDARD,
            target_id,
            n_seasons,
            _decorate(pd.DataFrame(columns=["PLAYER_ID", "distance"]), careers, df),
            len(careers),
            [f"Only {len(careers)} players reached season {n_seasons}; too few to fit PCA."],
        )
    if len(careers) < 50:
        warnings.append(
            f"Only {len(careers)} players reached season {n_seasons}, so this "
            "component space is fit on a small sample."
        )

    model = fit_season_pca(careers, FEATURES, n_components=N_COMPONENTS)
    scores = model.transform(careers)
    for i in range(N_COMPONENTS):
        careers[f"PC{i + 1}"] = scores[:, i]

    neighbors = _knn_query(careers, PC_COLS, target_id, k)
    return NeighborMatch(
        STANDARD, target_id, n_seasons, _decorate(neighbors, careers, df),
        len(careers), warnings,
    )


def _centroids(df: pd.DataFrame, n_seasons: int) -> pd.DataFrame:
    """Mean global-PC coordinates over each eligible player's window.

    The window is career years up to ``n_seasons``; ``coverage`` says how many
    of them the player was actually observed for, which is fewer than
    ``n_seasons`` for anyone whose record starts partway through.
    """
    subset = first_n_seasons(df, n_seasons)
    grouped = subset.groupby("PLAYER_ID")
    centroids = grouped[PC_COLS].mean()
    centroids["coverage"] = grouped["coverage"].first()
    return centroids.reset_index()


def match_centroid_knn(
    df: pd.DataFrame,
    target_id: int,
    n_seasons: int | None = None,
    k: int = DEFAULT_N_NEIGHBORS,
) -> NeighborMatch:
    """Match on centroids in the fixed, global season-level PC space."""
    n_seasons = n_seasons or _target_n_seasons(df, target_id)
    centroids = _centroids(df, n_seasons)
    _require_in_pool(centroids, target_id, n_seasons)

    if len(centroids) < 2:
        return NeighborMatch(
            CENTROID, target_id, n_seasons,
            _decorate(pd.DataFrame(columns=["PLAYER_ID", "distance"]), centroids, df),
            len(centroids),
            [f"Only {len(centroids)} players reached season {n_seasons}."],
        )

    neighbors = _knn_query(centroids, PC_COLS, target_id, k)
    return NeighborMatch(
        CENTROID, target_id, n_seasons, _decorate(neighbors, centroids, df),
        len(centroids),
    )


def match_epsilon(
    df: pd.DataFrame,
    target_id: int,
    n_seasons: int | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> NeighborMatch:
    """Return every player whose centroid lies within ``epsilon`` of the target."""
    n_seasons = n_seasons or _target_n_seasons(df, target_id)
    centroids = _centroids(df, n_seasons)
    _require_in_pool(centroids, target_id, n_seasons)

    target_row = centroids.loc[centroids["PLAYER_ID"] == target_id, PC_COLS]

    offsets = centroids[PC_COLS].to_numpy() - target_row.to_numpy()[0]
    centroids = centroids.assign(distance=np.linalg.norm(offsets, axis=1))

    within = centroids[
        (centroids["PLAYER_ID"] != target_id) & (centroids["distance"] <= epsilon)
    ][["PLAYER_ID", "distance"]]

    warnings: list[str] = []
    if within.empty:
        warnings.append(
            f"No player sits within ε={epsilon:g} of this profile. "
            "Widen the radius to find comparisons."
        )
    return NeighborMatch(
        EPSILON, target_id, n_seasons, _decorate(within, centroids, df),
        len(centroids), warnings,
    )


MATCHERS = {STANDARD: match_standard_knn, CENTROID: match_centroid_knn}
