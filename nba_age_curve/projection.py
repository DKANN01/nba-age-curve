"""Projecting a metric forward from a player's comparison set.

The projection is a *path of changes*, not a path of levels. For each future
season the model takes the year-over-year change each comparison player made at
that same point in their career, averages those changes weighted by how close
the comparison is, and adds the result to the running total. The path therefore
starts exactly where the target player actually is, and inherits only the
*shape* of its peers' careers -- not their level.

Two consequences worth knowing:

* Anchoring on a single season means one unusual year (an injury, a career
  season) propagates through the whole projection.
* A season is only projected while at least one comparison player has data for
  both it and the season before. Once they have all retired, the path stops
  rather than extrapolating past its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_age_curve.similarity import NeighborMatch

#: Guards against a division by zero when a comparison sits at distance 0.
_EPSILON = 1e-5


@dataclass
class Projection:
    """A projected path for one metric under one matching method."""

    method: str
    metric: str
    anchor_season_no: int
    anchor_value: float
    seasons: list[int]
    values: list[float]
    spread: list[float]
    neighbors_used: list[int]

    def __len__(self) -> int:
        return len(self.seasons)

    def to_frame(self) -> pd.DataFrame:
        """Tabular form, with the anchor season prepended so lines connect."""
        return pd.DataFrame(
            {
                "Season_No": [self.anchor_season_no] + self.seasons,
                self.metric: [self.anchor_value] + self.values,
                "spread": [0.0] + self.spread,
                "neighbors_used": [0] + self.neighbors_used,
                "method": self.method,
            }
        )


def inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
    """Normalized 1/d weights: closer comparisons count for more."""
    weights = 1.0 / (np.asarray(distances, dtype=float) + _EPSILON)
    total = weights.sum()
    if total == 0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def match_weights(match: NeighborMatch) -> pd.Series:
    """The comparison set's inverse-distance weights, keyed by player.

    These are exactly the weights ``project_metric`` applies to each
    comparison's year-over-year changes. Handing the same vector to
    :meth:`~nba_age_curve.gamm.GAMM.refit_weighted` is what lets a matching
    method drive the GAMM rather than sitting beside it: the same comp set and
    the same weights, used to fit whole smooth curves instead of one-season
    deltas.
    """
    if len(match) == 0:
        return pd.Series(dtype=float)
    return pd.Series(inverse_distance_weights(match.distances), index=match.ids)


def project_with_gamm(
    model,
    df: pd.DataFrame,
    match: NeighborMatch,
    metric: str,
    cutoff: int | None = None,
    n_ahead: int = 8,
    pc_cols: list[str] | None = None,
):
    """Forecast ``metric`` by refitting ``model`` on a comparison set.

    The comparison players (and only they) carry weight in the fit, in
    proportion to closeness. Because a handful of players cannot support the
    full basis, ``refit_weighted`` shrinks the result toward the global curve in
    proportion to how thin the weighting is -- so a tight comp set yields a
    gentle tilt rather than a wild extrapolation.

    Returns ``(seasons, mean, sd, ess)`` or ``None`` when the set is empty.
    ``ess`` is the effective number of weighted players, which says how much the
    comparisons were actually able to move the curve.
    """
    cutoff = cutoff or match.n_seasons
    weights = match_weights(match)
    if weights.empty:
        return None

    fit_rows = df.dropna(subset=[metric])
    row_w = fit_rows["PLAYER_ID"].map(weights).fillna(0.0).to_numpy()
    if row_w.sum() <= 0:
        return None

    local = model.refit_weighted(
        fit_rows,
        fit_rows[metric].to_numpy(),
        fit_rows["PLAYER_ID"].to_numpy(),
        row_w,
    )

    target = df[df["PLAYER_ID"] == match.target_id]
    history = target[target["Season_No"] <= cutoff].dropna(subset=[metric])
    if history.empty:
        return None

    future = type(model).future_frame(history, n_ahead=n_ahead, pc_cols=pc_cols)
    if len(future) == 0:
        return None

    mean, sd = local.forecast(history, future, y_col=metric)
    w = weights.to_numpy()
    ess = float(w.sum() ** 2 / max((w**2).sum(), 1e-12))
    return future["Season_No"].to_numpy(), mean, sd, ess


def _delta_table(
    df: pd.DataFrame, player_ids: list[int], metric: str
) -> pd.DataFrame:
    """Year-over-year change in ``metric`` per comparison player and season.

    Built once per projection rather than re-filtering the full table inside the
    season loop, which is what made the original O(seasons x neighbours) in
    full-table scans.
    """
    hist = df[df["PLAYER_ID"].isin(player_ids)][
        ["PLAYER_ID", "Season_No", metric]
    ].sort_values(["PLAYER_ID", "Season_No"])

    grouped = hist.groupby("PLAYER_ID")
    hist = hist.assign(
        prev_value=grouped[metric].shift(1),
        prev_season=grouped["Season_No"].shift(1),
    )
    # Only true consecutive seasons count as a year-over-year change.
    consecutive = hist["prev_season"] == hist["Season_No"] - 1
    hist = hist[consecutive].copy()
    hist["delta"] = hist[metric] - hist["prev_value"]
    return hist[["PLAYER_ID", "Season_No", "delta"]]


def project_metric(
    df: pd.DataFrame,
    match: NeighborMatch,
    metric: str,
    cutoff: int | None = None,
    min_support: int = 1,
) -> Projection:
    """Project ``metric`` forward for the player in ``match``.

    ``cutoff`` is the career season the projection starts from, defaulting to
    the number of seasons the comparison was built on. Setting it below a
    retired player's career length turns the projection into a backtest: the
    actual path is already known and can be laid over it.

    ``min_support`` stops the path once fewer than that many comparison players
    still have data. With the default of 1 a long projection can end up riding
    on a single peer's career; raising it to 2 or 3 trades length for evidence.
    """
    cutoff = cutoff or match.n_seasons
    target = df[df["PLAYER_ID"] == match.target_id]
    anchor_rows = target[target["Season_No"] == cutoff]
    if anchor_rows.empty:
        raise ValueError(
            f"Player has no season {cutoff} to project from "
            f"(career is {int(target['Season_No'].max())} seasons)."
        )
    anchor_value = float(anchor_rows.iloc[0][metric])
    if pd.isna(anchor_value):
        raise ValueError(
            f"{metric} is missing for season {cutoff}, so there is no value to "
            "project from. Pick a different cutoff or metric."
        )

    empty = Projection(match.method, metric, cutoff, anchor_value, [], [], [], [])
    if len(match) == 0:
        return empty

    weights = pd.Series(
        inverse_distance_weights(match.distances), index=match.ids, name="weight"
    )
    deltas = _delta_table(df, match.ids, metric)
    if deltas.empty:
        return empty
    deltas = deltas.join(weights, on="PLAYER_ID").dropna(subset=["delta", "weight"])

    seasons: list[int] = []
    values: list[float] = []
    spread: list[float] = []
    counts: list[int] = []

    running = anchor_value
    variance = 0.0
    last_season = int(deltas["Season_No"].max())

    for season in range(cutoff + 1, last_season + 1):
        step = deltas[deltas["Season_No"] == season]
        if len(step) < max(1, min_support):
            # Too few comparison players still have a change for this season:
            # stop here rather than inventing one.
            break

        w = step["weight"].to_numpy()
        w = w / w.sum()
        d = step["delta"].to_numpy()

        mean_delta = float(np.dot(w, d))
        # Weighted variance of the peer changes, accumulated in quadrature.
        # Treats each season's change as independent, which overstates the
        # band's precision but gives an honest sense of how much the
        # comparisons disagree.
        variance += float(np.dot(w, (d - mean_delta) ** 2))

        running += mean_delta
        seasons.append(season)
        values.append(running)
        spread.append(float(np.sqrt(variance)))
        counts.append(len(step))

    return Projection(
        method=match.method,
        metric=metric,
        anchor_season_no=cutoff,
        anchor_value=anchor_value,
        seasons=seasons,
        values=values,
        spread=spread,
        neighbors_used=counts,
    )


def peer_percentiles(
    df: pd.DataFrame, metric: str, n_seasons: int | None = None
) -> pd.DataFrame:
    """League-wide 25th/50th/75th percentile of ``metric`` by career season.

    Context for a projected path: whether it is tracking above or below what a
    typical career looks like at the same stage.
    """
    frame = df if n_seasons is None else df[df["Season_No"] <= n_seasons]
    pct = (
        frame.groupby("Season_No")[metric]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
        .sort_index()
    )
    return pct.rename(columns={0.25: "p25", 0.50: "p50", 0.75: "p75"})
