"""Backtest every forecasting method against held-out players.

The test is deliberately strict: players are split by *identity*, so the GAMM
never sees a test player during fitting, and the k-NN methods may only draw
comparisons from training players. Each method sees a player's first ``--cutoff``
seasons and forecasts the rest.

    python scripts/backtest.py
    python scripts/backtest.py --cutoff 8 --seeds 1 2 3

Two baselines are included on purpose. "Flat" carries the last observed value
forward and "League-avg delta" applies the population year-over-year change. A
forecaster that cannot beat both is not earning its complexity.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nba_age_curve.config import GAMM_SMOOTHS, N_COMPONENTS  # noqa: E402
from nba_age_curve.data import load_player_seasons  # noqa: E402
from nba_age_curve.gamm import GAMM, Smooth  # noqa: E402
from nba_age_curve.pca import attach_scores, fit_season_pca  # noqa: E402
from nba_age_curve.projection import match_weights, project_metric  # noqa: E402
from nba_age_curve.similarity import (  # noqa: E402
    CENTROID,
    EPSILON,
    STANDARD,
    match_centroid_knn,
    match_epsilon,
    match_standard_knn,
)


def run_split(df, cutoff: int, seed: int, metric: str, fit_k: int = 80) -> pd.DataFrame:
    lengths = df.groupby("PLAYER_ID")["Season_No"].max()
    last_year = df.groupby("PLAYER_ID")["SEASON_YEAR"].max()
    # Retired players only: their futures are known, so they can be scored.
    eligible = sorted(
        set(lengths[lengths >= cutoff + 1].index)
        & set(last_year[last_year < df["SEASON_YEAR"].max()].index)
    )
    rng = np.random.default_rng(seed)
    test = set(rng.choice(eligible, size=int(0.2 * len(eligible)), replace=False).tolist())
    train_ids = set(df["PLAYER_ID"].unique()) - test

    train = df[df["PLAYER_ID"].isin(train_ids)]
    model = GAMM([Smooth(n, k) for n, k in GAMM_SMOOTHS]).fit(
        train, train[metric].to_numpy(), train["PLAYER_ID"].to_numpy()
    )

    deltas = train.sort_values(["PLAYER_ID", "Season_No"]).copy()
    deltas["d"] = deltas.groupby("PLAYER_ID")[metric].diff()
    league_delta = deltas.groupby("Season_No")["d"].mean()

    rows = []
    for pid in sorted(test):
        g = df[df["PLAYER_ID"] == pid].sort_values("Season_No")
        hist, future = g[g.Season_No <= cutoff], g[g.Season_No > cutoff]
        if hist.empty or future.empty:
            continue
        truth = dict(zip(future["Season_No"], future[metric]))

        mean, sd = model.forecast(hist, future, y_col=metric)
        for s, m_, sd_, actual in zip(future["Season_No"], mean, sd, future[metric]):
            rows.append(("GAMM", pid, s, actual, m_, sd_))

        anchor = float(hist.iloc[-1][metric])
        running = anchor
        for s, actual in truth.items():
            running += league_delta.get(s, 0.0)
            rows.append(("Flat (carry last)", pid, s, actual, anchor, np.nan))
            rows.append(("League-avg delta", pid, s, actual, running, np.nan))

        pool = df[df["PLAYER_ID"].isin(train_ids | {pid})]
        train_pid = train["PLAYER_ID"].to_numpy()
        train_y = train[metric].to_numpy()
        for label, fn, kw in (
            (STANDARD, match_standard_knn, {}),
            (CENTROID, match_centroid_knn, {}),
            (EPSILON, match_epsilon, {"epsilon": 0.9}),
        ):
            try:
                match = fn(pool, int(pid), cutoff, **kw)
                if len(match) == 0:
                    continue
                proj = project_metric(pool, match, metric, cutoff=cutoff, min_support=1)
            except (ValueError, KeyError):
                continue
            for s, v in zip(proj.seasons, proj.values):
                if s in truth:
                    rows.append((f"{label} -> delta", pid, s, truth[s], v, np.nan))

            # The same comparison set and weights, fed to the GAMM instead.
            # Epsilon defines its own set by radius; the k-NN methods are
            # widened to fit_k, since five players cannot support the basis.
            try:
                fit_match = (
                    match
                    if label == EPSILON
                    else fn(pool, int(pid), cutoff, k=fit_k, **kw)
                )
                weights = match_weights(fit_match)
                row_w = pd.Series(train_pid).map(weights).fillna(0.0).to_numpy()
                if row_w.sum() <= 0:
                    continue
                local = model.refit_weighted(train, train_y, train_pid, row_w)
                mean, sd = local.forecast(hist, future, y_col=metric)
            except (ValueError, KeyError, np.linalg.LinAlgError):
                continue
            for s, m_, sd_, actual in zip(
                future["Season_No"], mean, sd, future[metric]
            ):
                rows.append((f"{label} -> GAMM", pid, s, actual, m_, sd_))

    out = pd.DataFrame(rows, columns=["method", "PLAYER_ID", "Season_No",
                                      "actual", "pred", "sd"])
    out["cutoff"], out["seed"] = cutoff, seed
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", type=int, nargs="+", default=[3, 5, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    ap.add_argument("--metric", default="ws")
    ap.add_argument("--fit-k", type=int, default=80,
                    help="Comparisons used in the weighted GAMM fit (default 80).")
    args = ap.parse_args()

    df = load_player_seasons()
    df = attach_scores(df, fit_season_pca(df), N_COMPONENTS)
    df = df.dropna(subset=[args.metric])

    frames = [run_split(df, c, s, args.metric, args.fit_k)
              for c in args.cutoff for s in args.seeds]
    res = pd.concat(frames, ignore_index=True)
    res["err"] = res["pred"] - res["actual"]
    res["horizon"] = res["Season_No"] - res["cutoff"]

    print(f"\n=== OVERALL ({len(args.cutoff)} cutoffs x {len(args.seeds)} seeds) ===")
    summary = (
        res.groupby("method")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "MAE": g.err.abs().mean(),
            "RMSE": float(np.sqrt((g.err ** 2).mean())),
            "bias": g.err.mean(),
        }), include_groups=False)
        .sort_values("MAE")
    )
    print(summary.round(3).to_string())

    print("\n=== MAE by horizon (seasons ahead) ===")
    print(
        res[res.horizon <= 6]
        .pivot_table(index="method", columns="horizon", values="err",
                     aggfunc=lambda e: e.abs().mean())
        .round(2)
        .to_string()
    )

    gamm = res[(res.method == "GAMM") & res.sd.notna()]
    if len(gamm):
        z = (gamm.err / gamm.sd).to_numpy()
        print("\n=== GAMM interval calibration ===")
        for level, k in ((0.50, 0.6745), (0.80, 1.2816), (0.95, 1.9600)):
            print(f"  nominal {level:.0%}  ->  actual {np.mean(np.abs(z) <= k):.1%}")


if __name__ == "__main__":
    main()
