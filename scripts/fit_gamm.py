"""Fit the GAMM once and save it, so the app never fits at runtime.

    python scripts/fit_gamm.py

Writes ``data/gamm_model.npz`` (a few hundred KB: knots, coefficients and
variance components). The app loads that and only evaluates basis functions,
which is fast enough to run per keystroke.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nba_age_curve.config import GAMM_PATH, GAMM_SMOOTHS, N_COMPONENTS  # noqa: E402
from nba_age_curve.data import load_player_seasons  # noqa: E402
from nba_age_curve.gamm import GAMM, Smooth  # noqa: E402
from nba_age_curve.pca import attach_scores, fit_season_pca  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=GAMM_PATH)
    ap.add_argument("--metric", default="ws")
    args = ap.parse_args()

    df = load_player_seasons()
    df = attach_scores(df, fit_season_pca(df), N_COMPONENTS)
    df = df.dropna(subset=[args.metric])
    print(f"fitting on {len(df):,} player-seasons, {df['PLAYER_ID'].nunique():,} players")

    model = GAMM([Smooth(name, k) for name, k in GAMM_SMOOTHS])
    model.fit(df, df[args.metric].to_numpy(), df["PLAYER_ID"].to_numpy())

    print(f"  sigma_b = {model.sigma_b_:.3f}  (spread of player levels)")
    print(f"  sigma_e = {model.sigma_e_:.3f}  (season-to-season noise)")
    print(f"  lambdas = {np.round(model.lambdas_, 1)}")

    # The headline diagnostic: the within-player aging curve, which should
    # decline far more steeply than the survivor-biased raw averages.
    ages = np.arange(21, 41)
    grid = pd.DataFrame(
        {
            "Age": ages,
            "Season_No": np.clip(ages - 22, 1, 20),
            "SEASON_YEAR": 2015,
            **{f"PC{i}": 0.0 for i in range(1, N_COMPONENTS + 1)},
        }
    )
    curve = model.predict_fixed(grid)
    curve -= curve.max()
    raw = df.groupby("Age")[args.metric].mean().reindex(ages)
    raw -= raw.max()
    print(f"\n  age          {'  '.join(f'{a:>5}' for a in ages[::3])}")
    print(f"  raw mean     {'  '.join(f'{raw.values[i]:5.1f}' for i in range(0, len(ages), 3))}  <- survivor-biased")
    print(f"  within-player{'  '.join(f'{curve[i]:5.1f}' for i in range(0, len(ages), 3))}")
    print(f"  peak age: {ages[int(np.argmax(curve))]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
