"""Fetch the source spreadsheet and bake it into the parquet the app reads.

The app deliberately does not pull from Google Sheets at runtime: a public
dashboard should not depend on a third-party URL being reachable on every cold
start, and the data only changes when the season does. Re-run this script to
refresh:

    python scripts/build_dataset.py

Pass ``--source path/to/file.xlsx`` to build from a local copy instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nba_age_curve.config import (  # noqa: E402
    DATASET_PATH,
    DEBUT_PATH,
    FEATURES,
    IDENTITY_COLS,
    PROJECTABLE_METRICS,
    SHEET_EXPORT_URL,
)


def build_debut_years(df: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    """Derive each player's true first professional season.

    The source spreadsheet starts at 1979-80, so a player already in the league
    by then looks like a rookie. The master workbook carries ``year_min``, the
    real debut year (including ABA), which fixes that.

    Joining is on name and season, not age: the two sources disagree about age
    by a year on roughly 40% of rows. Names are not unique either -- there are
    two Eddie Johnsons and two Mike Dunleavys -- so candidate debuts are
    filtered to those at or before a player's first observed season, and the
    latest surviving candidate is taken. That picks the right player of the
    pair, since the younger one debuted later.
    """
    master = pd.read_excel(master_path, usecols=["player", "season", "year_min"])
    master = master.rename(
        columns={"player": "PLAYER_NAME", "season": "SEASON_YEAR"}
    ).dropna()
    master["year_min"] = master["year_min"].astype(int)

    first_seen = df.groupby("PLAYER_ID")["SEASON_YEAR"].min()
    joined = df[["PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR"]].merge(
        master, on=["PLAYER_NAME", "SEASON_YEAR"], how="inner"
    )
    joined["first_seen"] = joined["PLAYER_ID"].map(first_seen)
    # A debut cannot postdate the first season we observe.
    joined = joined[joined["year_min"] <= joined["first_seen"]]
    derived = joined.groupby("PLAYER_ID")["year_min"].max()

    out = pd.DataFrame({"PLAYER_ID": first_seen.index})
    out["debut_year"] = out["PLAYER_ID"].map(derived)
    out["source"] = np.where(out["debut_year"].notna(), "master", "first_observed")
    out["debut_year"] = (
        out["debut_year"].fillna(out["PLAYER_ID"].map(first_seen)).astype(int)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=SHEET_EXPORT_URL,
        help="Spreadsheet URL or local path (default: the published Google Sheet).",
    )
    parser.add_argument(
        "--out", type=Path, default=DATASET_PATH, help="Destination parquet file."
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=None,
        help=(
            "Optional path to nba_age_curve_master.xlsx. When given, true debut "
            f"years are re-derived from it and written to {DEBUT_PATH.name}."
        ),
    )
    args = parser.parse_args()

    print(f"Reading {args.source}")
    df = pd.read_excel(args.source)

    wanted = IDENTITY_COLS + FEATURES + list(PROJECTABLE_METRICS)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise SystemExit(f"Source is missing expected columns: {missing}")

    df = df[wanted].copy()

    # One row per player-season is the assumption every downstream step makes.
    duplicates = df.duplicated(subset=["PLAYER_ID", "SEASON"]).sum()
    if duplicates:
        raise SystemExit(
            f"{duplicates} duplicate PLAYER_ID/SEASON rows in the source; "
            "resolve these before building."
        )

    df = df.sort_values(["PLAYER_ID", "SEASON"]).reset_index(drop=True)

    if args.master is not None:
        df["SEASON_YEAR"] = df["SEASON"].str.slice(0, 4).astype(int) + 1
        debuts = build_debut_years(df, args.master)
        df = df.drop(columns="SEASON_YEAR")
        DEBUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        debuts.to_csv(DEBUT_PATH, index=False)
        known = (debuts["source"] == "master").sum()
        print(
            f"Wrote {DEBUT_PATH} -- {known:,} of {len(debuts):,} debut years from "
            f"the master workbook, the rest from first observed season"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")

    size_mb = args.out.stat().st_size / 1e6
    print(
        f"Wrote {args.out} -- {len(df):,} player-seasons, "
        f"{df['PLAYER_ID'].nunique():,} players, "
        f"{df['SEASON'].min()} to {df['SEASON'].max()} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
