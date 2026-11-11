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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nba_age_curve.config import (  # noqa: E402
    DATASET_PATH,
    FEATURES,
    IDENTITY_COLS,
    PROJECTABLE_METRICS,
    SHEET_EXPORT_URL,
)


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
