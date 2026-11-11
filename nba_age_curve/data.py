"""Loading and shaping the player-season table.

Everything downstream keys on ``PLAYER_ID``, never on ``PLAYER_NAME``: 25 names
in the source data are shared by two or three different players (there are two
Eddie Johnsons, three Charles Smiths), and grouping by name would splice them
into single impossible careers.
"""

from __future__ import annotations

import pandas as pd

from nba_age_curve.config import (
    DATASET_PATH,
    FEATURES,
    FIRST_SEASON,
    PRE_1980_PATH,
)


def season_end_year(season: pd.Series) -> pd.Series:
    """Convert ``"1979-80"`` season labels to the calendar year they end in."""
    return season.str.slice(0, 4).astype(int) + 1


def add_season_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Season_No``: the 1-based index of each season within a career.

    This counts *seasons played*, not years elapsed. A player who misses a full
    season has that gap collapsed, so season 5 means "fifth season on the
    floor". Career trajectories are compared on this index.
    """
    df = df.sort_values(["PLAYER_ID", "SEASON"]).copy()
    df["Season_No"] = df.groupby("PLAYER_ID").cumcount() + 1
    return df


def flag_left_censored(df: pd.DataFrame) -> pd.DataFrame:
    """Mark players whose careers started before the data does.

    The source data begins at 1979-80, so for a player already in the league by
    then, ``Season_No`` understates their true experience. Two signals are
    combined: a curated list of confirmed pre-1980 debuts, plus the fallback
    heuristic that *any* player whose first observed season is 1979-80 may have
    debuted earlier. Flagged players are kept, not dropped -- the app labels
    them wherever they turn up as a comparison.
    """
    first_season = df.groupby("PLAYER_ID")["SEASON"].transform("min")
    censored = first_season == FIRST_SEASON

    if PRE_1980_PATH.exists():
        known = set(pd.read_csv(PRE_1980_PATH)["PLAYER_NAME"])
        censored |= df["PLAYER_NAME"].isin(known)

    df = df.copy()
    df["left_censored"] = censored
    return df


def add_player_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``label``: a display name that is unique per ``PLAYER_ID``.

    Players with a name all to themselves keep it as-is. Shared names get their
    season span appended, so the dropdown can tell them apart.
    """
    spans = df.groupby("PLAYER_ID").agg(
        PLAYER_NAME=("PLAYER_NAME", "first"),
        first=("SEASON", "min"),
        last=("SEASON", "max"),
    )
    shared = spans["PLAYER_NAME"].duplicated(keep=False)
    spans["label"] = spans["PLAYER_NAME"].where(
        ~shared,
        spans["PLAYER_NAME"] + " (" + spans["first"] + " – " + spans["last"] + ")",
    )
    return df.merge(
        spans[["label"]], left_on="PLAYER_ID", right_index=True, how="left"
    )


def load_player_seasons(path=None) -> pd.DataFrame:
    """Load the baked dataset, ready for PCA.

    Returns one row per player-season, sorted by player then season, with
    ``Season_No``, ``left_censored`` and ``label`` added and the PCA features
    filled. Blank ratios in the source are genuine 0/0 cases -- a player with no
    assists and no turnovers has no ``AST/TO`` -- and are filled with 0, which
    is the convention the original analysis used.
    """
    path = path or DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/build_dataset.py` first."
        )

    df = pd.read_parquet(path)
    df[FEATURES] = df[FEATURES].fillna(0)
    df["SEASON_YEAR"] = season_end_year(df["SEASON"])
    df = add_season_index(df)
    df = flag_left_censored(df)
    df = add_player_labels(df)
    return df.reset_index(drop=True)


def career_lengths(df: pd.DataFrame) -> pd.Series:
    """Total seasons played, indexed by ``PLAYER_ID``."""
    return df.groupby("PLAYER_ID")["Season_No"].max()


def first_n_seasons(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rows covering the first ``n`` seasons of every player who played ``n``.

    Restricting to players who reached season ``n`` is what makes the comparison
    fair: a target five seasons into their career is matched against other
    players' first five seasons, not against careers that ended after two.
    """
    lengths = career_lengths(df)
    eligible = lengths.index[lengths >= n]
    return df[df["PLAYER_ID"].isin(eligible) & (df["Season_No"] <= n)]


def aggregate_careers(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Collapse player-seasons to one row per player, averaging ``value_cols``.

    This is the player-level view the season-level table cannot provide, and
    what the "standard" matcher runs PCA over.
    """
    present = [c for c in value_cols if c in df.columns]
    agg = {c: "mean" for c in present}
    agg["Season_No"] = "max"
    out = df.groupby("PLAYER_ID").agg(agg)
    meta = df.groupby("PLAYER_ID").agg(
        PLAYER_NAME=("PLAYER_NAME", "first"),
        label=("label", "first"),
        left_censored=("left_censored", "max"),
    )
    return meta.join(out).rename(columns={"Season_No": "Seasons"}).reset_index()


def player_directory(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player for the search dropdown, most recent players first."""
    latest_season = df["SEASON"].max()
    directory = df.groupby("PLAYER_ID").agg(
        label=("label", "first"),
        PLAYER_NAME=("PLAYER_NAME", "first"),
        first_season=("SEASON", "min"),
        last_season=("SEASON", "max"),
        seasons=("Season_No", "max"),
        left_censored=("left_censored", "max"),
    )
    directory["is_active"] = directory["last_season"] == latest_season
    return directory.sort_values(
        ["is_active", "last_season", "label"], ascending=[False, False, True]
    ).reset_index()
