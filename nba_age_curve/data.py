"""Loading and shaping the player-season table.

Everything downstream keys on ``PLAYER_ID``, never on ``PLAYER_NAME``: 25 names
in the source data are shared by two or three different players (there are two
Eddie Johnsons, three Charles Smiths), and grouping by name would splice them
into single impossible careers.
"""

from __future__ import annotations

import pandas as pd

from nba_age_curve.config import DATASET_PATH, DEBUT_PATH, FEATURES


def season_end_year(season: pd.Series) -> pd.Series:
    """Convert ``"1979-80"`` season labels to the calendar year they end in."""
    return season.str.slice(0, 4).astype(int) + 1


def load_debut_years() -> pd.DataFrame | None:
    """Each player's true first professional season, if the table is present.

    Built by ``scripts/build_dataset.py --master`` from the master workbook's
    ``year_min``, which records the real debut including ABA seasons.
    """
    if not DEBUT_PATH.exists():
        return None
    return pd.read_csv(DEBUT_PATH)


def add_season_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Season_No``: the 1-based *career year* of each season.

    This counts years since the player's true professional debut, not rows in
    this dataset. The distinction matters because the data starts at 1979-80:
    counting rows would make Kareem Abdul-Jabbar's 1979-80 his rookie season
    when it was his eleventh, and would line him up against actual rookies.

    Two consequences follow. A player who misses a season leaves a gap in the
    index rather than having it collapsed, which is the honest representation
    and which the projection code already handles (it only differences across
    genuinely consecutive years). And a player who debuted before the data
    begins simply has no low career years, so they drop out of early-career
    comparisons on their own instead of polluting them.

    Where no debut year is known, the first observed season is used, which
    reduces to the old row-counting behaviour.
    """
    df = df.sort_values(["PLAYER_ID", "SEASON"]).copy()
    debuts = load_debut_years()
    first_seen = df.groupby("PLAYER_ID")["SEASON_YEAR"].transform("min")

    if debuts is None:
        debut_year = first_seen
    else:
        mapped = df["PLAYER_ID"].map(
            debuts.set_index("PLAYER_ID")["debut_year"]
        )
        debut_year = mapped.fillna(first_seen).astype(int)

    df["debut_year"] = debut_year
    df["Season_No"] = df["SEASON_YEAR"] - debut_year + 1
    return df


def flag_left_censored(df: pd.DataFrame) -> pd.DataFrame:
    """Mark players whose careers began before this dataset does.

    With true debut years available this is exact rather than heuristic: a
    player is censored when they debuted before their first season here, so
    their early career is missing however it is indexed. They are kept, not
    dropped -- their later seasons are correctly placed on the career-year
    axis, and the app labels them wherever they appear.
    """
    df = df.copy()
    first_seen = df.groupby("PLAYER_ID")["SEASON_YEAR"].transform("min")
    df["left_censored"] = df["debut_year"] < first_seen
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
    """Career span in years, indexed by ``PLAYER_ID``.

    This is the last career year observed, so for a player whose early seasons
    predate the dataset it exceeds the number of seasons present here.
    """
    return df.groupby("PLAYER_ID")["Season_No"].max()


def first_n_seasons(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rows covering career years 1..``n`` for players observed over all of them.

    Two conditions, both needed for a fair comparison. The player must have
    *reached* career year ``n``, so a target five years in is not matched
    against careers that ended after two. And the player must be observed from
    career year 1, which excludes anyone who debuted before this dataset
    begins -- their early seasons are missing, so there is nothing to compare
    against however the index is computed.
    """
    span = df.groupby("PLAYER_ID")["Season_No"].agg(["min", "max"])
    eligible = span.index[(span["min"] == 1) & (span["max"] >= n)]
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
        observed=("SEASON", "size"),
        left_censored=("left_censored", "max"),
    )
    directory["is_active"] = directory["last_season"] == latest_season
    return directory.sort_values(
        ["is_active", "last_season", "label"], ascending=[False, False, True]
    ).reset_index()
