"""Shared configuration: paths, the PCA feature set, and model defaults."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"

DATASET_PATH = DATA_DIR / "player_seasons.parquet"
DEBUT_PATH = DATA_DIR / "debut_years.csv"

# Source of truth for the raw data. `scripts/build_dataset.py` turns this into
# the parquet file that the app actually reads.
SHEET_ID = "1e44hf4TYKM1inD3OJ672Zoi8nmm8sxqRogvAbmJAnSA"
SHEET_EXPORT_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
)

#: The 12 per-minute / ratio stats that define a player's style. Every one is
#: z-scored before PCA so each counts equally.
FEATURES = [
    "MIN/GM",
    "PTS/MIN",
    "FGA / MIN",
    "AST/MIN",
    "TOV/MIN",
    "AST/TO",
    "OREB/MIN",
    "DREB/MIN",
    "BLK/MIN",
    "FTr",
    "STL/PF",
    "3PAr",
]

#: Columns carried through from the source data but never used as PCA inputs.
IDENTITY_COLS = ["PLAYER_ID", "PLAYER_NAME", "SEASON", "Age"]

#: Components retained. The source data yields 4 components with eigenvalue > 1
#: (Kaiser rule); see `scripts/analysis.py` for the scree plot that shows this.
N_COMPONENTS = 4
PC_COLS = [f"PC{i}" for i in range(1, N_COMPONENTS + 1)]

#: Outcome metrics that can be projected forward. Keys are source columns.
PROJECTABLE_METRICS = {
    "ws": "Win Shares",
    "ws_per_48": "Win Shares / 48 min",
    "ows": "Offensive Win Shares",
    "dws": "Defensive Win Shares",
    "bpm": "Box Plus/Minus",
    "obpm": "Offensive BPM",
    "dbpm": "Defensive BPM",
    "vorp": "VORP",
    "usg_pct": "Usage %",
}
DEFAULT_METRIC = "ws"

#: Metrics that are rates rather than season totals. These are averaged over a
#: career, never summed -- a career "total" usage percentage is meaningless.
RATE_METRICS = {"ws_per_48", "bpm", "obpm", "dbpm", "usg_pct"}

DEFAULT_N_NEIGHBORS = 5
DEFAULT_EPSILON = 0.9

#: The first season present in the source data. Players already in the league
#: before this point have truncated careers (see `data.flag_left_censored`).
FIRST_SEASON = "1979-80"
