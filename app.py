"""Streamlit dashboard: find a player's closest comparisons and project forward.

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nba_age_curve.config import (
    DEFAULT_EPSILON,
    DEFAULT_METRIC,
    DEFAULT_N_NEIGHBORS,
    N_COMPONENTS,
    PROJECTABLE_METRICS,
    RATE_METRICS,
)
from nba_age_curve.config import GAMM_METHOD, GAMM_PATH, PC_COLS
from nba_age_curve.data import load_player_seasons, player_directory
from nba_age_curve.gamm import GAMM
from nba_age_curve.pca import attach_scores, fit_season_pca
from nba_age_curve.projection import peer_percentiles, project_metric
from nba_age_curve.similarity import (
    CENTROID,
    EPSILON,
    STANDARD,
    match_centroid_knn,
    match_epsilon,
    match_standard_knn,
)

st.set_page_config(
    page_title="NBA Career Projections",
    page_icon="\N{BASKETBALL AND HOOP}",
    layout="wide",
)

METHOD_COLORS = {
    STANDARD: "#4C72B0",
    CENTROID: "#DD8452",
    EPSILON: "#55A868",
    GAMM_METHOD: "#8172B2",
}
METHOD_DASHES = {
    STANDARD: "dash",
    CENTROID: "dot",
    EPSILON: "dashdot",
    GAMM_METHOD: "solid",
}

#: Mean absolute error on held-out retired players, from the backtest in
#: `scripts/backtest.py`. Shown in the UI so the methods are not presented as
#: equally trustworthy when they measurably are not.
BACKTEST_MAE = {
    GAMM_METHOD: 1.33,
    EPSILON: 2.37,
    STANDARD: 2.53,
    CENTROID: 2.55,
}


def rgba(hex_color: str, alpha: float) -> str:
    """``"#4C72B0", 0.15`` -> ``"rgba(76,114,176,0.15)"`` for Plotly fills."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------
# Data and model (built once per server process)
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading player-seasons and fitting PCA...")
def get_model_and_data():
    """Load the dataset and fit the global season-level PCA exactly once."""
    df = load_player_seasons()
    model = fit_season_pca(df)
    return attach_scores(df, model), model


@st.cache_data(show_spinner=False)
def get_directory() -> pd.DataFrame:
    df, _ = get_model_and_data()
    return player_directory(df)


@st.cache_data(show_spinner=False)
def get_match(method: str, player_id: int, n_seasons: int, k: int, epsilon: float):
    """Comparison set for one player. Cached on the arguments that define it."""
    df, _ = get_model_and_data()
    if method == STANDARD:
        return match_standard_knn(df, player_id, n_seasons, k=k)
    if method == CENTROID:
        return match_centroid_knn(df, player_id, n_seasons, k=k)
    return match_epsilon(df, player_id, n_seasons, epsilon=epsilon)


@st.cache_resource(show_spinner=False)
def get_gamm():
    """The pre-fitted GAMM, or None if `scripts/fit_gamm.py` has not been run."""
    if not GAMM_PATH.exists():
        return None
    return GAMM.load(GAMM_PATH)


@st.cache_data(show_spinner=False)
def get_percentiles(metric: str) -> pd.DataFrame:
    df, _ = get_model_and_data()
    return peer_percentiles(df, metric)


df, pca_model = get_model_and_data()
directory = get_directory()
label_by_id = dict(zip(directory["PLAYER_ID"], directory["label"]))
latest_season = df["SEASON"].max()


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    metric = st.selectbox(
        "Metric to project",
        options=list(PROJECTABLE_METRICS),
        index=list(PROJECTABLE_METRICS).index(DEFAULT_METRIC),
        format_func=lambda c: PROJECTABLE_METRICS[c],
    )

    st.subheader("Forecast methods")
    gamm_model = get_gamm()
    use_gamm = st.checkbox(
        f"{GAMM_METHOD} (MAE {BACKTEST_MAE[GAMM_METHOD]})",
        value=gamm_model is not None,
        disabled=gamm_model is None,
        help=(
            "Additive model: smooth curves in age, experience, era and playing "
            "style, plus a per-player random intercept. Only available for the "
            "win-shares metric it was fit on."
        ),
    )
    if gamm_model is None:
        st.caption("Run `python scripts/fit_gamm.py` to enable the GAMM.")

    st.caption("Comparison-based (MAE in brackets):")
    use_standard = st.checkbox(f"{STANDARD} ({BACKTEST_MAE[STANDARD]})", value=True)
    use_centroid = st.checkbox(f"{CENTROID} ({BACKTEST_MAE[CENTROID]})", value=True)
    use_epsilon = st.checkbox(f"{EPSILON} ({BACKTEST_MAE[EPSILON]})", value=True)

    gamm_horizon = st.slider(
        "GAMM seasons to project",
        min_value=1,
        max_value=15,
        value=8,
        help="The GAMM projects to a chosen horizon or age 40, whichever comes first.",
    )

    k = st.slider(
        "Comparisons per k-NN method",
        min_value=3,
        max_value=15,
        value=DEFAULT_N_NEIGHBORS,
        help="How many nearest players the two k-NN methods return.",
    )
    epsilon = st.slider(
        "ε radius",
        min_value=0.2,
        max_value=3.0,
        value=float(DEFAULT_EPSILON),
        step=0.1,
        help=(
            "The ε-neighbourhood returns every player within this distance "
            "in component space, however many that is."
        ),
    )
    min_support = st.slider(
        "Minimum comparisons per projected season",
        min_value=1,
        max_value=5,
        value=2,
        help=(
            "Stop a projection once fewer than this many comparison players "
            "still have data. Higher values give shorter but better-evidenced "
            "projections."
        ),
    )

    show_band = st.checkbox(
        "Show disagreement band",
        value=True,
        help="±1 SD of the comparison players' year-over-year changes.",
    )
    show_peers = st.checkbox("Show comparison players' careers", value=False)
    show_league = st.checkbox("Show league percentiles", value=True)

    st.divider()
    st.caption(
        f"{len(df):,} player-seasons — {df['PLAYER_ID'].nunique():,} players, "
        f"{df['SEASON'].min()} to {latest_season}. "
        f"PCA on {len(pca_model.features)} style stats, "
        f"{N_COMPONENTS} components "
        f"({pca_model.explained_variance_ratio[:N_COMPONENTS].sum():.0%} of variance)."
    )


# --------------------------------------------------------------------------
# Player picker
# --------------------------------------------------------------------------

st.title("NBA Career Projections")
st.markdown(
    "Pick a player to see the most stylistically similar careers in the modern "
    f"era, and where four methods project their "
    f"**{PROJECTABLE_METRICS[metric].lower()}** going forward. The bracketed "
    "numbers are each method's mean absolute error on held-out players — "
    "lower is better, and they are not close."
)

preferred = directory.loc[directory["PLAYER_NAME"] == "Nikola Jokic", "PLAYER_ID"]
default_id = int(preferred.iloc[0] if len(preferred) else directory["PLAYER_ID"].iloc[0])

player_id = st.selectbox(
    "Player",
    options=directory["PLAYER_ID"].tolist(),
    index=directory["PLAYER_ID"].tolist().index(default_id),
    format_func=lambda pid: label_by_id[pid],
    help="Start typing to filter. Active players are listed first.",
)

player = directory[directory["PLAYER_ID"] == player_id].iloc[0]
history = df[df["PLAYER_ID"] == player_id].sort_values("Season_No")
career_length = int(player["seasons"])
is_active = bool(player["is_active"])

st.caption(
    f"**{player['PLAYER_NAME']}** — {player['first_season']} to "
    f"{player['last_season']}"
)

cols = st.columns(4)
cols[0].metric("Seasons played", career_length)
cols[1].metric("Status", "Active" if is_active else "Retired")
cols[2].metric("Age range", f"{int(history['Age'].min())}–{int(history['Age'].max())}")
# Totals only make sense for counting stats; rates are averaged instead.
if metric in RATE_METRICS:
    cols[3].metric(f"Average {metric}", f"{history[metric].mean():.2f}")
else:
    cols[3].metric(f"Career {metric}", f"{history[metric].sum():.1f}")

if player["left_censored"]:
    st.warning(
        f"**{player['PLAYER_NAME']} was already in the league before 1979-80**, "
        "where this dataset begins. Their earlier seasons are missing, so "
        '"career season 1" here is not their real rookie year and the '
        "comparisons are matched against the wrong stage of their career.",
        icon="⚠️",
    )

# How many seasons the comparison is built on, and where projection starts.
if is_active:
    default_cutoff = career_length
    cutoff_help = (
        "Comparisons are built on the player's first N seasons, and the "
        "projection starts from season N. Lower it to see what the model "
        "would have said earlier."
    )
else:
    default_cutoff = max(2, career_length // 2)
    cutoff_help = (
        "This player is retired, so the projection is a backtest: it runs "
        "from the season you choose, and their actual career is drawn over it."
    )

if career_length > 1:
    cutoff = st.slider(
        "Build comparisons from the first N career seasons",
        min_value=1,
        max_value=career_length,
        value=default_cutoff,
        help=cutoff_help,
    )
else:
    cutoff = 1
    st.caption("This player has a single season, so comparisons use season 1.")

if not is_active and cutoff == career_length:
    st.info(
        "This player is retired and the cutoff is their final season, so there "
        "is nothing left to project. Lower the cutoff to backtest the model "
        "against what actually happened.",
        icon="ℹ️",
    )


# --------------------------------------------------------------------------
# Comparisons and projections
# --------------------------------------------------------------------------

enabled = [
    m
    for m, on in ((STANDARD, use_standard), (CENTROID, use_centroid), (EPSILON, use_epsilon))
    if on
]

show_gamm = bool(use_gamm and gamm_model is not None)
if show_gamm and metric != DEFAULT_METRIC:
    st.info(
        f"The GAMM was fit on {PROJECTABLE_METRICS[DEFAULT_METRIC].lower()}, so "
        f"it is not shown for {PROJECTABLE_METRICS[metric].lower()}. Re-run "
        f"`scripts/fit_gamm.py --metric {metric}` to fit it for this one.",
        icon="ℹ️",
    )
    show_gamm = False

if not enabled and not show_gamm:
    st.error("Enable at least one forecast method in the sidebar.")
    st.stop()

# The GAMM needs no comparison set: it carries the player's own level forward
# along a population aging curve, so it is computed on its own.
gamm_forecast = None
if show_gamm:
    hist = history[history["Season_No"] <= cutoff]
    future = GAMM.future_frame(hist, n_ahead=gamm_horizon, pc_cols=PC_COLS)
    if len(future):
        mean, sd = gamm_model.forecast(hist, future, y_col=metric)
        gamm_forecast = (future["Season_No"].to_numpy(), mean, sd)
    else:
        st.info(
            f"{player['PLAYER_NAME']} is already past the model's age-40 "
            "horizon, so there is nothing left to project.",
            icon="ℹ️",
        )

matches, projections = {}, {}
for method in enabled:
    match = get_match(method, int(player_id), cutoff, k, epsilon)
    matches[method] = match
    for note in match.warnings:
        st.warning(f"**{method}** — {note}", icon="⚠️")
    if not len(match):
        continue
    try:
        projection = project_metric(
            df, match, metric, cutoff=cutoff, min_support=min_support
        )
    except ValueError as exc:
        st.warning(f"**{method}** — {exc}", icon="⚠️")
        continue

    projections[method] = projection
    if len(projection) == 0:
        plural = "s" if len(match) != 1 else ""
        st.info(
            f"**{method}** found {len(match)} comparison{plural}, but fewer "
            f"than {min_support} of them have a season past {cutoff}, so there "
            "is nothing to project from. Lower the minimum-support slider, "
            "widen ε, or pick an earlier cutoff.",
            icon="ℹ️",
        )

st.subheader(f"Projected {PROJECTABLE_METRICS[metric].lower()}")

figure = go.Figure()

if show_league:
    pct = get_percentiles(metric)
    figure.add_trace(
        go.Scatter(
            x=list(pct.index) + list(pct.index[::-1]),
            y=list(pct["p75"]) + list(pct["p25"][::-1]),
            fill="toself",
            fillcolor="rgba(140,140,140,0.12)",
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            name="League 25th–75th pct",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=pct.index,
            y=pct["p50"],
            mode="lines",
            line={"color": "#8c8c8c", "width": 1.5},
            name="League median",
            hovertemplate="Season %{x}: %{y:.1f}<extra>League median</extra>",
        )
    )

if show_peers:
    seen = set()
    for method in enabled:
        for pid in matches[method].ids:
            if pid in seen:
                continue
            seen.add(pid)
            peer = df[df["PLAYER_ID"] == pid].sort_values("Season_No")
            figure.add_trace(
                go.Scatter(
                    x=peer["Season_No"],
                    y=peer[metric],
                    mode="lines",
                    line={"color": "rgba(120,120,120,0.45)", "width": 1},
                    name=label_by_id[pid],
                    legendgroup="peers",
                    legendgrouptitle_text="Comparison players",
                    hovertemplate=(
                        f"<b>{label_by_id[pid]}</b><br>"
                        "Season %{x}: %{y:.1f}<extra></extra>"
                    ),
                )
            )

figure.add_trace(
    go.Scatter(
        x=history["Season_No"],
        y=history[metric],
        mode="lines+markers",
        line={"color": "#111111", "width": 3.5},
        marker={"size": 8, "symbol": "star"},
        name=f"{player['PLAYER_NAME']} (actual)",
        customdata=history[["SEASON", "Age"]],
        hovertemplate=(
            "Season %{x} (%{customdata[0]}, age %{customdata[1]})"
            "<br>%{y:.1f}<extra>Actual</extra>"
        ),
    )
)

for method, projection in projections.items():
    if len(projection) == 0:
        continue
    frame = projection.to_frame()
    color = METHOD_COLORS[method]

    if show_band:
        upper = frame[metric] + frame["spread"]
        lower = frame[metric] - frame["spread"]
        figure.add_trace(
            go.Scatter(
                x=list(frame["Season_No"]) + list(frame["Season_No"][::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor=rgba(color, 0.15),
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=frame["Season_No"],
            y=frame[metric],
            mode="lines+markers",
            line={"color": color, "width": 3, "dash": METHOD_DASHES[method]},
            marker={"size": 7},
            name=method,
            customdata=frame[["neighbors_used"]],
            hovertemplate=(
                "Season %{x}: %{y:.1f}"
                "<br>%{customdata[0]} comparisons"
                f"<extra>{method}</extra>"
            ),
        )
    )

if gamm_forecast is not None:
    gs, gmean, gsd = gamm_forecast
    color = METHOD_COLORS[GAMM_METHOD]
    anchor_x = float(history.loc[history["Season_No"] == cutoff, "Season_No"].iloc[0])
    anchor_y = float(history.loc[history["Season_No"] == cutoff, metric].iloc[0])
    xs = np.concatenate([[anchor_x], gs])
    ys = np.concatenate([[anchor_y], gmean])

    if show_band:
        # 80% interval; backtested coverage is 78.6%.
        lo = np.concatenate([[anchor_y], gmean - 1.2816 * gsd])
        hi = np.concatenate([[anchor_y], gmean + 1.2816 * gsd])
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([xs, xs[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=rgba(color, 0.15),
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": color, "width": 3.5},
            marker={"size": 7, "symbol": "diamond"},
            name=f"{GAMM_METHOD} (MAE {BACKTEST_MAE[GAMM_METHOD]})",
            customdata=np.concatenate([[0.0], gsd]),
            hovertemplate=(
                "Season %{x}: %{y:.1f}"
                "<br>±1 SD: %{customdata:.1f}"
                f"<extra>{GAMM_METHOD}</extra>"
            ),
        )
    )

figure.add_vline(
    x=cutoff,
    line={"color": "#999999", "width": 1, "dash": "dot"},
    annotation_text="projection starts",
    annotation_position="top",
)

upper_x = max(
    [career_length]
    + [max(p.seasons) for p in projections.values() if len(p)]
    + ([int(gamm_forecast[0].max())] if gamm_forecast is not None else [])
)
figure.update_layout(
    xaxis_title="Season number in career",
    yaxis_title=PROJECTABLE_METRICS[metric],
    xaxis={"range": [0.5, upper_x + 0.5], "dtick": 1},
    hovermode="closest",
    height=560,
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "x": 0},
    margin={"l": 60, "r": 30, "t": 90, "b": 50},
)
st.plotly_chart(figure, width="stretch")

if projections or gamm_forecast is not None:
    summary = []
    if gamm_forecast is not None:
        gs, gmean, gsd = gamm_forecast
        row = {
            "Method": GAMM_METHOD,
            "Seasons projected": len(gs),
            "Next season": round(float(gmean[0]), 2),
            "Final projected": round(float(gmean[-1]), 2),
        }
        if metric not in RATE_METRICS:
            row["Remaining total"] = round(float(gmean.sum()), 1)
        row["Backtest MAE"] = BACKTEST_MAE[GAMM_METHOD]
        summary.append(row)

    for method, projection in projections.items():
        if len(projection) == 0:
            continue
        row = {
            "Method": method,
            "Seasons projected": len(projection),
            "Next season": round(projection.values[0], 2),
            "Final projected": round(projection.values[-1], 2),
        }
        # A remaining-career total is only meaningful for counting stats.
        if metric not in RATE_METRICS:
            row["Remaining total"] = round(sum(projection.values), 1)
        row["Thinnest support"] = min(projection.neighbors_used)
        row["Backtest MAE"] = BACKTEST_MAE[method]
        summary.append(row)
    if summary:
        st.dataframe(
            pd.DataFrame(summary), hide_index=True, width="stretch"
        )
    else:
        st.caption("No seasons could be projected at the current settings.")


# --------------------------------------------------------------------------
# Comparison tables
# --------------------------------------------------------------------------

if enabled:
    st.subheader("Most similar players")
    st.caption(
        f"Matched on the first {cutoff} season{'s' if cutoff != 1 else ''} of each "
        f"career, against the {matches[enabled[0]].pool_size:,} players who also "
        f"reached season {cutoff}."
    )

tabs = st.tabs(enabled) if enabled else []
for tab, method in zip(tabs, enabled):
    with tab:
        match = matches[method]
        if len(match) == 0:
            st.info("No comparisons found at these settings.")
            continue
        table = match.neighbors.assign(
            Player=lambda t: t["label"],
            Distance=lambda t: t["distance"].round(3),
            Seasons=lambda t: t["total_seasons"].astype(int),
            Span=lambda t: t["first_season"] + " – " + t["last_season"],
            Weight=lambda t: (
                (1 / (t["distance"] + 1e-5)) / (1 / (t["distance"] + 1e-5)).sum()
            ).round(3),
            Censored=lambda t: t["left_censored"].map({True: "⚠️", False: ""}),
        )
        st.dataframe(
            table[["Player", "Distance", "Weight", "Seasons", "Span", "Censored"]],
            hide_index=True,
            width="stretch",
            column_config={
                "Weight": st.column_config.NumberColumn(
                    help="Share of the projection this player drives (1/distance, normalized)."
                ),
                "Censored": st.column_config.TextColumn(
                    "Pre-1980",
                    help="This player debuted before the data starts, so their career season numbers are not their real ones.",
                ),
            },
        )
        if (match.neighbors["left_censored"]).any():
            st.caption(
                "⚠️ Flagged players were already in the league before "
                "1979-80, so they are being compared at the wrong career stage."
            )


# --------------------------------------------------------------------------
# Method notes
# --------------------------------------------------------------------------

with st.expander("How this works, and what to distrust"):
    st.markdown(
        f"""
**The style space.** Every player-season is described by {len(pca_model.features)}
per-minute and ratio stats, z-scored so each counts equally, then reduced by PCA
to {N_COMPONENTS} components (eigenvalue > 1, the Kaiser rule) holding
{pca_model.explained_variance_ratio[:N_COMPONENTS].sum():.0%} of the variance.
Broadly, PC1 separates bigs from guards, PC2 tracks scoring volume and usage,
PC3 perimeter shooting, and PC4 defence and rebounding.

**The three methods disagree on purpose.** They build the comparison space
differently, so treat agreement between them as the signal:

- *{STANDARD}* averages the raw stats over each player's first N seasons and
  fits a **fresh PCA on just that peer group**, so the space changes with the
  query.
- *{CENTROID}* uses the one global PCA fit over all player-seasons and takes
  each player's mean position across their first N seasons.
- *{EPSILON}* uses those same global centroids but returns **everyone within a
  radius** instead of a fixed count. If a player has no close peers it returns
  few or none, which is the honest answer where k-NN would hand back five
  distant players and call them neighbours.

**The comparison-based projection is a path of changes, not levels.** For each
future season it takes the year-over-year change each comparison player made at
that point in *their* career, weights it by 1/distance, and adds it to a running
total that starts at the player's actual current value. So it inherits the
*shape* of its comparisons' careers, never their level.

**The GAMM works differently, and forecasts about twice as well.** It is an
additive model with penalized smooth curves in age, career season, era and
playing style, plus a random intercept for every player. Two things follow:

- *It separates the aging curve from the player.* Raw average win shares by age
  barely fall after 30, which would say players hardly decline. That is
  survivorship — only the good ones are still playing at 36. The random
  intercept absorbs each player's own level so the age curve is estimated from
  *within-player* change. Peak age comes out at 27.
- *It pools partially.* A player's level is estimated from their seasons and
  shrunk toward the population curve by `n / (n + 2.8)`, so a player
  with three noisy seasons is pulled most of the way back to average while one
  with twelve keeps nearly all of their own level. The comparison-based methods
  have no equivalent: they anchor 100% on a single season.

On held-out retired players the GAMM's mean absolute error is **1.33 win
shares** against **2.53** / **2.55** / **2.37** for the three comparison
methods — all three of which also lose to simply applying the league-average
year-over-year change (**2.23**). Its 80% intervals cover 78% of outcomes.

**Where to be sceptical.**

- It anchors on a single season, so one injury year or one career year
  propagates through the entire projection.
- Support thins out fast. Hover any projected point to see how many comparison
  players are still behind it; the sidebar's minimum-support slider cuts the
  line off once it gets too thin.
- `Season_No` counts *seasons played*, not years elapsed, so a missed season is
  collapsed rather than represented as a gap.
- Players who debuted before 1979-80 are flagged wherever they appear, because
  the data cannot see their real early careers.
- Style similarity is not quality similarity. Two players can occupy the same
  point in this space and be very different players; the model has no idea who
  is good.
- The GAMM gives each player a single level for their whole career, so a player
  who improved a lot has that level dragged down by their rookie years and will
  look under-forecast from their peak. A random *slope* per player would fix
  this; it is not in yet.
- Every method here forecasts performance **conditional on still playing**.
  None of them models retirement, and retirement is strongly tied to
  performance — a player's final season averages 0.7 win shares against 3.1
  otherwise. Expected value over a career needs a hazard model on top.
"""
    )
