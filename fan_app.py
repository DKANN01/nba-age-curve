"""Fan-facing dashboard: who does this player play like, and what happens next?

The same model as ``app.py``, with the statistics vocabulary translated. Three
matching methods become one consensus line with a range; distances become
closeness words; the knobs move out of the way.

Run locally with:

    streamlit run fan_app.py
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from nba_age_curve.config import (
    DEFAULT_EPSILON,
    DEFAULT_METRIC,
    DEFAULT_N_NEIGHBORS,
    PC_COLS,
    PROJECTABLE_METRICS,
    RATE_METRICS,
)
from nba_age_curve.data import load_player_seasons, player_directory
from nba_age_curve.pca import attach_scores, fit_season_pca
from nba_age_curve.projection import peer_percentiles, project_metric
from nba_age_curve.similarity import (
    CENTROID,
    EPSILON,
    STANDARD,
    TargetNotComparable,
    _centroids,
    match_centroid_knn,
    match_epsilon,
    match_standard_knn,
)

st.set_page_config(
    page_title="Who does he play like?",
    page_icon="\N{BASKETBALL AND HOOP}",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Vocabulary
#
# Everything a fan reads is defined here rather than inline, so the plain
# language stays consistent across the chart, the board and the copy.
# --------------------------------------------------------------------------

WALL = "#E7EAE3"
SLATE = "#17211E"
LEATHER = "#BC5524"
KEY = "#2F6A5B"
FAINT = "#8D978F"
CARD = "#F3F5EF"

#: Short name, then one sentence saying what the stat actually measures.
METRIC_WORDS = {
    "ws": ("Win shares", "how many of his team's wins he was responsible for"),
    "ws_per_48": (
        "Win shares per 48 minutes",
        "the same thing, per full game of court time",
    ),
    "ows": ("Offensive win shares", "the share of those wins he earned on offence"),
    "dws": ("Defensive win shares", "the share he earned on defence"),
    "bpm": (
        "Box plus/minus",
        "points better than an average player, per 100 possessions",
    ),
    "obpm": ("Offensive box plus/minus", "the offensive half of that"),
    "dbpm": ("Defensive box plus/minus", "the defensive half of that"),
    "vorp": ("Value over replacement", "value above a freely available bench player"),
    "usg_pct": ("Usage rate", "the share of his team's plays that he finishes"),
}

#: Words for the four quartiles of match quality, closest first.
CLOSENESS_WORDS = ["Uncanny", "Strong", "Fair", "Loose"]


def closeness(distance: float, cuts: np.ndarray) -> str:
    """Which quartile of achievable match quality this distance falls in.

    The cuts are recomputed per career year rather than fixed, because match
    distances grow as the pool thins: the median top-5 distance runs about
    0.46 at career year 3 and 0.76 at year 15. A fixed threshold would call
    every late-career comparison "loose" purely because fewer players are
    left to compare against, which says nothing about the match.
    """
    return CLOSENESS_WORDS[int(np.searchsorted(cuts, distance))]


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=Newsreader:opsz,wght@6..72,300;6..72,400&display=swap');

.stApp {{ background: {WALL}; }}

.block-container {{
    max-width: 1080px;
    padding-top: 4.5rem;
    padding-bottom: 5rem;
}}
header[data-testid="stHeader"] {{ background: transparent; }}

html, body, [class*="st-"], .stMarkdown, button, input, select, textarea {{
    font-family: 'Archivo', system-ui, sans-serif;
    color: {SLATE};
}}

/* The rule above also catches Streamlit's icon spans, whose glyphs are
   ligatures in the Material font -- without this they render as raw text. */
span[data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Rounded' !important;
}}

/* ---- masthead ---- */
.masthead {{
    font-size: .95rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: {KEY};
    margin-bottom: .1rem;
}}
.masthead-sub {{
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.02rem;
    font-weight: 300;
    color: {FAINT};
    margin-bottom: 1.1rem;
    max-width: 44em;
}}

/* ---- hero ---- */
.player-name {{
    font-size: clamp(2.6rem, 7vw, 4.6rem);
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: .95;
    margin: .2rem 0 .45rem 0;
}}
.player-meta {{
    font-size: 1.02rem;
    font-weight: 500;
    color: {KEY};
    margin-bottom: 1.5rem;
}}
.verdict {{
    font-family: 'Newsreader', Georgia, serif;
    font-weight: 300;
    font-size: clamp(1.25rem, 2.5vw, 1.75rem);
    line-height: 1.45;
    max-width: 30em;
    margin-bottom: .4rem;
}}
.verdict b {{ font-weight: 400; color: {LEATHER}; font-variant-numeric: tabular-nums; }}
.verdict-note {{
    font-size: .88rem;
    color: {FAINT};
    max-width: 44em;
    margin-bottom: 1.6rem;
}}

/* ---- section headings ---- */
.section {{
    font-size: 1.32rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 2.6rem 0 .25rem 0;
}}
.section-note {{
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1rem;
    font-weight: 300;
    color: {FAINT};
    max-width: 46em;
    margin-bottom: 1rem;
}}

/* ---- comparison board ---- */
.board {{ margin-top: .3rem; }}
/* Six tracks for six cells: rank, name, closeness, share, bar, span. */
.board-row {{
    display: grid;
    grid-template-columns: 1.6rem minmax(0,1fr) 5.5rem 2.8rem 6rem 8.6rem;
    align-items: baseline;
    gap: .9rem;
    padding: .66rem 0;
    border-bottom: 1px solid rgba(23,33,30,.09);
}}
.board-row:first-child {{ border-top: 1px solid rgba(23,33,30,.18); }}
.bd-rank {{
    font-size: .88rem;
    font-weight: 600;
    color: {FAINT};
    font-variant-numeric: tabular-nums;
}}
.bd-name {{
    font-size: 1.06rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.bd-close {{ font-size: .86rem; font-weight: 600; }}
.bd-share {{
    font-size: .86rem;
    color: {FAINT};
    font-variant-numeric: tabular-nums;
}}
.bd-bar {{
    height: 5px;
    border-radius: 3px;
    background: rgba(23,33,30,.09);
    align-self: center;
    overflow: hidden;
}}
.bd-bar > span {{ display: block; height: 100%; background: {KEY}; opacity: .75; }}
.bd-span {{
    font-size: .86rem;
    color: {FAINT};
    font-variant-numeric: tabular-nums;
    text-align: right;
    white-space: nowrap;
}}
.board-foot {{
    font-size: .86rem;
    color: {FAINT};
    margin-top: .75rem;
    max-width: 46em;
}}

@media (max-width: 760px) {{
    .board-row {{ grid-template-columns: 1.4rem minmax(0,1fr) 4.8rem 2.6rem; }}
    .bd-bar, .bd-span {{ display: none; }}
}}

/* ---- widgets ---- */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {{
    font-size: .82rem !important;
    font-weight: 600 !important;
    color: {KEY} !important;
}}
div[data-baseweb="select"] > div {{
    background: {CARD};
    border-color: rgba(23,33,30,.18);
    border-radius: 3px;
}}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background: {LEATHER};
}}

.stAlert {{ border-radius: 3px; }}

details > summary {{ font-weight: 600; }}
.stExpander {{
    border: 1px solid rgba(23,33,30,.14) !important;
    border-radius: 3px;
    background: transparent;
}}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data and model
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Reading 20,000 player-seasons...")
def get_model_and_data():
    frame = load_player_seasons()
    model = fit_season_pca(frame)
    return attach_scores(frame, model), model


@st.cache_data(show_spinner=False)
def get_directory() -> pd.DataFrame:
    frame, _ = get_model_and_data()
    return player_directory(frame)


@st.cache_data(show_spinner=False)
def get_match(method: str, player_id: int, n_seasons: int, k: int, eps: float):
    frame, _ = get_model_and_data()
    if method == STANDARD:
        return match_standard_knn(frame, player_id, n_seasons, k=k)
    if method == CENTROID:
        return match_centroid_knn(frame, player_id, n_seasons, k=k)
    return match_epsilon(frame, player_id, n_seasons, epsilon=eps)


@st.cache_data(show_spinner=False)
def get_percentiles(metric: str) -> pd.DataFrame:
    frame, _ = get_model_and_data()
    return peer_percentiles(frame, metric)


@st.cache_data(show_spinner=False)
def closeness_cuts(n_seasons: int, k: int) -> np.ndarray:
    """Quartiles of the top-``k`` match distance across everyone at this stage.

    Gives the board's closeness words a yardstick drawn from what is actually
    achievable at this career year, instead of a constant.
    """
    frame, _ = get_model_and_data()
    points = _centroids(frame, n_seasons)[PC_COLS].to_numpy()
    n_neighbors = min(k + 1, len(points))
    distances, _ = (
        NearestNeighbors(n_neighbors=n_neighbors).fit(points).kneighbors(points)
    )
    # Column 0 is each player's distance to themselves.
    return np.percentile(distances[:, 1:].ravel(), [25, 50, 75])


df, pca_model = get_model_and_data()
directory = get_directory()
label_by_id = dict(zip(directory["PLAYER_ID"], directory["label"]))


# --------------------------------------------------------------------------
# Settings, tucked away
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    st.caption("Defaults are sensible. Change these only if you want to.")
    k = st.slider(
        "Comparison players",
        3,
        15,
        DEFAULT_N_NEIGHBORS,
        help="How many similar careers to average over.",
    )
    epsilon = st.slider(
        "Similarity radius",
        0.2,
        3.0,
        float(DEFAULT_EPSILON),
        0.1,
        help=(
            "One of the three methods returns everyone within this distance "
            "instead of a fixed count. Widen it to admit looser matches."
        ),
    )
    min_support = st.slider(
        "Stop projecting below",
        1,
        5,
        2,
        help=(
            "Cut the projection off once fewer than this many comparison "
            "players are still in the league."
        ),
    )
    show_peers = st.checkbox("Draw each comparison career", value=True)
    show_league = st.checkbox("Draw the league's typical range", value=True)


# --------------------------------------------------------------------------
# Masthead and search
# --------------------------------------------------------------------------

st.markdown('<div class="masthead">Who does he play like?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="masthead-sub">Pick a player. We find the careers that looked '
    "most like his at the same age, then follow those careers forward to see "
    "where his might go.</div>",
    unsafe_allow_html=True,
)

preferred = directory.loc[directory["PLAYER_NAME"] == "Nikola Jokic", "PLAYER_ID"]
default_id = int(preferred.iloc[0] if len(preferred) else directory["PLAYER_ID"].iloc[0])
ids = directory["PLAYER_ID"].tolist()

player_id = st.selectbox(
    "Search for a player",
    options=ids,
    index=ids.index(default_id),
    format_func=lambda pid: label_by_id[pid],
    help="Start typing a name. Current players come first.",
)

player = directory[directory["PLAYER_ID"] == player_id].iloc[0]
history = df[df["PLAYER_ID"] == player_id].sort_values("Season_No")
career_length = int(player["seasons"])
is_active = bool(player["is_active"])
first_career_year = int(history["Season_No"].min())
name = str(player["PLAYER_NAME"])

hero = st.container()


# --------------------------------------------------------------------------
# What to show, and from when
# --------------------------------------------------------------------------

left, right = st.columns([1, 1.45], gap="large")

with left:
    metric = st.selectbox(
        "Measure",
        options=list(PROJECTABLE_METRICS),
        index=list(PROJECTABLE_METRICS).index(DEFAULT_METRIC),
        format_func=lambda c: METRIC_WORDS[c][0],
    )

metric_name, metric_gloss = METRIC_WORDS[metric]

with right:
    if career_length > 1:
        label = (
            "Judge him on his first N seasons"
            if is_active
            else "Rewind: what would we have said after N seasons?"
        )
        cutoff = st.slider(
            label,
            min_value=max(1, first_career_year),
            max_value=career_length,
            value=career_length if is_active else max(first_career_year, career_length // 2),
            help=(
                "Everything to the left of this is used to find his "
                "comparisons. Everything to the right is projected."
            ),
        )
    else:
        cutoff = first_career_year

st.caption(f"{metric_name} is {metric_gloss}.")

if player["left_censored"]:
    st.info(
        f"{name} was already in the league when this data starts in 1979-80, "
        f"so his first {first_career_year - 1} seasons are missing. Everything "
        "below uses only the seasons on record.",
        icon=":material/history:",
    )


# --------------------------------------------------------------------------
# Comparisons and the consensus projection
# --------------------------------------------------------------------------

matches, projections = {}, {}
blocked: list[str] = []

for method in (STANDARD, CENTROID, EPSILON):
    try:
        match = get_match(method, int(player_id), cutoff, k, epsilon)
    except TargetNotComparable as exc:
        blocked.append(str(exc))
        continue
    if not len(match):
        continue
    matches[method] = match
    try:
        projections[method] = project_metric(
            df, match, metric, cutoff=cutoff, min_support=min_support
        )
    except ValueError:
        continue

if not matches:
    with hero:
        st.markdown(f'<div class="player-name">{html.escape(name)}</div>', unsafe_allow_html=True)
    st.warning(
        blocked[0]
        if blocked
        else "No comparable careers at this setting. Try a later season.",
        icon=":material/search_off:",
    )
    st.stop()

# One consensus path: average the methods season by season, and take the range
# as the widest disagreement between them, including each method's own spread.
by_season: dict[int, list[tuple[float, float]]] = {}
for projection in projections.values():
    for season, value, spread in zip(
        projection.seasons, projection.values, projection.spread
    ):
        by_season.setdefault(season, []).append((value, spread))

anchor_value = float(history.loc[history["Season_No"] == cutoff, metric].iloc[0])
proj_seasons = sorted(by_season)
proj_values = [float(np.mean([v for v, _ in by_season[s]])) for s in proj_seasons]
proj_low = [min(v - sd for v, sd in by_season[s]) for s in proj_seasons]
proj_high = [max(v + sd for v, sd in by_season[s]) for s in proj_seasons]

# The comparison set shown on the board: the closest players across all three
# methods, each kept once at its best distance.
best: dict[int, float] = {}
for match in matches.values():
    for pid, dist in zip(match.ids, match.distances):
        best[pid] = min(dist, best.get(pid, float("inf")))
board = sorted(best.items(), key=lambda kv: kv[1])
inv = np.array([1 / (d + 1e-5) for _, d in board])
shares = inv / inv.sum()


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

age_at_cutoff = int(history.loc[history["Season_No"] == cutoff, "Age"].iloc[0])
played_through = int(history["Season_No"].max())

with hero:
    st.markdown(
        f'<div class="player-name">{html.escape(name)}</div>', unsafe_allow_html=True
    )
    if is_active:
        meta = (
            f"{age_at_cutoff} years old, in his {ordinal(cutoff)} season, "
            "still playing"
        )
    else:
        meta = (
            f"Played {player['first_season']} to {player['last_season']}. "
            f"Judged on his first {cutoff} seasons, when he was {age_at_cutoff}."
        )
    st.markdown(f'<div class="player-meta">{meta}</div>', unsafe_allow_html=True)

    if not proj_seasons:
        verdict = (
            f"His {len(board)} closest comparisons were all out of the league "
            f"by this point, so there is nothing left to project."
        )
        note = "Try an earlier season, or widen the settings in the sidebar."
    else:
        last = proj_values[-1]
        change = last - anchor_value
        direction = (
            "holding steady at"
            if abs(change) < 0.05 * max(abs(anchor_value), 1)
            else ("climbing to" if change > 0 else "easing down to")
        )
        years = len(proj_seasons)
        span = "one more season" if years == 1 else f"{years} more seasons"

        if metric in RATE_METRICS:
            verdict = (
                f"Players who started out like him kept going for about "
                f"<b>{span}</b>, {direction} <b>{last:.1f}</b> "
                f"{metric_name.lower()} from {anchor_value:.1f} today."
            )
        else:
            total = sum(proj_values)
            verdict = (
                f"Players who started out like him were worth about "
                f"<b>{total:.0f} more {metric_name.lower()}</b> across "
                f"<b>{span}</b> — {direction} <b>{last:.1f}</b> a year, "
                f"from {anchor_value:.1f} today."
            )
        note = (
            f"That is the average of {len(board)} comparable careers. The "
            "shaded band on the chart is how much they disagree — where it is "
            "wide, the model is guessing."
        )
    st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="verdict-note">{note}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# The arc
# --------------------------------------------------------------------------

st.markdown('<div class="section">His career, and where it is heading</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Solid line is what actually happened. '
    "Dotted line is where comparable careers went next.</div>",
    unsafe_allow_html=True,
)

figure = go.Figure()

if show_league:
    pct = get_percentiles(metric)
    figure.add_trace(
        go.Scatter(
            x=list(pct.index) + list(pct.index[::-1]),
            y=list(pct["p75"]) + list(pct["p25"][::-1]),
            fill="toself",
            fillcolor=rgba(FAINT, 0.13),
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            name="Middle half of the league",
        )
    )

if show_peers:
    for rank, (pid, _) in enumerate(board):
        peer = df[df["PLAYER_ID"] == pid].sort_values("Season_No")
        figure.add_trace(
            go.Scatter(
                x=peer["Season_No"],
                y=peer[metric],
                mode="lines",
                line={"color": rgba(KEY, 0.26), "width": 1},
                name=label_by_id[pid],
                legendgroup="peers",
                showlegend=rank == 0,
                legendgrouptitle_text="",
                hovertemplate=(
                    f"<b>{html.escape(label_by_id[pid])}</b><br>"
                    "Season %{x}: %{y:.1f}<extra></extra>"
                ),
            )
        )
    # Relabel the single legend entry that stands for all of them.
    figure.data[-len(board)].name = "Comparable careers"

if proj_seasons:
    figure.add_trace(
        go.Scatter(
            x=[cutoff] + proj_seasons + proj_seasons[::-1] + [cutoff],
            y=[anchor_value] + proj_high + proj_low[::-1] + [anchor_value],
            fill="toself",
            fillcolor=rgba(LEATHER, 0.13),
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            name="Range of outcomes",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[cutoff] + proj_seasons,
            y=[anchor_value] + proj_values,
            mode="lines+markers",
            line={"color": LEATHER, "width": 3, "dash": "dot"},
            marker={"size": 7},
            name="Projected",
            hovertemplate=(
                "Season %{x}<br>%{y:.1f} projected<extra></extra>"
            ),
        )
    )

figure.add_trace(
    go.Scatter(
        x=history["Season_No"],
        y=history[metric],
        mode="lines+markers",
        line={"color": SLATE, "width": 3.4},
        marker={"size": 7},
        name=f"{name}, actual",
        customdata=history[["SEASON", "Age"]],
        hovertemplate=(
            "Season %{x} — %{customdata[0]}, age %{customdata[1]}"
            "<br>%{y:.1f}<extra></extra>"
        ),
    )
)

figure.add_vline(x=cutoff, line={"color": rgba(SLATE, 0.35), "width": 1, "dash": "dot"})

upper_x = max([career_length] + proj_seasons) if proj_seasons else career_length
figure.update_layout(
    height=480,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"family": "Archivo, sans-serif", "color": SLATE, "size": 13},
    hovermode="closest",
    hoverlabel={"bgcolor": CARD, "bordercolor": rgba(SLATE, 0.2)},
    margin={"l": 54, "r": 18, "t": 44, "b": 48},
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.0,
        "x": 0,
        "bgcolor": "rgba(0,0,0,0)",
    },
    xaxis={
        "title": "Season of his career",
        "range": [0.4, upper_x + 0.6],
        "dtick": 1 if upper_x <= 22 else 2,
        "showgrid": False,
        "zeroline": False,
        "linecolor": rgba(SLATE, 0.25),
        "ticks": "outside",
        "tickcolor": rgba(SLATE, 0.25),
    },
    yaxis={
        "title": metric_name,
        "showgrid": True,
        "gridcolor": rgba(SLATE, 0.08),
        "zeroline": False,
    },
)
st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


# --------------------------------------------------------------------------
# The board
# --------------------------------------------------------------------------

st.markdown('<div class="section">Players like him</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-note">Ranked by how closely their first {cutoff} '
    "seasons resembled his — not by how good they were. Uncanny, strong, fair "
    "and loose are the four quarters of how well anyone can be matched this "
    "deep into a career. The bar is how much each one pulls the "
    "projection.</div>",
    unsafe_allow_html=True,
)

career_meta = df.groupby("PLAYER_ID").agg(
    first_season=("SEASON", "min"),
    last_season=("SEASON", "max"),
)

cuts = closeness_cuts(cutoff, k)
tints = {"Uncanny": KEY, "Strong": SLATE, "Fair": FAINT, "Loose": FAINT}

rows = []
for rank, ((pid, dist), share) in enumerate(zip(board, shares), start=1):
    word = closeness(dist, cuts)
    meta = career_meta.loc[pid]
    tint = tints[word]
    rows.append(
        f'<div class="board-row">'
        f'<div class="bd-rank">{rank}</div>'
        f'<div class="bd-name">{html.escape(str(label_by_id[pid]))}</div>'
        f'<div class="bd-close" style="color:{tint}">{word}</div>'
        f'<div class="bd-share">{share * 100:.0f}%</div>'
        f'<div class="bd-bar"><span style="width:{share / shares.max() * 100:.0f}%"></span></div>'
        f'<div class="bd-span">{meta["first_season"]} – {meta["last_season"]}</div>'
        f"</div>"
    )
st.markdown(f'<div class="board">{"".join(rows)}</div>', unsafe_allow_html=True)

if float(np.median([d for _, d in board])) > cuts[2]:
    st.markdown(
        '<div class="board-foot">These are the closest players available, but '
        "none of them is really a match. Treat the projection as a rough "
        "guide.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# For anyone who wants the machinery
# --------------------------------------------------------------------------

with st.expander("How this works"):
    st.markdown(
        f"""
Every season a player has ever had is described by twelve rate stats — minutes,
shots, assists, turnovers, rebounds, blocks, steals, free throws and threes, all
per minute so that a starter and a bench player can be compared. Those twelve
numbers get boiled down to four that capture most of what separates one style
from another: roughly, big versus guard, high usage versus role player, perimeter
versus post, and defence versus offence.

To find comparisons for **{html.escape(name)}**, the model averages his first
{cutoff} seasons into a single point and looks for the players sitting nearest
to it — among the
{matches[next(iter(matches))].pool_size:,} players who also made it to season
{cutoff}, so he is compared against careers at the same stage, not against
players who were out of the league by then.

The projection then follows those careers forward. For each future season it
takes the change each comparison player made *at that point in his own career*,
averages those changes with the closest players counting most, and adds the
result to where {html.escape(name)} actually is today. So the projection borrows
the **shape** of their careers, never their level — a player better than all his
comparisons stays better than them.

Three slightly different methods run underneath, and the line you see is their
average. Where they disagree, the shaded band widens.

**What it cannot do.** It has no idea who is good — only who plays alike. Two
players can sit in the same spot and be very different. It anchors entirely on
one season, so an injury year drags the whole projection down. And the further
right you go, the fewer comparison careers are still running: by the end, the
line can rest on one or two players.
"""
    )

st.markdown(
    f'<div class="board-foot" style="margin-top:2rem">'
    f"{len(df):,} player-seasons from {df['PLAYER_ID'].nunique():,} players, "
    f"{df['SEASON'].min()} through {df['SEASON'].max()}.</div>",
    unsafe_allow_html=True,
)
