# NBA Career Projections

Find a player's closest stylistic comparisons in the modern NBA, then project
their future win shares from how those comparisons' careers actually went.

Type a player's name into the dashboard and you get:

- the most similar careers under three different matching methods,
- a projected path for win shares (or BPM, VORP, usage, …) with a band showing
  how much the comparisons disagree,
- league 25th/50th/75th percentile context by career season.

## How it works

Every player-season is described by **12 per-minute and ratio stats** — minutes
per game, points, field-goal attempts, assists, turnovers, assist-to-turnover
ratio, offensive and defensive rebounds, blocks, free-throw rate, steals per
foul, and three-point attempt rate. Each is z-scored so no stat dominates on
scale alone, then PCA reduces them to **4 components** (every component with
eigenvalue > 1 under the Kaiser rule), holding **65.7%** of the variance.

The components are interpretable:

| Component | Variance | High end | Low end |
|---|---|---|---|
| PC1 | 27.9% | Bigs — Manute Bol, Mutombo, Biyombo | Guards — Tyus Jones, Gilbert Arenas |
| PC2 | 17.6% | High usage — Embiid, Giannis, Westbrook | Role players — P.J. Tucker |
| PC3 | 10.9% | Perimeter specialists — Steve Novak | Post players — Bill Cartwright |
| PC4 | 9.4% | Defence/rebounding — Ben Wallace | Low-event guards |

### Three matching methods, on purpose

They build the comparison space differently and will not agree. Treat agreement
between them as the signal.

| Method | Space | Returns |
|---|---|---|
| **Standard k-NN** | Averages raw stats over each player's first *N* seasons, then fits a **fresh PCA on just that peer group** — the space changes with the query | *k* nearest |
| **Centroid k-NN** | The one global season-level PCA; each player's mean position across their first *N* seasons | *k* nearest |
| **ε-neighbourhood** | The same global centroids | **Everyone within radius ε** |

The ε method exists because k-NN always hands back exactly *k* players, however
distant. For a player with no real comparisons, returning two matches — or none
— is the honest answer.

A player is only matched against others who **also reached career year *N***, so
someone five seasons in is compared to other players' first five seasons rather
than to careers that ended after two. Full coverage of that window is not
required — a player whose record starts partway through contributes the part
that was recorded (see left-censoring below).

### The projection is a path of changes, not levels

For each future season, the model takes the year-over-year change each
comparison player made *at that point in their own career*, weights it by
1/distance, and adds it to a running total that starts at the target player's
actual current value.

So the projection inherits the **shape** of its comparisons' careers, never
their level. A player outperforming all their comps keeps that gap.

## What to distrust

- **It anchors on a single season.** One injury year or one career year
  propagates through the whole projection.
- **Support thins out fast.** Hover any projected point to see how many
  comparisons are still behind it. The minimum-support slider cuts the line off
  once it drops too low; without it, a long projection can ride on one peer.
- **Career year is years since the player's true debut**, not a count of rows.
  The debut comes from the master record's `year_min`, which includes ABA
  seasons, so Kareem Abdul-Jabbar's 1979-80 is career year 11 rather than a
  rookie season. A missed season therefore leaves a real gap in the index
  rather than being collapsed; 660 players have at least one.
- **Pre-1980 debuts are left-censored.** 237 players were already in the league
  when the data starts. Their seasons sit correctly on the career-year axis,
  but their early years are missing, so they are matched on whatever part of
  the window was recorded — Robert Parish's record begins at career year 4, so
  at *N* = 5 he is matched on two seasons where his peers get five. The
  "Matched on" column reports this. A player whose record starts *after*
  career year *N* contributes nothing at that cutoff and cannot be a target
  there: Kareem Abdul-Jabbar's record begins at career year 11, so he needs a
  cutoff of 11 or more.
- **Style similarity is not quality similarity.** Two players can sit at the
  same point in this space and be very different players. The model has no idea
  who is good — it only knows who plays alike.
- **Backtests are not out-of-sample.** The PCA is fit on all seasons including
  the ones being "projected", so a backtest flatters the method somewhat.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads `data/player_seasons.parquet`, which is committed, so no network
access is needed at runtime.

### Rebuilding the data

The source is a published Google Sheet of 20,546 player-seasons (3,816 players,
1979-80 → 2025-26). The app deliberately does **not** fetch it at runtime — a
public dashboard shouldn't depend on a third-party URL on every cold start.

```bash
pip install -r requirements-dev.txt
python scripts/build_dataset.py                      # from the Google Sheet
python scripts/build_dataset.py --source local.xlsx  # or a local copy

# re-derive true debut years from the master workbook (updates
# data/debut_years.csv, which is committed so the build needs it only once)
python scripts/build_dataset.py --master ../nba_age_curve_master.xlsx
```

### Regenerating the PCA figures

```bash
python scripts/analysis.py --outdir figures
```

Writes the scree plot, loadings heatmap and PC1/PC2 biplot, and prints the
variance table, loadings, and the player-seasons at each end of every component.

## Deploying

The dashboard is a standard Streamlit app and runs on
[Streamlit Community Cloud](https://share.streamlit.io) for free:

1. Go to share.streamlit.io and sign in with GitHub.
2. **New app** → pick this repo, branch `main`, main file `app.py`.
3. Deploy. Dependencies come from `requirements.txt`.

Free-tier apps sleep after about a week idle and need a click to wake.

## Layout

```
app.py                      Streamlit dashboard
nba_age_curve/
  config.py                 Feature list, paths, model defaults
  data.py                   Loading, season indexing, censoring flags, labels
  pca.py                    SeasonPCA: fit, transform, loadings, scree inputs
  similarity.py             The three matchers
  projection.py             Year-over-year delta projection, league percentiles
scripts/
  build_dataset.py          Google Sheet -> data/player_seasons.parquet
  analysis.py               Exploratory PCA figures and component extremes
data/
  player_seasons.parquet    Baked dataset (committed)
  debut_years.csv           True debut year per player, for career-year indexing
```

## Notes on the original script

This started as a Colab notebook (`nba_age_curve.py`). Beyond restructuring,
the rewrite fixes several real bugs:

- **Players were grouped by `PLAYER_NAME`.** 25 names in the data belong to two
  or three different players — there are two Eddie Johnsons and three Charles
  Smiths — so grouping by name spliced them into single impossible careers
  (one "Eddie Johnson" spanning 1979-80 to 1998-99, 25 seasons). Everything now
  keys on `PLAYER_ID`, and the dropdown disambiguates shared names by season
  span.
- **`df_player` was referenced but never defined**, so every code path for a
  retired player crashed. Retired players now work, and the cutoff slider turns
  their projection into a backtest against what actually happened.
- **The scaler, PCA and k-NN model were fit twice**, the second fit silently
  discarding the first.
- **`get_projection_data_*` never checked whether a player was active**, so they
  produced results for retired players with no indication they were meaningless.
- **`display()` is Colab-only**, so the script ran nowhere else.
- The three methods used different PC spaces without that being stated —
  "standard" refits PCA per query while the other two use the global space.
  That difference is preserved (it is a legitimate design choice) but is now
  documented in the code and surfaced in the app.
- The projection loop re-scanned the full 20k-row table once per neighbour per
  season; it now builds a delta table once per projection.
