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

A player is only matched against others who **also reached season *N***, so
someone five seasons in is compared to other players' first five seasons rather
than to careers that ended after two.

### The projection is a path of changes, not levels

For each future season, the model takes the year-over-year change each
comparison player made *at that point in their own career*, weights it by
1/distance, and adds it to a running total that starts at the target player's
actual current value.

So the projection inherits the **shape** of its comparisons' careers, never
their level. A player outperforming all their comps keeps that gap.


## The GAMM forecaster

The comparison-based projection above is the classic "delta method" from the
aging-curve literature: differencing cancels the player-level effect, which is
why it is a sensible starting point. But it anchors entirely on one season, it
accumulates error like a random walk, and it does no pooling.

The GAMM (`nba_age_curve/gamm.py`) is an additive model with penalized smooth
curves in age, career season, era and the four style components, plus a **random
intercept for every player**:

```
ws_ij = a + f_age(age) + f_exp(season_no) + f_era(year) + sum_k f_k(PC_k) + b_i + e_ij
b_i ~ N(0, sigma_b^2)   e_ij ~ N(0, sigma_e^2)
```

Fitted on all 20,546 player-seasons: **sigma_b = 0.90** (spread of player
levels) and **sigma_e = 1.50** (season-to-season noise). The fitted career path
peaks around season 7 / age 27 -- but see **Limits** before reading that as an
age effect.

**Why the random intercept is the whole point.** Raw average win shares by age
rise into the late twenties and barely fall afterwards — a 40-year-old looks as
good as a 23-year-old. That is survivorship: only the good ones are still
playing at 40. The random intercept absorbs each player's own level, so the age
curve is estimated from *within-player* change:

| age | 21 | 24 | 27 | 30 | 33 | 36 | 39 |
|---|---|---|---|---|---|---|---|
| raw mean (survivor-biased) | -1.4 | -1.1 | -0.2 | -0.1 | -0.4 | -0.7 | -1.1 |
| within-player (GAMM) | -0.7 | -0.4 | 0.0 | -0.2 | -0.5 | -1.0 | -1.5 |

**Forecasting a player** estimates their level from the seasons you have and
shrinks it toward the population curve by `n / (n + sigma_e^2/sigma_b^2)` =
`n / (n + 2.8)`. Three noisy seasons get pulled most of the way back to average;
twelve seasons keep nearly all of their own level.

### Feeding the comparison sets through it

The two parts of this project are not rivals. Each matcher produces a
comparison set and a vector of inverse-distance weights; the delta method
averages those comparisons' year-over-year changes, and the GAMM can take the
*same* weights and use them to fit whole smooth curves instead
(`projection.match_weights` -> `GAMM.refit_weighted`). Same comps, same
weights, different engine.

Because a handful of players cannot support a 68-column basis, the weighted fit
shrinks toward the global model in proportion to how thin the weighting is:

```
minimize  ||y - X.beta||^2_W  +  (beta - beta_global)' S (beta - beta_global)
                              +  tau ||beta - beta_global||^2 ,   tau ~ p / ESS
```

so a tight comparison set yields a gentle tilt on the population curve rather
than a wild extrapolation. Without that term the weighted normal equations go
singular and single-player errors reach 49 win shares.

### Does it actually help?

`scripts/backtest.py` splits players by identity -- the GAMM never sees a test
player, and the matchers may only draw comparisons from training players. Mean
absolute error in win shares:

| Method | MAE | vs. plain GAMM |
|---|---|---|
| **Centroid k-NN -> GAMM** | **1.27** | better |
| Standard k-NN -> GAMM | 1.27 | better |
| epsilon-neighbourhood -> GAMM | 1.30 | better |
| GAMM alone (no comparisons) | 1.32 | — |
| League-avg delta *(baseline)* | 2.23 | |
| epsilon-neighbourhood -> delta | 2.37 | |
| Standard k-NN -> delta | 2.53 | |
| Centroid k-NN -> delta | 2.55 | |
| Flat, carry last *(baseline)* | 2.68 | |

**Routing the same comparison sets through the GAMM cuts their error by about
44%** (2.55 -> 1.27). The comparison machinery was never the problem; averaging
one-season deltas was.

### How many comparisons?

Five is far too few. Sweeping the comparison-set size through the identical
pipeline (mean per-player MAE, 3 cutoffs x 2 seeds, paired against the plain
GAMM):

| k | effective sample | MAE | vs. plain GAMM | t |
|---|---|---|---|---|
| 5 | 4.7 | 1.214 | +0.029 | +3.8 *(worse)* |
| 15 | 13.7 | 1.205 | +0.020 | +2.6 *(worse)* |
| 40 | 35.9 | 1.177 | -0.008 | -1.1 *(ns)* |
| **80** | **70.9** | **1.160** | **-0.024** | **-3.4** |
| 160 | 139.9 | 1.162 | -0.023 | -3.4 |

At k=5 the comparisons make the forecast *worse* than ignoring them. The app
therefore fits on 80 by default while still showing only the nearest few in the
table. A smooth kernel over the same distances does slightly better again
(tricube, bandwidth 2.0: MAE 1.122, -0.058 vs global, t=-9.0), since
inverse-distance weighting is peaked and has a hard edge at the k-th neighbour.

A varying-coefficient term letting the age curve scale with style
(`Smooth("Age", by="PC1")`) is statistically significant but practically
negligible: -0.011 MAE, t=-3.2.

The implementation was cross-checked against R's `mgcv`, the reference GAMM
package. Fitting the same specification with `bam()` gives sigma_b = 0.9043
(CI 0.870-0.939) and sigma_e = 1.4708, against 0.901 and 1.497 here.

### Limits

- **One level per career.** The random intercept is constant, so a player who
  improved a lot has it dragged down by their rookie years and looks
  under-forecast from their peak (Jokic is the obvious case). A random *slope*
  per player is the natural next step.
- **No retirement model.** Every method here forecasts performance *conditional
  on still playing*, and retirement is strongly performance-related — a final
  season averages 0.67 win shares against 3.07 otherwise. Expected career value
  needs a companion hazard model.
- **Smoothing parameters** are chosen by GCV with Nelder-Mead, and this is the
  weakest part of the implementation. It drives some lambdas to zero, and it
  splits the age/experience signal differently from `mgcv`: fitting the same
  specification with REML in `mgcv` gives age a 6.6 win-share swing from peak to
  40, while GCV here gives it 0.24 and hands the decline to career season
  instead. The two are correlated at 0.861, so either can carry it, and
  forecasts are barely affected -- but **the age curve from this model should
  not be read as an age effect**, and "peak age 27" is not a claim it supports.
  Moving to REML is the fix.

### Rebuilding it

```bash
python scripts/fit_gamm.py        # writes data/gamm_model.npz (30 KB)
python scripts/backtest.py        # reproduces the table above
```

The app loads the saved `.npz` and only evaluates basis functions, so nothing is
re-fit at runtime.

## What to distrust

- **It anchors on a single season.** One injury year or one career year
  propagates through the whole projection.
- **Support thins out fast.** Hover any projected point to see how many
  comparisons are still behind it. The minimum-support slider cuts the line off
  once it drops too low; without it, a long projection can ride on one peer.
- **`Season_No` counts seasons *played*, not years elapsed**, so a missed season
  is collapsed rather than represented as a gap.
- **Pre-1980 debuts are left-censored.** The data starts at 1979-80, so for the
  ~300 players already in the league by then, "career season 1" is not their
  rookie year. They are kept but flagged ⚠ wherever they appear.
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
  gamm.py                   Penalized splines + player random intercept
  similarity.py             The three matchers
  projection.py             Year-over-year delta projection, league percentiles
scripts/
  build_dataset.py          Google Sheet -> data/player_seasons.parquet
  analysis.py               Exploratory PCA figures and component extremes
  fit_gamm.py               Fits the GAMM -> data/gamm_model.npz
  backtest.py               Scores every forecaster on held-out players
data/
  player_seasons.parquet    Baked dataset (committed)
  gamm_model.npz            Fitted GAMM coefficients (committed)
  pre_1980_debuts.csv       Confirmed pre-1980 debuts, for the censoring flag
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
