# Pre-registration: do the comparison methods predict?

**Status:** written before any evaluation code was run. No outcome in this
document has been observed. Deviations go in the log at the bottom.

**Frozen against commit:** `405a718`
**Master seed:** `20261119`

---

## 1. What is being asked

1. Does any of the three comparison methods predict future performance better
   than a null that ignores similarity?
2. If more than one does, is any of them reliably better than the others?
3. What are the best settings for each (`k`, `ε`, `min_support`)?

Question 3 is answered **first**, on players who are never used for questions
1 and 2. Reporting a p-value from the same data used to pick `k` would report
the maximum over a grid as though it were a single pre-specified test.

---

## 2. What is actually being tested

Each method produces a path of *changes*, not levels. For a player `i` with
cutoff season `c`, the forecast at horizon `h` is

```
ŷ(i, c+h) = y(i, c) + Σ(j=1..h) Δ̄(j)
```

where `Δ̄(j)` is the inverse-distance-weighted mean of the comparison players'
year-over-year changes at career year `c+j`.

The anchor `y(i, c)` is the player's own observed value and is **identical
across all methods and baselines**. It contributes no error of its own and
cancels in any paired comparison. Everything below therefore tests the
estimated deltas and nothing else.

This also means the anchor-override feature in the dashboard is out of scope.
It changes the level by construction and has no effect on the quantity under
test.

---

## 3. Data and eligibility

Panel: 20,546 player-seasons, 3,816 players, 1979-80 through 2025-26. `ws` has
no missing values.

A `(player, cutoff)` pair is eligible when:

- `c ≥ 3`. Below three seasons the career centroid is too thin to be a profile.
- The player has observed seasons at every one of `c+1 … c+h` for the horizons
  being scored. Gap seasons disqualify the pair rather than being interpolated.
- The player is not left-censored past the window, i.e. `match_*` does not
  raise `TargetNotComparable`.

Eligible pairs at these thresholds:

| horizon | pairs | players |
|---|---|---|
| h=1 | 13,473 | 2,202 |
| h=3 | 9,371 | 1,686 |
| h=5 | 6,190 | 1,312 |

Careers are short — median 4 seasons, p75 = 10 — so eligibility is the binding
constraint, not compute.

---

## 4. Sampling and splits

**The unit of clustering is the player, not the player-season.** A single long
career offers up to ~18 eligible cutoffs that share an anchor, a career and
largely the same comparison set. Treating those as independent observations
would overstate precision badly.

From the 1,686 players eligible at h=3, drawn once under the master seed:

| split | players | use |
|---|---|---|
| **TUNE** | 600 | grid search for Q3. Read as often as needed. |
| **TEST** | 300 | Q1 and Q2. Read **once**, at the end. |
| unused | remainder | held in reserve; not touched |

Within each sampled player, draw **up to 3 cutoffs** uniformly from that
player's eligible set, without replacement. Multiple cutoffs per player are
kept for efficiency but never counted as independent — every standard error,
test and bootstrap resamples **players**.

TEST is not to be loaded until the tuning grid is finalised and committed.

---

## 5. Methods and baselines

Under test, all from `nba_age_curve.similarity`:

- **Standard k-NN** — career averages over the window, PCA refit per query.
- **Centroid k-NN** — global season-level PCA, player centroids.
- **ε-neighborhood** — same centroids, fixed radius.

Baselines, both built with the same delta-accumulation machinery so that the
comparison set is the only thing that differs:

- **B0 persistence** — `ŷ = y(i, c)`. All deltas zero.
- **B1 league aging curve** — deltas are the unweighted league-mean change at
  each career year, computed over all players who have that transition. This
  is the real null: it is the same model with the weights spread uniformly
  over the whole pool instead of concentrated on neighbors.

**B1 is the primary null.** Beating B0 shows only that players change with age,
which is not in dispute. Beating B1 is the claim that *these particular
comparisons* carry information.

B1 must be computed with the target player excluded from the league mean, for
the same reason the matchers exclude the target from its own neighbor set
(verified: `similarity.py:138` and `similarity.py:238`).

---

## 6. Outcome measures

**Primary endpoint.** Mean absolute error on `ws`, averaged over h ∈ {1,2,3}.
One metric, one horizon window, fixed in advance.

**Secondary, reported but not used for any accept/reject decision.** MAE at
each of h = 1…5 separately; RMSE on the same; the primary endpoint recomputed
on `ws_per_48`.

`ws` is the primary despite scaling with minutes played, because it is the
dashboard's default and the quantity a reader will act on. `ws_per_48` is
carried as a secondary precisely because it does not have that scaling, and a
disagreement between the two is itself worth reporting.

---

## 7. Common support

Score only those `(player, cutoff, h)` cells where **all three methods and both
baselines** produce a value.

This matters more than it looks. ε-neighborhood returns nothing when the radius
is empty, and every method truncates its path once too few comparison players
remain. Those abstentions are not random — they concentrate on unusual players
and late-career seasons, which are the hard cases. Without a common-support
rule, a method that quietly declines the hard cases would post a better MAE for
having attempted less.

Report, alongside the main results, the **attempt rate** of each method: the
fraction of eligible cells it would have scored before intersection. A method
that wins on common support while attempting far fewer cells has bought its
accuracy with coverage, and the write-up must say so.

---

## 8. Hypotheses and tests

All tests operate on **player-level mean loss**: collapse each player's cells
to one number, then test across the 300 TEST players.

**Q1 — does any method beat the null?**

> H₀: E[loss(method)] ≥ E[loss(B1)]  H₁: E[loss(method)] < E[loss(B1)]

One-sided Wilcoxon signed-rank on the per-player loss differential
`loss(method) − loss(B1)`. Three tests, Holm-corrected at family-wise α = 0.05.
Accompanied by a cluster bootstrap over players (10,000 resamples, percentile
interval) for the MAE difference, which is the number actually reported.

**Q2 — is any method better than the others?**

Friedman test across the three methods on player-level means. If it rejects at
α = 0.05, Holm-corrected pairwise Wilcoxon over the three pairs. If it does not
reject, the finding is that the methods are indistinguishable at this sample
size, and no pairwise claim is made.

**Q3 — best settings.**

Grid on TUNE only: `k ∈ {3,5,8,10,15}`, `ε ∈ {0.3,0.5,0.75,1.0,1.5}`,
`min_support ∈ {1,2,3}`. Select by primary-endpoint MAE. `k` applies to the two
k-NN methods, `ε` to the radius method, `min_support` to all three. Ties within
one bootstrap standard error are broken toward the more conservative setting
(larger `min_support`, smaller `k`).

---

## 9. Power

With 300 clusters, a paired test detects an effect of roughly **0.16 SD** at
80% power, α = 0.05 two-sided.

If the true edge over B1 is smaller than that, the correct outcome is a
confidence interval containing zero. That is a result, not a failure, and will
be reported as one. The pre-committed interpretation: an interval that excludes
zero but whose upper bound is under 0.1 win shares of MAE improvement is
statistically detectable and practically negligible, and will be described that
way.

---

## 10. Known leakage, declared rather than fixed

**The estimand is retrospective, not prospective.** Two distinct problems:

1. The comparison pool contains players whose careers were still in progress at
   the simulated forecast date. Projecting 1998 from a pool that includes
   careers finishing in 2024 is not something a 1998 analyst could have done.
2. The global season-level PCA (`fit_season_pca` over all 20,546 rows) is fit
   on the entire panel, including the test player's own post-cutoff seasons.

Problem 2 is asymmetric across the methods and is a genuine confound:
`match_standard_knn` refits PCA on the window's career averages only
(`similarity.py:169`), whereas `match_centroid_knn` and `match_epsilon` both
read the globally-fit component space (`similarity.py:190`). **Standard k-NN is
therefore less leaky than the other two**, and any Q2 result favouring centroid
or ε must be read with that in mind. This will be stated wherever Q2 is
reported.

Making the design genuinely prospective means restricting each forecast to
careers complete as of that date, which collapses the pool in the early years
to the point of uselessness. The chosen estimand is: *does the comparison idea
carry structural signal in this panel*, not *could you have traded on it live*.

---

## 11. What would falsify the project's premise

Stated now so it cannot be renegotiated later: if no method's bootstrap
interval for MAE improvement over **B1** excludes zero at the primary endpoint,
the conclusion is that similarity-based comparison adds nothing detectable
beyond a league-average aging curve at this sample size. The dashboards remain
useful as descriptive tools — "who did this player resemble" is a real question
with a real answer — but the forward projection would not be supportable as a
predictive claim, and the write-up would say so plainly.

---

## 12. Analysis order

1. Build the eligible-pair frame; freeze TUNE/TEST split to disk with the seed.
2. Run the Q3 grid on TUNE. Commit the chosen settings.
3. Compute predictions for all methods and baselines on TEST under those
   settings.
4. Apply common support; record attempt rates.
5. Run Q1, then Q2.
6. Write up, including every secondary endpoint regardless of direction.

Steps 3–5 are run exactly once.

---

## 13. Deviation log

Any departure from the above is recorded here with a date and a reason, before
the affected result is reported.

| date | deviation | reason |
|---|---|---|
| — | none yet | — |
