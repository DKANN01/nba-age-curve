"""A Gaussian GAMM: penalized B-splines plus a player random intercept.

Pure numpy/scipy, so the fitted model ships anywhere scikit-learn does. The
model is

    ws_ij = a + sum_k f_k(x_ijk) + b_i + e_ij
    b_i ~ N(0, sigma_b^2),  e_ij ~ N(0, sigma_e^2)

where each ``f_k`` is a penalized cubic B-spline and ``b_i`` is a per-player
random intercept.

**Why a random intercept matters here.** Raw mean win shares by age *rises*
into the late twenties and barely falls after, which would suggest players
hardly decline. That is survivorship: only the good ones are still playing at
36. The random intercept absorbs each player's own level, so the age smooth is
estimated from *within-player* change -- the same logic that makes the
year-over-year delta method work, but smoothed and with partial pooling.

**Forecasting an unseen player.** Their level is estimated from the seasons you
have and shrunk toward the population curve by the standard BLUP factor,
``n / (n + sigma_e^2/sigma_b^2)``. A player with two noisy seasons is pulled
most of the way back to average; one with ten keeps nearly all of their own
level. The weighted-average-of-deltas approach has no equivalent -- it anchors
100% on a single season.

**What it does not do.** It predicts performance *conditional on playing*. It
has no opinion on when a career ends, and retirement is strongly tied to
performance, so a full treatment needs a companion hazard model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize


def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """Design matrix of B-splines of ``degree`` evaluated at ``x``."""
    lo, hi = knots[0], knots[-1]
    t = np.concatenate([[lo] * degree, knots, [hi] * degree])
    n_basis = len(t) - degree - 1
    out = np.empty((len(x), n_basis))
    xc = np.clip(x, lo, hi)  # constant extrapolation beyond the data range
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        out[:, j] = BSpline(t, c, degree, extrapolate=True)(xc)
    return out


def _difference_penalty(n: int, order: int = 2) -> np.ndarray:
    """``D'D`` for the ``order``-th difference of adjacent coefficients.

    Penalizing second differences shrinks each smooth toward a straight line,
    so the data decides how much curvature to buy.
    """
    d = np.eye(n)
    for _ in range(order):
        d = np.diff(d, axis=0)
    return d.T @ d


@dataclass
class Smooth:
    """One penalized spline term, optionally a varying-coefficient term.

    With ``by`` set, the basis is multiplied by that column, so the term is
    ``f(x) * by`` rather than ``f(x)``: the shape of the curve scales with
    another variable. ``Smooth("Age", by="PC1")`` lets the aging curve differ
    between the two ends of PC1 -- bigs declining on a different schedule from
    guards -- which a plain additive ``s(Age) + s(PC1)`` cannot express.
    """

    name: str
    n_knots: int = 10
    degree: int = 3
    by: str | None = None
    knots_: np.ndarray | None = None
    center_: np.ndarray | None = None
    cols_: tuple[int, int] | None = None

    @property
    def label(self) -> str:
        return self.name if self.by is None else f"{self.name}:{self.by}"

    def basis(self, x, by_values=None, fit: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if fit:
            self.knots_ = np.unique(np.quantile(x, np.linspace(0, 1, self.n_knots)))
        b = _bspline_basis(x, self.knots_, self.degree)
        if fit:
            # Centre on the TRAINING mean and reuse it at predict time. Centring
            # each batch on its own mean would silently shift predictions.
            self.center_ = b.mean(axis=0)
        b = b - self.center_
        if self.by is not None:
            b = b * np.asarray(by_values, dtype=float)[:, None]
        return b

    @property
    def size(self) -> int:
        return len(self.knots_) + self.degree - 1


@dataclass
class GAMM:
    """Penalized additive model with one grouping factor's random intercept."""

    smooths: list[Smooth]
    max_iter: int = 50
    tol: float = 1e-7

    beta_: np.ndarray | None = None
    vbeta_: np.ndarray | None = None
    lambdas_: np.ndarray | None = None
    group_effects_: dict = field(default_factory=dict)
    sigma_b_: float = 1.0
    sigma_e_: float = 1.0

    # -- design -------------------------------------------------------------
    def design(self, frame, fit: bool = False) -> np.ndarray:
        blocks, pos = [], 1  # column 0 is the global intercept
        for s in self.smooths:
            by = np.asarray(frame[s.by]) if s.by is not None else None
            b = s.basis(np.asarray(frame[s.name]), by, fit=fit)
            if fit:
                s.cols_ = (pos, pos + b.shape[1])
            pos += b.shape[1]
            blocks.append(b)
        return np.column_stack([np.ones(len(frame))] + blocks)

    def _penalty(self, lambdas: np.ndarray) -> np.ndarray:
        p = 1 + sum(s.cols_[1] - s.cols_[0] for s in self.smooths)
        S = np.zeros((p, p))
        for lam, s in zip(lambdas, self.smooths):
            lo, hi = s.cols_
            S[lo:hi, lo:hi] = lam * _difference_penalty(hi - lo)
        return S

    def _solve(self, X, y, S):
        XtX = X.T @ X
        A = XtX + S
        # GCV will happily drive a lambda to zero, and with varying-coefficient
        # terms the basis is close to collinear (f(Age) and f(Age)*PC1 share
        # their marginal). Either can make A singular, so carry a nugget that is
        # negligible against A's own scale, and fall back to a pseudo-inverse.
        A = A + np.eye(A.shape[0]) * (1e-9 * max(np.trace(A), 1.0) / A.shape[0])
        try:
            beta = np.linalg.solve(A, X.T @ y)
            edf = float(np.trace(np.linalg.solve(A, XtX)))
        except np.linalg.LinAlgError:
            pinv = np.linalg.pinv(A)
            beta = pinv @ (X.T @ y)
            edf = float(np.trace(pinv @ XtX))
        if not np.isfinite(beta).all():
            pinv = np.linalg.pinv(A)
            beta = pinv @ (X.T @ y)
            edf = float(np.trace(pinv @ XtX))
        return beta, edf, A

    def _gcv(self, X, y, log_lam):
        # Nelder-Mead probes extreme lambdas; a failure there is a bad proposal,
        # not a bad model, so score it as infinitely bad rather than crashing.
        try:
            beta, edf, _ = self._solve(X, y, self._penalty(np.exp(np.clip(log_lam, -30, 30))))
        except np.linalg.LinAlgError:
            return np.inf
        rss = float(((y - X @ beta) ** 2).sum())
        score = len(y) * rss / max(len(y) - edf, 1e-6) ** 2
        return score if np.isfinite(score) else np.inf

    # -- fitting ------------------------------------------------------------
    def fit(self, frame, y, groups) -> "GAMM":
        y = np.asarray(y, dtype=float)
        X = self.design(frame, fit=True)
        levels, gidx = np.unique(np.asarray(groups), return_inverse=True)
        counts = np.bincount(gidx, minlength=len(levels)).astype(float)
        n, q = len(y), len(levels)

        # Smoothing parameters by GCV, then refined once the player effects are
        # out of the residual.
        start = np.full(len(self.smooths), np.log(10.0))
        opt = minimize(lambda L: self._gcv(X, y, L), start, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-2, "fatol": 1e-2})
        self.lambdas_ = np.exp(opt.x)
        S = self._penalty(self.lambdas_)

        b = np.zeros(q)
        group_mean = np.bincount(gidx, weights=y, minlength=q) / counts
        s_b2 = max(float(np.var(group_mean)), 1e-3)
        s_e2 = max(float(np.var(y)), 1e-3)

        for outer in range(2):
            for _ in range(self.max_iter):
                beta, edf, A = self._solve(X, y - b[gidx], S)
                resid = y - X @ beta
                totals = np.bincount(gidx, weights=resid, minlength=q)

                denom = counts + s_e2 / s_b2
                b_new = totals / denom           # BLUP for each player
                post_var = s_e2 / denom          # and its posterior variance

                # EM updates. sigma_b^2 must add back the posterior variance:
                # the BLUPs are shrunken, so mean(b^2) alone understates it and
                # drives sigma_b to zero, collapsing this to a plain GAM.
                s_b2 = max((float((b_new**2).sum()) + float(post_var.sum())) / q, 1e-8)
                s_e2 = max(
                    (float(((resid - b_new[gidx]) ** 2).sum())
                     + float((counts * post_var).sum())) / n,
                    1e-8,
                )
                converged = np.max(np.abs(b_new - b)) < self.tol
                b = b_new
                if converged:
                    break

            if outer == 0:
                opt = minimize(lambda L: self._gcv(X, y - b[gidx], L),
                               np.log(self.lambdas_), method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-2, "fatol": 1e-2})
                self.lambdas_ = np.exp(opt.x)
                S = self._penalty(self.lambdas_)

        self.beta_ = beta
        # pinv rather than inv, and symmetrized: A is penalized and can be
        # near-singular when a smooth is heavily shrunk.
        vb = s_e2 * np.linalg.pinv(A)
        self.vbeta_ = (vb + vb.T) / 2.0
        self.sigma_b_, self.sigma_e_ = float(np.sqrt(s_b2)), float(np.sqrt(s_e2))
        self.group_effects_ = dict(zip(levels.tolist(), b.tolist()))
        return self

    # -- locally weighted refit ---------------------------------------------
    def refit_weighted(self, frame, y, groups, weights) -> "GAMM":
        """Refit the curves with per-row weights, reusing basis and penalties.

        This is what turns a comparison set into a *model*. Rather than picking
        five neighbours and averaging their year-over-year changes, every player
        in the league contributes to the fit in proportion to how similar they
        are to the target -- a locally weighted regression in style space. The
        weighted-average-of-deltas is the degenerate case: a top-hat kernel over
        five players, fit one season at a time with no smoothing.

        Knots, smoothing parameters and variance components are held at their
        values from the global fit. They describe the population, and
        re-estimating them from a reweighted subsample is noisy and slow; fixing
        them makes this a couple of linear solves, fast enough to run per query.
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(weights, dtype=float)
        X = self.design(frame)
        S = self._penalty(self.lambdas_)

        levels, gidx = np.unique(np.asarray(groups), return_inverse=True)
        wsum = np.bincount(gidx, weights=w, minlength=len(levels))
        ratio = self.sigma_e_**2 / self.sigma_b_**2

        local = GAMM(smooths=self.smooths, max_iter=self.max_iter, tol=self.tol)
        local.lambdas_ = self.lambdas_
        local.sigma_b_, local.sigma_e_ = self.sigma_b_, self.sigma_e_

        Xw = X * w[:, None]
        XwX = Xw.T @ X
        p = X.shape[1]

        # Shrink the local fit toward the global one, hard when the kernel is
        # concentrated. Effective sample size (Kish) says how many rows the
        # weights are really worth: when that approaches the number of basis
        # columns the weighted normal equations are near-singular and the
        # coefficients explode, so tau grows to pull beta back to the global
        # solution. With diffuse weights tau is negligible and this is an
        # ordinary weighted fit.
        ess = float(w.sum() ** 2 / max((w**2).sum(), 1e-12))
        tau = (p / max(ess, 1.0)) * float(np.mean(np.diag(XwX)))
        A = XwX + S + np.eye(p) * (tau + 1e-9 * max(np.trace(XwX), 1.0) / p)
        anchor = (S + np.eye(p) * tau) @ self.beta_

        b = np.zeros(len(levels))
        beta = self.beta_.copy()
        for _ in range(self.max_iter):
            rhs = Xw.T @ (y - b[gidx]) + anchor
            try:
                beta = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(A) @ rhs
            if not np.isfinite(beta).all():
                beta = self.beta_.copy()
                break
            resid = y - X @ beta
            b_new = np.bincount(gidx, weights=w * resid, minlength=len(levels)) / (
                wsum + ratio
            )
            if np.max(np.abs(b_new - b)) < self.tol:
                b = b_new
                break
            b = b_new

        local.beta_ = beta
        local.vbeta_ = self.vbeta_  # population-level, not re-derived
        local.group_effects_ = dict(zip(levels.tolist(), b.tolist()))
        return local

    # -- prediction ---------------------------------------------------------
    def predict_fixed(self, frame) -> np.ndarray:
        """The population curve: what this model expects of an unseen player."""
        return self.design(frame) @ self.beta_

    def fixed_se(self, frame) -> np.ndarray:
        """Standard error of the population curve itself."""
        X = self.design(frame)
        var = np.einsum("ij,jk,ik->i", X, self.vbeta_, X)
        # The penalized information matrix is ill-conditioned enough that this
        # quadratic form can come back a hair below zero; clip before the root
        # so a stray NaN cannot silently poison every interval downstream.
        return np.sqrt(np.maximum(var, 0.0))

    def blup(self, residuals) -> tuple[float, float]:
        """Shrunk intercept for a player, and its posterior SD.

        ``residuals`` are that player's observed values minus the population
        curve. Returns ``(level, sd)``.
        """
        residuals = np.asarray(residuals, dtype=float)
        k = len(residuals)
        if k == 0:
            return 0.0, self.sigma_b_
        denom = k + self.sigma_e_**2 / self.sigma_b_**2
        return float(residuals.sum() / denom), float(np.sqrt(self.sigma_e_**2 / denom))

    def forecast(self, history, future, y_col: str = "ws"):
        """Project ``y_col`` for one player.

        ``history`` holds their observed seasons, ``future`` the covariates of
        the seasons to project. Returns ``(mean, sd)`` arrays, where the SD
        combines uncertainty in the population curve, in the player's level,
        and the season-to-season residual.
        """
        level, level_sd = self.blup(
            np.asarray(history[y_col], dtype=float) - self.predict_fixed(history)
        )
        mean = self.predict_fixed(future) + level
        sd = np.sqrt(self.fixed_se(future) ** 2 + level_sd**2 + self.sigma_e_**2)
        return mean, sd

    # -- persistence --------------------------------------------------------
    @staticmethod
    def future_frame(history, n_ahead: int = 10, max_age: int = 40, pc_cols=None):
        """Covariates for a player's next ``n_ahead`` seasons.

        Age, career season and calendar year each advance by one. Playing style
        is held at the player's own average, so the forecast answers "if they
        keep playing like this, what does the aging curve do to them?" rather
        than guessing at a change of role.
        """
        import pandas as pd

        pc_cols = pc_cols or [c for c in history.columns if c.startswith("PC")]
        last = history.sort_values("Season_No").iloc[-1]
        n_ahead = max(0, min(n_ahead, max_age - int(last["Age"])))
        if n_ahead == 0:
            return pd.DataFrame(columns=["Age", "Season_No", "SEASON_YEAR", *pc_cols])

        steps = np.arange(1, n_ahead + 1)
        frame = pd.DataFrame(
            {
                "Age": int(last["Age"]) + steps,
                "Season_No": int(last["Season_No"]) + steps,
                "SEASON_YEAR": int(last["SEASON_YEAR"]) + steps,
            }
        )
        for c in pc_cols:
            frame[c] = float(history[c].mean())
        return frame

    # -- serialization ------------------------------------------------------
    def save(self, path) -> None:
        """Serialize to a small .npz so the app never refits at runtime."""
        payload = {
            "beta": self.beta_,
            "vbeta": self.vbeta_,
            "lambdas": self.lambdas_,
            "sigma_b": self.sigma_b_,
            "sigma_e": self.sigma_e_,
            "names": np.array([s.name for s in self.smooths]),
            "degrees": np.array([s.degree for s in self.smooths]),
            # "" stands in for None: npz has no null, and a varying-coefficient
            # term that lost its `by` on reload would silently become a plain
            # smooth with the wrong coefficients.
            "bys": np.array([s.by or "" for s in self.smooths]),
        }
        for i, s in enumerate(self.smooths):
            payload[f"knots_{i}"] = s.knots_
            payload[f"center_{i}"] = s.center_
            payload[f"cols_{i}"] = np.array(s.cols_)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path) -> "GAMM":
        z = np.load(path, allow_pickle=False)
        bys = z["bys"] if "bys" in z.files else [""] * len(z["names"])
        smooths = []
        for i, name in enumerate(z["names"]):
            s = Smooth(name=str(name), degree=int(z["degrees"][i]),
                       by=str(bys[i]) or None)
            s.knots_ = z[f"knots_{i}"]
            s.center_ = z[f"center_{i}"]
            s.cols_ = tuple(int(v) for v in z[f"cols_{i}"])
            smooths.append(s)
        model = cls(smooths=smooths)
        model.beta_ = z["beta"]
        model.vbeta_ = z["vbeta"]
        model.lambdas_ = z["lambdas"]
        model.sigma_b_ = float(z["sigma_b"])
        model.sigma_e_ = float(z["sigma_e"])
        return model
