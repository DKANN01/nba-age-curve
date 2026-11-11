"""Reproduce the exploratory PCA analysis as static figures.

This is the notebook half of the original script -- the scree plot, loadings,
biplot and component extremes -- rewritten to run anywhere (the original used
Colab's `display`) and to write figures to disk instead of a notebook cell.

    python scripts/analysis.py --outdir figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nba_age_curve.config import FEATURES, N_COMPONENTS  # noqa: E402
from nba_age_curve.data import load_player_seasons  # noqa: E402
from nba_age_curve.pca import attach_scores, fit_season_pca  # noqa: E402

#: Minimum minutes per game for a player-season to appear in the extremes list.
#: This filters the *report* only -- it never changes the PCA, which is fit on
#: every player-season.
MIN_MPG = 15


def plot_scree(model, outdir: Path) -> None:
    evr = model.explained_variance_ratio
    n = len(evr)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n + 1), evr, color="#4C72B0", label="Individual")
    ax.plot(range(1, n + 1), evr.cumsum(), "o-", color="#DD8452", label="Cumulative")
    # Under a z-scored PCA, eigenvalue 1 is exactly 1/n of total variance,
    # so the Kaiser cutoff is a horizontal line at 1/n of explained variance.
    ax.axhline(1 / n, ls="--", c="gray", lw=0.8, label="Kaiser (eigenvalue = 1)")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Share of variance")
    ax.set_xticks(range(1, n + 1))
    ax.legend()
    ax.set_title("Scree plot")
    fig.tight_layout()
    fig.savefig(outdir / "scree.png", dpi=150)
    plt.close(fig)


def plot_loadings(model, outdir: Path, k: int = N_COMPONENTS) -> None:
    loadings = model.loadings(k)
    evr = model.explained_variance_ratio

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(loadings, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"PC{i + 1}\n({evr[i]:.0%})" for i in range(k)])
    ax.set_yticks(range(len(loadings.index)))
    ax.set_yticklabels(loadings.index)
    for (r, c), v in np.ndenumerate(loadings.values):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Loadings")
    fig.tight_layout()
    fig.savefig(outdir / "loadings.png", dpi=150)
    plt.close(fig)


def plot_biplot(df, model, outdir: Path) -> None:
    scores = model.transform(df)[:, :2]
    loadings = model.loadings(2)
    evr = model.explained_variance_ratio

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(scores[:, 0], scores[:, 1], s=4, alpha=0.15, c="#888")

    scale = np.percentile(np.abs(scores), 99) * 1.2
    for feature in FEATURES:
        x = loadings.loc[feature, "PC1"] * scale
        y = loadings.loc[feature, "PC2"] * scale
        ax.arrow(0, 0, x, y, color="#C44E52", head_width=0.08, length_includes_head=True)
        ax.text(x * 1.1, y * 1.1, feature, color="#C44E52", fontsize=9, ha="center")

    # A handful of very-low-minute seasons are extreme outliers; zoom to the
    # middle 99% so the bulk of the cloud is actually readable.
    lo, hi = np.percentile(scores, [0.5, 99.5], axis=0)
    ax.set_xlim(lo[0] - 0.5, hi[0] + 0.5)
    ax.set_ylim(lo[1] - 0.5, hi[1] + 0.5)
    ax.axhline(0, c="k", lw=0.4)
    ax.axvline(0, c="k", lw=0.4)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("Player-seasons on PC1 vs PC2 (middle 99% shown)")
    fig.tight_layout()
    fig.savefig(outdir / "biplot.png", dpi=150)
    plt.close(fig)


def report_extremes(df, k: int = N_COMPONENTS) -> None:
    regulars = df[df["MIN/GM"] >= MIN_MPG]
    print(f"\nComponent extremes (player-seasons at >= {MIN_MPG} MPG)")
    for i in range(k):
        col = f"PC{i + 1}"
        for end, rows in (
            ("high", regulars.nlargest(5, col)),
            ("low ", regulars.nsmallest(5, col)),
        ):
            names = [f"{r.PLAYER_NAME} {r.SEASON}" for r in rows.itertuples()]
            print(f"  {col} {end}: {', '.join(names)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_player_seasons()
    print(f"{len(df):,} player-seasons, {df['PLAYER_ID'].nunique():,} players")

    model = fit_season_pca(df)
    print("\nVariance explained")
    print(model.variance_table().round(3).to_string())
    print(
        f"\nComponents with eigenvalue > 1 (Kaiser): {model.n_kaiser} "
        f"({model.explained_variance_ratio[: model.n_kaiser].sum():.1%} of variance)"
    )
    print("\nLoadings")
    print(model.loadings(N_COMPONENTS).round(2).to_string())

    plot_scree(model, args.outdir)
    plot_loadings(model, args.outdir)
    plot_biplot(df, model, args.outdir)
    report_extremes(attach_scores(df, model))
    print(f"Figures written to {args.outdir}/")


if __name__ == "__main__":
    main()
