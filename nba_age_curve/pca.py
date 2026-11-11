"""PCA over the 12 style features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nba_age_curve.config import FEATURES, N_COMPONENTS


@dataclass
class SeasonPCA:
    """A fitted scaler + PCA pair, with the feature order it was fit on."""

    scaler: StandardScaler
    pca: PCA
    features: list[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Project rows of ``df`` into component space."""
        return self.pca.transform(self.scaler.transform(df[self.features]))

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.pca.explained_variance_

    @property
    def n_kaiser(self) -> int:
        """Components to retain under the Kaiser rule (eigenvalue > 1)."""
        return int((self.eigenvalues > 1).sum())

    def loadings(self, n_components: int | None = None) -> pd.DataFrame:
        """Feature weights per component: what each component measures."""
        n = n_components or self.pca.n_components_
        return pd.DataFrame(
            self.pca.components_[:n].T,
            index=self.features,
            columns=[f"PC{i + 1}" for i in range(n)],
        )

    def variance_table(self) -> pd.DataFrame:
        evr = self.explained_variance_ratio
        return pd.DataFrame(
            {
                "explained_var": evr,
                "cumulative": evr.cumsum(),
                "eigenvalue": self.eigenvalues,
            },
            index=[f"PC{i + 1}" for i in range(len(evr))],
        )


def fit_season_pca(
    df: pd.DataFrame,
    features: list[str] | None = None,
    n_components: int | None = None,
) -> SeasonPCA:
    """Standardize ``features`` and fit PCA.

    Features are z-scored first so that every stat contributes equally,
    regardless of its natural scale. With ``n_components=None`` all components
    are kept, which is what the scree and loadings plots need.
    """
    features = features or FEATURES
    scaler = StandardScaler().fit(df[features])
    pca = PCA(n_components=n_components).fit(scaler.transform(df[features]))
    return SeasonPCA(scaler=scaler, pca=pca, features=list(features))


def attach_scores(
    df: pd.DataFrame, model: SeasonPCA, n_components: int = N_COMPONENTS
) -> pd.DataFrame:
    """Return ``df`` with ``PC1``..``PCn`` columns added."""
    scores = model.transform(df)
    out = df.copy()
    for i in range(n_components):
        out[f"PC{i + 1}"] = scores[:, i]
    return out
