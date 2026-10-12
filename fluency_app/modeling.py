"""Small sklearn-compatible helpers shared by research notebooks and serving."""

from __future__ import annotations

import numpy as np
import warnings
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin


class SpearmanTopKSelector(BaseEstimator, TransformerMixin):
    """Select top features by absolute Spearman association, fit on training data only."""

    def __init__(self, k: int = 25):
        self.k = k

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        scores = []
        for column in range(values.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = spearmanr(values[:, column], y).statistic
            scores.append(abs(float(rho)) if np.isfinite(rho) else 0.0)
        self.scores_ = np.asarray(scores)
        self.indices_ = np.argsort(self.scores_)[::-1][: min(self.k, values.shape[1])]
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices_]

    def get_support(self):
        return self.indices_
