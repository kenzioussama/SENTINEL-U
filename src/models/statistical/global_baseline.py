import numpy as np
import pandas as pd


class GlobalBaseline:

    FEATURES = [
        "failed_logins",
        "mb_out",
        "dns_queries"
    ]

    def __init__(self, n_sigma=3):

        self.n_sigma = n_sigma

        self.statistics_ = None
        self.thresholds_ = None

        self.is_fitted_ = False

    # ---------------------------------
    # 1. Transformation des variables
    # ---------------------------------

    def _transform(self, df):

        X = df[self.FEATURES].copy()

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if X.isnull().any().any():
            raise ValueError(
                "Missing or non-numeric values detected."
            )

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                "Infinite values detected."
            )

        if (X < 0).any().any():
            raise ValueError(
                "Negative values cannot be transformed with log1p."
            )

        return np.log1p(X)

    # ---------------------------------
    # 2. Apprentissage du modèle
    # ---------------------------------

    def fit(self, train_df):

        X = self._transform(train_df)

        self.statistics_ = pd.DataFrame({

            "mean": X.mean(),

            "std": X.std(ddof=1)

        })

        if self.statistics_["std"].isna().any():
            raise ValueError(
                "Not enough training observations."
            )

        self.thresholds_ = (
            self.statistics_["mean"]
            + self.n_sigma * self.statistics_["std"]
        )

        self.is_fitted_ = True

        return self

    # ---------------------------------
    # 3. Calcul du score d'anomalie
    # ---------------------------------

    def score_samples(self, df):

        if not self.is_fitted_:
            raise RuntimeError(
                "Model must be fitted before scoring."
            )

        X = self._transform(df)

        mean = self.statistics_["mean"]

        std = self.statistics_["std"]

        # Protection contre les écarts-types nuls
        safe_std = std.mask(std == 0, 1e-12)

        z_scores = (X - mean) / safe_std

        scores = z_scores.max(axis=1)

        return scores.to_numpy()

    # ---------------------------------
    # 4. Détection des anomalies
    # ---------------------------------

    def predict(self, df):

        if not self.is_fitted_:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        X = self._transform(df)

        alerts = (
            X > self.thresholds_
        ).any(axis=1)

        return alerts.to_numpy()