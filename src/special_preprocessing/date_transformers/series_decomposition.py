from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import seasonal_decompose


class SeriesDecompositionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model: str = "additive", extrapolate_trend: str = "freq") -> None:
        self.model: str = model
        self.extrapolate_trend: str = extrapolate_trend
        self.X_train_: Optional[pd.DataFrame] = None

    def fit(self, X_united: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "SeriesDecompositionTransformer":
        X_train = X_united[X_united["mark"] == "train"]
        decomposed_data = []

        for key, group in X_train.groupby("key"):
            group = group.set_index("date").sort_index()

            result = seasonal_decompose(
                group["ship"],
                model=self.model,
                extrapolate_trend=self.extrapolate_trend,
            )

            group["trend"] = result.trend
            group["seasonal"] = result.seasonal + result.resid

            decomposed_data.append(group.reset_index())

        X_train = pd.concat(decomposed_data)
        self.X_train_ = X_train
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_test = X[X["mark"] == "test"].copy()
        X_test["trend"] = 0
        X_test["seasonal"] = 0
        return pd.concat([self.X_train_, X_test])


class Separation(BaseEstimator, TransformerMixin):
    def __init__(self, production_mode: bool = True) -> None:
        self.production_mode: bool = production_mode
        self.X_train_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "Separation":
        self.X_train_ = X[X["mark"] == "train"].drop(columns=["mark", "ship"])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        if not self.production_mode:
            return X
        if self.X_train_ is not None:
            train = self.X_train_
            self.X_train_ = None
            return train
        return X[X["mark"] == "test"].drop(columns=["ship", "trend", "seasonal", "mark"])
