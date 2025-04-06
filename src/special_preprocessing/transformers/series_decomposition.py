import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import seasonal_decompose


class SeriesDecompositionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model: str = "additive", extrapolate_trend: str = "freq") -> None:
        self.model = model
        self.extrapolate_trend = extrapolate_trend
        self.X_train = None

    def fit(self, X: pd.DataFrame, y=None) -> "Self":
        decomposed_data = []

        for key, group in X.groupby("key"):
            group = group.set_index("date").sort_index()

            result = seasonal_decompose(
                group["ship"],
                model=self.model,
                extrapolate_trend=self.extrapolate_trend,
            )

            group["trend"] = result.trend
            group["seasonal"] = result.seasonal + result.resid

            decomposed_data.append(group.reset_index())

        self.X_train = pd.concat(decomposed_data)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.X_train is not None:
            transformed = self.X_train
            self.X_train = None
            return transformed
        return X
