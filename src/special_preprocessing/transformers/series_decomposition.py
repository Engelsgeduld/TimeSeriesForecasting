import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import seasonal_decompose


class SeriesDecompositionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model="additive", extrapolate_trend="freq"):
        self.model = model
        self.extrapolate_trend = extrapolate_trend

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

        return pd.concat(decomposed_data)
