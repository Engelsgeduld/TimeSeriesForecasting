import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupByDateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> "GroupByDateTransformer":
        self._is_fitted_: bool = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        special_columns = ["discount", "price", "discount.1", "key", "date", "ship"]
        other = [col for col in X.columns if col not in special_columns]
        new_data = pd.DataFrame()
        keys = X["key"].unique()
        for key in keys:
            df_key = X[X["key"] == key]
            grouped = df_key.groupby(["key", "date"], as_index=False).agg(
                {
                    **{col: "max" for col in other},
                    "ship": "sum",
                    "discount.1": "mean",
                    "price": "mean",
                }
            )
            new_data = pd.concat([new_data, grouped], ignore_index=True)
        return new_data


class DateRangeFilledTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> "DateRangeFilledTransformer":
        self._is_fitted_: bool = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        test_separator = X[X["mark"] == "train"]["date"].max()
        special_columns = ["discount", "price", "discount.1", "key", "date", "ship", "mark"]
        other = [col for col in X.columns if col not in special_columns]

        date_range = pd.date_range(X["date"].min(), X["date"].max(), freq="D")
        missing_data = []
        unique_keys = X["key"].unique()

        for key in unique_keys:
            product_data = X[X["key"] == key]
            existing_dates = list(set(product_data["date"]))
            missing_dates = date_range.difference(existing_dates)

            if missing_dates.empty:
                continue
            new_dt = pd.DataFrame(
                {
                    "date": missing_dates,
                    "ship": 0,
                    "discount": 0,
                    "discount.1": 0,
                    "price": product_data["price"].mean(),
                    "key": key,
                    "mark": np.where(missing_dates <= test_separator, "train", "test"),
                }
            )

            for col in other:
                new_dt[col] = product_data[col].iloc[0]
            missing_data.append(new_dt)

        if missing_data:
            X = pd.concat([X] + missing_data, ignore_index=True)
        return X
