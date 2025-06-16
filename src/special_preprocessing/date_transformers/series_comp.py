from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class GroupByDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "GroupByDateTransformer":
        self._is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        special_columns = ["price", "discount.1", "key", "date", "ship"]
        other = [col for col in X.columns if col not in special_columns]
        new_data = pd.DataFrame()
        keys = X["key"].unique()
        for key in keys:
            df_key = X[X["key"] == key]
            grouped = df_key.groupby(["key", "date", "mark"], as_index=False).agg(
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
    def __init__(self, fill_config: dict[str, Any]) -> None:
        self._is_fitted_: bool = False
        self.test_separator_: Optional[pd.DatetimeIndex] = None
        self.statistics_: Optional[dict] = None
        self.fixed_columns_: Optional[list[str]] = None
        self.first_values_: Optional[pd.DataFrame] = None
        self.fill_config = fill_config

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "DateRangeFilledTransformer":
        train_data = X[X["mark"] == "train"].copy()
        self.test_separator_ = train_data["date"].max()
        self.statistics_ = {}
        cols_to_agg = {col: method for col, method in self.fill_config.items() if isinstance(method, str)}
        if cols_to_agg:
            aggregated_stats = train_data.groupby("key").agg(cols_to_agg)
            for col in aggregated_stats.columns:
                self.statistics_[col] = aggregated_stats[col]
        special_cols = ["key", "date", "mark"] + list(self.fill_config.keys())
        self.fixed_columns_ = [col for col in X.columns if col not in special_cols]

        if self.fixed_columns_:
            self.first_values_ = train_data.groupby("key")[self.fixed_columns_].first()

        self._is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.test_separator_ is None or self.statistics_ is None or self.first_values_ is None:
            raise NotFittedError()
        date_range = pd.date_range(X["date"].min(), X["date"].max(), freq="D")
        unique_keys = X["key"].unique()

        missing_data_list = []

        for key in unique_keys:
            product_data = X[X["key"] == key]
            existing_dates = list(set(product_data["date"]))
            missing_dates = date_range.difference(existing_dates)

            if missing_dates.empty:
                continue

            new_rows_data = {
                "date": missing_dates,
                "key": key,
                "mark": np.where(missing_dates <= self.test_separator_, "train", "test"),
            }

            for col, method in self.fill_config.items():
                if isinstance(method, (int, float)):
                    new_rows_data[col] = method
                elif isinstance(method, str):
                    value = self.statistics_[col].get(key, 0)
                    new_rows_data[col] = value
            if self.fixed_columns_:
                for col in self.fixed_columns_:
                    if key in self.first_values_.index:
                        value = self.first_values_.loc[key, col]
                    else:
                        value = product_data[col].iloc[0]
                    new_rows_data[col] = value

            missing_data_list.append(pd.DataFrame(new_rows_data))

        if missing_data_list:
            X = pd.concat([X] + missing_data_list, ignore_index=True).reset_index(drop=True)
        return X
