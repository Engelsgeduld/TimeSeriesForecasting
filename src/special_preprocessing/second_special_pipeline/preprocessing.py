from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class RenameColumns(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "RenameColumns":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.rename(
            columns={
                "CUSTOMER_CODE": "channel",
                "PROMO_FAMILY_CD": "product",
                "MASTER_FAMILY_CD": "group",
                "PLAN_INVEST_V": "discount.1",
                "PHY_CS_V": "ship",
            }
        )
        return X


class KeyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "KeyTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["key"] = X["channel"] + "/" + X["group"] + "/" + X["product"]
        return X


class DiscountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "DiscountTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["discount"] = X["discount.1"].apply(lambda x: True if x > 0 else False)
        return X


class CategoricalFeaturesTransform(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._transformer: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "CategoricalFeaturesTransform":
        ohe_features = ["channel", "group"]
        label_features = ["product"]
        self._transformer = ColumnTransformer(
            transformers=[
                (
                    "OHE",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ohe_features,
                ),
                ("OrdinalTransform", OrdinalEncoder(handle_unknown="error"), label_features),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
            force_int_remainder_cols=False,
        )
        self._transformer.set_output(transform="pandas")
        self._transformer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._transformer is None:
            raise NotFittedError()
        return self._transformer.transform(X)


class DateRangeFilledTransformerSec(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "DateRangeFilledTransformerSec":
        self._is_fitted_ = True
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
