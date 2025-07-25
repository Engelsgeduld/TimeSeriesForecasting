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
        X["key"] = X["channel"].astype("str") + "/" + X["group"].astype("str") + "/" + X["product"].astype("str")
        return X


class DiscountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "DiscountTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["discount"] = X["discount.1"].apply(lambda x: 1 if x > 0 else 0)
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
