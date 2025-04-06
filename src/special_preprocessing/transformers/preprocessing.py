import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ConcatenateTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X_train = None

    def fit(self, X: pd.DataFrame, y=pd.DataFrame) -> "Self":
        self.X_train = pd.concat([X, y], axis=1)
        print(self.X_train)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.X_train is not None:
            transformed = self.X_train
            self.X_train = None
            return transformed
        X["ship"] = np.nan
        return X


class NaNHandlerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> "Self":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["discount"] = X["discount"].fillna(0)
        X["discount"] = X["discount"].replace("promo", 1)
        X = X.dropna(subset=["date"])
        return X


class ChangeTypesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> "Self":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["discount"] = X["discount"].astype(bool)
        return X


class KeyIndexTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> "Self":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        columns = ["channel", "level_1", "level_2", "level_3", "brend"]
        X["key"] = X[columns].astype(str).agg("/".join, axis=1)
        return X


class DropDuplicatesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> "Self":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop_duplicates()
