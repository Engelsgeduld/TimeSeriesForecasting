from sklearn.base import BaseEstimator, TransformerMixin


class NaNHandlerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["discount"] = X["discount"].fillna(0)
        X["discount"] = X["discount"].replace("promo", 1)
        X = X.dropna(subset=["date"])
        return X


class ChangeTypesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["discount"] = X["discount"].astype(bool)
        return X


class KeyIndexTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        columns = ["channel", "level_1", "level_2", "level_3", "brend"]
        X["key"] = X[columns].astype(str).agg("/".join, axis=1)
        return X


class DropDuplicatesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates()
