import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupByDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X_train = None

    def fit(self, X: pd.DataFrame, y=None) -> "Self":
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
        self.X_train = new_data
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.X_train is not None:
            transformed = self.X_train
            self.X_train = None
            return transformed
        return X


class DateRangeFilledTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X_train = None

    def fit(self, X: pd.DataFrame, y=None) -> "Self":
        special_columns = ["discount", "price", "discount.1", "key", "date", "ship"]
        other = [col for col in X.columns if col not in special_columns]

        date_range = pd.date_range(X["date"].min(), X["date"].max(), freq="D")
        missing_data = []
        unique_keys = X["key"].unique()

        for key in unique_keys:
            product_data = X[X["key"] == key]
            existing_dates = set(product_data["date"])
            missing_dates = date_range.difference(existing_dates)

            if missing_dates.empty:
                continue
            new_dt = pd.DataFrame(
                {
                    "date": missing_dates,
                    "ship": 0,
                    "discount": 0,
                    "discount.1": 0,
                    "price": product_data["price"].iloc[0],
                    "key": key,
                }
            )

            for col in other:
                new_dt[col] = product_data[col].iloc[0]

            missing_data.append(new_dt)

        if missing_data:
            self.X_train = pd.concat([X] + missing_data, ignore_index=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.X_train is not None:
            transformed = self.X_train
            self.X_train = None
            return transformed.sort_values(by=["key", "date"]).reset_index(drop=True)
        return X
