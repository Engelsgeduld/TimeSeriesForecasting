import datetime
from typing import Any, Never, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from workalendar.europe import Russia


class HolidayTransformer(BaseEstimator, TransformerMixin):
    @staticmethod
    def get_holidays(year: Any) -> list[tuple[datetime.datetime, str]]:
        names = [
            "New year",
            "Christmas",
            "Defendence of the Fatherland",
            "International Women's Day",
            "The Day of Spring and Labour",
            "Victory Day",
            "National Day",
            "Day of Unity",
        ]
        calc = {y: x for x, y in Russia().holidays(year)}
        holidays = []
        for name in names:
            holidays.append((calc[name], name))
            holidays.append((calc[name] - datetime.timedelta(days=1), f"Day before {name}"))
            holidays.append((calc[name] - datetime.timedelta(days=2), f"Two days before {name}"))
        return holidays

    @staticmethod
    def check_holiday(date: datetime.datetime, holidays: list[tuple[datetime.datetime, str]]) -> str:
        for holiday in holidays:
            diff: datetime.timedelta = date.date() - holiday[0]
            if diff.days <= 21:
                return holiday[1]
        return "No holiday"

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "HolidayTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        years = X["date"].dt.year.unique()
        holidays = [h for year in years for h in self.get_holidays(year)]
        X = X.copy()
        X["holiday"] = X["date"].apply(self.check_holiday, args=(holidays,))
        X["dayofyear"] = X["date"].dt.dayofyear
        return X


class MeanWeekMonthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.month_avg: Optional[pd.DataFrame] = None
        self.week_avg: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "MeanWeekMonthTransformer":
        X_train = X[X["mark"] == "train"].copy()
        X_train["month"] = X_train["date"].dt.month
        X_train["week"] = X_train["date"].dt.isocalendar().week
        self.month_avg = X_train.groupby(["key", "month"], as_index=False).agg({"ship": "mean"})
        self.week_avg = X_train.groupby(["key", "week"], as_index=False).agg({"ship": "mean"})
        self.month_avg["month_avg"] = self.month_avg["ship"]
        self.week_avg["week_avg"] = self.week_avg["ship"]
        self.month_avg.drop(columns=["ship"], inplace=True)
        self.week_avg.drop(columns=["ship"], inplace=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.month_avg is None or self.week_avg is None:
            raise NotFittedError()
        X["month"] = X["date"].dt.month
        X["week"] = X["date"].dt.isocalendar().week
        X = X.merge(self.month_avg, on=["key", "month"], how="left")
        X = X.merge(self.week_avg, on=["key", "week"], how="left")
        X.drop(columns=["month", "week"], inplace=True)
        return X


class FourierFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, order: int = 4) -> None:
        self.order: int = order
        self._is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "FourierFeaturesTransformer":
        self._is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        date_index = pd.date_range(X["date"].min(), X["date"].max(), freq="D")
        fourier = CalendarFourier(freq="YE", order=self.order)
        dp = DeterministicProcess(
            index=date_index,
            constant=False,
            order=0,
            seasonal=True,
            additional_terms=[fourier],
            drop=True,
        )
        fourier_features = dp.in_sample()
        X = X.merge(fourier_features, how="left", left_on="date", right_index=True)

        for i in range(1, self.order + 1):
            cos_col, sin_col = f"cos({i},freq=YE-DEC)", f"sin({i},freq=YE-DEC)"
            X[f"season_cos_{i}"] = X[cos_col]
            X[f"season_neg_cos_{i}"] = -X[cos_col]
            X[f"season_sin_{i}"] = X[sin_col]
            X[f"season_neg_sin_{i}"] = -X[sin_col]

        columns_to_drop = [f"cos({i},freq=YE-DEC)" for i in range(1, self.order + 1)] + [
            f"sin({i},freq=YE-DEC)" for i in range(1, self.order + 1)
        ]
        X.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        return X


class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.pipe: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "FeatureExtractionTransformer":
        self.pipe = self.features_pipeline()
        self.pipe.fit(X, y)
        return self

    @staticmethod
    def features_pipeline() -> Pipeline:
        ohe = ColumnTransformer(
            transformers=[
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["holiday"],
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
            force_int_remainder_cols=False,
        )
        ohe.set_output(transform="pandas")
        pipline = Pipeline(
            steps=[
                ("date_feature_transform", HolidayTransformer()),
                ("ohe", ohe),
                ("mean_ship_feature", MeanWeekMonthTransformer()),
                (
                    "fourier_features",
                    FourierFeaturesTransformer(),
                ),
            ]
        )
        return pipline

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipe is None:
            raise NotFittedError()
        return self.pipe.transform(X)
