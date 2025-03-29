import datetime

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from workalendar.europe import Russia


class HolidayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.holidays = None

    def get_holidays(self, year):
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
            holidays.append(
                (calc[name] - datetime.timedelta(days=1), f"Day before {name}")
            )
            holidays.append(
                (calc[name] - datetime.timedelta(days=2), f"Two days before {name}")
            )
        return holidays

    def check_holiday(self, date):
        for holiday in self.holidays:
            if (date.date() - holiday[0]).days <= 21:
                return holiday[1]
        return "No holiday"

    def fit(self, X, y=None):
        years = X["date"].dt.year.unique()
        self.holidays = [h for year in years for h in self.get_holidays(year)]
        return self

    def transform(self, X):
        X = X.copy()
        X["holiday"] = X["date"].apply(self.check_holiday)
        X["dayofyear"] = X["date"].dt.dayofyear
        return X


class MeanWeekMonthTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["month"] = X["date"].dt.month
        X["week"] = X["date"].dt.isocalendar().week
        X["mean_month_ship"] = X.groupby(["key", "month"])["ship"].transform("mean")
        X["mean_week_ship"] = X.groupby(["key", "week"])["ship"].transform("mean")
        X.drop(["week", "month"], axis=1, inplace=True)
        return X


class FourierFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, order=4):
        self.order = order
        self.dp = None

    def fit(self, X, y=None):
        date_index = pd.DatetimeIndex(sorted(X["date"].unique()))
        fourier = CalendarFourier(freq="A", order=self.order)
        self.dp = DeterministicProcess(
            index=date_index,
            constant=False,
            order=0,
            seasonal=True,
            additional_terms=[fourier],
            drop=True,
        )
        return self

    def transform(self, X):
        X = X.copy()
        fourier_features = self.dp.in_sample()
        X = X.merge(fourier_features, how="left", left_on="date", right_index=True)

        for i in range(1, self.order + 1):
            cos_col, sin_col = f"cos({i},freq=YE-DEC)", f"sin({i},freq=YE-DEC)"
            X[f"season_cos_{i}"] = X[cos_col]
            X[f"season_neg_cos_{i}"] = -X[cos_col]
            X[f"season_sin_{i}"] = X[sin_col]
            X[f"season_neg_sin_{i}"] = -X[sin_col]

        columns_to_drop = [
            f"cos({i},freq=YE-DEC)" for i in range(1, self.order + 1)
        ] + [f"sin({i},freq=YE-DEC)" for i in range(1, self.order + 1)]
        X.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        return X
