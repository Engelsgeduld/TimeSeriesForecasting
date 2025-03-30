from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.metrics.forecast_accuracy import forecast_accuracy


class TimeSeriesModel(BaseEstimator, RegressorMixin):
    """
    A time series forecasting model that fits separate trend and seasonal models.

    This model selects the best trend and seasonal models for each unique key
    using cross-validation and evaluates performance using a given metric.
    """

    def __init__(
        self,
        trend_models: list[tuple[Any, dict[str, list[float]]]],
        seasonal_models: list[tuple[Any, dict[str, list[float]]]],
        keys_index: str = "key",
        date_index: str = "date",
        trend_index: str = "trend",
        seasonal_index: str = "seasonal",
        cv: Any = 3,
        scoring: str = "neg_mean_absolute_error",
    ):
        """
        Initializes the TimeSeriesModel with trend and seasonal models.

        Args:
            trend_models (list[tuple[BaseEstimator, dict[str, list[float]]]]):
                List of machine learning models and params grids.
            seasonal_models (list[tuple[Any, dict[str, list[float]]]]):
                List of machine learning models and params grids.
            keys_index (str, optional):
                Column name in `X` representing unique identifiers for different time series.
                Defaults to `"key"`.
            date_index (str, optional):
                Column name in `X` representing dates for different time series.
                Defaults to `"date"`.
            trend_index (str, optional):
                Column name in `y` that contains trend values. Defaults to `"trend"`.
            seasonal_index (str, optional):
                Column name in `y` that contains seasonal values. Defaults to `"seasonal"`.
            cv (Any, optional):
                Cross-validation strategy. Defaults to `3`.
            scoring (str, optional):
                Scoring metric for model selection. Defaults to `"neg_mean_absolute_error"`.
        """
        self.trend_models = trend_models
        self.seasonal_models = seasonal_models
        self.keys_index = keys_index
        self.date_index = date_index
        self.trend_index = trend_index
        self.seasonal_index = seasonal_index
        self.cv = cv
        self.scoring = scoring
        self.best_models_: dict[str, tuple[BaseEstimator, BaseEstimator]] = dict()
        self.known_keys: Optional[NDArray] = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> "TimeSeriesModel":
        """
        Trains the model by selecting the best trend and seasonal models for each key.

        This method performs the following steps:
        - Validates input data.
        - Sets up cross-validated grid searches for trend and seasonal models.
        - Fits the best models for each unique key in the dataset.
        - Stores the best models for future predictions.

        Args:
            X (pd.DataFrame):
                Feature dataframe containing a column with unique keys for different time series.
            y (pd.DataFrame | pd.Series):
                Target dataframe containing trend and seasonal values.

        Returns:
            TimeSeriesModel:
                The trained instance of the model.

        Raises:
            ValueError: If `X` or `y` does not contain the expected columns.
        """
        self._validate_X(X)
        self._validate_y(y)
        keys = X[self.keys_index].unique()
        for key in keys:
            grid_search_trend, grid_search_seasonal = self._setup_searchers()
            train, trend, season = self._setup_train_sets(X, y, key)
            grid_search_trend.fit(train, trend)
            grid_search_seasonal.fit(train, season)
            trend_model = grid_search_trend.best_estimator_
            seasonal_model = grid_search_seasonal.best_estimator_
            trend_model.set_params(**grid_search_trend.best_params_)
            seasonal_model.set_params(**grid_search_seasonal.best_params_)
            trend_model.fit(train, trend)
            seasonal_model.fit(train, season)
            self.best_models_[key] = trend_model, seasonal_model
        self.known_keys = keys
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates forecasts using the best trained trend and seasonal models for each key.

        Args:
            X (pd.DataFrame):
                Feature dataframe containing a column with unique keys for different time series.

        Returns:
            pd.DataFrame:
                DataFrame with keys, dates and forecasts.

        Raises:
            NotFittedError: If the model has not been trained before calling `predict`.
            ValueError: If `X` contains an unknown key.
        """
        self._validate_X(X)
        if self.known_keys is None or len(self.best_models_) == 0:
            raise NotFittedError
        if not np.all(np.isin(X[self.keys_index].unique(), self.known_keys)):
            raise ValueError("Got unknown key")
        forecast_list: list[tuple[Any, Any, Any]] = []
        for key in X[self.keys_index].unique():
            mask = X[self.keys_index] == key
            trend_pred = self.best_models_[key][0].predict(X[mask].drop(columns=[self.keys_index, self.date_index]))
            seasonal_pred = self.best_models_[key][1].predict(X[mask].drop(columns=[self.keys_index, self.date_index]))
            forecast_values = trend_pred + seasonal_pred
            forecast_list.extend(zip(X.loc[mask, self.keys_index], X.loc[mask, self.date_index], forecast_values))
        forecast_df = pd.DataFrame(forecast_list, columns=[self.keys_index, self.date_index, "Forecast"])
        return forecast_df

    def score(self, X: pd.DataFrame, y: pd.Series, time_index: str = "date", sample_weight: Any = None) -> pd.DataFrame:
        """
        Computes the performance score of the model for each key using the forecast accuracy metric.

        Args:
            X (pd.DataFrame):
                Feature dataframe.
            y (pd.DataFrame):
                Target dataframe containing true values.
            time_index (str):
                Name of date column in X
            sample_weight (optional):
                Sample weights for scoring. Not currently used.

        Returns:
           pd.DataFrame: A DataFrame with computed metrics:
            - `{y_true_index}`: Sum of actual values.
            - `{y_pred_index}`: Sum of forecasted values.
            - `AE`: Absolute error.
            - `Acc`: Forecast accuracy (1 - AE / Actual), handling division by zero.

        Raises:
            NotFittedError: If the model has not been trained before calling `score`.
            ValueError: If `X` contains an unknown key.
        """
        y_pred = self.predict(X)
        data = X.copy()
        data["Actual"] = y.values
        data = data.merge(y_pred, on=[self.keys_index, self.date_index], how="left")
        scores = forecast_accuracy(data, time_index=time_index, ae_gr_cols=[self.keys_index])
        return scores

    def _validate_X(self, X: pd.DataFrame) -> None:
        """
        Validates the input feature dataframe.

        Ensures that the `keys_index` column exists in `X`.

        Args:
            X (pd.DataFrame):
                The input feature dataframe.

        Raises:
            ValueError: If `keys_index` is not found in `X`.
            ValueError: If `date_index` is not found in `X`.
        """
        if not self.keys_index in X.columns:
            raise ValueError(f"{self.keys_index} not in X")
        if not self.date_index in X.columns:
            raise ValueError(f"{self.date_index} not in X")

    def _validate_y(self, y: pd.DataFrame) -> None:
        """
        Validates the target dataframe.

        Ensures that both the `trend_index` and `seasonal_index` columns exist in `y`.

        Args:
            y (pd.DataFrame):
                The target dataframe.

        Raises:
            ValueError: If `trend_index` or `seasonal_index` is not found in `y`.
        """
        if not self.trend_index in y.columns:
            raise ValueError(f"{self.trend_index} not in y")
        if not self.seasonal_index in y.columns:
            raise ValueError(f"{self.seasonal_index} not in y")

    def _setup_searchers(self) -> tuple[GridSearchCV, GridSearchCV]:
        """
        Creates and configures GridSearchCV objects for trend and seasonal models.

        Uses a base pipeline with a `LinearRegression` model as a placeholder.
        The grid search selects the best models from `trend_models` and `seasonal_models`
        based on cross-validation.

        Returns:
            tuple[GridSearchCV, GridSearchCV]:
                A tuple containing the GridSearchCV objects for trend and seasonal models.
        """
        trend_pipe = Pipeline([("Regressor", LinearRegression())])
        seasonal_pipe = Pipeline([("Regressor", LinearRegression())])

        t_models = self._create_pipelines(self.trend_models)
        s_models = self._create_pipelines(self.seasonal_models)

        grid_search_trend = GridSearchCV(
            trend_pipe,
            t_models,
            cv=self.cv,
            scoring=self.scoring,
            verbose=0,
            n_jobs=-1,
        )
        grid_search_seasonal = GridSearchCV(
            seasonal_pipe,
            s_models,
            cv=self.cv,
            scoring=self.scoring,
            verbose=0,
            n_jobs=-1,
        )
        return grid_search_trend, grid_search_seasonal

    def _setup_train_sets(
        self, X: pd.DataFrame, y: pd.DataFrame, key: str
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Extracts training data for a specific time series key.

        Filters `X` and `y` to create training sets for trend and seasonal forecasting.

        Args:
            X (pd.DataFrame):
                The input feature dataframe.
            y (pd.DataFrame):
                The target dataframe containing trend and seasonal values.
            key (str):
                The unique identifier for the time series.

        Returns:
            tuple[pd.DataFrame, pd.Series, pd.Series]:
                - `train`: The feature subset for the given key.
                - `trend`: The target trend values for the given key.
                - `season`: The target seasonal values for the given key.
        """
        mask = X[self.keys_index] == key
        train = X[mask].drop(columns=[self.keys_index, self.date_index])
        trend = y[mask][self.trend_index]
        season = y[mask][self.seasonal_index]
        return train, trend, season

    def _create_pipelines(self, models: list[tuple[Any, dict]]) -> list[dict[str, list]]:
        """
        Creates a list of pipelines for regression models with their corresponding parameters.

        Args:
            models (list[tuple[Any, dict]]):
                A list of tuples, where each tuple consists of a model (a class, not an instance)
                and a dictionary of parameters.
                Example input:
                [
                    (SomeRegressor, {"param1": 10, "param2": 0.5}),
                    (AnotherRegressor, {"alpha": 0.01})
                ]

        Returns:
            list[dict[str, list]]:
                A list of dictionaries, where each dictionary represents a regression model
                with its parameters formatted in a way compatible with `Pipeline` in `scikit-learn`.
                Example output:
                [
                    {"Regressor": [SomeRegressor()], "Regressor__param1": 10, "Regressor__param2": 0.5},
                    {"Regressor": [AnotherRegressor()], "Regressor__alpha": 0.01}
                ]

        Notes:
            - The function assumes that the provided models are classes, not already instantiated objects.
            - The parameters are transformed into the format "Regressor__parameter_name,"
              which is compatible with `Pipeline` in `scikit-learn`.
        """
        set_models = []
        for pair in models:
            model, params = pair
            named_params = dict(("Regressor__" + key, value) for (key, value) in params.items())
            set_models.append({"Regressor": [model()], **named_params})
        return set_models
