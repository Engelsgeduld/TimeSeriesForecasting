import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error

from src.models.model import TimeSeriesModel


class TestTimeSeriesModel:
    def setup_model(self):
        trend_models = {"Regressor": [DummyRegressor(strategy="mean")]}
        seasonal_models = {"Regressor": [DummyRegressor(strategy="mean")]}
        return TimeSeriesModel(trend_models, seasonal_models, root_mean_squared_error)

    def create_x_dataframe(self, data, features_number, key_name):
        keys_column = [key for key, count in data for _ in range(count)]
        features = [f"feature_{i}" for i in range(features_number)]
        dt_signature = {key_name: keys_column}
        for feature in features:
            dt_signature[feature] = np.random.rand(sum([pair[1] for pair in data]))
        return pd.DataFrame(data=dt_signature)

    def create_y_dataframe(self, count, trend_name, seasonal_name):
        data = np.random.rand(count, 2)
        return pd.DataFrame(data=data, columns=[trend_name, seasonal_name])

    def setup_dataframe(
        self,
        data,
        features_number=5,
        key_name="key",
        trend_name="trend",
        seasonal_name="seasonal",
    ):
        X = self.create_x_dataframe(data, features_number, key_name)
        y = self.create_y_dataframe(len(X), trend_name, seasonal_name)
        keys = list(X["key"].unique())
        return X, y, keys

    @given(
        data=st.lists(
            st.tuples(st.text(min_size=1), st.integers(min_value=1, max_value=100)),
            min_size=1,
            max_size=10,
            unique_by=lambda x: x[0],
        )
    )
    def test_setup_data(self, data):
        model = self.setup_model()
        X, y, _ = self.setup_dataframe(data)
        for pair in data:
            train, trend, seasonal = model._setup_train_sets(X, y, pair[0])
            assert len(train) == pair[1]
            assert len(trend) == pair[1]
            assert len(seasonal) == pair[1]

    @settings(deadline=None)
    @given(
        data=st.lists(
            st.tuples(st.text(min_size=1), st.integers(min_value=4, max_value=100)),
            min_size=1,
            max_size=10,
            unique_by=lambda x: x[0],
        )
    )
    def test_fit_normal_scenario(self, data):
        model = self.setup_model()
        X, y, keys = self.setup_dataframe(data)
        model.fit(X, y)
        assert len(model.best_models_) == len(data)
        assert all([key in model.best_models_ for key in keys])

    @settings(deadline=None)
    @given(
        train_data=st.lists(
            st.tuples(st.text(min_size=1), st.integers(min_value=4, max_value=100)),
            min_size=5,
            max_size=10,
            unique_by=lambda x: x[0],
        )
    )
    def test_prediction_normal_scenario(self, train_data):
        X, y, train_keys = self.setup_dataframe(train_data)
        indexes = np.random.choice(range(len(train_data)), size=4)
        test_data = self.create_x_dataframe([train_data[index] for index in indexes], 5, "key")
        test_keys = list(test_data["key"].unique())
        model = self.setup_model()
        model.fit(X, y)
        predict = model.predict(test_data)
        assert all(key in predict for key in test_keys)

    @pytest.mark.parametrize(
        "key_name, trend_name, seasonal_name",
        [
            ("wrong_key_name", "trend", "seasonal"),
            ("key", "wrong_trend_name", "seasonal"),
            ("key", "trend", "wrong_seasonal_name"),
        ],
    )
    def test_fit_exceptions(self, key_name, trend_name, seasonal_name):
        data = [("1", 3), ("2", 10), ("milk", 12)]
        X = self.create_x_dataframe(data, 5, key_name)
        y = self.create_y_dataframe(len(X), trend_name, seasonal_name)
        model = self.setup_model()
        with pytest.raises(ValueError):
            model.fit(X, y)

    @pytest.mark.parametrize(
        "test_data",
        [([("1", 4), ("2", 10), ("3", 12)]), ([("3", 4), ("4", 10), ("5", 12)])],
    )
    def test_predict_exceptions(self, test_data):
        data = [("1", 3), ("2", 10), ("milk", 12)]
        X, y, _ = self.setup_dataframe(data)
        model = self.setup_model()
        model.fit(X, y)
        test_df = self.create_x_dataframe(test_data, 5, "key")
        with pytest.raises(ValueError):
            model.predict(test_df)
