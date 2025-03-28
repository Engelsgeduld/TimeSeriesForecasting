import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given

from src.metrics.forecast_accuracy import forecast_accuracy


@st.composite
def same_len_lists(draw, min_length=1, max_length=30):

    n = draw(st.integers(min_value=min_length, max_value=max_length))
    fixed_length_list = st.lists(st.floats(min_value=1, max_value=100_000), min_size=n, max_size=n)

    return draw(fixed_length_list), draw(fixed_length_list)


class TestForecastAccuracyMetric:
    @staticmethod
    def calc_metric(actual, forecast):
        expected_ae = abs(sum(actual) - sum(forecast))
        expected_acc = (1 - expected_ae / sum(actual)) if sum(actual) != 0 else 0
        return expected_ae, expected_acc

    @given(same_len_lists())
    def test_basic_accuracy(self, lists):
        actual, forecast = lists
        data = pd.DataFrame(
            {
                "date": pd.date_range(start="2024-01-01", periods=len(actual), freq="D"),
                "Actual": actual,
                "Forecast": forecast,
            }
        )

        expected_ae, expected_acc = self.calc_metric(actual, forecast)

        result = forecast_accuracy(data)
        assert "AE" in result.columns
        assert "Acc" in result.columns

        assert np.isclose(result["AE"].iloc[0], expected_ae, atol=1e-4)

        assert np.isclose(result["Acc"].iloc[0], expected_acc, atol=1e-4)

    @given(same_len_lists(40, 160))
    def test_grouping_by_month_with_accuracy(self, lists):
        actual, forecast = lists

        dates = pd.date_range("2024-01-01", periods=len(actual), freq="D").tolist()

        data = pd.DataFrame({"date": pd.to_datetime(dates), "Actual": actual, "Forecast": forecast})

        expected_results = {}
        data["Month"] = data["date"].dt.strftime("%Y-%m")
        for month in data["Month"].unique():
            month_data = data[data["Month"] == month]
            expected_ae, expected_acc = self.calc_metric(month_data["Actual"].tolist(), month_data["Forecast"].tolist())
            expected_results[month] = {"AE": expected_ae, "Acc": expected_acc}

        result = forecast_accuracy(data, time_index="date", time_period="M")

        result["Month"] = result["date"].dt.strftime("%Y-%m")
        result_months = result["Month"].unique().tolist()

        expected_months = list(expected_results.keys())
        assert set(result_months) == set(expected_months)

        for month, expected in expected_results.items():
            month_data = result[result["Month"] == month]

            assert not month_data.empty

            computed_ae = month_data["AE"]
            computed_acc = month_data["Acc"]

            assert np.isclose(computed_ae, expected["AE"])
            assert np.isclose(computed_acc, expected["Acc"])

    def test_grouping_by_month_and_key(self):
        dates = (
            pd.date_range("2024-01-01", periods=6, freq="D").tolist()
            + pd.date_range("2024-02-01", periods=6, freq="D").tolist()
        )

        keys = ["A", "B", "C"] * 4

        actual = [100, 200, 300] * 4
        forecast = [80, 190, 300] * 4

        expected_results = {
            ("2024-01", "A"): {"AE": 40, "Acc": 0.80},
            ("2024-01", "B"): {"AE": 20, "Acc": 0.95},
            ("2024-01", "C"): {"AE": 0, "Acc": 1.00},
            ("2024-02", "A"): {"AE": 40, "Acc": 0.80},
            ("2024-02", "B"): {"AE": 20, "Acc": 0.95},
            ("2024-02", "C"): {"AE": 0, "Acc": 1.00},
        }

        data = pd.DataFrame({"date": pd.to_datetime(dates), "Key": keys, "Actual": actual, "Forecast": forecast})

        result = forecast_accuracy(data, time_index="date", time_period="M", ae_gr_cols=["Key"])

        result["Month"] = result["date"].dt.strftime("%Y-%m")

        for (month, key), expected in expected_results.items():
            group_data = result[(result["Month"] == month) & (result["Key"] == key)]

            assert not group_data.empty

            computed_ae = group_data["AE"].iloc[0]
            computed_acc = group_data["Acc"].iloc[0]

            assert np.isclose(computed_ae, expected["AE"], atol=1e-4)
            assert np.isclose(computed_acc, expected["Acc"], atol=1e-4)

    def test_grouping_by_channel(self):
        data = {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "Channel": [0, 0, 1, 1, 0, 0, 1, 1],
            "Actual": [100, 200, 150, 250, 100, 200, 150, 250],
            "Forecast": [80, 210, 140, 280, 110, 190, 160, 240],
        }
        df = pd.DataFrame(data)

        result = forecast_accuracy(df, out_gr_cols=["Channel"])

        assert set(result["Channel"]) == {0, 1}

        expected_metrics = {0: {"AE": 10, "Acc": 1 - 10 / 600}, 1: {"AE": 20, "Acc": 1 - 20 / 800}}

        for channel in [0, 1]:
            row = result[result["Channel"] == channel].iloc[0]
            assert row["AE"] == expected_metrics[channel]["AE"]
            assert np.isclose(row["Acc"], expected_metrics[channel]["Acc"], atol=1e-4)
