from typing import Sequence

import numpy as np
import pandas as pd

def forecast_accuracy(
    df: pd.DataFrame,
    y_true_index: str = "Actual",
    y_pred_index: str = "Forecast",
    time_index: str = "date",
    time_period: str = "M",
    zero_filler: float = 0,
    ae_gr_cols: Sequence = [],
    out_gr_cols: Sequence = [],
) -> pd.DataFrame:
    """Calculates forecast accuracy.

    This function computes absolute error (AE) and forecast accuracy (Acc)
    at the specified aggregation levels. It supports time-based grouping.

    Args:
        df (pd.DataFrame): Input DataFrame containing actual and forecast values.
        y_true_index (str): Column name for actual values. Defaults to "Actual".
        y_pred_index (str): Column name for forecast values. Defaults to "Forecast".
        time_index (str): Column name for the time index. Defaults to "date".
        time_period (str): Frequency for time-based grouping (e.g., 'M' for monthly). Defaults to "M".
        zero_filler (float): Value to replace NaN in accuracy calculation. Defaults to 0.
        ae_gr_cols (Sequence): Columns used for calculating absolute error (AE).
        out_gr_cols (Sequence): Columns used for aggregating the final accuracy metrics.

    Returns:
        pd.DataFrame: A DataFrame with computed metrics:
            - `{y_true_index}`: Sum of actual values.
            - `{y_pred_index}`: Sum of forecasted values.
            - `AE`: Absolute error.
            - `Acc`: Forecast accuracy (1 - AE / Actual), handling division by zero.

    Raises:
        ValueError: If any of `time_index`, `ae_gr_cols`, or `out_gr_cols` columns are missing.
    """
    ae_gr_cols, out_gr_cols = list(ae_gr_cols), list(out_gr_cols) # For mypy
    required_cols = {time_index, y_true_index, y_pred_index} | set(ae_gr_cols) | set(out_gr_cols)
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"The following required columns are missing in the DataFrame: {missing_cols}")

    used_cols = list(required_cols)
    group_columns = [pd.Grouper(key=time_index, freq=time_period)] + list(set(ae_gr_cols + out_gr_cols))
    gr_df = df[used_cols].groupby(group_columns, as_index=False).sum()
    gr_df["AE"] = np.abs(gr_df[y_true_index] - gr_df[y_pred_index])
    out_df = gr_df.copy()
    if len(out_gr_cols) != 0:
        out_df = gr_df.groupby(list(set(out_gr_cols)), as_index=False).agg(
            {y_true_index: "sum", y_pred_index: "sum", "AE": "sum"}
        )
    out_df["Acc"] = np.where(out_df[y_true_index] == 0, zero_filler, 1 - out_df["AE"] / out_df[y_true_index])

    return out_df
