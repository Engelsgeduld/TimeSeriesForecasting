import numpy as np
import pandas as pd


def train_test_split_by_months(df: pd.DataFrame, date_column: str, test_months: int) -> pd.DataFrame:
    """
    Splits a DataFrame into train and test sets, where the test set consists of the last `test_months` months,
    and also splits the target `y` into corresponding `y_train` and `y_test`.

    Args:
        df (pd.DataFrame): Input DataFrame containing a date column.
        y (pd.Series): Target variable corresponding to the DataFrame `df`.
        date_column (str): Name of the date column.
        test_months (int): Number of last months to include in the test set.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            A tuple containing two DataFrames (train, test) and two Series (y_train, y_test),
            where train contains data before the test period, and test contains the last `test_months` months.
    """
    df = df.copy()

    max_date = df[date_column].max()
    max_date = max_date.replace(day=1)
    test_start = max_date - pd.DateOffset(months=test_months - 1)
    test_start = test_start.replace(day=1)

    df["mark"] = np.where(df[date_column] < test_start, "train", "test")
    return df
