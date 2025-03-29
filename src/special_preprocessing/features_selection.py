import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor


def select_features(
    X,
    y,
    spearman_threshold=None,
    spearman_quantile=0.75,
    vif_threshold=None,
    vif_quantile=0.75,
):
    def safe_spearman(col):
        if col.nunique() <= 1:
            return np.nan
        return spearmanr(col, y)[0]

    spearman_corr = X.apply(safe_spearman).dropna()
    if spearman_corr.empty:
        return []

    if spearman_threshold is None:
        spearman_threshold = np.quantile(spearman_corr.abs(), spearman_quantile)

    selected_features = spearman_corr[
        abs(spearman_corr) > spearman_threshold
    ].index.tolist()
    if not selected_features:
        return []

    X_selected = X[selected_features].copy()

    def compute_vif(df):
        return pd.Series(
            [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],
            index=df.columns,
        )

    vif_series = compute_vif(X_selected)

    if vif_threshold is None:
        vif_threshold = np.quantile(vif_series, vif_quantile)

    while vif_series.max() >= vif_threshold and len(vif_series) > 1:
        feature_to_remove = vif_series.idxmax()
        X_selected.drop(columns=[feature_to_remove], inplace=True)
        vif_series = compute_vif(X_selected)

    return X_selected.columns.tolist()
