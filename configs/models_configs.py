import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

XGBRegressorConfig = (
    xgb.XGBRegressor,
    {
        "n_estimators": [100, 200, 500],
        "learning_rate": np.logspace(-2, -1, 3),
        "max_depth": [3, 5, 7],
        "subsample": np.linspace(0.7, 1.0, 3),
        "colsample_bytree": np.linspace(0.7, 1.0, 3),
    },
)

LassoConfig = (Lasso, {"alpha": np.logspace(-4, 1, 6)})

RidgeConfig = (Ridge, {"alpha": np.logspace(-1, 3, 5)})

GradientBoostingRegressorConfig = (
    GradientBoostingRegressor,
    {
        "n_estimators": [100, 200, 500],
        "learning_rate": np.logspace(-2, -1, 3),
        "max_depth": [3, 5, 7],
        "subsample": np.linspace(0.7, 1.0, 3),
    },
)

KNeighborsRegressorConfig = (
    KNeighborsRegressor,
    {
        "n_neighbors": [3, 5, 10, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    },
)

SVRConfig = (
    SVR,
    {"C": np.logspace(-1, 2, 4), "epsilon": np.linspace(0.01, 0.5, 4), "kernel": ["linear", "rbf", "poly"]},
)


ModelsConfigs = {
    "XGBRegressor": XGBRegressorConfig,
    "Lasso": LassoConfig,
    "Ridge": RidgeConfig,
    "GradientBoostingRegressor": GradientBoostingRegressorConfig,
    "KNeighborsRegressor": KNeighborsRegressorConfig,
    "SVR": SVRConfig,
}
