import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

XGBRegressorConfig = (
    xgb.XGBRegressor,
    {
        "learning_rate": np.logspace(-2, -1, 3),
        "max_depth": [3, 5, 7],
    },
)

LassoConfig = (Lasso, {"alpha": np.logspace(-3, 2, 6)})

RidgeConfig = (Ridge, {"alpha": np.logspace(-3, 2, 5)})

GradientBoostingRegressorConfig = (
    GradientBoostingRegressor,
    {
        "learning_rate": np.logspace(-2, -1, 3),
        "max_depth": [3, 5, 7],
    },
)

KNeighborsRegressorConfig = (
    KNeighborsRegressor,
    {
        "n_neighbors": [3, 5, 10, 15],
    },
)

SVRConfig = (
    SVR,
    {"C": np.logspace(-1, 2, 4), "epsilon": np.linspace(0.01, 0.5, 4)},
)

ModelsConfigs = {
    "XGBRegressor": XGBRegressorConfig,
    "Lasso": LassoConfig,
    "Ridge": RidgeConfig,
    "GradientBoostingRegressor": GradientBoostingRegressorConfig,
    "KNeighborsRegressor": KNeighborsRegressorConfig,
    "SVR": SVRConfig,
}
