from typing import Optional

import pandas as pd

from configs.models_collector import CONFIG_TYPE, ModelsCollector
from configs.models_configs import ModelsConfigs
from src.models.time_series_model import TimeSeriesModel


class ModelConfigurator:
    def __init__(self) -> None:
        self._trend_config: Optional[CONFIG_TYPE] = None
        self._seasonal_config: Optional[CONFIG_TYPE] = None
        self._model: Optional[TimeSeriesModel] = None
        self._collector: ModelsCollector = ModelsCollector(ModelsConfigs)
        self.config_names: Optional[dict] = None

    def set_config(self, trend_models: list[str], seasonal_models: list[str]) -> None:
        self._trend_config = self._collector.get_configs(trend_models)
        self._seasonal_config = self._collector.get_configs(seasonal_models)
        self.config_names = {"trend_models": trend_models, "seasonal_models": seasonal_models}

    def fit_model(self, X: pd.DataFrame) -> None:
        if self._trend_config is None or self._seasonal_config is None:
            raise ValueError("Configs unsetted")
        model = TimeSeriesModel(self._trend_config, self._seasonal_config)
        self._model = model.fit(X)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Unfitted")
        return self._model.predict(X)
