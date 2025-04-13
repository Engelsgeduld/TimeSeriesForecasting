from typing import Any


class ModelsCollector:
    """
    Collects and provides configurations for machine learning models.

    Attributes:
        models_config (dict[str, tuple[Any, list[Any]]]):
            A dictionary where each key is a model name, and each value is a tuple
            containing a model configuration object and a list of related parameters.
    """

    def __init__(self, models_config: dict[str, tuple[Any, list[Any]]]) -> None:
        """
        Initializes the ModelsCollector with a dictionary of model configurations.

        Args:
            models_config (dict[str, tuple[Any, list[Any]]]):
                A mapping of model names to their corresponding configurations.
                Each configuration is a tuple containing a model object and a list
                of parameters or metadata.
        """
        self.models_config = models_config

    def get_configs(self, models_names: list[str]) -> list[tuple[Any, list[Any]]]:
        """
        Retrieves unique configurations for the specified model names.

        Duplicate names in the input will be ignored. The order of returned
        configurations is not guaranteed.

        Args:
            models_names (list[str]):
                A list of model names to retrieve configurations for.

        Returns:
            list[tuple[Any, list[Any]]]:
                A list of configurations corresponding to the specified model names.
                Each configuration is a tuple containing a model object and a list
                of associated parameters.

        Raises:
            ValueError: If any of the requested model names are not found
                        in the stored configuration.
        """
        unique_names = set(models_names)
        missing = unique_names - self.models_config.keys()

        if missing:
            raise ValueError(f"Unexpected model names: {missing}")

        return [self.models_config[name] for name in unique_names]
