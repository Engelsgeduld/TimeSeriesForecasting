from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..date_transformers.features_extraction import FeatureExtractionTransformer
from ..date_transformers.series_comp import DateRangeFilledTransformer, GroupByDateTransformer
from ..date_transformers.series_decomposition import Separation, SeriesDecompositionTransformer
from ..first_special_pipeline.preprocessing import (
    ChangeTypesTransformer,
    DropDuplicatesTransformer,
    KeyIndexTransformer,
    NaNHandlerTransformer,
)


def preprocessing() -> Pipeline:
    cat_cols = ["channel", "level_2", "level_3", "brend", "level_1", "unit"]
    ohe = ColumnTransformer(
        [
            (
                "OHE",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        force_int_remainder_cols=False,
    )
    ohe.set_output(transform="pandas")
    pipline = Pipeline(
        steps=[
            ("nan_handel", NaNHandlerTransformer()),
            ("change_types", ChangeTypesTransformer()),
            ("key_index", KeyIndexTransformer()),
            ("drop_duplicates", DropDuplicatesTransformer()),
            ("ohe", ohe),
        ],
    )
    return pipline


preprocessing_pipeline = Pipeline(
    steps=[
        ("base preprocessing", preprocessing()),
        ("fill_data_range", DateRangeFilledTransformer()),
        ("grouping", GroupByDateTransformer()),
        ("features extraction", FeatureExtractionTransformer()),
        ("decomposition", SeriesDecompositionTransformer()),
        ("separation", Separation()),
    ]
)
