from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.special_preprocessing.transformers.features_extraction import (
    FourierFeaturesTransformer,
    HolidayTransformer,
    MeanWeekMonthTransformer,
)
from src.special_preprocessing.transformers.preprocessing import (
    ChangeTypesTransformer,
    ConcatenateTransform,
    DropDuplicatesTransformer,
    KeyIndexTransformer,
    NaNHandlerTransformer,
)
from src.special_preprocessing.transformers.series_comp import (
    DateRangeFilledTransformer,
    GroupByDateTransformer,
)
from src.special_preprocessing.transformers.series_decomposition import SeriesDecompositionTransformer


def features():
    ohe = ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["holiday"],
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ohe.set_output(transform="pandas")
    pipline = Pipeline(
        steps=[
            ("date_feature_transform", HolidayTransformer()),
            ("ohe", ohe),
            ("mean_ship_feature", MeanWeekMonthTransformer()),
            (
                "fourier_features",
                FourierFeaturesTransformer(),
            ),
        ]
    )
    return pipline


def preprocessing():
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
    )
    ohe.set_output(transform="pandas")
    pipline = Pipeline(
        steps=[
            ("concatenate", ConcatenateTransform()),
            ("nan_handel", NaNHandlerTransformer()),
            ("change_types", ChangeTypesTransformer()),
            ("key_index", KeyIndexTransformer()),
            ("drop_duplicates", DropDuplicatesTransformer()),
            ("ohe", ohe),
        ],
    )
    return pipline


def grouping():
    pipeline = Pipeline(
        steps=[
            ("fill_data_range", DateRangeFilledTransformer()),
            ("group", GroupByDateTransformer()),
        ]
    )
    return pipeline


def decomposition():
    pipline = Pipeline(steps=[("series_decomposition", SeriesDecompositionTransformer())])
    return pipline


preprocessing_pipeline = Pipeline(
    steps=[
        ("base preprocessing", preprocessing()),
        ("grouping", grouping()),
        ("features extraction", features()),
        ("decomposition", decomposition()),
    ]
)
