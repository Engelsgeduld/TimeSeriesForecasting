from sklearn.pipeline import Pipeline

from src.special_preprocessing.date_transformers.features_extraction import FeatureExtractionTransformer
from src.special_preprocessing.date_transformers.series_decomposition import Separation, SeriesDecompositionTransformer
from src.special_preprocessing.second_special_pipeline.preprocessing import (
    CategoricalFeaturesTransform,
    DateRangeFilledTransformerSec,
    DiscountTransformer,
    KeyTransformer,
    RenameColumns,
)

preprocessing_pipeline = Pipeline(
    steps=[
        ("rename", RenameColumns()),
        ("key", KeyTransformer()),
        ("discount", DiscountTransformer()),
        ("fill_data_range", DateRangeFilledTransformerSec()),
        ("categorical_features_prep", CategoricalFeaturesTransform()),
        ("features extraction", FeatureExtractionTransformer()),
        ("decomposition", SeriesDecompositionTransformer()),
        ("separation", Separation()),
    ]
)
