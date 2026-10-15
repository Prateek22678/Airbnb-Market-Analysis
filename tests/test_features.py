"""
tests/test_features.py
Unit tests for geospatial feature calculations, ratio safeguards,
missing value imputations, and ColumnTransformer preprocessing.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features import (
    calculate_haversine_distance,
    engineer_features,
    build_preprocessor,
    get_feature_names,
    AUSTIN_DOWNTOWN_LAT,
    AUSTIN_DOWNTOWN_LON,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES
)


# =====================================================================
# 1. Haversine Distance Unit Tests
# =====================================================================

class TestHaversineDistance:
    """Test suite for geospatial distance calculations."""

    def test_downtown_to_itself(self):
        lat = pd.Series([AUSTIN_DOWNTOWN_LAT])
        lon = pd.Series([AUSTIN_DOWNTOWN_LON])
        dist = calculate_haversine_distance(lat, lon, AUSTIN_DOWNTOWN_LAT, AUSTIN_DOWNTOWN_LON)
        assert np.isclose(dist.iloc[0], 0.0, atol=1e-3), "Distance to identical coordinate should be 0.0 km"

    def test_known_landmark_distance(self):
        # Austin-Bergstrom International Airport (AUS): ~30.1975 N, -97.6664 W
        airport_lat = pd.Series([30.1975])
        airport_lon = pd.Series([-97.6664])
        dist = calculate_haversine_distance(airport_lat, airport_lon, AUSTIN_DOWNTOWN_LAT, AUSTIN_DOWNTOWN_LON)
        # Distance from Downtown Austin to Airport is approximately 10 to 14 km
        assert 10.0 <= dist.iloc[0] <= 15.0, f"Expected AUS distance ~10-15 km, got {dist.iloc[0]:.2f} km"

    def test_non_negative_and_vectorized(self):
        lats = pd.Series([30.25, 30.30, 30.15, 30.40])
        lons = pd.Series([-97.75, -97.70, -97.80, -97.65])
        dists = calculate_haversine_distance(lats, lons)
        assert len(dists) == 4
        assert (dists >= 0.0).all(), "Haversine distance must always be non-negative"


# =====================================================================
# 2. Feature Engineering & Ratio Safeguards
# =====================================================================

class TestFeatureEngineering:
    """Test suite for domain features and mathematical edge cases."""

    def test_engineer_features_adds_columns(self, sample_cleaned_df):
        df_feat = engineer_features(sample_cleaned_df)
        expected_cols = [
            'distance_to_downtown_km',
            'bathrooms_per_bedroom',
            'beds_per_accommodates',
            'accommodates_per_bedroom',
            'has_reviews',
            'property_type_group'
        ]
        for col in expected_cols:
            assert col in df_feat.columns, f"Missing engineered column: {col}"

    def test_zero_division_protection(self):
        # Test extreme boundary case: 0 bedrooms and 0 accommodates (studio or non-standard space)
        edge_data = pd.DataFrame({
            'latitude': [30.2672],
            'longitude': [-97.7431],
            'bathrooms': [1.0],
            'bedrooms': [0.0],       # 0 bedrooms
            'beds': [1.0],
            'accommodates': [0],      # 0 accommodates
            'number_of_reviews': [0],
            'reviews_per_month': [np.nan],
            'review_scores_rating': [np.nan],
            'review_scores_cleanliness': [np.nan],
            'review_scores_location': [np.nan],
            'review_scores_value': [np.nan],
            'property_type': ['Entire rental unit'],
            'room_type': ['Entire home/apt'],
            'neighbourhood': ['78701'],
            'host_is_superhost': [0]
        })
        df_feat = engineer_features(edge_data)

        # Ratios must be finite numbers, not inf or nan
        assert np.isfinite(df_feat['bathrooms_per_bedroom'].iloc[0])
        assert np.isfinite(df_feat['beds_per_accommodates'].iloc[0])
        assert np.isfinite(df_feat['accommodates_per_bedroom'].iloc[0])
        assert df_feat['bathrooms_per_bedroom'].iloc[0] == 1.0  # 1.0 / max(0, 1.0) == 1.0

    def test_review_imputation_and_flag(self, sample_cleaned_df):
        df_feat = engineer_features(sample_cleaned_df)
        # Listing 3 has 0 reviews
        unreviewed = df_feat[df_feat['number_of_reviews'] == 0]
        assert (unreviewed['has_reviews'] == 0).all()
        assert (unreviewed['reviews_per_month'] == 0.0).all()

        # Review ratings must have no missing values after imputation
        rating_cols = [
            'review_scores_rating', 'review_scores_cleanliness',
            'review_scores_location', 'review_scores_value'
        ]
        for col in rating_cols:
            assert df_feat[col].notnull().all(), f"Column {col} still has nulls after imputation"

    def test_property_type_grouping(self):
        df = pd.DataFrame({
            'latitude': [30.26, 30.26, 30.26],
            'longitude': [-97.74, -97.74, -97.74],
            'bathrooms': [1.0, 1.0, 1.0],
            'bedrooms': [1.0, 1.0, 1.0],
            'beds': [1.0, 1.0, 1.0],
            'accommodates': [2, 2, 2],
            'number_of_reviews': [5, 5, 5],
            'reviews_per_month': [1.0, 1.0, 1.0],
            'review_scores_rating': [4.9, 4.9, 4.9],
            'review_scores_cleanliness': [4.9, 4.9, 4.9],
            'review_scores_location': [4.9, 4.9, 4.9],
            'review_scores_value': [4.9, 4.9, 4.9],
            'property_type': ['Entire home', 'Treehouse', 'Yurt'],
            'room_type': ['Entire home/apt', 'Entire home/apt', 'Entire home/apt'],
            'neighbourhood': ['78701', '78701', '78701'],
            'host_is_superhost': [1, 0, 0]
        })
        df_feat = engineer_features(df)
        assert df_feat['property_type_group'].tolist() == ['Entire home', 'Other', 'Other']


# =====================================================================
# 3. Preprocessor Pipeline Unit Tests
# =====================================================================

class TestPreprocessorPipeline:
    """Test suite for scikit-learn ColumnTransformer structure and fitting."""

    def test_preprocessor_fit_transform(self, sample_cleaned_df):
        df_feat = engineer_features(sample_cleaned_df)
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

        # Fit and transform
        X_trans = preprocessor.fit_transform(df_feat)
        assert isinstance(X_trans, np.ndarray)
        assert X_trans.shape[0] == len(df_feat)
        assert np.isfinite(X_trans).all(), "Transformed array contains non-finite values (NaN or Inf)"

        # Check feature names match transformed column count
        names = get_feature_names(preprocessor)
        assert len(names) == X_trans.shape[1], f"Feature name count ({len(names)}) != transformed matrix cols ({X_trans.shape[1]})"
