"""
src/features.py
Feature engineering and transformation pipeline for Austin Airbnb price prediction.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Geographic reference: Austin Downtown / City Hall (Congress Ave & Cesar Chavez)
AUSTIN_DOWNTOWN_LAT = 30.2672
AUSTIN_DOWNTOWN_LON = -97.7431

# Feature Column Definitions
NUMERIC_FEATURES: List[str] = [
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'bathrooms_per_bedroom',
    'beds_per_accommodates',
    'accommodates_per_bedroom',
    'distance_to_downtown_km',
    'minimum_nights',
    'number_of_reviews',
    'number_of_reviews_ltm',
    'reviews_per_month',
    'days_since_first_review',
    'review_scores_rating',
    'review_scores_cleanliness',
    'review_scores_location',
    'review_scores_value',
    'amenities_count'
]

BINARY_FEATURES: List[str] = [
    'host_is_superhost',
    'host_identity_verified',
    'instant_bookable',
    'has_reviews',
    'has_wifi',
    'has_pool',
    'has_hot_tub',
    'has_free_parking',
    'has_air_conditioning',
    'has_workspace',
    'has_kitchen',
    'has_patio_or_balcony',
    'has_washer_dryer',
    'has_self_checkin',
    'has_pets_allowed'
]

CATEGORICAL_FEATURES: List[str] = [
    'room_type',
    'neighbourhood',
    'property_type_group'
]


def calculate_haversine_distance(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: float = AUSTIN_DOWNTOWN_LAT,
    lon2: float = AUSTIN_DOWNTOWN_LON
) -> pd.Series:
    """
    Calculate the great circle distance between points on the earth in kilometers
    using the Haversine formula.
    """
    R = 6371.0088  # Mean Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific features from the cleaned listings dataset.
    
    Creates:
      - distance_to_downtown_km: Great-circle distance to Downtown Austin.
      - Capacity ratios: bathrooms_per_bedroom, beds_per_accommodates, accommodates_per_bedroom.
      - Review imputation: Fills review nulls and flags unreviewed listings.
      - property_type_group: Groups long-tail property categories into top categories.
    """
    df = df_raw.copy()

    # 1. Geographic distance
    df['distance_to_downtown_km'] = calculate_haversine_distance(
        df['latitude'], df['longitude'], AUSTIN_DOWNTOWN_LAT, AUSTIN_DOWNTOWN_LON
    )

    # 2. Capacity & space ratios (safeguarded against divide-by-zero)
    df['bathrooms_per_bedroom'] = df['bathrooms'] / np.maximum(df['bedrooms'], 1.0)
    df['beds_per_accommodates'] = df['beds'] / np.maximum(df['accommodates'], 1.0)
    df['accommodates_per_bedroom'] = df['accommodates'] / np.maximum(df['bedrooms'], 1.0)

    # 3. Review indicators and safe imputations
    df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0.0)
    
    # Impute missing rating subscores with domain medians
    rating_median = df['review_scores_rating'].median() if df['review_scores_rating'].notnull().any() else 4.88
    clean_median = df['review_scores_cleanliness'].median() if df['review_scores_cleanliness'].notnull().any() else 4.88
    loc_median = df['review_scores_location'].median() if df['review_scores_location'].notnull().any() else 4.90
    val_median = df['review_scores_value'].median() if df['review_scores_value'].notnull().any() else 4.78

    df['review_scores_rating'] = df['review_scores_rating'].fillna(rating_median)
    df['review_scores_cleanliness'] = df['review_scores_cleanliness'].fillna(clean_median)
    df['review_scores_location'] = df['review_scores_location'].fillna(loc_median)
    df['review_scores_value'] = df['review_scores_value'].fillna(val_median)

    # 4. Property type aggregation
    top_properties = [
        'Entire home',
        'Entire rental unit',
        'Entire condo',
        'Private room in home',
        'Entire guesthouse',
        'Room in hotel'
    ]
    df['property_type_group'] = df['property_type'].apply(
        lambda x: x if x in top_properties else 'Other'
    )

    # 5. Type casting for binary indicators
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for leak-free preprocessing.
    - Scales numeric continuous variables (StandardScaler).
    - Passes binary indicators directly through.
    - One-hot encodes nominal categoricals with drop='first' and handle_unknown='ignore'.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract readable feature names after ColumnTransformer fitting.
    """
    num_names = NUMERIC_FEATURES
    bin_names = BINARY_FEATURES
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + bin_names + cat_names
