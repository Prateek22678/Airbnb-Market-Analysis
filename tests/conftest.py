"""
tests/conftest.py
Shared Pytest fixtures for Austin Airbnb testing suite.
Provides synthetic datasets, realistic edge cases, and pre-trained model fixtures.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd
import joblib


@pytest.fixture
def sample_raw_listings_df() -> pd.DataFrame:
    """
    Returns a synthetic raw DataFrame resembling Inside Airbnb raw schema
    with typical real-world dirty data patterns (currency strings, dirty bathrooms,
    raw JSON amenities, missing values, outliers).
    """
    data = {
        'id': [101, 102, 103, 104, 105, 106, 107],
        'price': ['$150.00', '$2,450.50', ' $75.00 ', '$5.00', '$8,500.00', np.nan, 'invalid'],
        'bathrooms_text': ['1 bath', '2.5 baths', 'Half-bath', 'Shared half-bath', '3 baths', '1 private bath', None],
        'host_is_superhost': ['t', 'f', 't', 'f', None, 't', 'unknown'],
        'host_identity_verified': ['t', 't', 'f', 't', 'f', None, '1'],
        'instant_bookable': ['t', 'f', 'f', 't', 't', 'f', '0'],
        'accommodates': [2, 8, 1, 4, 12, 2, 4],
        'bedrooms': [1.0, 4.0, 0.0, 2.0, 6.0, np.nan, 2.0],
        'beds': [1.0, 4.0, 1.0, 2.0, 8.0, 1.0, np.nan],
        'latitude': [30.2672, 30.2500, 30.3000, 30.2800, 30.2200, 30.2700, 30.2650],
        'longitude': [-97.7431, -97.7500, -97.7200, -97.7600, -97.7100, -97.7400, -97.7450],
        'neighbourhood': ['78701', '78704', '78702', '78705', '78745', '78701', '78701'],
        'room_type': ['Entire home/apt', 'Entire home/apt', 'Private room', 'Entire home/apt', 'Entire home/apt', 'Private room', 'Entire home/apt'],
        'property_type': ['Entire home', 'Entire rental unit', 'Private room in home', 'Entire condo', 'Entire home', 'Treehouse', 'Entire home'],
        'amenities': [
            '["Wifi", "Pool", "Air conditioning", "Kitchen"]',
            '["Wifi", "Hot tub", "Free parking", "Dedicated workspace", "Washer", "Dryer"]',
            '["Wifi", "Air conditioning"]',
            '["Self check-in", "Pets allowed", "Patio or balcony", "Wifi"]',
            '["Pool", "Hot tub", "Kitchen", "Free parking", "Air conditioning", "Wifi"]',
            '[]',
            np.nan
        ],
        'number_of_reviews': [45, 120, 0, 15, 80, 0, 5],
        'number_of_reviews_ltm': [12, 30, 0, 8, 25, 0, 2],
        'reviews_per_month': [1.5, 3.2, np.nan, 0.8, 2.1, np.nan, 0.4],
        'review_scores_rating': [4.92, 4.98, np.nan, 4.75, 4.88, np.nan, 4.60],
        'review_scores_cleanliness': [4.90, 5.00, np.nan, 4.80, 4.90, np.nan, 4.50],
        'review_scores_location': [4.95, 4.90, np.nan, 4.85, 4.80, np.nan, 4.70],
        'review_scores_value': [4.85, 4.90, np.nan, 4.60, 4.75, np.nan, 4.50],
        'minimum_nights': [2, 3, 1, 30, 4, 1, 2],
        'first_review': ['2021-05-10', '2019-03-15', np.nan, '2023-01-20', '2020-08-14', np.nan, '2024-02-01'],
        'last_review': ['2026-09-01', '2026-09-15', np.nan, '2026-08-20', '2026-09-10', np.nan, '2026-08-30']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_cleaned_df() -> pd.DataFrame:
    """
    Returns a clean synthetic listings DataFrame ready for feature engineering.
    """
    data = {
        'id': [1, 2, 3, 4, 5],
        'price': [120.0, 250.0, 85.0, 450.0, 180.0],
        'accommodates': [2, 6, 1, 10, 4],
        'bathrooms': [1.0, 2.5, 1.0, 3.5, 2.0],
        'bedrooms': [1.0, 3.0, 1.0, 5.0, 2.0],
        'beds': [1.0, 3.0, 1.0, 6.0, 2.0],
        'latitude': [30.2672, 30.2500, 30.2900, 30.2600, 30.2800],
        'longitude': [-97.7431, -97.7500, -97.7300, -97.7400, -97.7600],
        'minimum_nights': [2, 3, 1, 4, 2],
        'number_of_reviews': [25, 60, 0, 110, 15],
        'number_of_reviews_ltm': [10, 22, 0, 35, 8],
        'reviews_per_month': [1.2, 2.5, np.nan, 3.8, 0.9],
        'days_since_first_review': [850, 1400, 0, 1900, 400],
        'review_scores_rating': [4.90, 4.95, np.nan, 4.88, 4.75],
        'review_scores_cleanliness': [4.85, 4.90, np.nan, 4.90, 4.80],
        'review_scores_location': [4.95, 4.90, np.nan, 4.85, 4.80],
        'review_scores_value': [4.80, 4.85, np.nan, 4.75, 4.70],
        'amenities_count': [25, 45, 12, 65, 30],
        'host_is_superhost': [1, 1, 0, 1, 0],
        'host_identity_verified': [1, 1, 1, 1, 0],
        'instant_bookable': [1, 0, 0, 1, 0],
        'has_wifi': [1, 1, 1, 1, 1],
        'has_pool': [0, 1, 0, 1, 0],
        'has_hot_tub': [0, 1, 0, 1, 0],
        'has_free_parking': [1, 1, 0, 1, 1],
        'has_air_conditioning': [1, 1, 1, 1, 1],
        'has_workspace': [1, 1, 0, 1, 1],
        'has_kitchen': [1, 1, 0, 1, 1],
        'has_patio_or_balcony': [0, 1, 0, 1, 1],
        'has_washer_dryer': [1, 1, 0, 1, 1],
        'has_self_checkin': [1, 1, 1, 1, 0],
        'has_pets_allowed': [0, 1, 0, 0, 1],
        'room_type': ['Entire home/apt', 'Entire home/apt', 'Private room', 'Entire home/apt', 'Entire home/apt'],
        'neighbourhood': ['78701', '78704', '78702', '78701', '78705'],
        'property_type': ['Entire rental unit', 'Entire home', 'Private room in home', 'Entire home', 'Entire condo']
    }
    return pd.DataFrame(data)


@pytest.fixture
def trained_model_bundle():
    """
    Loads pre-trained XGBoost model bundle if present on disk.
    """
    model_path = os.path.join("models", "price_model_xgboost.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None
