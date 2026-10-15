"""
tests/test_cleaning.py
Unit tests for data sanitization, parsing, outlier treatment, and data quality.
"""

import os
import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    clean_price_string,
    clean_boolean,
    clean_bathrooms,
    parse_amenities,
    treat_price_outliers
)


# =====================================================================
# 1. Price Cleaning Unit Tests
# =====================================================================

class TestPriceCleaning:
    """Test suite for currency string parsing and type sanitization."""

    @pytest.mark.parametrize("raw_val, expected", [
        ("$1,200.00", 1200.0),
        ("$79.74", 79.74),
        ("$0.00", 0.0),
        ("$5.00", 5.0),
        (" $450.50 ", 450.5),
        ("$12,345,678.90", 12345678.9),
        (150, 150.0),
        (250.75, 250.75),
        ("100", 100.0)
    ])
    def test_clean_price_valid_inputs(self, raw_val, expected):
        result = clean_price_string(raw_val)
        assert np.isclose(result, expected), f"Failed for input '{raw_val}': got {result}, expected {expected}"

    @pytest.mark.parametrize("invalid_val", [
        None,
        np.nan,
        float('nan'),
        "",
        "   ",
        "N/A",
        "Free",
        "Contact Host"
    ])
    def test_clean_price_invalid_inputs(self, invalid_val):
        result = clean_price_string(invalid_val)
        assert pd.isna(result), f"Expected NaN for '{invalid_val}', got {result}"


# =====================================================================
# 2. Boolean Sanitization Unit Tests
# =====================================================================

class TestBooleanCleaning:
    """Test suite for boolean flag normalization ('t'/'f' -> 1/0)."""

    @pytest.mark.parametrize("val", [
        "t", "T", "true", "True", "TRUE", "1", 1, True, "yes", "YES"
    ])
    def test_clean_boolean_truthy(self, val):
        assert clean_boolean(val) == 1, f"Expected 1 for truthy '{val}'"

    @pytest.mark.parametrize("val", [
        "f", "F", "false", "False", "0", 0, False, "no", "unknown", None, np.nan, ""
    ])
    def test_clean_boolean_falsy(self, val):
        assert clean_boolean(val) == 0, f"Expected 0 for falsy '{val}'"


# =====================================================================
# 3. Bathrooms Extraction Unit Tests
# =====================================================================

class TestBathroomsCleaning:
    """Test suite for regex parsing of free-form bathroom strings."""

    @pytest.mark.parametrize("raw_bath, expected", [
        ("1 bath", 1.0),
        ("2 baths", 2.0),
        ("2.5 baths", 2.5),
        ("3.5 baths", 3.5),
        ("0 baths", 0.0),
        ("1 private bath", 1.0),
        ("2 shared baths", 2.0),
        ("Half-bath", 0.5),
        ("half-bath", 0.5),
        ("Shared half-bath", 0.5),
        ("Private half-bath", 0.5),
        ("1 half-bath", 1.5),
        ("2 half-baths", 2.5),
        (2.0, 2.0),
        (1, 1.0)
    ])
    def test_clean_bathrooms_valid(self, raw_bath, expected):
        result = clean_bathrooms(raw_bath)
        assert np.isclose(result, expected), f"Failed for '{raw_bath}': got {result}, expected {expected}"

    @pytest.mark.parametrize("invalid_bath", [
        None,
        np.nan,
        "",
        "   ",
        "unknown",
        "no bathroom listed"
    ])
    def test_clean_bathrooms_invalid(self, invalid_bath):
        result = clean_bathrooms(invalid_bath)
        assert pd.isna(result), f"Expected NaN for '{invalid_bath}', got {result}"


# =====================================================================
# 4. Amenities Parsing Unit Tests
# =====================================================================

class TestAmenitiesParsing:
    """Test suite for JSON & string amenities parsing and indicator flags."""

    def test_parse_amenities_json_string(self):
        raw = '["Wifi", "Pool", "Free parking", "Air conditioning", "Kitchen"]'
        count, indicators = parse_amenities(raw)
        assert count == 5
        assert indicators["has_wifi"] == 1
        assert indicators["has_pool"] == 1
        assert indicators["has_free_parking"] == 1
        assert indicators["has_air_conditioning"] == 1
        assert indicators["has_kitchen"] == 1
        assert indicators["has_hot_tub"] == 0
        assert indicators["has_workspace"] == 0

    def test_parse_amenities_python_list(self):
        items = ["Dedicated workspace", "Hot tub", "Washer", "Dryer", "Self check-in"]
        count, indicators = parse_amenities(items)
        assert count == 5
        assert indicators["has_workspace"] == 1
        assert indicators["has_hot_tub"] == 1
        assert indicators["has_washer_dryer"] == 1
        assert indicators["has_self_checkin"] == 1
        assert indicators["has_pool"] == 0

    def test_parse_amenities_empty_or_null(self):
        assert parse_amenities(None) == (0, {})
        assert parse_amenities(np.nan) == (0, {})
        assert parse_amenities("") == (0, {})
        assert parse_amenities("[]") == (0, {
            "has_wifi": 0, "has_pool": 0, "has_hot_tub": 0, "has_free_parking": 0,
            "has_air_conditioning": 0, "has_workspace": 0, "has_kitchen": 0,
            "has_patio_or_balcony": 0, "has_washer_dryer": 0, "has_self_checkin": 0,
            "has_pets_allowed": 0
        })

    def test_parse_amenities_messy_string_fallback(self):
        raw = '{Pets allowed, Wifi, "Patio or balcony"}'
        count, indicators = parse_amenities(raw)
        assert count >= 3
        assert indicators["has_pets_allowed"] == 1
        assert indicators["has_wifi"] == 1
        assert indicators["has_patio_or_balcony"] == 1


# =====================================================================
# 5. Outlier Treatment Unit Tests
# =====================================================================

class TestOutlierTreatment:
    """Test suite for sub-minimum price removal and upper percentile trimming."""

    def test_treat_price_outliers_thresholds(self):
        df = pd.DataFrame({
            "id": list(range(100)),
            "price": [5.0, 10.0] + [100.0] * 96 + [5000.0, 10000.0]
        })
        cleaned_df, audit = treat_price_outliers(df, min_price=15.0, upper_percentile=0.99)
        
        # Sub-minimum listings ($5, $10) must be dropped
        assert cleaned_df["price"].min() >= 15.0
        assert audit["dropped_sub_min"] == 2
        
        # Upper luxury cutoff must be enforced
        assert cleaned_df["price"].max() <= audit["upper_cutoff_val"]
        assert len(cleaned_df) < len(df)
        assert audit["retention_rate_pct"] > 90.0


# =====================================================================
# 6. Processed Data Smoke & Quality Assertions
# =====================================================================

class TestProcessedDataQuality:
    """Integration data assertions verifying clean dataset integrity."""

    @pytest.fixture(scope="class")
    def cleaned_data(self):
        csv_path = os.path.join("data", "processed", "listings_cleaned.csv")
        if not os.path.exists(csv_path):
            pytest.skip(f"Cleaned dataset not found at {csv_path}")
        return pd.read_csv(csv_path)

    def test_dataset_size_and_primary_key(self, cleaned_data):
        assert len(cleaned_data) > 5000, "Cleaned dataset should contain at least 5,000 listings"
        assert cleaned_data["id"].is_unique, "Listing IDs must be strictly unique"

    def test_price_sanity(self, cleaned_data):
        assert (cleaned_data["price"] >= 15.0).all(), "No price should be below the $15 economic threshold"
        assert cleaned_data["price"].notnull().all(), "Price column must contain zero null values"

    def test_austin_geographic_bounding_box(self, cleaned_data):
        # Austin metropolitan latitude and longitude boundaries
        assert cleaned_data["latitude"].between(30.0, 30.6).all(), "Latitudes outside Austin bounding box"
        assert cleaned_data["longitude"].between(-98.2, -97.4).all(), "Longitudes outside Austin bounding box"

    def test_binary_columns_strict_values(self, cleaned_data):
        binary_cols = [
            "host_is_superhost", "host_identity_verified", "instant_bookable",
            "has_wifi", "has_pool", "has_kitchen"
        ]
        for col in binary_cols:
            if col in cleaned_data.columns:
                unique_vals = set(cleaned_data[col].dropna().unique())
                assert unique_vals.issubset({0, 1}), f"Column '{col}' contains non-binary values: {unique_vals}"
