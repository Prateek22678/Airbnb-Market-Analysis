"""
tests/test_india_market.py
Unit and integration tests for Indian market dataset, postal PIN/ZIP codes,
and investment analytics.
"""

import pytest
import pandas as pd
from src.india_market import (
    get_indian_listings_df,
    get_indian_city_summary,
    get_indian_zipcode_summary,
    get_indian_zipcodes,
    get_indian_hubs_df,
    calculate_india_investment_roi,
    CITY_CAMERA_COORDINATES
)


def test_indian_listings_structure():
    """Verify Indian listings dataset structure, row count, and essential fields."""
    df = get_indian_listings_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 2000
    assert df['city'].nunique() >= 15
    assert 'price_inr' in df.columns
    assert 'price_usd' in df.columns
    assert 'occupancy_rate_365' in df.columns


def test_indian_postal_zipcodes_present():
    """Verify authentic 6-digit Indian postal PIN / ZIP codes are populated and valid."""
    df = get_indian_listings_df()
    assert 'zipcode' in df.columns
    assert 'pincode' in df.columns
    assert 'postal_code' in df.columns
    
    zipcodes = get_indian_zipcodes()
    assert len(zipcodes) >= 20
    # Verify every PIN code is 6 digits
    for z in zipcodes:
        assert len(str(z)) == 6, f"Invalid postal PIN code: {z}"
        assert str(z).isdigit(), f"PIN code must be numeric: {z}"


def test_indian_city_summary_no_keyerror():
    """Verify get_indian_city_summary executes without KeyError and aggregates accurately."""
    summary = get_indian_city_summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) >= 15
    assert 'avg_gross_annual_lakhs' in summary.columns
    assert 'median_adr_inr' in summary.columns
    assert (summary['total_listings'] > 0).all()


def test_indian_zipcode_summary():
    """Verify get_indian_zipcode_summary provides micro-market PIN code level metrics."""
    zip_summary = get_indian_zipcode_summary()
    assert isinstance(zip_summary, pd.DataFrame)
    assert len(zip_summary) >= 25
    assert 'zipcode' in zip_summary.columns
    assert 'median_adr_inr' in zip_summary.columns
    assert 'avg_occupancy' in zip_summary.columns


def test_indian_hubs_and_roi_calculator():
    """Verify investment hubs dataframe and financial ROI simulator calculation."""
    hubs = get_indian_hubs_df()
    assert isinstance(hubs, pd.DataFrame)
    assert 'hub' in hubs.columns
    assert 'avg_price_inr' in hubs.columns

    roi = calculate_india_investment_roi(
        property_cost_lakhs=80.0,
        nightly_rate_inr=10000.0,
        occupancy_pct=70.0,
        opex_pct=25.0
    )
    assert isinstance(roi, dict)
    assert roi['gross_annual_rev_lakhs'] > 0
    assert roi['gross_yield_pct'] > 0
    assert roi['payback_years'] > 0
    assert roi['booked_nights_per_year'] in (255, 256)
