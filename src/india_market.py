"""
src/india_market.py
Market Intelligence, Geospatial Benchmarks, and Investment Analytics
for Premier Indian Short-Term Rental & Vacation Hubs across 25+ Destinations.
"""

import os
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

INDIA_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "india_listings_cleaned.csv")

# City / State View Coordinates for Camera Zoom
CITY_CAMERA_COORDINATES: Dict[str, Tuple[float, float, float, int]] = {
    'All India (National Overview)': (21.8, 79.2, 4.3, 35),
    'Goa': (15.42, 73.88, 9.6, 32),
    'Mumbai': (19.05, 72.84, 11.0, 35),
    'Alibaug': (18.75, 72.88, 11.2, 30),
    'Lonavala & Khandala': (18.75, 73.41, 11.0, 35),
    'Jaipur': (26.91, 75.80, 11.4, 30),
    'Udaipur': (24.58, 73.68, 11.8, 35),
    'Manali': (32.25, 77.18, 11.0, 40),
    'Shimla & Mashobra': (31.11, 77.19, 11.0, 35),
    'Dharamshala & McLeodGanj': (32.24, 76.32, 11.3, 35),
    'Bengaluru': (12.96, 77.63, 11.0, 30),
    'Coorg (Kodagu)': (12.38, 75.80, 10.0, 30),
    'Kerala (Alleppey & Kochi)': (9.75, 76.35, 9.3, 30),
    'Delhi NCR': (28.53, 77.18, 10.8, 30),
    'Rishikesh': (30.13, 78.33, 11.6, 35),
    'Mussoorie & Landour': (30.46, 78.08, 11.8, 40),
    'Pondicherry': (11.93, 79.83, 12.0, 30),
    'Varanasi': (25.31, 83.00, 11.8, 30),
}


def get_indian_listings_df() -> pd.DataFrame:
    """
    Load the clean Indian listings dataset (2,480 listings across 17 cities).
    Generates dataset on the fly if not yet present on disk.
    """
    if os.path.exists(INDIA_CSV_PATH):
        return pd.read_csv(INDIA_CSV_PATH)
    
    from src.generate_india_dataset import generate_dataset
    return generate_dataset(INDIA_CSV_PATH)


def get_indian_city_summary() -> pd.DataFrame:
    """
    Returns aggregated market benchmarks by city (ADR, occupancy, revenue).
    """
    df = get_indian_listings_df()
    
    # Safe revenue column resolution
    rev_col = 'annual_gross_revenue_lakhs' if 'annual_gross_revenue_lakhs' in df.columns else (
        'annual_gross_lakhs' if 'annual_gross_lakhs' in df.columns else None
    )
    if rev_col is None:
        df['annual_gross_lakhs'] = (365 * df.get('occupancy_rate_365', 0.65) * df['price_inr']) / 100000.0
        rev_col = 'annual_gross_lakhs'

    summary = df.groupby(['city', 'state', 'region']).agg(
        total_listings=('id', 'count'),
        avg_adr_inr=('price_inr', 'mean'),
        median_adr_inr=('price_inr', 'median'),
        avg_adr_usd=('price_usd', 'mean'),
        avg_occupancy=('occupancy_rate_365', 'mean'),
        pct_superhost=('host_is_superhost', lambda x: (x.mean() * 100.0)),
        pct_with_pool=('has_pool', lambda x: (x.mean() * 100.0)),
        pct_power_backup=('has_inverter_power_backup', lambda x: (x.mean() * 100.0)),
        avg_gross_annual_lakhs=(rev_col, 'mean'),
        avg_rating=('review_scores_rating', 'mean')
    ).reset_index()

    summary['avg_adr_inr'] = summary['avg_adr_inr'].round(0)
    summary['avg_adr_usd'] = summary['avg_adr_usd'].round(0)
    summary['avg_gross_annual_lakhs'] = summary['avg_gross_annual_lakhs'].round(2)
    summary['avg_rating'] = summary['avg_rating'].round(2)
    summary['avg_occupancy'] = (summary['avg_occupancy'] * 100.0).round(1)
    summary['pct_superhost'] = summary['pct_superhost'].round(1)
    summary['pct_with_pool'] = summary['pct_with_pool'].round(1)
    summary['pct_power_backup'] = summary['pct_power_backup'].round(1)

    return summary.sort_values(by='median_adr_inr', ascending=False)


def get_indian_zipcode_summary() -> pd.DataFrame:
    """
    Returns aggregated market benchmarks by Indian postal PIN / ZIP code.
    """
    df = get_indian_listings_df()
    if 'zipcode' not in df.columns:
        if 'pincode' in df.columns:
            df['zipcode'] = df['pincode']
        elif 'postal_code' in df.columns:
            df['zipcode'] = df['postal_code']
        else:
            df['zipcode'] = '110001'

    rev_col = 'annual_gross_revenue_lakhs' if 'annual_gross_revenue_lakhs' in df.columns else (
        'annual_gross_lakhs' if 'annual_gross_lakhs' in df.columns else 'price_inr'
    )

    zip_summary = df.groupby(['zipcode', 'neighborhood', 'city', 'state']).agg(
        total_listings=('id', 'count'),
        median_adr_inr=('price_inr', 'median'),
        avg_adr_inr=('price_inr', 'mean'),
        avg_occupancy=('occupancy_rate_365', 'mean'),
        pct_superhost=('host_is_superhost', lambda x: (x.mean() * 100.0)),
        pct_pool=('has_pool', lambda x: (x.mean() * 100.0)),
        pct_power_backup=('has_inverter_power_backup', lambda x: (x.mean() * 100.0)),
        avg_gross_annual_lakhs=(rev_col, 'mean'),
        avg_rating=('review_scores_rating', 'mean')
    ).reset_index()

    zip_summary['median_adr_inr'] = zip_summary['median_adr_inr'].round(0)
    zip_summary['avg_adr_inr'] = zip_summary['avg_adr_inr'].round(0)
    zip_summary['avg_occupancy'] = (zip_summary['avg_occupancy'] * 100.0).round(1)
    zip_summary['pct_superhost'] = zip_summary['pct_superhost'].round(1)
    zip_summary['pct_pool'] = zip_summary['pct_pool'].round(1)
    zip_summary['pct_power_backup'] = zip_summary['pct_power_backup'].round(1)
    zip_summary['avg_gross_annual_lakhs'] = zip_summary['avg_gross_annual_lakhs'].round(2)
    zip_summary['avg_rating'] = zip_summary['avg_rating'].round(2)

    return zip_summary.sort_values(by='median_adr_inr', ascending=False)


def get_indian_zipcodes() -> List[str]:
    """Returns sorted list of unique postal PIN / ZIP codes in the Indian dataset."""
    df = get_indian_listings_df()
    col = 'zipcode' if 'zipcode' in df.columns else ('pincode' if 'pincode' in df.columns else 'postal_code')
    if col in df.columns:
        return sorted(df[col].dropna().astype(str).unique().tolist())
    return []


def get_indian_hubs_df() -> pd.DataFrame:
    """
    Returns city hubs with columns compatible with the investment simulator.
    """
    df = get_indian_city_summary()
    df['hub'] = df['city'] + " (" + df['state'] + ")"
    df['avg_price_inr'] = df['avg_adr_inr']
    df['occupancy_rate'] = df['avg_occupancy'] / 100.0
    return df


def calculate_india_investment_roi(
    property_cost_lakhs: float,
    nightly_rate_inr: float,
    occupancy_pct: float,
    opex_pct: float = 25.0
) -> Dict[str, float]:
    """
    Calculate financial feasibility, gross yield, and payback period for an Indian rental investment.
    """
    days_per_year = 365
    annual_booked_nights = days_per_year * (occupancy_pct / 100.0)
    gross_annual_rev_inr = annual_booked_nights * nightly_rate_inr
    opex_annual_inr = gross_annual_rev_inr * (opex_pct / 100.0)
    net_operating_income_inr = gross_annual_rev_inr - opex_annual_inr

    property_cost_inr = property_cost_lakhs * 100000.0
    gross_yield_pct = (gross_annual_rev_inr / property_cost_inr) * 100.0 if property_cost_inr > 0 else 0.0
    net_yield_pct = (net_operating_income_inr / property_cost_inr) * 100.0 if property_cost_inr > 0 else 0.0
    payback_years = (property_cost_inr / net_operating_income_inr) if net_operating_income_inr > 0 else float('inf')

    return {
        'gross_annual_rev_lakhs': round(gross_annual_rev_inr / 100000.0, 2),
        'net_annual_income_lakhs': round(net_operating_income_inr / 100000.0, 2),
        'gross_yield_pct': round(gross_yield_pct, 1),
        'net_yield_pct': round(net_yield_pct, 1),
        'payback_years': round(payback_years, 1),
        'booked_nights_per_year': int(round(annual_booked_nights))
    }
