"""
Data Cleaning Module for Inside Airbnb Austin Dataset.

Handles structured data transformations:
- Currency string parsing to float
- Boolean text normalisation ('t'/'f' -> 1/0)
- Bathrooms text parsing to numeric values
- Amenities JSON parsing, count extraction, and key feature indicators
- Documented missing value imputation
- Outlier detection and treatment
"""
import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def clean_price_string(val: Any) -> float:
    """
    Sanitize and convert raw currency strings to floating point numbers.
    
    Examples:
        '$1,200.00' -> 1200.0
        '$79.74' -> 79.74
        np.nan -> np.nan
        '0' -> 0.0
    """
    if pd.isna(val) or val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    
    cleaned = str(val).strip().replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return np.nan


def clean_boolean(val: Any) -> int:
    """
    Convert text or numeric boolean flags ('t'/'f', 1/0) to integer 1 or 0.
    """
    if pd.isna(val) or val is None:
        return 0
    val_str = str(val).strip().lower()
    if val_str in ("t", "true", "1", "yes"):
        return 1
    return 0


def clean_bathrooms(val: Any) -> float:
    """
    Extract numeric bathroom count from free-form text.
    
    Examples:
        '1 bath' -> 1.0
        '2.5 baths' -> 2.5
        'Half-bath' -> 0.5
        'Shared half-bath' -> 0.5
        '1 private bath' -> 1.0
    """
    if pd.isna(val) or val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    
    s = str(val).strip().lower()
    if "half" in s:
        # Check for numbers before half (e.g., '1 half-bath')
        match = re.search(r"(\d+(\.\d+)?)", s)
        if match:
            return float(match.group(1)) + 0.5
        return 0.5
    
    match = re.search(r"(\d+(\.\d+)?)", s)
    if match:
        return float(match.group(1))
    return np.nan


def parse_amenities(amenities_raw: Any) -> Tuple[int, Dict[str, int]]:
    """
    Parse amenities string/JSON array to extract total amenity count and
    binary indicator flags for key amenities known to influence price and satisfaction.
    """
    if amenities_raw is None:
        return 0, {}
    if isinstance(amenities_raw, float) and np.isnan(amenities_raw):
        return 0, {}
    if isinstance(amenities_raw, (list, tuple, set, dict, str)) and len(amenities_raw) == 0:
        return 0, {}
    
    items: List[str] = []
    if isinstance(amenities_raw, list):
        items = [str(x).strip().lower() for x in amenities_raw]
    elif isinstance(amenities_raw, str):
        raw = amenities_raw.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                items = [str(x).strip().lower() for x in parsed]
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, set)):
                    items = [str(x).strip().lower() for x in parsed]
            except Exception:
                # Fallback: regex split on commas ignoring braces
                cleaned = re.sub(r'[{}\[\]"\'\\]', '', raw)
                items = [x.strip().lower() for x in cleaned.split(',') if x.strip()]
    
    count = len(items)
    combined_text = " ".join(items)
    
    indicators = {
        "has_wifi": int(any("wifi" in item or "internet" in item for item in items)),
        "has_pool": int(any("pool" in item for item in items)),
        "has_hot_tub": int(any("hot tub" in item or "jacuzzi" in item for item in items)),
        "has_free_parking": int(any("free parking" in item or "free driveway parking" in item or "free carport" in item for item in items)),
        "has_air_conditioning": int(any("air conditioning" in item or "central air" in item or "ac" == item for item in items)),
        "has_workspace": int(any("dedicated workspace" in item or "desk" in item for item in items)),
        "has_kitchen": int(any("kitchen" in item for item in items)),
        "has_patio_or_balcony": int(any("patio" in item or "balcony" in item for item in items)),
        "has_washer_dryer": int(any("washer" in item or "dryer" in item for item in items)),
        "has_self_checkin": int(any("self check-in" in item or "keypad" in item or "smart lock" in item for item in items)),
        "has_pets_allowed": int(any("pets allowed" in item or "pet friendly" in item for item in items)),
    }
    
    return count, indicators


def treat_price_outliers(
    df: pd.DataFrame,
    min_price: float = 15.0,
    upper_percentile: float = 0.995,
    price_col: str = "price"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter erroneous non-positive prices and cap/filter extreme luxury outliers.
    
    Rule Explanation:
    1. Listings with price < $15/night are non-commercial placeholders, test listings,
       or data errors and are excluded.
    2. Prices above the 99.5th percentile (or extreme luxury estates exceeding $1,500-$2,500/night)
       heavily distort ordinary market regression and variance tests.
       We filter or mark them to ensure models learn general market dynamics.
    """
    initial_count = len(df)
    valid_price_df = df[df[price_col].notnull()].copy()
    
    # 1. Minimum price filter
    non_zero_df = valid_price_df[valid_price_df[price_col] >= min_price].copy()
    dropped_low = len(valid_price_df) - len(non_zero_df)
    
    # 2. Upper cutoff
    cutoff_val = float(non_zero_df[price_col].quantile(upper_percentile))
    cleaned_df = non_zero_df[non_zero_df[price_col] <= cutoff_val].copy()
    dropped_high = len(non_zero_df) - len(cleaned_df)
    
    audit_stats = {
        "initial_rows": initial_count,
        "valid_price_rows": len(valid_price_df),
        "min_price_threshold": min_price,
        "dropped_sub_min": dropped_low,
        "upper_cutoff_percentile": upper_percentile,
        "upper_cutoff_val": round(cutoff_val, 2),
        "dropped_upper_outliers": dropped_high,
        "final_cleaned_rows": len(cleaned_df),
        "retention_rate_pct": round(len(cleaned_df) / initial_count * 100, 2)
    }
    
    return cleaned_df, audit_stats


def clean_listings(
    raw_csv_path: str = "data/raw/listings.csv.gz",
    output_csv_path: Optional[str] = "data/processed/listings_cleaned.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end data cleaning pipeline for listings table.
    """
    print(f"Loading raw listings from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, low_memory=False)
    
    audit_log: Dict[str, Any] = {"raw_shape": df.shape}
    
    # 1. Deduplicate by listing ID
    dup_count = int(df["id"].duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates(subset=["id"]).copy()
    audit_log["duplicates_removed"] = dup_count
    
    # 2. Clean numeric price
    df["price_raw"] = df["price"]
    df["price"] = df["price"].apply(clean_price_string)
    
    # 3. Clean booleans
    df["host_is_superhost"] = df["host_is_superhost"].apply(clean_boolean)
    df["host_identity_verified"] = df["host_identity_verified"].apply(clean_boolean)
    if "instant_bookable" in df.columns:
        df["instant_bookable"] = df["instant_bookable"].apply(clean_boolean)
    else:
        df["instant_bookable"] = 0
        
    # 4. Clean bathrooms
    df["bathrooms"] = df["bathrooms_text"].apply(clean_bathrooms)
    
    # Impute missing bedrooms, beds, bathrooms with sensible median based on accommodates
    # Group medians by accommodates for accuracy
    med_bath = df.groupby("accommodates")["bathrooms"].transform("median").fillna(1.0)
    med_bed = df.groupby("accommodates")["bedrooms"].transform("median").fillna(1.0)
    med_beds = df.groupby("accommodates")["beds"].transform("median").fillna(1.0)
    
    df["bathrooms"] = df["bathrooms"].fillna(med_bath)
    df["bedrooms"] = df["bedrooms"].fillna(med_bed)
    df["beds"] = df["beds"].fillna(med_beds)
    
    # 5. Clean & Parse Amenities
    print("Parsing amenities and generating feature flags...")
    amenity_counts = []
    amenity_flags = {
        "has_wifi": [],
        "has_pool": [],
        "has_hot_tub": [],
        "has_free_parking": [],
        "has_air_conditioning": [],
        "has_workspace": [],
        "has_kitchen": [],
        "has_patio_or_balcony": [],
        "has_washer_dryer": [],
        "has_self_checkin": [],
        "has_pets_allowed": [],
    }
    
    for raw_am in df["amenities"]:
        cnt, indicators = parse_amenities(raw_am)
        amenity_counts.append(cnt)
        for k in amenity_flags:
            amenity_flags[k].append(indicators.get(k, 0))
            
    df["amenities_count"] = amenity_counts
    for k, vals in amenity_flags.items():
        df[k] = vals
        
    # 6. Parse dates
    date_cols = ["first_review", "last_review"]
    for dc in date_cols:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            
    # Calculate days since first review (host listing age proxy)
    ref_date = pd.to_datetime("2026-09-17")
    df["days_since_first_review"] = (ref_date - df["first_review"]).dt.days.fillna(0)
    
    # 7. Normalize neighborhood names and night constraints
    df["neighbourhood"] = df["neighbourhood_cleansed"].astype(str).str.strip()
    df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    df["maximum_nights"] = pd.to_numeric(df["maximum_nights"], errors="coerce").fillna(365).clip(lower=1).astype(int)
    
    # 8. Treat Outliers on price
    df_cleaned, price_audit = treat_price_outliers(df, min_price=15.0, upper_percentile=0.995)
    audit_log["price_outlier_treatment"] = price_audit
    
    # Add log price for ML modelling
    df_cleaned["log_price"] = np.log(df_cleaned["price"])
    
    # Select prioritized columns for downstream analytics
    selected_columns = [
        "id", "name", "host_id", "host_name", "host_is_superhost",
        "host_identity_verified", "neighbourhood", "latitude", "longitude",
        "property_type", "room_type", "accommodates", "bathrooms", "bedrooms",
        "beds", "price", "log_price", "minimum_nights", "maximum_nights",
        "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
        "first_review", "last_review", "days_since_first_review",
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication", "review_scores_location",
        "review_scores_value", "instant_bookable", "amenities_count",
        "has_wifi", "has_pool", "has_hot_tub", "has_free_parking",
        "has_air_conditioning", "has_workspace", "has_kitchen",
        "has_patio_or_balcony", "has_washer_dryer", "has_self_checkin",
        "has_pets_allowed"
    ]
    
    # Filter to existing
    final_cols = [c for c in selected_columns if c in df_cleaned.columns]
    final_df = df_cleaned[final_cols].copy()
    
    if output_csv_path:
        final_df.to_csv(output_csv_path, index=False)
        print(f"Cleaned listings saved to {output_csv_path} ({len(final_df):,} records).")
        
    audit_log["final_shape"] = final_df.shape
    return final_df, audit_log


def clean_calendar(
    raw_csv_path: str = "data/raw/calendar.csv.gz",
    output_csv_path: Optional[str] = "data/processed/calendar_cleaned.csv",
    sample_listings: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean calendar availability data and compute occupancy proxies.
    
    Occupancy Proxy Definition:
    In Inside Airbnb, available='f' signifies the date is booked or blocked.
    Occupancy proxy is calculated as:
        occupancy_rate = count(available == 'f') / total_days
    over 30-day, 90-day, and 365-day forward horizons.
    """
    print(f"Loading calendar from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, low_memory=False)
    
    # Clean available column
    df["is_available"] = df["available"].apply(clean_boolean)
    df["is_booked"] = 1 - df["is_available"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Price if available
    if "price" in df.columns:
        df["price"] = df["price"].apply(clean_price_string)
        
    print(f"Calendar processed: {len(df):,} availability rows.")
    
    # Compute listing-level forward occupancy metrics
    print("Computing listing forward occupancy metrics (30d, 90d, 365d)...")
    min_date = df["date"].min()
    d30 = min_date + pd.Timedelta(days=30)
    d90 = min_date + pd.Timedelta(days=90)
    
    agg_365 = df.groupby("listing_id").agg(
        total_days=("date", "count"),
        booked_days_365=("is_booked", "sum")
    ).reset_index()
    agg_365["occupancy_rate_365"] = (agg_365["booked_days_365"] / agg_365["total_days"]).round(4)
    
    df_30 = df[df["date"] <= d30]
    agg_30 = df_30.groupby("listing_id")["is_booked"].agg(
        booked_days_30="sum",
        days_count_30="count"
    ).reset_index()
    agg_30["occupancy_rate_30"] = (agg_30["booked_days_30"] / agg_30["days_count_30"]).round(4)
    
    df_90 = df[df["date"] <= d90]
    agg_90 = df_90.groupby("listing_id")["is_booked"].agg(
        booked_days_90="sum",
        days_count_90="count"
    ).reset_index()
    agg_90["occupancy_rate_90"] = (agg_90["booked_days_90"] / agg_90["days_count_90"]).round(4)
    
    # Merge aggregations
    occupancy_df = agg_365.merge(agg_30[["listing_id", "booked_days_30", "occupancy_rate_30"]], on="listing_id", how="left")
    occupancy_df = occupancy_df.merge(agg_90[["listing_id", "booked_days_90", "occupancy_rate_90"]], on="listing_id", how="left")
    occupancy_df["booked_days_30"] = occupancy_df["booked_days_30"].fillna(0).astype(int)
    occupancy_df["booked_days_90"] = occupancy_df["booked_days_90"].fillna(0).astype(int)
    occupancy_df["occupancy_rate_30"] = occupancy_df["occupancy_rate_30"].fillna(0.0)
    occupancy_df["occupancy_rate_90"] = occupancy_df["occupancy_rate_90"].fillna(0.0)
    
    if output_csv_path:
        occupancy_df.to_csv(output_csv_path, index=False)
        print(f"Listing occupancy proxies saved to {output_csv_path} ({len(occupancy_df):,} listings).")
        
    audit_stats = {
        "raw_calendar_rows": len(df),
        "unique_listings_in_calendar": df["listing_id"].nunique(),
        "mean_occupancy_30d": float(occupancy_df["occupancy_rate_30"].mean().round(4)),
        "mean_occupancy_90d": float(occupancy_df["occupancy_rate_90"].mean().round(4)),
        "mean_occupancy_365d": float(occupancy_df["occupancy_rate_365"].mean().round(4)),
    }
    return occupancy_df, audit_stats


def clean_reviews(
    raw_csv_path: str = "data/raw/reviews.csv.gz",
    output_csv_path: Optional[str] = "data/processed/reviews_cleaned.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean review records and filter invalid/blank guest comments.
    """
    print(f"Loading reviews from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, low_memory=False)
    initial_count = len(df)
    
    # Drop records with missing comments
    df = df.dropna(subset=["comments"]).copy()
    df["comments"] = df["comments"].astype(str).str.strip()
    df = df[df["comments"].str.len() > 3].copy()
    
    # Parse review date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Deduplicate on review id
    df = df.drop_duplicates(subset=["id"]).copy()
    
    if output_csv_path:
        # Save a clean sample or full dataset (for NLP Phase)
        df.to_csv(output_csv_path, index=False)
        print(f"Cleaned reviews saved to {output_csv_path} ({len(df):,} records).")
        
    audit_stats = {
        "raw_reviews_count": initial_count,
        "clean_reviews_count": len(df),
        "dropped_blank_or_short": initial_count - len(df),
        "unique_listings_reviewed": df["listing_id"].nunique()
    }
    return df, audit_stats
