"""
Execute Phase 2 Structured Data Cleaning Pipeline.
Saves processed files to data/processed/ and prints audit logs.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
from src.cleaning import clean_listings, clean_calendar, clean_reviews

def main():
    print("==================================================")
    print("STARTING PHASE 2 DATA CLEANING PIPELINE")
    print("==================================================")
    t0 = time.time()
    
    # 1. Clean listings
    listings_df, listings_audit = clean_listings(
        raw_csv_path="data/raw/listings.csv.gz",
        output_csv_path="data/processed/listings_cleaned.csv"
    )
    print("\n[AUDIT] Listings Cleaning Summary:")
    print(json.dumps(listings_audit, indent=2))
    
    # 2. Clean calendar & compute occupancy proxies
    calendar_df, calendar_audit = clean_calendar(
        raw_csv_path="data/raw/calendar.csv.gz",
        output_csv_path="data/processed/calendar_cleaned.csv"
    )
    print("\n[AUDIT] Calendar Occupancy Summary:")
    print(json.dumps(calendar_audit, indent=2))
    
    # 3. Clean reviews (sample/full)
    reviews_df, reviews_audit = clean_reviews(
        raw_csv_path="data/raw/reviews.csv.gz",
        output_csv_path="data/processed/reviews_cleaned.csv"
    )
    print("\n[AUDIT] Reviews Summary:")
    print(json.dumps(reviews_audit, indent=2))
    
    total_time = round(time.time() - t0, 1)
    print(f"\n==================================================")
    print(f"PHASE 2 DATA CLEANING COMPLETE in {total_time}s!")
    print(f"Cleaned Listings: {len(listings_df):,} rows")
    print(f"Cleaned Listings in Calendar: {len(calendar_df):,} rows")
    print(f"Cleaned Reviews: {len(reviews_df):,} rows")
    print(f"==================================================")

if __name__ == "__main__":
    main()
