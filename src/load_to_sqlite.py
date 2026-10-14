"""
Load cleaned CSV datasets into SQLite relational database.
Creates database at data/processed/airbnb_austin.db using sql/schema.sql.
"""
import os
import sqlite3
import time
import pandas as pd

DB_PATH = "data/processed/airbnb_austin.db"
SCHEMA_PATH = "sql/schema.sql"

def init_database() -> sqlite3.Connection:
    """Initialize SQLite database with schema DDL."""
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass
            
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()
    print(f"Database schema initialized from {SCHEMA_PATH}.")
    return conn

def load_data_to_sqlite() -> None:
    t0 = time.time()
    conn = init_database()
    
    # 1. Load listings_cleaned
    listings_csv = "data/processed/listings_cleaned.csv"
    print(f"Loading {listings_csv} into 'listings' table...")
    df_listings = pd.read_csv(listings_csv)
    df_listings.to_sql("listings", conn, if_exists="append", index=False)
    print(f"Loaded {len(df_listings):,} listings.")
    
    # 2. Load calendar_cleaned
    calendar_csv = "data/processed/calendar_cleaned.csv"
    print(f"Loading {calendar_csv} into 'calendar_occupancy' table...")
    df_cal = pd.read_csv(calendar_csv)
    df_cal.to_sql("calendar_occupancy", conn, if_exists="append", index=False)
    print(f"Loaded {len(df_cal):,} calendar occupancy records.")
    
    # 3. Load reviews_cleaned in chunks
    reviews_csv = "data/processed/reviews_cleaned.csv"
    print(f"Loading {reviews_csv} into 'reviews' table (streaming chunks)...")
    chunk_size = 50_000
    total_reviews = 0
    for chunk in pd.read_csv(reviews_csv, chunksize=chunk_size, low_memory=False):
        # Filter reviews to listings that exist in our cleaned listings table
        valid_chunk = chunk[chunk["listing_id"].isin(df_listings["id"])].copy()
        valid_chunk.to_sql("reviews", conn, if_exists="append", index=False)
        total_reviews += len(valid_chunk)
        print(f"  Inserted {total_reviews:,} reviews...")
        
    conn.commit()
    
    # Print Table Row Counts Verification
    cur = conn.cursor()
    print("\n=== SQL DATABASE LOAD VERIFICATION ===")
    for table in ["listings", "calendar_occupancy", "reviews"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"Table '{table}': {count:,} rows")
        
    conn.close()
    elapsed = round(time.time() - t0, 1)
    print(f"Database loaded and indexed successfully in {elapsed}s at {DB_PATH}.")

if __name__ == "__main__":
    load_data_to_sqlite()
