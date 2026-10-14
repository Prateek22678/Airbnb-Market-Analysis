-- ====================================================================
-- SCHEMA DDL: Austin Airbnb Market Analytical Database
-- Relational SQLite Database: data/processed/airbnb_austin.db
-- ====================================================================

DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS calendar_occupancy;
DROP TABLE IF EXISTS listings;

-- 1. Listings Table: Cleaned Property and Host Dimensions
CREATE TABLE listings (
    id INTEGER PRIMARY KEY,
    name TEXT,
    host_id INTEGER NOT NULL,
    host_name TEXT,
    host_is_superhost INTEGER NOT NULL DEFAULT 0,
    host_identity_verified INTEGER NOT NULL DEFAULT 0,
    neighbourhood TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    property_type TEXT NOT NULL,
    room_type TEXT NOT NULL,
    accommodates INTEGER NOT NULL,
    bathrooms REAL NOT NULL,
    bedrooms REAL NOT NULL,
    beds REAL NOT NULL,
    price REAL NOT NULL,
    log_price REAL NOT NULL,
    minimum_nights INTEGER NOT NULL,
    maximum_nights INTEGER NOT NULL,
    number_of_reviews INTEGER NOT NULL,
    number_of_reviews_ltm INTEGER NOT NULL,
    reviews_per_month REAL,
    first_review TEXT,
    last_review TEXT,
    days_since_first_review REAL,
    review_scores_rating REAL,
    review_scores_accuracy REAL,
    review_scores_cleanliness REAL,
    review_scores_checkin REAL,
    review_scores_communication REAL,
    review_scores_location REAL,
    review_scores_value REAL,
    instant_bookable INTEGER NOT NULL DEFAULT 0,
    amenities_count INTEGER NOT NULL,
    has_wifi INTEGER NOT NULL DEFAULT 0,
    has_pool INTEGER NOT NULL DEFAULT 0,
    has_hot_tub INTEGER NOT NULL DEFAULT 0,
    has_free_parking INTEGER NOT NULL DEFAULT 0,
    has_air_conditioning INTEGER NOT NULL DEFAULT 0,
    has_workspace INTEGER NOT NULL DEFAULT 0,
    has_kitchen INTEGER NOT NULL DEFAULT 0,
    has_patio_or_balcony INTEGER NOT NULL DEFAULT 0,
    has_washer_dryer INTEGER NOT NULL DEFAULT 0,
    has_self_checkin INTEGER NOT NULL DEFAULT 0,
    has_pets_allowed INTEGER NOT NULL DEFAULT 0
);

-- 2. Calendar Occupancy Table: Forward-Looking Demand Metrics
CREATE TABLE calendar_occupancy (
    listing_id INTEGER PRIMARY KEY,
    total_days INTEGER NOT NULL,
    booked_days_365 INTEGER NOT NULL,
    occupancy_rate_365 REAL NOT NULL,
    booked_days_30 INTEGER NOT NULL,
    occupancy_rate_30 REAL NOT NULL,
    booked_days_90 INTEGER NOT NULL,
    occupancy_rate_90 REAL NOT NULL,
    FOREIGN KEY (listing_id) REFERENCES listings(id) ON DELETE CASCADE
);

-- 3. Reviews Table: Unstructured Guest Feedback Records
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    listing_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    reviewer_id INTEGER NOT NULL,
    reviewer_name TEXT,
    comments TEXT,
    FOREIGN KEY (listing_id) REFERENCES listings(id) ON DELETE CASCADE
);

-- 4. Analytical Performance Indexes
CREATE INDEX idx_listings_neighbourhood ON listings(neighbourhood);
CREATE INDEX idx_listings_room_type ON listings(room_type);
CREATE INDEX idx_listings_superhost ON listings(host_is_superhost);
CREATE INDEX idx_listings_price ON listings(price);
CREATE INDEX idx_reviews_listing_id ON reviews(listing_id);
CREATE INDEX idx_reviews_date ON reviews(date);
CREATE INDEX idx_calendar_occupancy_listing_id ON calendar_occupancy(listing_id);
