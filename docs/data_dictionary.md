# Data Dictionary: Austin Airbnb Market Analysis

Source: Inside Airbnb (http://insideairbnb.com/get-the-data/)  
Snapshot Date: September 17, 2026  
Location: Austin, Texas, USA  

---

## 1. Processed Tables Overview

| Table | File Format | Record Count | Description |
|---|---|---|---|
| `listings_cleaned` | CSV / SQLite | 9,945 | Cleaned listing profiles with sanitized pricing, imputed capacities, review ratings, and parsed amenity flags. |
| `calendar_cleaned` | CSV / SQLite | 11,111 | Forward-looking forward occupancy metrics calculated over 30d, 90d, and 365d windows per listing. |
| `reviews_cleaned` | CSV / SQLite | 650,134 | Cleaned guest review comments and review timestamps. |

---

## 2. Listings Schema (`listings_cleaned`)

| Column Name | Data Type | Null Count | Description & Value Domain |
|---|---|---|---|
| `id` | int64 | 0 | Primary Key. Unique Airbnb listing identifier. |
| `name` | string | 0 | Title of the listing on Airbnb. |
| `host_id` | int64 | 0 | Unique host identifier. |
| `host_name` | string | 0 | Host display name. |
| `host_is_superhost` | int64 (0/1) | 0 | Boolean indicator: 1 = Superhost status attained, 0 = Regular host. |
| `host_identity_verified`| int64 (0/1) | 0 | Boolean indicator: 1 = Host identity officially verified by Airbnb. |
| `neighbourhood` | string | 0 | Cleaned municipality neighborhood / ZIP code area in Austin. |
| `latitude` | float64 | 0 | WGS84 latitude coordinate. |
| `longitude` | float64 | 0 | WGS84 longitude coordinate. |
| `property_type` | string | 0 | Full property category (e.g., 'Entire home', 'Entire condo', 'Private room in home'). |
| `room_type` | string | 0 | Privacy tier: 'Entire home/apt', 'Private room', 'Hotel room', 'Shared room'. |
| `accommodates` | int64 | 0 | Maximum guest occupancy capacity. |
| `bathrooms` | float64 | 0 | Number of bathrooms (regex parsed from text; median imputed by capacity). |
| `bedrooms` | float64 | 0 | Number of bedrooms (median imputed by capacity). |
| `beds` | float64 | 0 | Number of beds provided (median imputed by capacity). |
| `price` | float64 | 0 | Nightly listing rate in USD ($15.00 to $3,080.62). |
| `log_price` | float64 | 0 | Natural logarithm of price: `ln(price)`. |
| `minimum_nights` | int64 | 0 | Minimum nights per booking reservation. |
| `maximum_nights` | int64 | 0 | Maximum nights per booking reservation. |
| `number_of_reviews` | int64 | 0 | Lifetime count of guest reviews received. |
| `number_of_reviews_ltm`| int64 | 0 | Count of reviews received in the Last Twelve Months. |
| `reviews_per_month` | float64 | 1,466 | Average review velocity per month (null for unreviewed listings). |
| `first_review` | date | 1,466 | Date of the earliest review received. |
| `last_review` | date | 1,466 | Date of the most recent review received. |
| `days_since_first_review`| float64 | 0 | Listing tenure age in days relative to snapshot date. |
| `review_scores_rating` | float64 | 1,466 | Overall guest rating (0.00 to 5.00 scale). |
| `review_scores_cleanliness`| float64 | 1,466 | Cleanliness rating subscore (0.00 to 5.00). |
| `review_scores_location` | float64 | 1,466 | Location rating subscore (0.00 to 5.00). |
| `review_scores_value` | float64 | 1,466 | Perceived value rating subscore (0.00 to 5.00). |
| `instant_bookable` | int64 (0/1) | 0 | Boolean indicator: 1 = Instant booking allowed without host review. |
| `amenities_count` | int64 | 0 | Total number of individual amenities offered. |
| `has_wifi` | int64 (0/1) | 0 | Indicator: Fast or standard Wifi available. |
| `has_pool` | int64 (0/1) | 0 | Indicator: Private or shared swimming pool available. |
| `has_hot_tub` | int64 (0/1) | 0 | Indicator: Hot tub or jacuzzi available. |
| `has_free_parking` | int64 (0/1) | 0 | Indicator: Free parking on premises/driveway. |
| `has_air_conditioning` | int64 (0/1) | 0 | Indicator: Central air conditioning or AC unit. |
| `has_workspace` | int64 (0/1) | 0 | Indicator: Dedicated workspace / desk and chair. |
| `has_kitchen` | int64 (0/1) | 0 | Indicator: Kitchen with cooking facilities. |
| `has_patio_or_balcony`| int64 (0/1) | 0 | Indicator: Outdoor patio, balcony, or deck. |
| `has_washer_dryer` | int64 (0/1) | 0 | Indicator: In-unit washer and dryer. |
| `has_self_checkin` | int64 (0/1) | 0 | Indicator: Keypad, smart lock, or lockbox entry. |
| `has_pets_allowed` | int64 (0/1) | 0 | Indicator: Pets welcome. |

---

## 3. Calendar Occupancy Schema (`calendar_cleaned`)

| Column Name | Data Type | Description |
|---|---|---|
| `listing_id` | int64 | Foreign Key to `listings_cleaned.id`. |
| `total_days` | int64 | Total calendar days recorded (typically 365). |
| `booked_days_365` | int64 | Days marked booked or blocked over the next 365 days. |
| `occupancy_rate_365` | float64 | Forward 365-day occupancy proxy (`booked_days_365 / total_days`). |
| `booked_days_30` | int64 | Days marked booked or blocked over the next 30 days. |
| `occupancy_rate_30` | float64 | Forward 30-day occupancy proxy (`booked_days_30 / 30`). |
| `booked_days_90` | int64 | Days marked booked or blocked over the next 90 days. |
| `occupancy_rate_90` | float64 | Forward 90-day occupancy proxy (`booked_days_90 / 90`). |

---

## 4. Reviews Schema (`reviews_cleaned`)

| Column Name | Data Type | Description |
|---|---|---|
| `listing_id` | int64 | Foreign Key to `listings_cleaned.id`. |
| `id` | int64 | Unique review identifier. |
| `date` | date | Date review was submitted. |
| `reviewer_id` | int64 | Unique guest reviewer identifier. |
| `reviewer_name` | string | Guest first name. |
| `comments` | string | Full unstructured text of the review. |
