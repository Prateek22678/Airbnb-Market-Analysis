# Data Dictionary: Austin Airbnb Market Analysis

Source: Inside Airbnb (http://insideairbnb.com/get-the-data/)  
Snapshot Date: September 17, 2026  
Location: Austin, Texas, USA  

---

## 1. Raw Data Tables Overview

| Table | File Format | Source URL Pattern | Description |
|---|---|---|---|
| `listings` | CSV / GZ | `data.insideairbnb.com/.../austin/2026-09-17/data/listings.csv.gz` | Detailed listing profiles including host information, amenities, spatial coordinates, property types, and aggregate review ratings. |
| `calendar` | CSV / GZ | `data.insideairbnb.com/.../austin/2026-09-17/data/calendar.csv.gz` | Forward-looking 365-day booking availability, nightly listing rates, and minimum/maximum night constraints per listing. |
| `reviews` | CSV / GZ | `data.insideairbnb.com/.../austin/2026-09-17/data/reviews.csv.gz` | Historical unstructured guest reviews including review ID, reviewer name, date, and full comment text. |

---

## 2. Key Listing Fields (Pre- and Post-Cleaning)

*(This dictionary will be continuously updated throughout Phase 2 Data Cleaning)*

| Column | Raw Type | Clean Type | Description |
|---|---|---|---|
| `id` | int64 | int64 / TEXT (PK) | Unique Airbnb listing identifier. |
| `name` | string | string | Title of the listing on Airbnb. |
| `host_id` | int64 | int64 | Unique host identifier. |
| `host_is_superhost` | string ('t'/'f') | boolean (1/0) | Whether the host has attained Airbnb Superhost badge status. |
| `neighbourhood_cleansed`| string | string | Standardized municipality neighborhood / planning area. |
| `latitude` | float64 | float64 | WGS84 latitude coordinate. |
| `longitude` | float64 | float64 | WGS84 longitude coordinate. |
| `property_type` | string | string | Categorical property description (e.g., Entire rental unit, Entire residential home). |
| `room_type` | string | string | Accommodation privacy tier (Entire home/apt, Private room, Shared room, Hotel room). |
| `accommodates` | int64 | int64 | Maximum number of guests allowed. |
| `bathrooms` | float64 | float64 | Number of bathrooms. |
| `bedrooms` | float64 | float64 | Number of bedrooms. |
| `beds` | float64 | float64 | Number of beds provided. |
| `amenities` | string (JSON array) | list / parsed | Parsed amenities tags (e.g. Wifi, Pool, Hot tub, Free parking). |
| `price` | string (e.g. "$150.00") | float64 | Nightly rate in USD. |
| `minimum_nights` | int64 | int64 | Minimum stay duration required. |
| `maximum_nights` | int64 | int64 | Maximum stay duration allowed. |
| `number_of_reviews` | int64 | int64 | Total lifetime review count. |
| `review_scores_rating` | float64 | float64 | Aggregate review score (0-5 scale). |
| `instant_bookable` | string ('t'/'f') | boolean (1/0) | Whether guests can book without prior host approval. |
