-- ====================================================================
-- AUSTIN AIRBNB MARKET ANALYSIS: 11 PRODUCTION ANALYTICAL QUERIES
-- Database: data/processed/airbnb_austin.db
-- Demonstrates: JOINs, GROUP BY, CTEs, Window Functions (ROW_NUMBER, DENSE_RANK, NTILE)
-- ====================================================================

-- --------------------------------------------------------------------
-- QUERY 1: Inventory & Pricing Breakdown by Room Type
-- Business Question: What is the market composition and pricing power across accommodation privacy tiers?
-- --------------------------------------------------------------------
-- name: query_1_room_type_summary
SELECT 
    room_type,
    COUNT(*) AS listing_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM listings), 2) AS market_share_pct,
    ROUND(AVG(price), 2) AS avg_nightly_price,
    ROUND(MIN(price), 2) AS min_price,
    ROUND(MAX(price), 2) AS max_price,
    ROUND(AVG(accommodates), 1) AS avg_guests_accommodated,
    ROUND(AVG(review_scores_rating), 2) AS avg_review_score
FROM listings
GROUP BY room_type
ORDER BY listing_count DESC;


-- --------------------------------------------------------------------
-- QUERY 2: Top 10 Neighbourhoods Ranked by Median Price
-- Business Question: Which Austin municipal zones command the highest price tiers?
-- Uses CTE and NTILE(2) window function to approximate the median for neighbourhoods with >= 50 listings.
-- --------------------------------------------------------------------
-- name: query_2_neighbourhood_median_ranks
WITH RankedNeighbourhoodPrices AS (
    SELECT 
        neighbourhood,
        price,
        ROW_NUMBER() OVER (PARTITION BY neighbourhood ORDER BY price) AS row_num,
        COUNT(*) OVER (PARTITION BY neighbourhood) AS total_count
    FROM listings
),
NeighbourhoodMedians AS (
    SELECT 
        neighbourhood,
        total_count AS active_listings,
        ROUND(AVG(price), 2) AS median_nightly_price
    FROM RankedNeighbourhoodPrices
    WHERE row_num IN ((total_count + 1) / 2, (total_count + 2) / 2)
      AND total_count >= 50
    GROUP BY neighbourhood, total_count
)
SELECT 
    DENSE_RANK() OVER (ORDER BY median_nightly_price DESC) AS price_rank,
    neighbourhood,
    active_listings,
    median_nightly_price
FROM NeighbourhoodMedians
ORDER BY price_rank ASC
LIMIT 10;


-- --------------------------------------------------------------------
-- QUERY 3: Superhost Premium by Neighbourhood
-- Business Question: Does achieving Superhost badge status allow hosts to charge higher rates in top neighbourhoods?
-- Uses CASE statements, conditional aggregation, and multi-table filtering.
-- --------------------------------------------------------------------
-- name: query_3_superhost_premium
WITH TopNeighbourhoods AS (
    SELECT neighbourhood
    FROM listings
    GROUP BY neighbourhood
    HAVING COUNT(*) >= 100
)
SELECT 
    l.neighbourhood,
    COUNT(CASE WHEN l.host_is_superhost = 1 THEN 1 END) AS superhost_count,
    COUNT(CASE WHEN l.host_is_superhost = 0 THEN 1 END) AS regular_host_count,
    ROUND(AVG(CASE WHEN l.host_is_superhost = 1 THEN l.price END), 2) AS avg_superhost_price,
    ROUND(AVG(CASE WHEN l.host_is_superhost = 0 THEN l.price END), 2) AS avg_regular_host_price,
    ROUND(
        AVG(CASE WHEN l.host_is_superhost = 1 THEN l.price END) - 
        AVG(CASE WHEN l.host_is_superhost = 0 THEN l.price END), 
        2
    ) AS absolute_premium_usd,
    ROUND(
        (AVG(CASE WHEN l.host_is_superhost = 1 THEN l.price END) - 
         AVG(CASE WHEN l.host_is_superhost = 0 THEN l.price END)) * 100.0 / 
         AVG(CASE WHEN l.host_is_superhost = 0 THEN l.price END), 
        2
    ) AS premium_pct
FROM listings l
INNER JOIN TopNeighbourhoods tn ON l.neighbourhood = tn.neighbourhood
GROUP BY l.neighbourhood
ORDER BY premium_pct DESC
LIMIT 10;


-- --------------------------------------------------------------------
-- QUERY 4: Commercial Operator Concentration (Host Portfolio Distribution)
-- Business Question: What percentage of Austin Airbnb inventory is operated by institutional multi-unit managers vs mom-and-pop hosts?
-- Demonstrates CTE, CASE portfolio categorization, and window function SUM() OVER ().
-- --------------------------------------------------------------------
-- name: query_4_host_concentration
WITH HostListingsCount AS (
    SELECT 
        host_id,
        host_name,
        COUNT(*) AS total_properties
    FROM listings
    GROUP BY host_id, host_name
),
CategorizedHosts AS (
    SELECT 
        host_id,
        total_properties,
        CASE 
            WHEN total_properties = 1 THEN '1. Single Listing (Mom & Pop)'
            WHEN total_properties BETWEEN 2 AND 4 THEN '2. Small Portfolio (2-4 units)'
            WHEN total_properties BETWEEN 5 AND 19 THEN '3. Medium Commercial (5-19 units)'
            ELSE '4. Mega Enterprise (20+ units)'
        END AS host_tier
    FROM HostListingsCount
)
SELECT 
    host_tier,
    COUNT(DISTINCT host_id) AS distinct_hosts,
    SUM(total_properties) AS total_listings_controlled,
    ROUND(SUM(total_properties) * 100.0 / (SELECT COUNT(*) FROM listings), 2) AS market_supply_share_pct
FROM CategorizedHosts
GROUP BY host_tier
ORDER BY host_tier ASC;


-- --------------------------------------------------------------------
-- QUERY 5: Price Tier vs. Forward Occupancy Elasticity
-- Business Question: Do more expensive listings suffer lower forward occupancy, or is demand price-inelastic?
-- Uses NTILE(4) to bucket listings into price quartiles and JOINs with calendar_occupancy.
-- --------------------------------------------------------------------
-- name: query_5_price_quartile_occupancy
WITH PriceQuartiles AS (
    SELECT 
        l.id,
        l.price,
        c.occupancy_rate_30,
        c.occupancy_rate_90,
        c.occupancy_rate_365,
        NTILE(4) OVER (ORDER BY l.price) AS price_quartile
    FROM listings l
    INNER JOIN calendar_occupancy c ON l.id = c.listing_id
)
SELECT 
    price_quartile,
    ROUND(MIN(price), 2) AS min_price_in_tier,
    ROUND(MAX(price), 2) AS max_price_in_tier,
    ROUND(AVG(price), 2) AS avg_price_in_tier,
    COUNT(*) AS listing_count,
    ROUND(AVG(occupancy_rate_30) * 100.0, 2) AS avg_occupancy_30d_pct,
    ROUND(AVG(occupancy_rate_90) * 100.0, 2) AS avg_occupancy_90d_pct,
    ROUND(AVG(occupancy_rate_365) * 100.0, 2) AS avg_occupancy_365d_pct
FROM PriceQuartiles
GROUP BY price_quartile
ORDER BY price_quartile ASC;


-- --------------------------------------------------------------------
-- QUERY 6: Top 10 Best-Value Listings (High Satisfaction + Accessible Pricing)
-- Business Question: Which properties provide exceptional guest satisfaction at competitive rates?
-- Uses DENSE_RANK() window function, subqueries, and multi-variable filtering.
-- --------------------------------------------------------------------
-- name: query_6_top_value_listings
WITH CityAverages AS (
    SELECT 
        AVG(price) AS city_avg_price,
        AVG(review_scores_rating) AS city_avg_rating
    FROM listings
),
RankedValue AS (
    SELECT 
        l.id,
        l.name,
        l.neighbourhood,
        l.room_type,
        l.price,
        l.review_scores_rating,
        l.number_of_reviews,
        ROUND(l.review_scores_rating / l.price * 100.0, 2) AS value_index,
        DENSE_RANK() OVER (ORDER BY (l.review_scores_rating / l.price) DESC) AS value_rank
    FROM listings l, CityAverages ca
    WHERE l.price < ca.city_avg_price
      AND l.review_scores_rating >= 4.90
      AND l.number_of_reviews >= 50
)
SELECT 
    value_rank,
    name,
    neighbourhood,
    room_type,
    price,
    review_scores_rating,
    number_of_reviews,
    value_index
FROM RankedValue
ORDER BY value_rank ASC
LIMIT 10;


-- --------------------------------------------------------------------
-- QUERY 7: Annual & Seasonal Review Volume Trends
-- Business Question: How has tourism volume in Austin evolved year-over-year, and which months reflect peak guest arrival?
-- Uses string functions, DATE formatting, and time-series aggregation on 600k+ reviews.
-- --------------------------------------------------------------------
-- name: query_7_review_seasonality
WITH MonthlyReviewTrends AS (
    SELECT 
        SUBSTR(date, 1, 4) AS review_year,
        SUBSTR(date, 6, 2) AS review_month,
        COUNT(*) AS total_reviews
    FROM reviews
    WHERE date >= '2021-01-01' AND date <= '2025-12-31'
    GROUP BY SUBSTR(date, 1, 4), SUBSTR(date, 6, 2)
)
SELECT 
    review_month,
    SUM(CASE WHEN review_year = '2022' THEN total_reviews ELSE 0 END) AS reviews_2022,
    SUM(CASE WHEN review_year = '2023' THEN total_reviews ELSE 0 END) AS reviews_2023,
    SUM(CASE WHEN review_year = '2024' THEN total_reviews ELSE 0 END) AS reviews_2024,
    SUM(CASE WHEN review_year = '2025' THEN total_reviews ELSE 0 END) AS reviews_2025,
    SUM(total_reviews) AS total_cumulative_reviews
FROM MonthlyReviewTrends
GROUP BY review_month
ORDER BY review_month ASC;


-- --------------------------------------------------------------------
-- QUERY 8: Luxury Amenity Lift on Realized Nightly Price
-- Business Question: What is the price premium associated with pool, hot tub, and dedicated workspace?
-- Uses CASE aggregations to isolate individual amenity contributions.
-- --------------------------------------------------------------------
-- name: query_8_amenity_pricing_lift
SELECT 
    'Pool' AS amenity_type,
    COUNT(CASE WHEN has_pool = 1 THEN 1 END) AS listings_with_amenity,
    COUNT(CASE WHEN has_pool = 0 THEN 1 END) AS listings_without_amenity,
    ROUND(AVG(CASE WHEN has_pool = 1 THEN price END), 2) AS avg_price_with,
    ROUND(AVG(CASE WHEN has_pool = 0 THEN price END), 2) AS avg_price_without,
    ROUND(AVG(CASE WHEN has_pool = 1 THEN price END) - AVG(CASE WHEN has_pool = 0 THEN price END), 2) AS dollar_lift,
    ROUND((AVG(CASE WHEN has_pool = 1 THEN price END) - AVG(CASE WHEN has_pool = 0 THEN price END)) * 100.0 / AVG(CASE WHEN has_pool = 0 THEN price END), 2) AS lift_pct
FROM listings
UNION ALL
SELECT 
    'Hot Tub' AS amenity_type,
    COUNT(CASE WHEN has_hot_tub = 1 THEN 1 END),
    COUNT(CASE WHEN has_hot_tub = 0 THEN 1 END),
    ROUND(AVG(CASE WHEN has_hot_tub = 1 THEN price END), 2),
    ROUND(AVG(CASE WHEN has_hot_tub = 0 THEN price END), 2),
    ROUND(AVG(CASE WHEN has_hot_tub = 1 THEN price END) - AVG(CASE WHEN has_hot_tub = 0 THEN price END), 2),
    ROUND((AVG(CASE WHEN has_hot_tub = 1 THEN price END) - AVG(CASE WHEN has_hot_tub = 0 THEN price END)) * 100.0 / AVG(CASE WHEN has_hot_tub = 0 THEN price END), 2)
FROM listings
UNION ALL
SELECT 
    'Dedicated Workspace' AS amenity_type,
    COUNT(CASE WHEN has_workspace = 1 THEN 1 END),
    COUNT(CASE WHEN has_workspace = 0 THEN 1 END),
    ROUND(AVG(CASE WHEN has_workspace = 1 THEN price END), 2),
    ROUND(AVG(CASE WHEN has_workspace = 0 THEN price END), 2),
    ROUND(AVG(CASE WHEN has_workspace = 1 THEN price END) - AVG(CASE WHEN has_workspace = 0 THEN price END), 2),
    ROUND((AVG(CASE WHEN has_workspace = 1 THEN price END) - AVG(CASE WHEN has_workspace = 0 THEN price END)) * 100.0 / AVG(CASE WHEN has_workspace = 0 THEN price END), 2)
FROM listings;


-- --------------------------------------------------------------------
-- QUERY 9: Estimated Annual Host Revenue by Neighbourhood
-- Business Question: What is the estimated annual gross rental yield across neighbourhoods?
-- Revenue Proxy Formula: Price * 365 * Annual Forward Occupancy Rate
-- Uses multi-table JOIN, CTE, and RANK() window function.
-- --------------------------------------------------------------------
-- name: query_9_estimated_annual_revenue
WITH ListingRevenues AS (
    SELECT 
        l.id,
        l.neighbourhood,
        l.price,
        c.occupancy_rate_365,
        ROUND(l.price * 365.0 * c.occupancy_rate_365, 2) AS estimated_annual_gross_revenue
    FROM listings l
    INNER JOIN calendar_occupancy c ON l.id = c.listing_id
    WHERE l.room_type = 'Entire home/apt'
)
SELECT 
    RANK() OVER (ORDER BY AVG(estimated_annual_gross_revenue) DESC) AS revenue_rank,
    neighbourhood,
    COUNT(*) AS entire_home_count,
    ROUND(AVG(price), 2) AS avg_nightly_price,
    ROUND(AVG(occupancy_rate_365) * 100.0, 2) AS avg_occupancy_pct,
    ROUND(AVG(estimated_annual_gross_revenue), 2) AS avg_estimated_annual_revenue_usd
FROM ListingRevenues
GROUP BY neighbourhood
HAVING COUNT(*) >= 50
ORDER BY avg_estimated_annual_revenue_usd DESC
LIMIT 10;


-- --------------------------------------------------------------------
-- QUERY 10: Repeat Guest & Loyalty Dynamics in Austin
-- Business Question: Are guests booking repeated stays with multiple Austin hosts, and what are their typical ratings?
-- Uses multi-table self-joins, aggregate counts, and subqueries.
-- --------------------------------------------------------------------
-- name: query_10_repeat_guests
SELECT 
    r.reviewer_id,
    r.reviewer_name,
    COUNT(r.id) AS total_reviews_written,
    COUNT(DISTINCT r.listing_id) AS distinct_properties_visited,
    MIN(r.date) AS earliest_review_date,
    MAX(r.date) AS latest_review_date
FROM reviews r
GROUP BY r.reviewer_id, r.reviewer_name
HAVING COUNT(DISTINCT r.listing_id) >= 8
ORDER BY distinct_properties_visited DESC
LIMIT 10;


-- --------------------------------------------------------------------
-- QUERY 11: Neighbourhood Price Dispersion (Interquartile Spread via NTILE)
-- Business Question: In which neighbourhoods is pricing most volatile/dispersed?
-- Uses NTILE(4) window function to calculate Q1, Q3, and IQR per neighbourhood.
-- --------------------------------------------------------------------
-- name: query_11_neighbourhood_iqr
WITH QuartiledListingPrices AS (
    SELECT 
        neighbourhood,
        price,
        NTILE(4) OVER (PARTITION BY neighbourhood ORDER BY price) AS quartile
    FROM listings
    WHERE neighbourhood IN (
        SELECT neighbourhood FROM listings GROUP BY neighbourhood HAVING COUNT(*) >= 150
    )
),
QuartileAggregates AS (
    SELECT 
        neighbourhood,
        ROUND(AVG(CASE WHEN quartile = 1 THEN price END), 2) AS q1_price,
        ROUND(AVG(CASE WHEN quartile = 4 THEN price END), 2) AS q4_price,
        COUNT(*) AS sample_size
    FROM QuartiledListingPrices
    GROUP BY neighbourhood
)
SELECT 
    neighbourhood,
    sample_size,
    q1_price,
    q4_price,
    ROUND(q4_price - q1_price, 2) AS price_spread_q4_minus_q1
FROM QuartileAggregates
ORDER BY price_spread_q4_minus_q1 DESC
LIMIT 10;
