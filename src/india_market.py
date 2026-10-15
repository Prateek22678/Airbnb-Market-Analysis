"""
src/india_market.py
Market Intelligence, Geospatial Benchmarks, and Investment Analytics
for Premier Indian Short-Term Rental & Vacation Hubs.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Premier Indian Short-Term Rental Hubs Dataset
# ---------------------------------------------------------------------
INDIAN_HUBS_DATA: List[Dict[str, Any]] = [
    {
        'hub': 'North Goa (Anjuna / Vagator / Candolim)',
        'city': 'Goa',
        'state': 'Goa',
        'region': 'Western Coast',
        'category': 'Coastal & Beaches',
        'latitude': 15.5847,
        'longitude': 73.7423,
        'avg_price_inr': 11500,
        'median_price_inr': 9200,
        'avg_price_usd': 138,
        'occupancy_rate': 0.82,
        'peak_season': 'Nov - Feb (Sunburn, Christmas & New Year)',
        'top_amenity': 'Private Swimming Pool & Beach Proximity',
        'property_focus': 'Luxury Beach Villas & Party Estates',
        'roi_tier': 'Tier 1 (High Yield)',
        'annual_rev_lakhs': 34.5,
        'avg_bedrooms': 3.2,
        'description': 'Premier coastal party and leisure corridor with the highest short-term villa rental yields in India.'
    },
    {
        'hub': 'South Goa (Palolem / Benaulim / Colva)',
        'city': 'Goa',
        'state': 'Goa',
        'region': 'Western Coast',
        'category': 'Coastal & Beaches',
        'latitude': 15.2285,
        'longitude': 73.9555,
        'avg_price_inr': 7800,
        'median_price_inr': 6500,
        'avg_price_usd': 94,
        'occupancy_rate': 0.74,
        'peak_season': 'Oct - Mar (Serene Beach Escape)',
        'top_amenity': 'Sea View & Yoga Deck',
        'property_focus': 'Boutique Homestays & Heritage Cottages',
        'roi_tier': 'Tier 1 (High Yield)',
        'annual_rev_lakhs': 21.0,
        'avg_bedrooms': 2.4,
        'description': 'Peaceful, family and digital nomad retreats with sustained long-stay demand and loyal repeat guests.'
    },
    {
        'hub': 'Udaipur (Lake Pichola / Fateh Sagar)',
        'city': 'Udaipur',
        'state': 'Rajasthan',
        'region': 'North India',
        'category': 'Royal & Heritage',
        'latitude': 24.5854,
        'longitude': 73.7125,
        'avg_price_inr': 9800,
        'median_price_inr': 7800,
        'avg_price_usd': 118,
        'occupancy_rate': 0.76,
        'peak_season': 'Sep - Mar (Luxury Romance & Royal Weddings)',
        'top_amenity': 'Lake View Terrace & Mewari Architecture',
        'property_focus': 'Lakeside Havelis & Boutique Palaces',
        'roi_tier': 'Tier 1 (High Yield)',
        'annual_rev_lakhs': 27.2,
        'avg_bedrooms': 2.8,
        'description': 'Venice of the East commanding top luxury premiums for lake-facing heritage properties and wedding travelers.'
    },
    {
        'hub': 'Jaipur (Pink City / C-Scheme / Civil Lines)',
        'city': 'Jaipur',
        'state': 'Rajasthan',
        'region': 'North India',
        'category': 'Royal & Heritage',
        'latitude': 26.9124,
        'longitude': 75.7873,
        'avg_price_inr': 6800,
        'median_price_inr': 5400,
        'avg_price_usd': 82,
        'occupancy_rate': 0.71,
        'peak_season': 'Oct - Mar (Winter Tourism & Literature Festival)',
        'top_amenity': 'Heritage Courtyard & Rooftop Dining',
        'property_focus': 'Restored Havelis & Art Suites',
        'roi_tier': 'Tier 2 (Moderate-High Yield)',
        'annual_rev_lakhs': 17.6,
        'avg_bedrooms': 2.6,
        'description': 'Cultural epicenter driving international inbound tourism, corporate retreats, and wedding groups.'
    },
    {
        'hub': 'Manali (Old Manali / Solang / Naggar)',
        'city': 'Manali',
        'state': 'Himachal Pradesh',
        'region': 'Himalayas',
        'category': 'Himalayan & Mountains',
        'latitude': 32.2432,
        'longitude': 77.1892,
        'avg_price_inr': 6400,
        'median_price_inr': 4900,
        'avg_price_usd': 77,
        'occupancy_rate': 0.78,
        'peak_season': 'Dec - Jan (Snow) & May - Jun (Summer Escape)',
        'top_amenity': 'Stone Fireplace & Himalayan Panoramic View',
        'property_focus': 'Cedar Wood Chalets & Apple Orchard Stays',
        'roi_tier': 'Tier 1 (High Yield)',
        'annual_rev_lakhs': 18.2,
        'avg_bedrooms': 2.5,
        'description': 'Year-round Himalayan hub with two distinct high-surge demand seasons catering to adventure and leisure seekers.'
    },
    {
        'hub': 'Shimla & Kasauli (Pine Forests)',
        'city': 'Shimla',
        'state': 'Himachal Pradesh',
        'region': 'Himalayas',
        'category': 'Himalayan & Mountains',
        'latitude': 31.1048,
        'longitude': 77.1734,
        'avg_price_inr': 5900,
        'median_price_inr': 4600,
        'avg_price_usd': 71,
        'occupancy_rate': 0.69,
        'peak_season': 'Apr - Jun & Dec - Jan (Colonial Summer Hill)',
        'top_amenity': 'Pine Forest Balcony & High-Speed Wi-Fi',
        'property_focus': 'British Era Colonial Bungalows',
        'roi_tier': 'Tier 2 (Moderate Yield)',
        'annual_rev_lakhs': 14.8,
        'avg_bedrooms': 2.3,
        'description': 'Accessible mountain weekend getaway hub for Delhi NCR travelers seeking quick summer and winter respite.'
    },
    {
        'hub': 'Mumbai (Bandra West / Juhu / Colaba)',
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'region': 'Western Coast',
        'category': 'Urban & Business Tech',
        'latitude': 19.0596,
        'longitude': 72.8295,
        'avg_price_inr': 10200,
        'median_price_inr': 8500,
        'avg_price_usd': 123,
        'occupancy_rate': 0.86,
        'peak_season': 'Year-Round (Corporate, Media & Bollywood)',
        'top_amenity': 'Sea-Facing Balcony & Fast Wi-Fi (300 Mbps)',
        'property_focus': 'Art Deco High-Rises & Designer Penthouses',
        'roi_tier': 'Tier 1 (High Yield / Capital Intensive)',
        'annual_rev_lakhs': 32.0,
        'avg_bedrooms': 1.8,
        'description': 'Highest baseline occupancy in India driven by media, entertainment, corporate travel, and medical tourism.'
    },
    {
        'hub': 'Bengaluru (Indiranagar / Koramangala / Whitefield)',
        'city': 'Bengaluru',
        'state': 'Karnataka',
        'region': 'South India',
        'category': 'Urban & Business Tech',
        'latitude': 12.9716,
        'longitude': 77.5946,
        'avg_price_inr': 5400,
        'median_price_inr': 4300,
        'avg_price_usd': 65,
        'occupancy_rate': 0.84,
        'peak_season': 'Year-Round (Tech Summits & Founder Stays)',
        'top_amenity': 'Ergonomic Workstation, Power Inverter & Fiber Internet',
        'property_focus': 'Modern Studio Lofts & Garden Terrace Units',
        'roi_tier': 'Tier 2 (High Steady Occupancy)',
        'annual_rev_lakhs': 16.5,
        'avg_bedrooms': 1.6,
        'description': 'Silicon Valley of India with massive steady demand for 14-30 day workation and corporate extended stays.'
    },
    {
        'hub': 'Delhi NCR (Hauz Khas / Saket / DLF Cyber City)',
        'city': 'Delhi',
        'state': 'Delhi NCR',
        'region': 'North India',
        'category': 'Urban & Business Tech',
        'latitude': 28.5494,
        'longitude': 77.2001,
        'avg_price_inr': 6200,
        'median_price_inr': 4900,
        'avg_price_usd': 75,
        'occupancy_rate': 0.77,
        'peak_season': 'Oct - Mar (Diplomatic Events & Heritage Trips)',
        'top_amenity': 'Air Purifier, Central Heating & Metro Access',
        'property_focus': 'Boho Artist Studios & Historic Monument Views',
        'roi_tier': 'Tier 2 (Moderate-High Yield)',
        'annual_rev_lakhs': 17.4,
        'avg_bedrooms': 2.0,
        'description': 'National capital region driving diplomatic visitors, heritage travelers, and corporate consultants.'
    },
    {
        'hub': 'Kerala (Alleppey Backwaters / Munnar / Fort Kochi)',
        'city': 'Kochi / Alleppey',
        'state': 'Kerala',
        'region': 'South India',
        'category': 'Backwaters & Wellness',
        'latitude': 9.5982,
        'longitude': 76.5222,
        'avg_price_inr': 7600,
        'median_price_inr': 6000,
        'avg_price_usd': 92,
        'occupancy_rate': 0.76,
        'peak_season': 'Sep - Mar (Monsoon Wellness & Winter Backwaters)',
        'top_amenity': 'Private Houseboat Deck & Ayurvedic Spa',
        'property_focus': 'Eco Heritage Villas & Tea Estate Bungalows',
        'roi_tier': 'Tier 1 (High Yield)',
        'annual_rev_lakhs': 21.0,
        'avg_bedrooms': 2.5,
        'description': 'God’s Own Country drawing sustained international ecotourism, honeymooners, and Ayurvedic wellness travelers.'
    },
    {
        'hub': 'Pondicherry (White Town / French Quarter)',
        'city': 'Puducherry',
        'state': 'Puducherry',
        'region': 'South India',
        'category': 'Coastal & Beaches',
        'latitude': 11.9338,
        'longitude': 79.8297,
        'avg_price_inr': 6600,
        'median_price_inr': 5200,
        'avg_price_usd': 80,
        'occupancy_rate': 0.74,
        'peak_season': 'Nov - Feb (Colonial Architecture & Auroville)',
        'top_amenity': 'Franco-Tamil Yellow Architecture & Bicycle Included',
        'property_focus': 'Colonial Villas & Pastel Heritage Suites',
        'roi_tier': 'Tier 2 (Moderate Yield)',
        'annual_rev_lakhs': 17.8,
        'avg_bedrooms': 2.2,
        'description': 'Distinct Franco-Tamil coastal quarter commanding strong weekend leisure premiums from Chennai & Bengaluru.'
    },
    {
        'hub': 'Rishikesh (Ganga Ghats / Tapovan)',
        'city': 'Rishikesh',
        'state': 'Uttarakhand',
        'region': 'Himalayas',
        'category': 'Spiritual & Wellness',
        'latitude': 30.0869,
        'longitude': 78.2676,
        'avg_price_inr': 4900,
        'median_price_inr': 3900,
        'avg_price_usd': 59,
        'occupancy_rate': 0.77,
        'peak_season': 'Sep - Nov & Mar - May (Yoga, River Rafting & Meditation)',
        'top_amenity': 'River-Facing Yoga Terrace & Organic Kitchen',
        'property_focus': 'Spiritual Retreat Homestays & River Lofts',
        'roi_tier': 'Tier 2 (Moderate-High Yield)',
        'annual_rev_lakhs': 13.7,
        'avg_bedrooms': 1.9,
        'description': 'Yoga Capital of the World drawing international spiritual long-stayers and weekend river-rafting youth.'
    }
]


def get_indian_hubs_df() -> pd.DataFrame:
    """Returns the curated benchmark dataset of India's premier short-term rental hubs."""
    return pd.DataFrame(INDIAN_HUBS_DATA)


def generate_indian_synthetic_listings(n_per_hub: int = 15, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic individual listing points clustered around
    the 12 premier Indian hubs to support fine-grained Pydeck 3D point rendering.
    """
    np.random.seed(random_state)
    hubs_df = get_indian_hubs_df()
    rows = []

    listing_id = 10001
    room_types = ['Entire villa', 'Entire serviced apartment', 'Private room in heritage homestay', 'Entire cottage']
    
    for _, hub_row in hubs_df.iterrows():
        base_lat = hub_row['latitude']
        base_lon = hub_row['longitude']
        base_price = hub_row['avg_price_inr']

        for i in range(n_per_hub):
            # Jitter coordinates within ~3-8 km of hub center
            jitter_lat = base_lat + np.random.normal(0, 0.035)
            jitter_lon = base_lon + np.random.normal(0, 0.035)
            
            price_multiplier = np.random.uniform(0.65, 1.45)
            price_inr = int(round(base_price * price_multiplier, -2))
            price_usd = int(round(price_inr / 83.0))

            rating = round(np.random.beta(15, 1.5) * 1.0 + 4.0, 2)
            rating = min(5.0, max(4.1, rating))

            accommodates = int(np.random.choice([2, 4, 6, 8, 12], p=[0.25, 0.35, 0.20, 0.15, 0.05]))
            is_superhost = int(np.random.choice([0, 1], p=[0.45, 0.55]))
            has_pool = int(hub_row['category'] == 'Coastal & Beaches' and np.random.rand() > 0.35)

            name_templates = [
                f"{hub_row['city']} Sunset {hub_row['property_focus'].split()[0]}",
                f"Heritage {hub_row['city']} Retreat with {hub_row['top_amenity'].split('&')[0].strip()}",
                f"Luxury {hub_row['category']} Villa ({accommodates} Guests)",
                f"Boutique {hub_row['city']} Suite near Center"
            ]

            rows.append({
                'id': listing_id,
                'name': name_templates[i % len(name_templates)],
                'hub': hub_row['hub'],
                'city': hub_row['city'],
                'state': hub_row['state'],
                'region': hub_row['region'],
                'category': hub_row['category'],
                'latitude': jitter_lat,
                'longitude': jitter_lon,
                'price_inr': price_inr,
                'price_usd': price_usd,
                'room_type': np.random.choice(room_types),
                'accommodates': accommodates,
                'host_is_superhost': is_superhost,
                'review_scores_rating': rating,
                'has_pool': has_pool,
                'top_amenity': hub_row['top_amenity'],
                'peak_season': hub_row['peak_season']
            })
            listing_id += 1

    return pd.DataFrame(rows)


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
