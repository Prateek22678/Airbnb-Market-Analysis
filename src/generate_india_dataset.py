"""
src/generate_india_dataset.py
Generate a comprehensive, realistic, production-quality dataset of Indian short-term rentals
spanning 25+ cities and vacation destinations across 8 key Indian states.
Outputs: data/processed/india_listings_cleaned.csv (~2,000 detailed listings)
"""

import os
import numpy as np
import pandas as pd

# Authentic 6-Digit Indian Postal PIN Codes (Postal Index Numbers / ZIP codes)
NEIGHBORHOOD_PINCODES = {
    # Goa (403xxx)
    'Assagao': '403507',
    'Anjuna': '403509',
    'Vagator': '403509',
    'Candolim': '403515',
    'Siolim': '403517',
    'Palolem': '403702',
    'Benaulim': '403716',

    # Mumbai (400xxx)
    'Bandra West': '400050',
    'Juhu': '400049',
    'Colaba': '400005',
    'Khar West': '400052',
    'Powai': '400076',

    # Alibaug (402xxx)
    'Mandwa': '402201',
    'Awas': '402201',
    'Kihim': '402201',

    # Lonavala & Khandala (410xxx)
    'Tungarli': '410401',
    'Gold Valley': '410401',
    'Khandala Ghat': '410301',

    # Jaipur (302xxx)
    'Pink City / Hawa Mahal': '302002',
    'C-Scheme': '302001',
    'Civil Lines': '302006',
    'Bani Park': '302016',

    # Udaipur (313xxx)
    'Lake Pichola': '313001',
    'Fateh Sagar': '313001',
    'Ambavgarh': '313001',

    # Manali (175xxx)
    'Old Manali': '175131',
    'Solang Valley': '175131',
    'Naggar': '175130',
    'Vashisht': '175131',

    # Shimla & Mashobra (171xxx)
    'The Mall Road': '171001',
    'Mashobra': '171007',
    'Kufri': '171012',

    # Dharamshala & McLeodGanj (176xxx)
    'McLeodGanj': '176219',
    'Dharamkot': '176219',
    'Bhagsunag': '176219',

    # Bengaluru (560xxx)
    'Indiranagar': '560038',
    'Koramangala': '560034',
    'Whitefield': '560066',
    'HSR Layout': '560102',

    # Coorg (571xxx)
    'Madikeri': '571201',
    'Kushalnagar': '571234',
    'Virajpet': '571218',

    # Kerala (68xxxx - 69xxxx)
    'Alleppey Backwaters': '688013',
    'Kumarakom': '686563',
    'Fort Kochi': '682001',
    'Munnar': '685612',
    'Varkala': '695141',

    # Delhi NCR (110xxx / 122xxx)
    'Hauz Khas Village': '110016',
    'Greater Kailash': '110048',
    'DLF Cyber City (Gurgaon)': '122002',
    'Saket': '110017',

    # Rishikesh (249xxx)
    'Tapovan': '249192',
    'Laxman Jhula': '249302',
    'Shivpuri': '249192',

    # Mussoorie & Landour (248xxx)
    'Landour / Char Dukan': '248179',
    'The Mall Mussoorie': '248179',

    # Pondicherry (605xxx)
    'White Town (French Quarter)': '605001',
    'Auroville': '605101',
    'Promenade Beach': '605001',

    # Varanasi (221xxx)
    'Assi Ghat': '221005',
    'Dashashwamedh': '221001',
    'Cantonment': '221002',
}

# Define Master Indian Micro-Markets & Coordinates
INDIAN_CITIES_SPECS = [
    # 1. Goa (Coastal & Party Hub)
    {
        'city': 'Goa',
        'state': 'Goa',
        'region': 'Western Coast',
        'neighborhoods': [
            ('Assagao', 15.5900, 73.7750, 14500, 'Luxury Designer Villas', 'North Goa'),
            ('Anjuna', 15.5780, 73.7420, 11500, 'Boho Beach Cottages', 'North Goa'),
            ('Vagator', 15.5990, 73.7440, 12800, 'Cliff-top Sunset Villas', 'North Goa'),
            ('Candolim', 15.5180, 73.7660, 9500, 'Family Beach Apartments', 'North Goa'),
            ('Siolim', 15.6200, 73.7680, 10800, 'Riverfront Portuguese Homes', 'North Goa'),
            ('Palolem', 15.0100, 74.0230, 7500, 'Serene Beach Huts & Studios', 'South Goa'),
            ('Benaulim', 15.2600, 73.9280, 8200, 'Peaceful Coastal Homestays', 'South Goa'),
        ],
        'count': 350
    },
    # 2. Mumbai & Maharashtra Getaways
    {
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'region': 'Western Coast',
        'neighborhoods': [
            ('Bandra West', 19.0596, 72.8295, 11200, 'Celebrity & Artist Studios', 'Urban Metro'),
            ('Juhu', 19.1075, 72.8263, 12500, 'Sea-Facing Luxury Suites', 'Urban Metro'),
            ('Colaba', 18.9067, 72.8147, 9800, 'Heritage South Mumbai Flats', 'Urban Metro'),
            ('Khar West', 19.0700, 72.8360, 8900, 'Modern Designer Lofts', 'Urban Metro'),
            ('Powai', 19.1176, 72.9060, 7200, 'Lake View Tech Residences', 'Urban Metro'),
        ],
        'count': 220
    },
    {
        'city': 'Alibaug',
        'state': 'Maharashtra',
        'region': 'Western Coast',
        'neighborhoods': [
            ('Mandwa', 18.7850, 72.8750, 18500, 'Ferry-Accessible Pool Villas', 'Coastal Weekend'),
            ('Awas', 18.7720, 72.8680, 16000, 'Private Coconut Grove Estates', 'Coastal Weekend'),
            ('Kihim', 18.7300, 72.8800, 14000, 'Beachside Luxury Bungalows', 'Coastal Weekend'),
        ],
        'count': 110
    },
    {
        'city': 'Lonavala & Khandala',
        'state': 'Maharashtra',
        'region': 'Western Coast',
        'neighborhoods': [
            ('Tungarli', 18.7610, 73.4190, 13500, 'Hill View Private Pool Villas', 'Monsoon Hill'),
            ('Gold Valley', 18.7520, 73.4310, 11800, 'Gated Community Mountain Stays', 'Monsoon Hill'),
            ('Khandala Ghat', 18.7590, 73.3760, 14200, 'Valley View Mist Chalets', 'Monsoon Hill'),
        ],
        'count': 120
    },
    # 3. Rajasthan (Royal Heritage)
    {
        'city': 'Jaipur',
        'state': 'Rajasthan',
        'region': 'North India',
        'neighborhoods': [
            ('Pink City / Hawa Mahal', 26.9239, 75.8267, 6200, 'Historic Bazaar Havelis', 'Royal Heritage'),
            ('C-Scheme', 26.9110, 75.8030, 7400, 'Upscale Urban Art Suites', 'Royal Heritage'),
            ('Civil Lines', 26.9020, 75.7820, 8200, 'Diplomatic Heritage Mansions', 'Royal Heritage'),
            ('Bani Park', 26.9310, 75.7950, 5800, 'Traditional Marwari Homestays', 'Royal Heritage'),
        ],
        'count': 180
    },
    {
        'city': 'Udaipur',
        'state': 'Rajasthan',
        'region': 'North India',
        'neighborhoods': [
            ('Lake Pichola', 24.5760, 73.6800, 12500, 'Lakeside Palace Facing Havelis', 'Royal Heritage'),
            ('Fateh Sagar', 24.6020, 73.6740, 9800, 'Waterfront Luxury Suites', 'Royal Heritage'),
            ('Ambavgarh', 24.5950, 73.6820, 8500, 'Panoramic Hill View Residences', 'Royal Heritage'),
        ],
        'count': 140
    },
    # 4. Himachal Pradesh (Himalayan Escapes)
    {
        'city': 'Manali',
        'state': 'Himachal Pradesh',
        'region': 'Himalayas',
        'neighborhoods': [
            ('Old Manali', 32.2540, 77.1780, 5800, 'Cedar Wood Backpacker Chalets', 'Himalayan Escape'),
            ('Solang Valley', 32.3160, 77.1570, 7900, 'Snow Mountain Adventure Cabins', 'Himalayan Escape'),
            ('Naggar', 32.1400, 77.1680, 6800, 'Apple Orchard Castle Retreats', 'Himalayan Escape'),
            ('Vashisht', 32.2610, 77.1890, 5400, 'Hot Spring View Cottages', 'Himalayan Escape'),
        ],
        'count': 160
    },
    {
        'city': 'Shimla & Mashobra',
        'state': 'Himachal Pradesh',
        'region': 'Himalayas',
        'neighborhoods': [
            ('The Mall Road', 31.1048, 77.1734, 6200, 'British Colonial Suites', 'Himalayan Escape'),
            ('Mashobra', 31.1300, 77.2340, 8400, 'Pine Forest Glass Cottages', 'Himalayan Escape'),
            ('Kufri', 31.0980, 77.2680, 7100, 'Snow Peak Ridge Lodges', 'Himalayan Escape'),
        ],
        'count': 120
    },
    {
        'city': 'Dharamshala & McLeodGanj',
        'state': 'Himachal Pradesh',
        'region': 'Himalayas',
        'neighborhoods': [
            ('McLeodGanj', 32.2426, 76.3213, 5200, 'Tibetan Monastery View Flats', 'Spiritual Mountain'),
            ('Dharamkot', 32.2510, 76.3260, 4800, 'Yoga Village Pine Cottages', 'Spiritual Mountain'),
            ('Bhagsunag', 32.2460, 76.3360, 4500, 'Waterfall Facing Homestays', 'Spiritual Mountain'),
        ],
        'count': 100
    },
    # 5. Karnataka (Tech & Coffee Escapes)
    {
        'city': 'Bengaluru',
        'state': 'Karnataka',
        'region': 'South India',
        'neighborhoods': [
            ('Indiranagar', 12.9784, 77.6408, 6200, 'Tech Founder Penthouses', 'Urban Tech'),
            ('Koramangala', 12.9352, 77.6245, 5800, 'Startup Workation Lofts', 'Urban Tech'),
            ('Whitefield', 12.9698, 77.7500, 4900, 'Corporate Long-Stay Suites', 'Urban Tech'),
            ('HSR Layout', 12.9121, 77.6446, 5200, 'Modern Designer Studios', 'Urban Tech'),
        ],
        'count': 170
    },
    {
        'city': 'Coorg (Kodagu)',
        'state': 'Karnataka',
        'region': 'South India',
        'neighborhoods': [
            ('Madikeri', 12.4244, 75.7382, 7400, 'Coffee Plantation Bungalows', 'Plantation Estate'),
            ('Kushalnagar', 12.4550, 75.9600, 6500, 'Kaveri River Homestays', 'Plantation Estate'),
            ('Virajpet', 12.2030, 75.8050, 8100, 'Historic Planter Mansions', 'Plantation Estate'),
        ],
        'count': 90
    },
    # 6. Kerala (God's Own Country)
    {
        'city': 'Kerala (Alleppey & Kochi)',
        'state': 'Kerala',
        'region': 'South India',
        'neighborhoods': [
            ('Alleppey Backwaters', 9.4981, 76.3388, 12500, 'Luxury Private Houseboats', 'Backwaters & Wellness'),
            ('Kumarakom', 9.6175, 76.4300, 11000, 'Vembanad Lake Eco Villas', 'Backwaters & Wellness'),
            ('Fort Kochi', 9.9658, 76.2421, 6800, 'Dutch & Portuguese Heritage Homes', 'Coastal Heritage'),
            ('Munnar', 10.0889, 77.0595, 7200, 'Misty Tea Plantation Lodges', 'Hill Plantation'),
            ('Varkala', 8.7379, 76.7163, 5600, 'North Cliff Ocean View Studios', 'Coastal Beach'),
        ],
        'count': 190
    },
    # 7. Delhi NCR (Capital Region)
    {
        'city': 'Delhi NCR',
        'state': 'Delhi NCR',
        'region': 'North India',
        'neighborhoods': [
            ('Hauz Khas Village', 28.5535, 77.1945, 6800, 'Bohemian Lake View Lofts', 'Urban Cultural'),
            ('Greater Kailash', 28.5380, 77.2350, 6200, 'Diplomatic Enclave Residences', 'Urban Metro'),
            ('DLF Cyber City (Gurgaon)', 28.4950, 77.0890, 7500, 'Executive High-Rise Suites', 'Corporate Tech'),
            ('Saket', 28.5240, 77.2160, 5400, 'Boutique Designer Apartments', 'Urban Metro'),
        ],
        'count': 150
    },
    # 8. Uttarakhand (Spiritual & Mountains)
    {
        'city': 'Rishikesh',
        'state': 'Uttarakhand',
        'region': 'Himalayas',
        'neighborhoods': [
            ('Tapovan', 30.1360, 78.3240, 5200, 'Yoga Center Mountain Flats', 'Spiritual & Wellness'),
            ('Laxman Jhula', 30.1260, 78.3300, 4800, 'River Ganga Facing Balconies', 'Spiritual & Wellness'),
            ('Shivpuri', 30.1480, 78.3880, 6500, 'White Water Rafting Retreats', 'Spiritual & Wellness'),
        ],
        'count': 110
    },
    {
        'city': 'Mussoorie & Landour',
        'state': 'Uttarakhand',
        'region': 'Himalayas',
        'neighborhoods': [
            ('Landour / Char Dukan', 30.4620, 78.0930, 8900, 'British Heritage Pine Chalets', 'Himalayan Escape'),
            ('The Mall Mussoorie', 30.4590, 78.0720, 6400, 'Valley Sunset Panorama Suites', 'Himalayan Escape'),
        ],
        'count': 80
    },
    # 9. Puducherry (French Quarter)
    {
        'city': 'Pondicherry',
        'state': 'Puducherry',
        'region': 'South India',
        'neighborhoods': [
            ('White Town (French Quarter)', 11.9338, 79.8297, 7200, 'Franco-Tamil Yellow Villas', 'Coastal Heritage'),
            ('Auroville', 12.0070, 79.8106, 5600, 'Spiritual Eco-Community Stays', 'Spiritual Eco'),
            ('Promenade Beach', 11.9280, 79.8340, 6500, 'Ocean Breeze Heritage Suites', 'Coastal Heritage'),
        ],
        'count': 100
    },
    # 10. Varanasi & Uttar Pradesh
    {
        'city': 'Varanasi',
        'state': 'Uttar Pradesh',
        'region': 'North India',
        'neighborhoods': [
            ('Assi Ghat', 25.2890, 83.0060, 4800, 'Ganga Aarti View Havelis', 'Spiritual Heritage'),
            ('Dashashwamedh', 25.3080, 83.0110, 4200, 'Ancient Old City Homestays', 'Spiritual Heritage'),
            ('Cantonment', 25.3340, 82.9780, 5500, 'Spacious Garden Bungalows', 'Spiritual Heritage'),
        ],
        'count': 90
    }
]


def generate_dataset(output_path: str = "data/processed/india_listings_cleaned.csv") -> pd.DataFrame:
    """Generate and write the comprehensive Indian listings dataset."""
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    current_id = 500001

    room_types_dist = [
        ('Entire villa', 0.32),
        ('Entire serviced apartment', 0.30),
        ('Private room in heritage homestay', 0.22),
        ('Entire mountain cottage', 0.12),
        ('Luxury houseboat', 0.04)
    ]
    r_types = [t[0] for t in room_types_dist]
    r_probs = [t[1] for t in room_types_dist]

    for city_spec in INDIAN_CITIES_SPECS:
        city_name = city_spec['city']
        state_name = city_spec['state']
        region_name = city_spec['region']
        target_count = city_spec['count']
        neighborhoods = city_spec['neighborhoods']

        for i in range(target_count):
            neigh_tuple = neighborhoods[i % len(neighborhoods)]
            neigh_name, base_lat, base_lon, base_adr, property_archetype, sub_zone = neigh_tuple

            # Realistic jitter around neighborhood center
            lat = round(base_lat + np.random.normal(0, 0.015), 5)
            lon = round(base_lon + np.random.normal(0, 0.015), 5)

            # Price modeling with log-normal market variation
            price_variance = np.random.lognormal(0, 0.28)
            price_inr = int(round(base_adr * price_variance, -2))
            price_inr = max(1800, min(55000, price_inr))
            price_usd = int(round(price_inr / 83.0))

            room_type = np.random.choice(r_types, p=r_probs)
            if 'Houseboat' in property_archetype:
                room_type = 'Luxury houseboat'
            elif 'Villa' in property_archetype and np.random.rand() > 0.3:
                room_type = 'Entire villa'
            elif 'Apartment' in property_archetype or 'Loft' in property_archetype:
                room_type = 'Entire serviced apartment'

            # Capacity features
            if room_type == 'Entire villa':
                bedrooms = int(np.random.choice([2, 3, 4, 5, 6], p=[0.15, 0.40, 0.25, 0.15, 0.05]))
                accommodates = bedrooms * 2 + int(np.random.choice([0, 2, 4]))
                bathrooms = float(bedrooms if np.random.rand() > 0.3 else bedrooms + 1)
                beds = accommodates - 1
            elif room_type == 'Entire serviced apartment':
                bedrooms = int(np.random.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
                accommodates = bedrooms * 2
                bathrooms = float(bedrooms)
                beds = bedrooms
            else:
                bedrooms = int(np.random.choice([1, 2], p=[0.75, 0.25]))
                accommodates = bedrooms * 2
                bathrooms = 1.0
                beds = bedrooms

            # Host flags
            is_superhost = int(np.random.choice([0, 1], p=[0.48, 0.52]))
            is_verified = int(np.random.choice([1, 0], p=[0.88, 0.12]))

            # Indian Amenity Multipliers
            has_pool = int(
                ('Villa' in room_type or 'Goa' in city_name or 'Alibaug' in city_name or 'Udaipur' in city_name) 
                and np.random.rand() > 0.40
            )
            has_inverter = int(np.random.choice([1, 0], p=[0.82, 0.18])) # Critical in India
            has_ac = int(np.random.choice([1, 0], p=[0.85, 0.15])) if 'Himalayas' not in region_name else int(np.random.rand() > 0.70)
            has_wifi = int(np.random.choice([1, 0], p=[0.94, 0.06]))
            has_parking = int(np.random.choice([1, 0], p=[0.78, 0.22]))
            has_kitchen = int(np.random.choice([1, 0], p=[0.72, 0.28]))
            has_geyser = int(np.random.choice([1, 0], p=[0.92, 0.08]))
            has_cook = int(has_pool == 1 or ('Villa' in room_type and np.random.rand() > 0.45))
            has_scenic = int('Lake' in neigh_name or 'Beach' in property_archetype or 'Mountain' in property_archetype or 'Sea' in property_archetype)

            # Reviews & Ratings
            num_reviews = int(np.random.exponential(35))
            if num_reviews > 0:
                rating = round(min(5.0, max(4.15, np.random.beta(20, 2) * 1.0 + 4.0)), 2)
                sentiment = round(float(np.clip(0.40 + (rating - 4.0) * 0.55 + np.random.normal(0, 0.08), -0.5, 0.98)), 3)
            else:
                rating = np.nan
                sentiment = np.nan

            # Occupancy & Revenues
            occ_base = 0.72 if is_superhost else 0.61
            if 'Goa' in city_name or 'Manali' in city_name or 'Udaipur' in city_name:
                occ_base += 0.08
            occ_30 = round(float(np.clip(occ_base + np.random.normal(0, 0.08), 0.35, 0.96)), 2)
            occ_365 = round(float(np.clip(occ_base - 0.05 + np.random.normal(0, 0.06), 0.38, 0.88)), 2)
            
            annual_gross_lakhs = round((365 * occ_365 * price_inr) / 100000.0, 2)

            # Realistic Name Generation
            prefix = np.random.choice(['Serene', 'Luxury', 'Heritage', 'Cozy', 'Boutique', 'Scenic', 'Boho', 'Grand'])
            name = f"{prefix} {neigh_name} {property_archetype.split()[0]} with {'Pool' if has_pool else ('View' if has_scenic else 'Fast Wi-Fi')}"

            pincode = NEIGHBORHOOD_PINCODES.get(neigh_name, '110001')

            records.append({
                'id': current_id,
                'name': name,
                'city': city_name,
                'state': state_name,
                'region': region_name,
                'neighborhood': neigh_name,
                'sub_zone': sub_zone,
                'property_archetype': property_archetype,
                'zipcode': pincode,
                'pincode': pincode,
                'postal_code': pincode,
                'latitude': lat,
                'longitude': lon,
                'room_type': room_type,
                'accommodates': accommodates,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'beds': beds,
                'price_inr': price_inr,
                'price_usd': price_usd,
                'minimum_nights': int(np.random.choice([1, 2, 3], p=[0.55, 0.35, 0.10])),
                'number_of_reviews': num_reviews,
                'review_scores_rating': rating,
                'sentiment_compound': sentiment,
                'host_is_superhost': is_superhost,
                'host_identity_verified': is_verified,
                'has_inverter_power_backup': has_inverter,
                'has_wifi': has_wifi,
                'has_air_conditioning': has_ac,
                'has_pool': has_pool,
                'has_kitchen': has_kitchen,
                'has_parking': has_parking,
                'has_geyser': has_geyser,
                'has_caretaker_or_cook': has_cook,
                'has_scenic_view': has_scenic,
                'occupancy_rate_30': occ_30,
                'occupancy_rate_365': occ_365,
                'annual_gross_lakhs': annual_gross_lakhs,
                'annual_gross_revenue_lakhs': annual_gross_lakhs
            })
            current_id += 1

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} Indian listings across {df['city'].nunique()} cities and {df['state'].nunique()} states.")
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
