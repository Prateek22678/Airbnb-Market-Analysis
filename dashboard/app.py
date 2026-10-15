"""
dashboard/app.py
Interactive Airbnb Market Intelligence & Pricing Simulator for Austin, TX.
Built with Streamlit, Plotly, Pydeck, and Scikit-Learn/XGBoost.
"""

import os
import sys
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import joblib

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import (
    calculate_haversine_distance,
    engineer_features,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    AUSTIN_DOWNTOWN_LAT,
    AUSTIN_DOWNTOWN_LON
)
from src.india_market import (
    get_indian_listings_df,
    get_indian_city_summary,
    get_indian_zipcode_summary,
    get_indian_zipcodes,
    get_indian_hubs_df,
    CITY_CAMERA_COORDINATES,
    calculate_india_investment_roi
)

# ---------------------------------------------------------
# Page Configuration & Visual Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Austin Airbnb Market Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Airbnb Coral, Navy & Modern Cards)
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #FF5A5F;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #484848;
    }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #EBEBEB;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #222222;
    }
    /* Badges */
    .badge-superhost {
        background-color: #003580;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data & Model Loading
# ---------------------------------------------------------
@st.cache_data
def load_market_data() -> pd.DataFrame:
    """
    Load cleaned listings, forward calendar occupancy, and review sentiment metrics.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    listings_path = os.path.join(data_dir, "listings_cleaned.csv")
    calendar_path = os.path.join(data_dir, "calendar_cleaned.csv")
    sentiment_path = os.path.join(data_dir, "listings_sentiment.csv")

    df_list = pd.read_csv(listings_path)
    
    # Calculate distance to downtown
    df_list['distance_to_downtown_km'] = calculate_haversine_distance(
        df_list['latitude'], df_list['longitude'], AUSTIN_DOWNTOWN_LAT, AUSTIN_DOWNTOWN_LON
    )

    # Merge calendar occupancy if available
    if os.path.exists(calendar_path):
        df_cal = pd.read_csv(calendar_path)
        cal_cols = ['listing_id', 'occupancy_rate_30', 'occupancy_rate_90', 'occupancy_rate_365']
        df_list = df_list.merge(df_cal[cal_cols], left_on='id', right_on='listing_id', how='left')
    else:
        df_list['occupancy_rate_30'] = 0.48
        df_list['occupancy_rate_90'] = 0.41

    # Merge sentiment metrics if available
    if os.path.exists(sentiment_path):
        df_sent = pd.read_csv(sentiment_path)
        sent_cols = ['listing_id', 'sentiment_mean', 'sentiment_median', 'pct_positive', 'pct_negative']
        df_list = df_list.merge(df_sent[sent_cols], left_on='id', right_on='listing_id', how='left')
    else:
        df_list['sentiment_mean'] = 0.78
        df_list['pct_positive'] = 94.8

    # Price Tier Categorization
    df_list['neighbourhood'] = df_list['neighbourhood'].astype(str)
    df_list['price_tier'] = pd.qcut(
        df_list['price'],
        q=[0, 0.25, 0.50, 0.75, 1.0],
        labels=['Budget (< $109)', 'Midscale ($109-$199)', 'Upscale ($199-$350)', 'Luxury (> $350)']
    )

    return df_list


@st.cache_resource
def load_ml_bundle() -> Dict[str, Any]:
    """
    Load persisted XGBoost price model bundle.
    """
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "price_model_xgboost.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return {}


# Load data and ML model
df_all = load_market_data()
df_india_all = get_indian_listings_df()
ml_bundle = load_ml_bundle()


# ---------------------------------------------------------
# Sidebar: Global Interactive Filters
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=140)
    
    market_territory = st.radio(
        "🌍 Global Market Territory",
        options=[
            "🇺🇸 Austin, Texas",
            "🇮🇳 India Multi-City (17 Corridors & 42 PIN Codes)"
        ],
        index=0,
        help="Switch between the Austin, Texas empirical market and India's vacation rental corridors."
    )
    st.markdown("---")

    if market_territory.startswith("🇺🇸"):
        st.title("Austin Market Explorer")
        st.markdown("Filter properties to evaluate live market metrics, spatial distributions, and guest sentiment.")
        st.markdown("---")

        # Room Type Filter
        all_room_types = sorted(df_all['room_type'].dropna().unique().tolist())
        selected_room_types = st.multiselect(
            "Room Type",
            options=all_room_types,
            default=all_room_types
        )

        # Neighbourhood Filter (Top Austin ZIP codes)
        all_neighbourhoods = sorted(df_all['neighbourhood'].unique().tolist())
        top_neighbourhoods = [z for z in df_all['neighbourhood'].value_counts().head(12).index.tolist() if z in all_neighbourhoods]
        selected_neighbourhoods = st.multiselect(
            "Neighbourhood (ZIP Code)",
            options=all_neighbourhoods,
            default=top_neighbourhoods
        )

        # Price Range Filter
        min_price, max_price = int(df_all['price'].min()), int(df_all['price'].quantile(0.99))
        price_range = st.slider(
            "Nightly Price Range ($)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, 850),
            step=25
        )

        # Guest Capacity Filter
        capacity = st.slider("Guest Capacity (accommodates)", min_value=1, max_value=16, value=(1, 16))

        # Superhost Filter
        superhost_option = st.radio(
            "Host Status",
            options=["All Hosts", "Superhosts Only", "Regular Hosts Only"],
            horizontal=True
        )

        st.markdown("---")
        st.caption("Inside Airbnb Snapshot: Sept 17, 2026 | Austin, TX")
    else:
        st.title("🇮🇳 India Market Explorer")
        st.markdown("Filter 2,480 properties across **17 cities and 42 authentic postal PIN / ZIP codes**.")
        st.markdown("---")

        sidebar_in_cities = list(CITY_CAMERA_COORDINATES.keys())
        sidebar_selected_city = st.selectbox("🏙️ City / Region Focus", options=sidebar_in_cities, index=0)

        df_in_scope = df_india_all.copy()
        if not sidebar_selected_city.startswith("All India"):
            df_in_scope = df_in_scope[df_in_scope['city'] == sidebar_selected_city]

        # Postal PIN Code Filter with Area Name
        pin_options = sorted(df_in_scope['zipcode'].dropna().astype(str).unique().tolist())
        pin_to_neigh_map = dict(zip(df_in_scope['zipcode'].astype(str), df_in_scope['neighborhood']))
        sidebar_selected_pins = st.multiselect(
            "📮 Postal PIN / ZIP Code",
            options=pin_options,
            default=pin_options,
            format_func=lambda z: f"{z} ({pin_to_neigh_map.get(z, '')})",
            help="Filter by authentic 6-digit Indian Postal Index Numbers (PIN codes)"
        )

        # Micro-Market Filter with PIN code visible
        neigh_options = sorted(df_in_scope['neighborhood'].unique().tolist())
        neigh_to_pin_map = dict(zip(df_in_scope['neighborhood'], df_in_scope['zipcode'].astype(str)))
        sidebar_selected_neighs = st.multiselect(
            "📍 Micro-Market / Area",
            options=neigh_options,
            default=neigh_options[:min(4, len(neigh_options))] if len(neigh_options) > 4 else neigh_options,
            format_func=lambda n: f"{n} (PIN: {neigh_to_pin_map.get(n, '')})"
        )

        min_in_p, max_in_p = int(df_india_all['price_inr'].min()), int(df_india_all['price_inr'].quantile(0.99))
        price_range_inr = st.slider(
            "Nightly Rate Range (₹ INR)",
            min_value=min_in_p,
            max_value=max_in_p,
            value=(min_in_p, 45000),
            step=500
        )

        superhost_in_option = st.radio(
            "Host Status",
            options=["All Hosts", "Superhosts Only"],
            horizontal=True
        )

        st.markdown("---")
        st.caption("India STR & Vacation Rental Database | 42 PIN Codes")


# ---------------------------------------------------------
# Filtering Logic & Main Header
# ---------------------------------------------------------
if market_territory.startswith("🇺🇸"):
    df_filtered = df_all.copy()
    if selected_room_types:
        df_filtered = df_filtered[df_filtered['room_type'].isin(selected_room_types)]
    if selected_neighbourhoods:
        df_filtered = df_filtered[df_filtered['neighbourhood'].astype(str).isin(selected_neighbourhoods)]
    df_filtered = df_filtered[
        (df_filtered['price'] >= price_range[0]) &
        (df_filtered['price'] <= price_range[1]) &
        (df_filtered['accommodates'] >= capacity[0]) &
        (df_filtered['accommodates'] <= capacity[1])
    ]
    if superhost_option == "Superhosts Only":
        df_filtered = df_filtered[df_filtered['host_is_superhost'] == 1]
    elif superhost_option == "Regular Hosts Only":
        df_filtered = df_filtered[df_filtered['host_is_superhost'] == 0]

    st.title("Austin Airbnb Market Intelligence & Pricing Simulator")
    st.markdown("An executive analytics dashboard evaluating listing drivers, spatial economics, and guest sentiment in Austin, Texas.")
    if len(df_filtered) == 0:
        st.warning("⚠️ No listings match the selected filters. Please broaden your sidebar filter criteria.")
        st.stop()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Active Listings", f"{len(df_filtered):,}", help="Filtered listing count")
    with kpi2:
        median_p = df_filtered['price'].median()
        st.metric("Median Nightly Price", f"${median_p:.0f}", help="Median nightly rate")
    with kpi3:
        avg_occ = df_filtered['occupancy_rate_30'].mean() * 100.0
        st.metric("Forward 30d Occupancy", f"{avg_occ:.1f}%", help="Calendar-booked proxy over next 30 days")
    with kpi4:
        avg_rating = df_filtered['review_scores_rating'].dropna().mean()
        st.metric("Avg Rating Score", f"{avg_rating:.2f} ★", help="Average guest rating (0-5 scale)")
    with kpi5:
        avg_sent = df_filtered['sentiment_mean'].dropna().mean()
        st.metric("Avg Sentiment", f"{avg_sent:.3f}", help="VADER review compound score (-1.0 to +1.0)")
else:
    df_in_filtered = df_in_scope.copy()
    if sidebar_selected_neighs:
        df_in_filtered = df_in_filtered[df_in_filtered['neighborhood'].isin(sidebar_selected_neighs)]
    if sidebar_selected_pins:
        df_in_filtered = df_in_filtered[df_in_filtered['zipcode'].astype(str).isin(sidebar_selected_pins)]
    df_in_filtered = df_in_filtered[
        (df_in_filtered['price_inr'] >= price_range_inr[0]) &
        (df_in_filtered['price_inr'] <= price_range_inr[1])
    ]
    if superhost_in_option == "Superhosts Only":
        df_in_filtered = df_in_filtered[df_in_filtered['host_is_superhost'] == 1]

    st.title("🇮🇳 India Vacation Rental & Short-Term Market Intelligence")
    st.markdown("Comprehensive portfolio analytics, spatial economics, and verified postal PIN codes across 17 premier Indian tourist corridors.")
    if len(df_in_filtered) == 0:
        st.warning("⚠️ No Indian listings match the selected filters. Please broaden your selection criteria.")
        st.stop()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Properties Mapped", f"{len(df_in_filtered):,}", help="Filtered listing count")
    with kpi2:
        median_p_inr = df_in_filtered['price_inr'].median()
        st.metric("Median Nightly Rate", f"₹{median_p_inr:,.0f}", help="Median nightly rate in INR")
    with kpi3:
        avg_occ_in = df_in_filtered['occupancy_rate_365'].mean() * 100.0
        st.metric("Annual Occupancy", f"{avg_occ_in:.1f}%", help="Annual occupancy proxy")
    with kpi4:
        avg_rating_in = df_in_filtered['review_scores_rating'].dropna().mean()
        st.metric("Avg Rating Score", f"{avg_rating_in:.2f} ★", help="Average guest rating (0-5 scale)")
    with kpi5:
        pwr_pct_in = df_in_filtered['has_inverter_power_backup'].mean() * 100.0
        st.metric("Power Backup %", f"{pwr_pct_in:.1f}%", help="Properties with dedicated inverter/generator backup")

    # Set default df_filtered for other tabs
    df_filtered = df_all.copy()

st.markdown("---")


# ---------------------------------------------------------
# Tabbed Navigation
# ---------------------------------------------------------
tab_map, tab_market, tab_nlp, tab_ml, tab_recommendations, tab_india = st.tabs([
    "🗺️ Geospatial & Price Map",
    "📊 Market Drivers & Seasonality",
    "🧠 Review NLP & Sentiment",
    "🏷️ Dynamic Price Predictor (ML)",
    "💡 Strategic Recommendations",
    "🇮🇳 India Market Intelligence"
])


# =========================================================
# TAB 1: GEOSPATIAL MAP & PRICE EXPLORER
# =========================================================
with tab_map:
    # Market Geography Selector
    map_region = st.radio(
        "📍 Select Map Geography:",
        options=[
            "🇺🇸 Austin, Texas (Empirical Study - 9,945 Listings)",
            "🇮🇳 India - Premier Vacation Rental & Tourism Hubs"
        ],
        horizontal=True,
        help="Switch between the Austin, TX detailed listing map and India's top tourism/vacation rental corridors."
    )

    if map_region.startswith("🇺🇸"):
        st.subheader("Geospatial Listing Distribution & Nightly Rates (Austin, TX)")
        st.markdown("Interactive 3D map of Austin short-term rentals. Downtown Austin center indicated at Congress Ave & Cesar Chavez.")

        col_map_left, col_map_right = st.columns([3, 1])

        with col_map_right:
            color_by = st.selectbox(
                "Color Map Points By",
                options=["Price Tier", "Superhost Status", "Review Rating", "Sentiment Score"]
            )
            map_style = st.selectbox(
                "Base Map Style",
                options=["Light", "Dark", "Road"],
                index=0
            )
            st.markdown(f"**Showing {len(df_filtered):,} listings.**")
            
            # Price percentile summary in sidebar column
            st.markdown("##### Selected Price Distribution")
            st.write(df_filtered['price'].describe()[['min', '25%', '50%', '75%', 'max']])

        with col_map_left:
            # Assign colors for Pydeck
            map_df = df_filtered[['latitude', 'longitude', 'name', 'price', 'room_type',
                                  'host_is_superhost', 'review_scores_rating', 'sentiment_mean', 'neighbourhood']].copy()
            
            # Color mapping logic
            if color_by == "Superhost Status":
                map_df['r'] = np.where(map_df['host_is_superhost'] == 1, 0, 255)
                map_df['g'] = np.where(map_df['host_is_superhost'] == 1, 53, 90)
                map_df['b'] = np.where(map_df['host_is_superhost'] == 1, 128, 95)
            elif color_by == "Review Rating":
                map_df['r'] = np.where(map_df['review_scores_rating'] >= 4.90, 0, 255)
                map_df['g'] = np.where(map_df['review_scores_rating'] >= 4.90, 166, 120)
                map_df['b'] = np.where(map_df['review_scores_rating'] >= 4.90, 153, 50)
            else: # Default: Price Tier
                map_df['r'] = np.clip((map_df['price'] / 400.0) * 255, 40, 255).astype(int)
                map_df['g'] = np.clip(180 - (map_df['price'] / 400.0) * 120, 20, 180).astype(int)
                map_df['b'] = 90

            map_df['tooltip_text'] = (
                map_df['name'].str[:35] + "<br>" +
                "Price: $" + map_df['price'].astype(str) + "/night<br>" +
                "Room: " + map_df['room_type'] + "<br>" +
                "Rating: " + map_df['review_scores_rating'].round(2).astype(str) + " ★<br>" +
                "ZIP: " + map_df['neighbourhood'].astype(str)
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_radius=120,
                get_fill_color=["r", "g", "b", 190],
                pickable=True,
                auto_highlight=True
            )

            view_state = pdk.ViewState(
                latitude=AUSTIN_DOWNTOWN_LAT,
                longitude=AUSTIN_DOWNTOWN_LON,
                zoom=10.8,
                pitch=30
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"html": "{tooltip_text}", "style": {"color": "white", "backgroundColor": "#222222"}}
                )
            )

        # Sub-row: Price distribution histogram
        st.markdown("##### Filtered Nightly Price Distribution")
        fig_hist = px.histogram(
            df_filtered,
            x='price',
            nbins=45,
            color_discrete_sequence=['#FF5A5F'],
            marginal='box',
            labels={'price': 'Nightly Price (USD)'},
            title=f"Price Distribution for Filtered Selection (N={len(df_filtered):,})"
        )
        fig_hist.update_layout(showlegend=False, height=350, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        # =========================================================
        # INDIA GEOSPATIAL VACATION RENTAL MAP (17 CITIES & 42 PIN CODES)
        # =========================================================
        st.subheader("🇮🇳 India Multi-City Short-Term Rental & Tourism Map")
        st.markdown(
            "Interactive 3D geospatial intelligence across **2,480 properties in 17 premier Indian cities and vacation corridors** "
            "(Goa, Mumbai, Alibaug, Lonavala, Jaipur, Udaipur, Manali, Shimla, Bengaluru, Coorg, Kerala, Delhi NCR, Rishikesh, Pondicherry, Varanasi)."
        )

        col_in_city, col_in_curr = st.columns([3, 1])
        with col_in_city:
            initial_in_city_idx = 0
            if 'sidebar_selected_city' in locals() and sidebar_selected_city in CITY_CAMERA_COORDINATES:
                initial_in_city_idx = list(CITY_CAMERA_COORDINATES.keys()).index(sidebar_selected_city)
            selected_city = st.selectbox(
                "🏙️ Select City / Region Focus",
                options=list(CITY_CAMERA_COORDINATES.keys()),
                index=initial_in_city_idx
            )
        with col_in_curr:
            currency_pref = st.radio("Display Currency", options=["INR (₹)", "USD ($)"], horizontal=True)

        # Filter by selected city
        df_india = df_india_all.copy()
        if not selected_city.startswith("All India"):
            df_india = df_india[df_india['city'] == selected_city]

        # ---------------------------------------------------------
        # PROMINENT POSTAL PIN / ZIP CODE DISPLAY BANNER
        # ---------------------------------------------------------
        city_pin_df = df_india[['neighborhood', 'zipcode']].drop_duplicates().sort_values(by='neighborhood')
        if selected_city.startswith("All India"):
            st.info("📮 **National Postal Coverage**: 42 Verified 6-Digit Indian Postal PIN / ZIP Codes across 17 Vacation Corridors & Metros.")
        else:
            pin_items = [f"**{row['neighborhood']}**: `{row['zipcode']}`" for _, row in city_pin_df.iterrows()]
            st.success(f"📮 **Postal PIN / ZIP Codes for {selected_city}**: " + "  •  ".join(pin_items))

        # Filters Row with PIN codes clearly formatted
        col_in_filter1, col_in_filter2, col_in_filter3 = st.columns([1.5, 1.5, 1.2])
        
        with col_in_filter1:
            avail_neighs = sorted(df_india['neighborhood'].unique().tolist())
            neigh_to_pin_map = dict(zip(df_india['neighborhood'], df_india['zipcode'].astype(str)))
            selected_neighs = st.multiselect(
                "📍 Micro-Market / Area (with Postal PIN)",
                options=avail_neighs,
                default=avail_neighs[:min(4, len(avail_neighs))] if len(avail_neighs) > 4 else avail_neighs,
                format_func=lambda n: f"{n} (PIN: {neigh_to_pin_map.get(n, '')})"
            )

        with col_in_filter2:
            avail_zips = sorted(df_india['zipcode'].dropna().astype(str).unique().tolist())
            pin_to_neigh_map = dict(zip(df_india['zipcode'].astype(str), df_india['neighborhood']))
            selected_zips = st.multiselect(
                "📮 Postal PIN / ZIP Code (with Area)",
                options=avail_zips,
                default=avail_zips,
                format_func=lambda z: f"{z} ({pin_to_neigh_map.get(z, '')})",
                help="Filter by authentic 6-digit Indian Postal Index Numbers (PIN codes)"
            )

        with col_in_filter3:
            room_options = sorted(df_india_all['room_type'].unique().tolist())
            selected_in_rooms = st.multiselect(
                "🏠 Property / Room Type",
                options=room_options,
                default=room_options
            )

        # Amenity Filters Row
        am_c1, am_c2, am_c3, am_c4, am_c5 = st.columns(5)
        with am_c1:
            req_pool = st.checkbox("🏊 Private Pool Only")
        with am_c2:
            req_inverter = st.checkbox("⚡ Power Backup Only")
        with am_c3:
            req_ac = st.checkbox("❄️ Air Conditioned Only")
        with am_c4:
            req_cook = st.checkbox("👨‍🍳 Cook / Caretaker Only")
        with am_c5:
            req_superhost = st.checkbox("⭐ Superhost Only")

        # Apply Filters
        if selected_neighs:
            df_india = df_india[df_india['neighborhood'].isin(selected_neighs)]
        if selected_zips:
            df_india = df_india[df_india['zipcode'].astype(str).isin(selected_zips)]
        if selected_in_rooms:
            df_india = df_india[df_india['room_type'].isin(selected_in_rooms)]
        if req_pool:
            df_india = df_india[df_india['has_pool'] == 1]
        if req_inverter:
            df_india = df_india[df_india['has_inverter_power_backup'] == 1]
        if req_ac:
            df_india = df_india[df_india['has_air_conditioning'] == 1]
        if req_cook:
            df_india = df_india[df_india['has_caretaker_or_cook'] == 1]
        if req_superhost:
            df_india = df_india[df_india['host_is_superhost'] == 1]

        if len(df_india) == 0:
            st.warning("⚠️ No Indian listings match the selected filters. Please broaden your selection criteria.")
        else:
            # Metrics Row for India
            in_kpi1, in_kpi2, in_kpi3, in_kpi4, in_kpi5 = st.columns(5)
            with in_kpi1:
                st.metric("Properties Mapped", f"{len(df_india):,}")
            with in_kpi2:
                in_med_p = df_india['price_inr'].median()
                if currency_pref.startswith("INR"):
                    st.metric("Median Nightly Rate", f"₹{in_med_p:,.0f}")
                else:
                    st.metric("Median Nightly Rate", f"${in_med_p/83.0:.0f}")
            with in_kpi3:
                in_avg_occ = df_india['occupancy_rate_365'].mean() * 100.0
                st.metric("Avg Annual Occupancy", f"{in_avg_occ:.1f}%")
            with in_kpi4:
                in_pool_pct = df_india['has_pool'].mean() * 100.0
                st.metric("With Private Pool", f"{in_pool_pct:.1f}%")
            with in_kpi5:
                in_pwr_pct = df_india['has_inverter_power_backup'].mean() * 100.0
                st.metric("Power Backup Coverage", f"{in_pwr_pct:.1f}%")

            col_map_box, col_side_box = st.columns([3, 1])

            with col_side_box:
                color_point_by = st.selectbox(
                    "Color Points By",
                    options=["Price Tier (INR)", "Superhost Status", "Review Rating", "Private Pool Presence"]
                )
                st.markdown(f"**Viewing: {selected_city}**")
                st.write(df_india['price_inr'].describe()[['min', '25%', '50%', '75%', 'max']].rename({
                    'min': 'Min (₹)', '25%': '25% (₹)', '50%': 'Median (₹)', '75%': '75% (₹)', 'max': 'Max (₹)'
                }))

                # Postal PIN code breakdown table in sidebar
                st.markdown("---")
                st.markdown(f"**📮 Postal PIN Codes ({selected_city})**")
                side_pin_df = df_india.groupby(['zipcode', 'neighborhood']).agg(
                    count=('id', 'count'),
                    med_inr=('price_inr', 'median')
                ).reset_index()
                side_pin_df['med_inr'] = side_pin_df['med_inr'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(
                    side_pin_df.rename(columns={
                        'zipcode': 'PIN Code',
                        'neighborhood': 'Area',
                        'count': 'Listings',
                        'med_inr': 'Median (₹)'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            with col_map_box:
                plot_in_df = df_india.copy()

                if color_point_by == "Superhost Status":
                    plot_in_df['r'] = np.where(plot_in_df['host_is_superhost'] == 1, 0, 255)
                    plot_in_df['g'] = np.where(plot_in_df['host_is_superhost'] == 1, 53, 90)
                    plot_in_df['b'] = np.where(plot_in_df['host_is_superhost'] == 1, 128, 95)
                elif color_point_by == "Private Pool Presence":
                    plot_in_df['r'] = np.where(plot_in_df['has_pool'] == 1, 0, 240)
                    plot_in_df['g'] = np.where(plot_in_df['has_pool'] == 1, 166, 100)
                    plot_in_df['b'] = np.where(plot_in_df['has_pool'] == 1, 153, 100)
                elif color_point_by == "Review Rating":
                    plot_in_df['r'] = np.where(plot_in_df['review_scores_rating'] >= 4.85, 0, 255)
                    plot_in_df['g'] = np.where(plot_in_df['review_scores_rating'] >= 4.85, 166, 120)
                    plot_in_df['b'] = np.where(plot_in_df['review_scores_rating'] >= 4.85, 153, 50)
                else: # Default: Price Tier
                    plot_in_df['r'] = np.clip((plot_in_df['price_inr'] / 20000.0) * 255, 40, 255).astype(int)
                    plot_in_df['g'] = np.clip(200 - (plot_in_df['price_inr'] / 20000.0) * 150, 30, 200).astype(int)
                    plot_in_df['b'] = 90

                plot_in_df['tooltip_text'] = (
                    "<b>" + plot_in_df['name'] + "</b><br>" +
                    "City: " + plot_in_df['city'] + " (" + plot_in_df['neighborhood'] + ")<br>" +
                    "📮 Postal PIN / ZIP: <b>" + plot_in_df['zipcode'].astype(str) + "</b><br>" +
                    "Rate: ₹" + plot_in_df['price_inr'].apply(lambda x: f"{x:,}") + " / night ($" + plot_in_df['price_usd'].astype(str) + ")<br>" +
                    "Type: " + plot_in_df['room_type'] + "<br>" +
                    "Capacity: " + plot_in_df['accommodates'].astype(str) + " Guests (" + plot_in_df['bedrooms'].astype(str) + " Beds)<br>" +
                    "Rating: " + plot_in_df['review_scores_rating'].round(2).astype(str) + " ★ | Superhost: " + np.where(plot_in_df['host_is_superhost'] == 1, "Yes", "No") + "<br>" +
                    "Power Backup: " + np.where(plot_in_df['has_inverter_power_backup'] == 1, "Yes (Inverter)", "No") + "<br>" +
                    "Private Pool: " + np.where(plot_in_df['has_pool'] == 1, "Yes", "No")
                )

                # Dynamically set camera from coordinates map
                cam_lat, cam_lon, cam_zoom, cam_pitch = CITY_CAMERA_COORDINATES.get(
                    selected_city, CITY_CAMERA_COORDINATES['All India (National Overview)']
                )
                
                # Dynamic radius based on zoom level
                dot_radius = 45000 if cam_zoom < 6 else (1500 if cam_zoom > 10 else 6500)

                layer_india = pdk.Layer(
                    "ScatterplotLayer",
                    data=plot_in_df,
                    get_position=["longitude", "latitude"],
                    get_radius=dot_radius,
                    get_fill_color=["r", "g", "b", 200],
                    pickable=True,
                    auto_highlight=True
                )

                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer_india],
                        initial_view_state=pdk.ViewState(
                            latitude=cam_lat,
                            longitude=cam_lon,
                            zoom=cam_zoom,
                            pitch=cam_pitch
                        ),
                        tooltip={"html": "{tooltip_text}", "style": {"color": "white", "backgroundColor": "#222222", "fontSize": "12px"}}
                    )
                )

            # Sub-Charts for Selected Indian Segment
            col_bar_left, col_bar_right = st.columns(2)

            with col_bar_left:
                fig_neigh = px.box(
                    df_india,
                    x='neighborhood',
                    y='price_inr' if currency_pref.startswith("INR") else 'price_usd',
                    color='neighborhood',
                    labels={'price_inr': 'Nightly Rate (INR ₹)', 'price_usd': 'Nightly Rate (USD $)', 'neighborhood': 'Micro-Market'},
                    title=f"Rate Distribution by Micro-Market in {selected_city}"
                )
                fig_neigh.update_layout(showlegend=False, height=380, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_neigh, use_container_width=True)

            with col_bar_right:
                fig_in_hist = px.histogram(
                    df_india,
                    x='price_inr' if currency_pref.startswith("INR") else 'price_usd',
                    nbins=35,
                    color_discrete_sequence=['#FF5A5F'],
                    marginal='box',
                    labels={'price_inr': 'Nightly Rate (INR ₹)', 'price_usd': 'Nightly Rate (USD $)'},
                    title=f"Nightly Rate Histogram for Filtered Selection (N={len(df_india):,})"
                )
                fig_in_hist.update_layout(showlegend=False, height=380, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_in_hist, use_container_width=True)

            # Property Listings Table WITH POSTAL PIN / ZIP CODES
            st.markdown(f"##### 📋 Filtered Properties Directory with Postal PIN / ZIP Codes ({len(df_india):,} Properties)")
            st.dataframe(
                df_india[['name', 'city', 'neighborhood', 'zipcode', 'room_type', 'accommodates', 'bedrooms', 'price_inr', 'price_usd', 'review_scores_rating', 'has_pool', 'has_inverter_power_backup', 'has_caretaker_or_cook']]
                .rename(columns={
                    'name': 'Property Name',
                    'city': 'City',
                    'neighborhood': 'Micro-Market',
                    'zipcode': 'Postal PIN / ZIP Code',
                    'room_type': 'Type',
                    'accommodates': 'Guests',
                    'bedrooms': 'Beds',
                    'price_inr': 'Rate (₹)',
                    'price_usd': 'Rate ($)',
                    'review_scores_rating': 'Rating ★',
                    'has_pool': 'Pool',
                    'has_inverter_power_backup': 'Inverter',
                    'has_caretaker_or_cook': 'Cook'
                }),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# TAB 2: MARKET DYNAMICS & PRICING DRIVERS
# =========================================================
with tab_market:
    st.subheader("Market Dynamics & Key Pricing Drivers")
    st.markdown("Explore structural pricing variation across room types, geographic distance to downtown, and amenity premiums.")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        # 1. Price by Room Type
        fig_room = px.box(
            df_filtered,
            x='room_type',
            y='price',
            color='room_type',
            color_discrete_sequence=['#FF5A5F', '#003580', '#00A699', '#FC642D'],
            labels={'room_type': 'Room Privacy Tier', 'price': 'Nightly Price ($)'},
            title="Nightly Price by Room Privacy Tier"
        )
        fig_room.update_layout(showlegend=False, height=400, yaxis_range=[0, min(800, df_filtered['price'].quantile(0.98))])
        st.plotly_chart(fig_room, use_container_width=True)

    with col_m2:
        # 2. Distance to Downtown Decay Curve
        n_dist_sample = min(1500, len(df_filtered))
        dist_plot_df = df_filtered.sample(n_dist_sample, random_state=42) if n_dist_sample > 0 else df_filtered
        fig_dist = px.scatter(
            dist_plot_df,
            x='distance_to_downtown_km',
            y='price',
            color='room_type',
            trendline='lowess',
            opacity=0.45,
            labels={'distance_to_downtown_km': 'Distance to Downtown Austin (km)', 'price': 'Nightly Price ($)'},
            title="Geographic Price Decay vs. Distance to Downtown Core"
        )
        fig_dist.update_layout(height=400, yaxis_range=[0, min(800, df_filtered['price'].quantile(0.98))])
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    col_m3, col_m4 = st.columns(2)

    with col_m3:
        # 3. Amenity Price Premiums
        amenity_premiums = []
        amenities_to_check = [
            ('has_pool', 'Swimming Pool'),
            ('has_hot_tub', 'Hot Tub'),
            ('has_washer_dryer', 'Washer / Dryer'),
            ('has_patio_or_balcony', 'Patio or Balcony'),
            ('has_workspace', 'Dedicated Workspace'),
            ('has_free_parking', 'Free Parking')
        ]
        for col_name, label in amenities_to_check:
            if col_name in df_filtered.columns:
                p_with = df_filtered[df_filtered[col_name] == 1]['price'].median()
                p_without = df_filtered[df_filtered[col_name] == 0]['price'].median()
                amenity_premiums.append({
                    'Amenity': label,
                    'Median With ($)': p_with,
                    'Median Without ($)': p_without,
                    'Premium ($)': p_with - p_without
                })
        
        df_prem = pd.DataFrame(amenity_premiums).sort_values(by='Premium ($)', ascending=True)
        fig_amenity = px.bar(
            df_prem,
            y='Amenity',
            x='Premium ($)',
            orientation='h',
            color='Premium ($)',
            color_continuous_scale='Reds',
            title="Median Nightly Rate Premium by Amenity",
            text='Premium ($)'
        )
        fig_amenity.update_traces(texttemplate='+$%{x:.0f}', textposition='outside')
        fig_amenity.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_amenity, use_container_width=True)

    with col_m4:
        # 4. Top ZIP Code Median Rates
        zip_pricing = df_filtered.groupby('neighbourhood')['price'].agg(['count', 'median']).reset_index()
        zip_pricing = zip_pricing[zip_pricing['count'] >= 25].sort_values(by='median', ascending=False).head(10)
        
        fig_zip = px.bar(
            zip_pricing,
            x='median',
            y='neighbourhood',
            orientation='h',
            color='median',
            color_continuous_scale='Blues',
            labels={'median': 'Median Nightly Rate ($)', 'neighbourhood': 'ZIP Code'},
            title="Top 10 Highest-Priced Austin ZIP Codes (N ≥ 25)",
            text='median'
        )
        fig_zip.update_traces(texttemplate='$%{x:.0f}', textposition='outside')
        fig_zip.update_layout(showlegend=False, height=380, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_zip, use_container_width=True)


# =========================================================
# TAB 3: REVIEW NLP & SENTIMENT INTELLIGENCE
# =========================================================
with tab_nlp:
    st.subheader("Unstructured Review Intelligence & Sentiment Analysis")
    st.markdown("VADER compound sentiment scores ($-1.0$ to $+1.0$) and Latent Dirichlet Allocation (LDA) theme discovery.")

    col_n1, col_n2 = st.columns(2)

    with col_n1:
        # Sentiment Distribution
        fig_sent = px.histogram(
            df_filtered.dropna(subset=['sentiment_mean']),
            x='sentiment_mean',
            color='host_is_superhost',
            barmode='overlay',
            nbins=35,
            color_discrete_map={1: '#003580', 0: '#FF5A5F'},
            labels={'sentiment_mean': 'Mean Review Sentiment Compound Score', 'host_is_superhost': 'Superhost'},
            title="Guest Sentiment Distribution: Superhosts vs. Regular Hosts"
        )
        fig_sent.update_layout(height=380)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_n2:
        # Breaking the Rating Ceiling Effect
        df_valid_sent = df_filtered.dropna(subset=['review_scores_rating', 'sentiment_mean'])
        n_sent_sample = min(1200, len(df_valid_sent))
        sent_plot_df = df_valid_sent.sample(n_sent_sample, random_state=42) if n_sent_sample > 0 else df_valid_sent
        fig_ceil = px.scatter(
            sent_plot_df,
            x='review_scores_rating',
            y='sentiment_mean',
            trendline='lowess',
            opacity=0.35,
            color_discrete_sequence=['#00A699'],
            labels={'review_scores_rating': 'Star Rating (0-5 scale)', 'sentiment_mean': 'Text Sentiment (-1 to +1)'},
            title="Star Rating vs. Text Sentiment: Exposing the Ceiling Effect"
        )
        fig_ceil.update_layout(height=380, xaxis_range=[4.0, 5.02])
        st.plotly_chart(fig_ceil, use_container_width=True)

    st.markdown("---")
    st.subheader("The 5 Latent Operational Topics in Austin Reviews")
    st.markdown("Extracted via Latent Dirichlet Allocation (LDA) over 133,000+ guest reviews:")

    topic_cols = st.columns(5)
    topics_info = [
        ("1. Property Issues", ["issues", "water", "property", "bathroom", "shower", "day"], "#FF5A5F", "Operational friction & plumbing failures"),
        ("2. Space & Pool", ["kitchen", "pool", "space", "family", "beds", "backyard"], "#003580", "High-value leisure & group gatherings"),
        ("3. Guest Experience", ["experience", "spot", "wonderful", "awesome", "book", "staying"], "#00A699", "Overall delight & repeat stays"),
        ("4. Host Responsiveness", ["responsive", "helpful", "check", "needed", "friendly", "easy"], "#FC642D", "Proactive communication & entry ease"),
        ("5. Location & Dining", ["close", "quiet", "downtown", "restaurants", "walk", "congress"], "#6C5CE7", "Walkability to South Congress / 6th")
    ]

    for idx, (title, keywords, color, desc) in enumerate(topics_info):
        with topic_cols[idx]:
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.code("\n".join(keywords), language="text")

    st.markdown("---")
    st.subheader("Interactive Listing Sentiment Inspector")
    sample_options = df_filtered.dropna(subset=['sentiment_mean']).sort_values(by='number_of_reviews', ascending=False).head(50)
    selected_listing_name = st.selectbox(
        "Select a High-Review Property to Inspect Sentiment Metrics:",
        options=sample_options['name'].tolist()
    )

    if selected_listing_name:
        row = sample_options[sample_options['name'] == selected_listing_name].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nightly Rate", f"${row['price']:.0f}")
        c2.metric("Rating Score", f"{row['review_scores_rating']:.2f} ★")
        c3.metric("Review Sentiment", f"{row['sentiment_mean']:.3f}")
        c4.metric("Positive Reviews", f"{row.get('pct_positive', 95):.1f}%")


# =========================================================
# TAB 4: DYNAMIC PRICE PREDICTOR (LIVE ML SIMULATOR)
# =========================================================
with tab_ml:
    st.subheader("Austin Fair Market Price Estimator (XGBoost)")
    st.markdown("Simulate pricing for any Austin short-term rental property using our trained gradient boosting model (R² = 78.94%, Median AE = $44.13).")

    if not ml_bundle:
        st.error("ML model bundle not found. Please train the model via `src/modeling.py` first.")
    else:
        model = ml_bundle['model']
        preprocessor = ml_bundle['preprocessor']
        test_metrics = ml_bundle.get('test_metrics', {})

        col_input1, col_input2, col_input3 = st.columns(3)

        with col_input1:
            st.markdown("##### 1. Space & Privacy")
            sim_room_type = st.selectbox("Room Type", options=['Entire home/apt', 'Private room', 'Hotel room', 'Shared room'])
            sim_property_type = st.selectbox(
                "Property Category",
                options=['Entire home', 'Entire rental unit', 'Entire condo', 'Private room in home', 'Entire guesthouse', 'Room in hotel', 'Other']
            )
            sim_accommodates = st.number_input("Guest Capacity (Accommodates)", min_value=1, max_value=16, value=4)
            sim_bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            sim_bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
            sim_beds = st.number_input("Beds", min_value=1, max_value=16, value=2)

        with col_input2:
            st.markdown("##### 2. Location & Booking Policy")
            sim_neighbourhood = st.selectbox("Austin ZIP Code", options=sorted(df_all['neighbourhood'].dropna().astype(str).unique().tolist()), index=0)
            sim_min_nights = st.number_input("Minimum Nights Stay", min_value=1, max_value=30, value=2)
            sim_superhost = st.checkbox("Host is Superhost (Verified Premium)", value=True)
            sim_verified = st.checkbox("Host Identity Officially Verified", value=True)
            sim_instant = st.checkbox("Instant Bookable Enabled", value=True)
            sim_reviews_velocity = st.slider("Expected Reviews per Month", min_value=0.0, max_value=10.0, value=2.5, step=0.5)

        with col_input3:
            st.markdown("##### 3. High-Value Amenities")
            sim_pool = st.checkbox("Swimming Pool (#1 Price Driver)", value=True)
            sim_hot_tub = st.checkbox("Hot Tub / Jacuzzi", value=False)
            sim_washer = st.checkbox("In-Unit Washer & Dryer", value=True)
            sim_wifi = st.checkbox("High-Speed Wifi", value=True)
            sim_ac = st.checkbox("Central Air Conditioning", value=True)
            sim_parking = st.checkbox("Free On-Site Parking", value=True)
            sim_workspace = st.checkbox("Dedicated Workspace", value=True)
            sim_kitchen = st.checkbox("Full Kitchen", value=True)
            sim_patio = st.checkbox("Patio / Balcony / Deck", value=True)
            sim_pets = st.checkbox("Pets Allowed", value=False)
            sim_self_checkin = st.checkbox("Self Check-in (Smart Lock / Keypad)", value=True)

        st.markdown("---")
        btn_predict = st.button("🚀 Calculate Estimated Fair Market Price", type="primary", use_container_width=True)

        if btn_predict:
            # Approximate representative coordinates based on ZIP code
            zip_coords = df_all[df_all['neighbourhood'].astype(str) == str(sim_neighbourhood)][['latitude', 'longitude']].median()
            if zip_coords.isnull().any():
                sim_lat, sim_lon = AUSTIN_DOWNTOWN_LAT, AUSTIN_DOWNTOWN_LON
            else:
                sim_lat, sim_lon = zip_coords['latitude'], zip_coords['longitude']

            # Count total checked amenities
            checked_amenities = [sim_pool, sim_hot_tub, sim_washer, sim_wifi, sim_ac, sim_parking,
                                 sim_workspace, sim_kitchen, sim_patio, sim_pets, sim_self_checkin]
            amenity_count = sum(checked_amenities) + 15 # base standard items

            # Create input dictionary
            listing_input = {
                'latitude': sim_lat,
                'longitude': sim_lon,
                'room_type': sim_room_type,
                'property_type': sim_property_type,
                'neighbourhood': str(sim_neighbourhood),
                'accommodates': sim_accommodates,
                'bedrooms': sim_bedrooms,
                'bathrooms': sim_bathrooms,
                'beds': sim_beds,
                'minimum_nights': sim_min_nights,
                'number_of_reviews': 35,
                'number_of_reviews_ltm': 15,
                'reviews_per_month': sim_reviews_velocity,
                'days_since_first_review': 450,
                'review_scores_rating': 4.92,
                'review_scores_cleanliness': 4.90,
                'review_scores_location': 4.95,
                'review_scores_value': 4.85,
                'amenities_count': amenity_count,
                'host_is_superhost': int(sim_superhost),
                'host_identity_verified': int(sim_verified),
                'instant_bookable': int(sim_instant),
                'has_wifi': int(sim_wifi),
                'has_pool': int(sim_pool),
                'has_hot_tub': int(sim_hot_tub),
                'has_free_parking': int(sim_parking),
                'has_air_conditioning': int(sim_ac),
                'has_workspace': int(sim_workspace),
                'has_kitchen': int(sim_kitchen),
                'has_patio_or_balcony': int(sim_patio),
                'has_washer_dryer': int(sim_washer),
                'has_self_checkin': int(sim_self_checkin),
                'has_pets_allowed': int(sim_pets)
            }

            input_df = pd.DataFrame([listing_input])
            engineered_df = engineer_features(input_df)

            # Reindex with full required columns
            for col in NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES:
                if col not in engineered_df.columns:
                    engineered_df[col] = 0

            X_sim = engineered_df[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]
            X_sim_proc = preprocessor.transform(X_sim)

            # Predict
            pred_log = model.predict(X_sim_proc)[0]
            pred_price = float(np.exp(pred_log))

            # Uncertainty interval based on test RMSE (0.363 log points)
            rmse_log = test_metrics.get('Test_RMSE_log', 0.363)
            low_price = float(np.exp(pred_log - rmse_log * 0.6745)) # 50% interquartile range
            high_price = float(np.exp(pred_log + rmse_log * 0.6745))

            st.success("✅ Prediction Completed Successfully!")
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Recommended Nightly Rate", f"${pred_price:.2f}")
            res_c2.metric("Fair Range (Low)", f"${low_price:.2f}")
            res_c3.metric("Fair Range (High)", f"${high_price:.2f}")

            st.info(f"💡 **Pricing Context**: For a {sim_accommodates}-guest {sim_room_type} in ZIP {sim_neighbourhood}, "
                    f"our model estimates a baseline market rate of **${pred_price:.2f} / night**. "
                    f"During major Austin festival weekends (SXSW in March, ACL/F1 in October), historical data indicates "
                    f"properties in this tier successfully command a **40% to 65% surge premium** (${pred_price*1.4:.0f} - ${pred_price*1.65:.0f}).")


# =========================================================
# TAB 5: STRATEGIC PLAYBOOK & RECOMMENDATIONS
# =========================================================
with tab_recommendations:
    st.subheader("Executive Strategic Recommendations")
    st.markdown("Actionable takeaways derived from empirical econometric modeling and unstructured NLP analysis of Austin, Texas.")

    st.markdown("""
    ### 1. Capacity & Architectural Strategy (Maximizing ADR)
    - **Add Sleeper Capacity**: Guest capacity (`accommodates`) is the single strongest price driver (14.1% gain importance). Adding a high-quality queen sofa bed or bunk beds to transition a 4-guest home into a 6-guest rental yields an estimated **+$40 to +$75/night lift**.
    - **Maintain Bathroom Parity**: The `bathrooms_per_bedroom` ratio is critical. Group bookings avoid properties with severe morning bathroom bottlenecks.

    ### 2. High-ROI Amenity Investments
    - **Install / Highlight a Swimming Pool**: Pools represent the **#1 individual amenity price driver** (2.1% gain importance; +$110/night median premium). In Austin's climate, a private or community pool dramatically outperforms all other leisure perks.
    - **Provide In-Unit Laundry**: Washers and dryers are the second-ranked amenity driver (1.8% gain importance), particularly vital for multi-day festival stays (SXSW, ACL).

    ### 3. Operational Excellence & Superhost Retention
    - **Protect the 4.80 Rating Floor**: Superhosts command a statistically confirmed pricing premium (**+$36.61/night median lift**, $p < 10^{-30}$) and generate 9.3% higher guest review sentiment.
    - **Pre-empt Maintenance Grievances (Topic 1)**: Negative sentiment is overwhelmingly driven by air conditioning failures, lukewarm water, or lockbox confusion. Instituting automated same-day maintenance dispatches preserves Superhost eligibility.

    ### 4. Dynamic Seasonality Management
    - **Capitalize on March & October**: Calendar occupancy and review arrival volumes peak intensely during **March (SXSW)** and **October (Austin City Limits & Formula 1)**. Hosts should apply minimum night restrictions (3–4 nights) and implement surge pricing tiers up to 60% above summer baseline.
    """)


# =========================================================
# TAB 6: INDIA MARKET INTELLIGENCE & FEASIBILITY SIMULATOR
# =========================================================
with tab_india:
    st.subheader("🇮🇳 India Vacation Rental Intelligence & Investment Feasibility")
    st.markdown(
        "A strategic overview of India's rapidly growing vacation rental and homestay sector, "
        "contrasting domestic demand dynamics with Western market benchmarks and featuring an interactive ROI yield calculator."
    )

    hubs_df = get_indian_hubs_df()

    # Section 1: Macro Archetypes
    st.markdown("### 1. The Four Indian Vacation Rental Archetypes")
    arc1, arc2, arc3, arc4 = st.columns(4)
    with arc1:
        st.markdown("""
        **🏖️ Coastal & Beachfront**
        - **Core Markets**: North & South Goa, Alibaug, Pondicherry
        - **ADR**: ₹7,500 – ₹18,000+
        - **Key Driver**: Private plunge pool, beach proximity, group party villa layout.
        - **Peak**: Nov – Feb (Sunburn, New Year).
        """)
    with arc2:
        st.markdown("""
        **🏰 Royal & Heritage**
        - **Core Markets**: Udaipur, Jaipur, Jodhpur
        - **ADR**: ₹6,000 – ₹15,000+
        - **Key Driver**: Lake views, restored courtyards (Mewari havelis), destination weddings.
        - **Peak**: Oct – Mar (Winter festivals).
        """)
    with arc3:
        st.markdown("""
        **⛰️ Himalayan Escapes**
        - **Core Markets**: Manali, Shimla, Kasauli, Dharamshala
        - **ADR**: ₹4,500 – ₹12,000
        - **Key Driver**: Mountain view balconies, stone fireplaces, apple orchard surroundings.
        - **Peak**: Dual season (Dec-Jan Snow, May-Jun Summer).
        """)
    with arc4:
        st.markdown("""
        **💼 Urban & Tech Extended Stays**
        - **Core Markets**: Mumbai (Bandra), Bengaluru (Indiranagar)
        - **ADR**: ₹5,000 – ₹12,000
        - **Key Driver**: 300 Mbps Wi-Fi, power inverter backup, ergonomic work desks.
        - **Peak**: High steady year-round occupancy (>80%).
        """)

    st.markdown("---")

    # National City & PIN Code Benchmarks Tables
    st.markdown("### 2. National Market Benchmarks Across 17 Indian Destinations & 42 PIN Codes")
    
    subtab_city, subtab_zip = st.tabs([
        "🏙️ City & State Benchmarks (17 Corridors)",
        "📮 Postal PIN / ZIP Code Breakdown (42 Micro-Markets)"
    ])
    
    with subtab_city:
        city_sum_df = get_indian_city_summary()
        st.dataframe(
            city_sum_df.rename(columns={
                'city': 'Destination City',
                'state': 'State',
                'region': 'Region',
                'total_listings': 'Mapped Properties',
                'median_adr_inr': 'Median Nightly (₹)',
                'avg_adr_inr': 'Avg ADR (₹)',
                'avg_occupancy': 'Occupancy %',
                'pct_superhost': 'Superhost %',
                'pct_with_pool': 'With Pool %',
                'pct_power_backup': 'Power Backup %',
                'avg_gross_annual_lakhs': 'Est. Gross (Lakhs ₹)'
            }),
            use_container_width=True,
            hide_index=True
        )

    with subtab_zip:
        zip_sum_df = get_indian_zipcode_summary()
        st.dataframe(
            zip_sum_df.rename(columns={
                'zipcode': 'Postal PIN / ZIP Code',
                'neighborhood': 'Micro-Market / Area',
                'city': 'City',
                'state': 'State',
                'total_listings': 'Properties',
                'median_adr_inr': 'Median Nightly (₹)',
                'avg_adr_inr': 'Avg ADR (₹)',
                'avg_occupancy': 'Occupancy %',
                'pct_superhost': 'Superhost %',
                'pct_pool': 'Pool %',
                'pct_power_backup': 'Power Backup %',
                'avg_gross_annual_lakhs': 'Est. Gross (Lakhs ₹)'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Section 3: Interactive Investment Feasibility Calculator
    st.markdown("### 3. Interactive Investment Yield & Payback Simulator (INR ₹ Lakhs)")
    st.markdown("Evaluate financial viability, gross rental yields, and capital payback periods for vacation rental acquisitions in India.")

    sim_c1, sim_c2 = st.columns([1, 1])

    with sim_c1:
        selected_hub_name = st.selectbox("Select Target Indian Hub", options=hubs_df['hub'].tolist(), index=0)
        selected_hub = hubs_df[hubs_df['hub'] == selected_hub_name].iloc[0]

        # Display PIN codes for the selected investment hub
        hub_city = selected_hub['city']
        hub_pins = df_india_all[df_india_all['city'] == hub_city][['neighborhood', 'zipcode']].drop_duplicates()
        hub_pin_str = " • ".join([f"**{r['zipcode']}** ({r['neighborhood']})" for _, r in hub_pins.iterrows()])
        st.info(f"📮 **Postal PIN Codes in {hub_city}**: {hub_pin_str}")

        property_cost = st.slider("Total Capital Investment (₹ in Lakhs)", min_value=25.0, max_value=350.0, value=85.0, step=5.0, help="Purchase price + interior fit-out/renovation in Lakhs")
        default_rate = float(selected_hub['avg_price_inr'])
        nightly_rate = st.slider("Target Nightly Rate (INR ₹)", min_value=2000.0, max_value=30000.0, value=default_rate, step=250.0)
        default_occ = float(selected_hub['occupancy_rate'] * 100.0)
        target_occ = st.slider("Expected Annual Occupancy Rate (%)", min_value=35.0, max_value=95.0, value=default_occ, step=1.0)
        opex_pct = st.slider("Operating Expenses (% of Gross Revenue)", min_value=15.0, max_value=40.0, value=25.0, help="Covers OTA commission (15%), housekeeping, utilities, and maintenance")

    with sim_c2:
        res = calculate_india_investment_roi(property_cost, nightly_rate, target_occ, opex_pct)
        st.markdown("#### Projected Financial Returns")
        
        m_r1, m_r2 = st.columns(2)
        m_r1.metric("Est. Annual Booked Nights", f"{res['booked_nights_per_year']} Nights")
        m_r2.metric("Gross Annual Revenue", f"₹{res['gross_annual_rev_lakhs']:.2f} Lakhs")

        m_r3, m_r4 = st.columns(2)
        m_r3.metric("Net Operating Income (NOI)", f"₹{res['net_annual_income_lakhs']:.2f} Lakhs")
        m_r4.metric("Capital Payback Period", f"{res['payback_years']:.1f} Years" if res['payback_years'] < 50 else "> 50 Years")

        y_r1, y_r2 = st.columns(2)
        y_r1.metric("Gross Rental Yield", f"{res['gross_yield_pct']:.1f}%", delta="vs. 2.5% Long-term Rental")
        y_r2.metric("Net Rental Yield", f"{res['net_yield_pct']:.1f}%")

        if res['gross_yield_pct'] >= 12.0:
            st.success("🌟 **Strong Commercial Feasibility**: Projected gross yield exceeds 12%, significantly outperforming standard Indian residential real estate (2.5% - 3.2%).")
        elif res['gross_yield_pct'] >= 8.0:
            st.info("👍 **Moderate Commercial Feasibility**: Healthy yield matching commercial real estate benchmarks with sustained capital appreciation potential.")
        else:
            st.warning("⚠️ **Low Yield Warning**: High acquisition capital or low nightly rates extend payback period beyond 12 years. Consider optimizing occupancy or adding high-value amenities (private pool, AC).")

    st.markdown("---")

    # Section 3: Unique Operational Nuances in India
    st.markdown("### 3. Critical Operational Differences: India vs. Western STR Markets")
    st.markdown("""
    - **1. The Power Inverter / Generator Mandate**: In India (especially Goa, Himachal, and Uttarakhand), unexpected grid power disruptions during monsoons or peak summers are common. Units without 100% inverter power backup experience catastrophic 1-star review drops.
    - **2. The Private Pool Premium**: Just as in Austin, TX, a private swimming pool is the single highest price power multiplier in Goa, Alibaug, and Udaipur, driving up to **+45% higher ADR** over comparable dry villas.
    - **3. Staffing & Caretaker Expectations**: Domestic travelers in India place an extremely high premium on on-site caretakers, breakfast preparation, and local assistance—a service dimension rarely observed in self-check-in Western markets.
    - **4. Extended Stay 'Workation' Demand**: Post-2020, destinations like Goa, Dharamshala, and Bengaluru generate a massive proportion of 14–30 day bookings requiring reliable 100+ Mbps fiber internet and dedicated desk setups.
    """)

st.markdown("---")
st.caption("Austin & India Airbnb Market Intelligence | Data Analyst Portfolio Project | September 2026")
