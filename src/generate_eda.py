"""
Exploratory Data Analysis (EDA) Script for Austin Airbnb Market.
Generates 8 high-resolution charts in reports/figures/ and prints empirical metrics.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    listings_path = "data/processed/listings_cleaned.csv"
    calendar_path = "data/processed/calendar_cleaned.csv"
    reviews_path = "data/processed/reviews_cleaned.csv"
    
    df_listings = pd.read_csv(listings_path)
    df_cal = pd.read_csv(calendar_path)
    
    # Merge forward occupancy into listings
    df = df_listings.merge(
        df_cal[["listing_id", "occupancy_rate_30", "occupancy_rate_90", "occupancy_rate_365"]],
        left_on="id", right_on="listing_id", how="left"
    )
    return df, df_cal

def plot_price_by_room_type(df):
    """Figure 1: Price distribution by room type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["Entire home/apt", "Private room", "Hotel room", "Shared room"]
    palette = ["#6C5CE7", "#00CEC9", "#FFA502", "#FF4757"]
    
    sns.boxplot(
        data=df, x="room_type", y="price", order=order,
        palette=palette, showfliers=False, width=0.5, ax=ax
    )
    # Add median labels
    medians = df.groupby("room_type")["price"].median()
    counts = df["room_type"].value_counts()
    for i, rt in enumerate(order):
        med = medians.get(rt, np.nan)
        cnt = counts.get(rt, 0)
        ax.text(i, med + 15, f"Median: ${med:,.0f}\n(n={cnt:,})", 
                horizontalalignment='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
        
    ax.set_title("Austin Airbnb Nightly Price by Room Type (Excluding Outliers)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Room Privacy Tier", fontsize=11, fontweight="bold")
    ax.set_ylabel("Nightly Price ($ USD)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 900)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_price_by_room_type.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def plot_top_neighbourhoods(df):
    """Figure 2: Top 15 neighbourhoods by median price."""
    counts = df["neighbourhood"].value_counts()
    valid_hoods = counts[counts >= 50].index
    hood_stats = df[df["neighbourhood"].isin(valid_hoods)].groupby("neighbourhood").agg(
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        listing_count=("id", "count")
    ).reset_index().sort_values(by="median_price", ascending=True).tail(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(hood_stats["neighbourhood"].astype(str), hood_stats["median_price"], color="#0984E3", height=0.65)
    
    for bar, (_, row) in zip(bars, hood_stats.iterrows()):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2, 
                f"${row['median_price']:,.0f} (n={int(row['listing_count']):,})",
                va="center", fontsize=9.5, fontweight="bold", color="#2d3436")
        
    ax.set_title("Top 15 Austin ZIP Codes by Median Nightly Rate (Min. 50 Listings)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Median Nightly Price ($ USD)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Austin Postal ZIP Code", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(hood_stats["median_price"]) * 1.25)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_top_neighbourhoods_pricing.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def plot_capacity_vs_price(df):
    """Figure 3: Guest Capacity (Accommodates) vs Nightly Price."""
    cap_stats = df.groupby("accommodates").agg(
        median_price=("price", "median"),
        mean_price=("price", "mean"),
        count=("id", "count")
    ).reset_index()
    cap_stats = cap_stats[cap_stats["accommodates"] <= 16]
    
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(cap_stats["accommodates"], cap_stats["median_price"], marker="o", color="#D63031", linewidth=2.5, label="Median Price ($)")
    ax1.plot(cap_stats["accommodates"], cap_stats["mean_price"], marker="s", linestyle="--", color="#E17055", linewidth=2, label="Mean Price ($)")
    ax2.bar(cap_stats["accommodates"], cap_stats["count"], alpha=0.25, color="#74B9FF", width=0.5, label="Listing Inventory (Volume)")
    
    ax1.set_title("Nightly Rate Scaling vs Guest Capacity (Accommodates 1–16 Guests)", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Guest Capacity (Accommodates)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Nightly Price ($ USD)", fontsize=11, fontweight="bold", color="#D63031")
    ax2.set_ylabel("Active Listing Count", fontsize=11, fontweight="bold", color="#0984E3")
    ax1.set_xticks(range(1, 17))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2.grid(False)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_capacity_vs_price.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def plot_superhost_comparison(df):
    """Figure 4: Superhost vs Non-Superhost performance metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Price comparison (box)
    sns.boxplot(data=df, x="host_is_superhost", y="price", palette=["#B2BEC3", "#00B894"], 
                showfliers=False, width=0.4, ax=axes[0])
    axes[0].set_xticklabels(["Regular Host", "Superhost"], fontweight="bold")
    axes[0].set_title("Nightly Price Distribution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Price ($ USD)", fontsize=10)
    axes[0].set_xlabel("")
    med_sh = df.groupby("host_is_superhost")["price"].median()
    axes[0].text(0, med_sh[0] + 15, f"Med: ${med_sh[0]:,.0f}", ha='center', fontweight='bold', bbox=dict(fc="white", ec="#ccc", boxstyle="round"))
    axes[0].text(1, med_sh[1] + 15, f"Med: ${med_sh[1]:,.0f}", ha='center', fontweight='bold', bbox=dict(fc="white", ec="#ccc", boxstyle="round"))
    
    # 2. Rating Score (KDE)
    sns.kdeplot(df[df["host_is_superhost"] == 1]["review_scores_rating"].dropna(), ax=axes[1], color="#00B894", label="Superhost", fill=True, alpha=0.3)
    sns.kdeplot(df[df["host_is_superhost"] == 0]["review_scores_rating"].dropna(), ax=axes[1], color="#B2BEC3", label="Regular Host", fill=True, alpha=0.3)
    axes[1].set_title("Overall Rating Score (0–5.0)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Review Score Rating", fontsize=10)
    axes[1].set_xlim(4.0, 5.05)
    axes[1].legend(loc="upper left")
    
    # 3. Monthly Review Velocity
    sns.boxplot(data=df, x="host_is_superhost", y="reviews_per_month", palette=["#B2BEC3", "#00B894"],
                showfliers=False, width=0.4, ax=axes[2])
    axes[2].set_xticklabels(["Regular Host", "Superhost"], fontweight="bold")
    axes[2].set_title("Review Velocity (Reviews / Month)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Reviews Per Month", fontsize=10)
    axes[2].set_xlabel("")
    
    plt.suptitle("Superhost Performance Premium: Pricing, Rating Quality & Booking Velocity", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_superhost_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def plot_correlation_heatmap(df):
    """Figure 5: High-resolution correlation matrix."""
    num_cols = [
        "price", "log_price", "accommodates", "bathrooms", "bedrooms", "beds",
        "amenities_count", "days_since_first_review", "number_of_reviews",
        "review_scores_rating", "review_scores_cleanliness", "review_scores_location",
        "review_scores_value", "occupancy_rate_30", "occupancy_rate_365"
    ]
    corr = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-0.4, vmax=1.0, center=0,
                square=True, linewidths=0.6, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"}, ax=ax)
    
    ax.set_title("Pearson Correlation Heatmap: Austin Airbnb Pricing, Features & Sentiment", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_correlation_heatmap.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def plot_seasonality(reviews_csv="data/processed/reviews_cleaned.csv"):
    """Figure 6: Historical seasonal review trends."""
    print("Computing seasonal review metrics...")
    df_rev = pd.read_csv(reviews_csv, usecols=["date"])
    df_rev["date"] = pd.to_datetime(df_rev["date"], errors="coerce")
    df_rev = df_rev.dropna(subset=["date"])
    df_rev["year"] = df_rev["date"].dt.year
    df_rev["month"] = df_rev["date"].dt.month
    
    # Filter 2021-2025
    df_sub = df_rev[(df_rev["year"] >= 2021) & (df_rev["year"] <= 2025)]
    pivot_season = df_sub.groupby(["month", "year"]).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    colors = ["#A29BFE", "#74B9FF", "#00CEC9", "#0984E3", "#6C5CE7"]
    for idx, year in enumerate(pivot_season.columns):
        ax.plot(months, pivot_season[year], marker="o", linewidth=2.2, label=str(year), color=colors[idx % len(colors)])
        
    # Annotate twin peaks
    oct_mean = pivot_season.loc[10].mean()
    mar_mean = pivot_season.loc[3].mean()
    ax.annotate("Peak 1: SXSW\n(March Surge)", xy=("Mar", mar_mean + 1000), xytext=("Feb", mar_mean + 2800),
                arrowprops=dict(facecolor="#FF7675", shrink=0.08, width=1.5, headwidth=7),
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#FFF0F0", ec="#FF7675"))
    
    ax.annotate("Peak 2: Austin City Limits & F1 USGP\n(Annual High: October)", xy=("Oct", oct_mean + 1000), xytext=("Aug", oct_mean + 2800),
                arrowprops=dict(facecolor="#D63031", shrink=0.08, width=1.5, headwidth=7),
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#FFF0F0", ec="#D63031"))
    
    ax.set_title("Austin Tourism Seasonality: Monthly Guest Reviews (2021–2025)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Calendar Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Monthly Guest Reviews Submitted", fontsize=11, fontweight="bold")
    ax.legend(title="Calendar Year", loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_seasonality_timeseries.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def plot_geospatial_price_map(df):
    """Figure 7: Geospatial mapping of Austin listings."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Clip coordinates to Austin metro bounding box
    austin_df = df[
        (df["latitude"].between(30.1, 30.55)) & 
        (df["longitude"].between(-98.05, -97.55))
    ].copy()
    
    scatter = ax.scatter(
        austin_df["longitude"], austin_df["latitude"],
        c=austin_df["price"], cmap="plasma",
        s=austin_df["accommodates"] * 4, alpha=0.65,
        edgecolors="none", vmin=50, vmax=800
    )
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Nightly Price ($ USD)", fontsize=11, fontweight="bold")
    
    # Key location annotations
    ax.annotate("Downtown Austin (78701)\nHigh Density / Premium", xy=(-97.7431, 30.2672), xytext=(-97.68, 30.29),
                arrowprops=dict(facecolor="#2D3436", shrink=0.08, width=1.2, headwidth=6),
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2D3436", alpha=0.9))
    
    ax.annotate("West Lake Hills (78746)\nLuxury Hill Country", xy=(-97.80, 30.29), xytext=(-97.98, 30.34),
                arrowprops=dict(facecolor="#2D3436", shrink=0.08, width=1.2, headwidth=6),
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2D3436", alpha=0.9))
    
    ax.annotate("South Congress / Zilker (78704)\nTourism & Leisure Epicenter", xy=(-97.76, 30.245), xytext=(-97.92, 30.21),
                arrowprops=dict(facecolor="#2D3436", shrink=0.08, width=1.2, headwidth=6),
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2D3436", alpha=0.9))
    
    ax.set_title("Geospatial Map: Austin Airbnb Price and Capacity Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude (°W)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_geospatial_price_map.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def plot_occupancy_vs_price_curve(df):
    """Figure 8: Forward Occupancy Rate vs Price Deciles."""
    df_valid = df[df["occupancy_rate_30"].notnull()].copy()
    df_valid["price_decile"] = pd.qcut(df_valid["price"], q=10, labels=[f"D{i}" for i in range(1, 11)])
    
    decile_stats = df_valid.groupby("price_decile").agg(
        min_p=("price", "min"),
        max_p=("price", "max"),
        avg_p=("price", "mean"),
        occ_30=("occupancy_rate_30", "mean"),
        occ_90=("occupancy_rate_90", "mean"),
        occ_365=("occupancy_rate_365", "mean")
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(11, 6))
    
    x = range(len(decile_stats))
    ax1.plot(x, decile_stats["occ_30"] * 100, marker="o", linewidth=2.5, color="#00CEC9", label="30-Day Forward Occupancy (%)")
    ax1.plot(x, decile_stats["occ_90"] * 100, marker="s", linestyle="--", linewidth=2.0, color="#0984E3", label="90-Day Forward Occupancy (%)")
    ax1.plot(x, decile_stats["occ_365"] * 100, marker="^", linestyle=":", linewidth=1.8, color="#6C5CE7", label="365-Day Forward Occupancy (%)")
    
    ax1.set_xticks(x)
    labels = [f"{row['price_decile']}\n(${row['min_p']:.0f}–${row['max_p']:.0f})" for _, row in decile_stats.iterrows()]
    ax1.set_xticklabels(labels, fontsize=9)
    
    ax1.set_title("Forward Occupancy Rate across Price Deciles (Demand Elasticity)", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Price Decile (Nightly Rate Range)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Occupancy Proxy Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(20, 65)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_occupancy_vs_price_curve.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[SAVED] {path}")

def run_all_eda():
    print("Starting EDA generation...")
    df, df_cal = load_data()
    print(f"Loaded {len(df):,} cleaned listings merged with calendar occupancy.")
    
    plot_price_by_room_type(df)
    plot_top_neighbourhoods(df)
    plot_capacity_vs_price(df)
    plot_superhost_comparison(df)
    plot_correlation_heatmap(df)
    plot_seasonality()
    plot_geospatial_price_map(df)
    plot_occupancy_vs_price_curve(df)
    print("\nALL 8 EDA FIGURES GENERATED SUCCESSFULLY IN reports/figures/!")

if __name__ == "__main__":
    run_all_eda()
