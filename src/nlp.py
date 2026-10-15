"""
src/nlp.py
Natural Language Processing & Sentiment Analysis Pipeline for Austin Airbnb Reviews.
Includes:
  1. Review text preprocessing and cleaning.
  2. VADER Sentiment Analysis (review-level and listing-level aggregation).
  3. Latent Dirichlet Allocation (LDA) Topic Modeling (5 distinct guest experience themes).
  4. Publication-grade visual figures and listing-level CSV export.
"""

import os
import sys
import re
from typing import Dict, List, Tuple, Any

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

# Configuration & Paths
REVIEWS_PATH = "data/processed/reviews_cleaned.csv"
LISTINGS_PATH = "data/processed/listings_cleaned.csv"
OUTPUT_SENTIMENT_CSV = "data/processed/listings_sentiment.csv"
FIGURES_DIR = "reports/figures"
RANDOM_STATE = 42

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
CORAL = '#FF5A5F'
NAVY = '#003580'
TEAL = '#00A699'
GOLD = '#FC642D'
GRAY = '#484848'

# Domain Stopwords to prevent generic adjectives from swamping thematic topics
DOMAIN_STOPWORDS = {
    'austin', 'stay', 'place', 'host', 'house', 'airbnb', 'room', 'home',
    'stayed', 'would', 'really', 'night', 'time', 'also', 'great', 'nice', 'good',
    'br', 'just', 'like', 'clean', 'perfect', 'recommend', 'definitely', 'highly',
    'super', 'amazing', 'everything', 'well', 'loved', 'enjoyed', 'felt', 'made',
    'location', 'comfortable', 'hosts', 'barbara', 'lois', 'peter', 'chris', 'thank',
    'thanks', 'us', 'we', 'our', 'had', 'were', 'was', 'did', 'very', 'one', 'two'
}


def clean_text_fast(text: str) -> str:
    """
    Rapid cleaning of review text:
    removes HTML line breaks, special characters, and digits.
    """
    if not isinstance(text, str):
        return ""
    # Strip HTML tags like <br/>
    text = re.sub(r'<[^>]+>', ' ', text)
    # Retain only alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower().strip()


def run_sentiment_analysis(
    df_reviews: pd.DataFrame,
    max_reviews_per_listing: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run VADER sentiment scoring on reviews, sampling up to max_reviews_per_listing
    most recent reviews per listing for balanced representation and speed.
    """
    print(f"Sampling up to {max_reviews_per_listing} most recent reviews per listing...")
    df_sample = df_reviews.groupby('listing_id').tail(max_reviews_per_listing).copy()
    print(f"Scoring {len(df_sample):,} reviews across {df_sample['listing_id'].nunique():,} unique listings...")

    sia = SentimentIntensityAnalyzer()
    
    # Compute compound sentiment scores
    df_sample['compound'] = [sia.polarity_scores(str(t))['compound'] for t in df_sample['comments']]
    
    # Assign discrete sentiment categories
    conditions = [
        df_sample['compound'] >= 0.05,
        df_sample['compound'] <= -0.05
    ]
    choices = ['Positive', 'Negative']
    df_sample['sentiment_category'] = np.select(conditions, choices, default='Neutral')

    # Aggregate to listing level
    print("Aggregating sentiment metrics at the listing level...")
    df_agg = df_sample.groupby('listing_id').agg(
        sentiment_mean=('compound', 'mean'),
        sentiment_median=('compound', 'median'),
        sentiment_std=('compound', 'std'),
        pct_positive=('compound', lambda s: float((s >= 0.05).mean() * 100.0)),
        pct_negative=('compound', lambda s: float((s <= -0.05).mean() * 100.0)),
        pct_neutral=('compound', lambda s: float(((s > -0.05) & (s < 0.05)).mean() * 100.0)),
        reviews_analyzed_count=('compound', 'count')
    ).reset_index()

    # Fill std for single-review listings
    df_agg['sentiment_std'] = df_agg['sentiment_std'].fillna(0.0)

    return df_sample, df_agg


def run_topic_modeling(
    df_sample: pd.DataFrame,
    n_topics: int = 5,
    sample_size: int = 30000
) -> Tuple[LatentDirichletAllocation, CountVectorizer, Dict[int, Dict[str, Any]]]:
    """
    Fit an online Latent Dirichlet Allocation (LDA) topic model to extract
    interpretable guest experience themes.
    """
    print(f"Fitting LDA Topic Model on {min(sample_size, len(df_sample)):,} sample reviews...")
    sample_texts = df_sample['comments'].dropna().sample(
        n=min(sample_size, len(df_sample)), random_state=RANDOM_STATE
    ).apply(clean_text_fast)

    all_stopwords = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))

    vectorizer = CountVectorizer(
        max_features=800,
        min_df=15,
        max_df=0.25,
        stop_words=all_stopwords
    )
    dtm = vectorizer.fit_transform(sample_texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online',
        batch_size=512,
        max_iter=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    lda.fit(dtm)

    vocab = vectorizer.get_feature_names_out()

    # Human-readable theme titles mapped to topic indices
    theme_names = [
        "1. Maintenance & Property Issues",
        "2. Space, Pool & Family Amenities",
        "3. Guest Experience & Repeat Booking",
        "4. Host Responsiveness & Easy Check-In",
        "5. Location, Downtown & Dining"
    ]

    topic_dict = {}
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [vocab[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        
        # Normalize weights for visualization
        norm_weights = [w / sum(top_weights) for w in top_weights]

        topic_dict[idx] = {
            'theme_title': theme_names[idx] if idx < len(theme_names) else f"Topic {idx + 1}",
            'words': top_words,
            'weights': norm_weights
        }

    return lda, vectorizer, topic_dict


def plot_sentiment_distribution(df_merged: pd.DataFrame, out_path: str):
    """
    Generate 2-panel chart:
      Panel 1: Overall distribution of listing-level compound sentiment.
      Panel 2: Superhost vs. Regular Host sentiment comparison (Violin + Boxplot).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Distribution
    ax1 = axes[0]
    sns.histplot(df_merged['sentiment_mean'], kde=True, color=NAVY, bins=35, ax=ax1, alpha=0.45)
    mean_val = df_merged['sentiment_mean'].mean()
    median_val = df_merged['sentiment_mean'].median()
    ax1.axvline(mean_val, color=CORAL, linestyle='--', linewidth=2, label=f'Mean Sentiment ({mean_val:.2f})')
    ax1.axvline(median_val, color=TEAL, linestyle='-', linewidth=2, label=f'Median Sentiment ({median_val:.2f})')
    ax1.set_title("1. Listing Compound Sentiment Distribution", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Average Review Sentiment Compound Score (-1.0 to +1.0)", fontsize=10)
    ax1.set_ylabel("Listing Count", fontsize=10)
    ax1.legend(loc='upper left', frameon=True)

    # 2. Superhost vs Regular Host
    ax2 = axes[1]
    df_merged['host_label'] = df_merged['host_is_superhost'].map({1: 'Superhost (N=3,790)', 0: 'Regular Host (N=4,554)'})
    sns.violinplot(
        x='host_label', y='sentiment_mean', hue='host_label', data=df_merged,
        palette=[CORAL, NAVY], ax=ax2, inner='box', cut=0, legend=False
    )
    sh_mean = df_merged[df_merged['host_is_superhost'] == 1]['sentiment_mean'].mean()
    reg_mean = df_merged[df_merged['host_is_superhost'] == 0]['sentiment_mean'].mean()
    ax2.set_title(f"2. Sentiment: Superhost ({sh_mean:.3f}) vs Regular ({reg_mean:.3f})", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Host Status", fontsize=10)
    ax2.set_ylabel("Mean Review Sentiment (Compound)", fontsize=10)

    plt.suptitle("Austin Airbnb Review Sentiment Analysis (VADER)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Sentiment distribution figure saved to {out_path}")


def plot_topic_keywords(topic_dict: Dict[int, Dict[str, Any]], out_path: str):
    """
    Generate 5-panel horizontal bar chart showing top keywords and weights for each LDA topic.
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 5.5), dpi=300, sharex=True)
    colors = [CORAL, NAVY, TEAL, GOLD, '#6C5CE7']

    for idx, ax in enumerate(axes):
        info = topic_dict[idx]
        words = info['words'][:8][::-1]
        weights = info['weights'][:8][::-1]

        bars = ax.barh(words, weights, color=colors[idx % len(colors)], height=0.65)
        ax.set_title(info['theme_title'], fontsize=9.5, fontweight='bold', pad=10)
        ax.set_xlabel("Relative Word Weight", fontsize=8.5)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    plt.suptitle("Austin Airbnb: 5 Latent Topics in Guest Reviews (LDA Topic Model)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Topic keywords figure saved to {out_path}")


def plot_sentiment_vs_ratings(df_merged: pd.DataFrame, out_path: str):
    """
    Generate 2-panel chart linking unstructured sentiment with structured ratings:
      Panel 1: Scatter plot with Lowess trend line.
      Panel 2: Boxplot by Rating Tier showing sentiment nuance below 4.9.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Scatter & Trend
    ax1 = axes[0]
    sns.regplot(
        x='review_scores_rating', y='sentiment_mean', data=df_merged,
        scatter_kws={'alpha': 0.25, 'color': NAVY, 's': 15},
        line_kws={'color': CORAL, 'linewidth': 2.5},
        lowess=True, ax=ax1
    )
    corr = df_merged['sentiment_mean'].corr(df_merged['review_scores_rating'])
    ax1.set_title(f"1. Sentiment vs. Review Rating Score (r = {corr:.3f})", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Overall Review Rating (0.00 to 5.00 Scale)", fontsize=10)
    ax1.set_ylabel("Mean Review Sentiment Compound Score", fontsize=10)
    ax1.set_xlim(3.8, 5.05)

    # 2. Rating Tiers vs Sentiment
    ax2 = axes[1]
    bins = [0.0, 4.79, 4.89, 4.99, 5.01]
    labels = ['< 4.80 (At Risk)', '4.80 - 4.89 (Standard)', '4.90 - 4.99 (High)', '5.00 (Flawless)']
    df_merged['rating_tier'] = pd.cut(df_merged['review_scores_rating'], bins=bins, labels=labels)

    sns.boxplot(
        x='rating_tier', y='sentiment_mean', hue='rating_tier',
        data=df_merged.dropna(subset=['rating_tier']),
        palette=[GOLD, TEAL, NAVY, CORAL], ax=ax2, width=0.55, legend=False
    )
    ax2.set_title("2. Review Sentiment by Rating Tier", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Rating Tier", fontsize=10)
    ax2.set_ylabel("Mean Review Sentiment Compound Score", fontsize=10)

    plt.suptitle("Validating Guest Satisfaction: Text Sentiment vs. Numerical Ratings", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Sentiment vs ratings figure saved to {out_path}")


def main():
    """
    Execute full Phase 7 NLP and Sentiment pipeline.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_SENTIMENT_CSV), exist_ok=True)

    print(f"Loading cleaned reviews from {REVIEWS_PATH}...")
    df_reviews = pd.read_csv(REVIEWS_PATH)
    print(f"Loaded {len(df_reviews):,} total reviews.")

    # 1. Sentiment Scoring & Aggregation
    df_sample, df_agg = run_sentiment_analysis(df_reviews, max_reviews_per_listing=20)

    # Save listing sentiment
    df_agg.to_csv(OUTPUT_SENTIMENT_CSV, index=False)
    print(f"Aggregated listing sentiment saved to {OUTPUT_SENTIMENT_CSV} ({len(df_agg):,} listings).")

    # 2. Merge with Listings for Analysis
    print(f"Merging with cleaned listings from {LISTINGS_PATH}...")
    df_listings = pd.read_csv(LISTINGS_PATH)
    df_merged = pd.merge(df_listings, df_agg, left_on='id', right_on='listing_id', how='inner')
    print(f"Merged dataset contains {len(df_merged):,} listings with full review sentiment.")

    # 3. Topic Modeling
    lda, vectorizer, topic_dict = run_topic_modeling(df_sample, n_topics=5, sample_size=30000)

    # 4. Generate Figures
    fig1_path = os.path.join(FIGURES_DIR, "05_sentiment_distribution.png")
    fig2_path = os.path.join(FIGURES_DIR, "05_topic_keywords.png")
    fig3_path = os.path.join(FIGURES_DIR, "05_sentiment_vs_ratings.png")

    plot_sentiment_distribution(df_merged, fig1_path)
    plot_topic_keywords(topic_dict, fig2_path)
    plot_sentiment_vs_ratings(df_merged, fig3_path)

    # Empirical Summary Print
    print("\n" + "=" * 60)
    print("PHASE 7 EMPIRICAL NLP & SENTIMENT SUMMARY (AUSTIN, TX)")
    print("=" * 60)
    print(f"Total Reviews Scored:                {len(df_sample):,}")
    print(f"Listings Evaluated:                  {len(df_merged):,}")
    print(f"Overall Mean Sentiment Score:        {df_merged['sentiment_mean'].mean():.4f}")
    print(f"Superhost Mean Sentiment:            {df_merged[df_merged['host_is_superhost'] == 1]['sentiment_mean'].mean():.4f}")
    print(f"Regular Host Mean Sentiment:          {df_merged[df_merged['host_is_superhost'] == 0]['sentiment_mean'].mean():.4f}")
    print(f"Correlation (Sentiment vs Rating):   {df_merged['sentiment_mean'].corr(df_merged['review_scores_rating']):.4f}")
    print(f"Positive Reviews (% >= +0.05):       {df_sample['compound'].ge(0.05).mean() * 100:.2f}%")
    print(f"Negative Reviews (% <= -0.05):       {df_sample['compound'].le(-0.05).mean() * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
