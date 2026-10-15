"""
tests/test_nlp.py
Unit tests for review text sanitization, VADER sentiment scoring,
and processed sentiment data integrity.
"""

import os
import pytest
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.nlp import clean_text_fast


# =====================================================================
# 1. Text Preprocessing Unit Tests
# =====================================================================

class TestTextCleaning:
    """Test suite for rapid review text regex cleaning."""

    def test_strip_html_tags(self):
        raw = "Great location!<br/><br/>Very clean and host was awesome.<p>5 stars</p>"
        cleaned = clean_text_fast(raw)
        assert "<br/>" not in cleaned
        assert "<p>" not in cleaned
        assert "</p>" not in cleaned
        assert "great location" in cleaned

    def test_strip_digits_and_symbols(self):
        raw = "10/10 stay!!! Loved the pool & downtown vibe ($15 Uber)."
        cleaned = clean_text_fast(raw)
        assert "10" not in cleaned
        assert "!" not in cleaned
        assert "&" not in cleaned
        assert "$" not in cleaned
        assert "loved the pool" in cleaned

    def test_non_string_inputs(self):
        assert clean_text_fast(None) == ""
        assert clean_text_fast(np.nan) == ""
        assert clean_text_fast(12345) == ""


# =====================================================================
# 2. VADER Sentiment Scoring Sanity Tests
# =====================================================================

class TestSentimentScoring:
    """Test suite for polarity scoring sanity."""

    @pytest.fixture(scope="class")
    def sia(self):
        return SentimentIntensityAnalyzer()

    def test_positive_review(self, sia):
        review = "Absolutely phenomenal stay! The house was spotless, beds were super comfy, and the pool was heavenly."
        scores = sia.polarity_scores(review)
        assert scores['compound'] > 0.6, f"Expected strong positive compound score, got {scores['compound']}"

    def test_negative_review(self, sia):
        review = "Horrible experience. The air conditioner broke, water was brown, and the host refused to respond."
        scores = sia.polarity_scores(review)
        assert scores['compound'] < -0.3, f"Expected negative compound score, got {scores['compound']}"


# =====================================================================
# 3. Processed Sentiment Dataset Integrity
# =====================================================================

class TestSentimentDataIntegrity:
    """Integration assertions verifying output listings_sentiment.csv."""

    @pytest.fixture(scope="class")
    def sentiment_data(self):
        csv_path = os.path.join("data", "processed", "listings_sentiment.csv")
        if not os.path.exists(csv_path):
            pytest.skip(f"Sentiment dataset not found at {csv_path}")
        return pd.read_csv(csv_path)

    def test_required_columns(self, sentiment_data):
        required = [
            'listing_id', 'sentiment_mean', 'sentiment_median', 'sentiment_std',
            'pct_positive', 'pct_negative', 'pct_neutral', 'reviews_analyzed_count'
        ]
        for col in required:
            assert col in sentiment_data.columns, f"Missing column {col} in listings_sentiment.csv"

    def test_sentiment_bounds(self, sentiment_data):
        assert sentiment_data['sentiment_mean'].between(-1.0, 1.0).all(), "Sentiment mean outside [-1.0, 1.0]"
        assert sentiment_data['listing_id'].is_unique, "Sentiment listing_ids must be unique"
