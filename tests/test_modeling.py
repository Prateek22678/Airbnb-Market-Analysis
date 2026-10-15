"""
tests/test_modeling.py
Unit tests for regression metrics calculation, model artifact persistence,
inference pipelines, and sensitivity sanity checks.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from src.modeling import calculate_metrics
from src.features import engineer_features


# =====================================================================
# 1. Metric Calculation Unit Tests
# =====================================================================

class TestMetricCalculation:
    """Test suite for evaluation metric computations (log and dollar scales)."""

    def test_perfect_predictions(self):
        y_true = np.array([4.5, 5.0, 5.5, 6.0])
        y_pred = np.array([4.5, 5.0, 5.5, 6.0])
        metrics = calculate_metrics(y_true, y_pred)

        assert np.isclose(metrics['R2_log'], 1.0)
        assert np.isclose(metrics['RMSE_log'], 0.0)
        assert np.isclose(metrics['MAE_log'], 0.0)
        assert np.isclose(metrics['MAE_dollar'], 0.0)
        assert np.isclose(metrics['MedAE_dollar'], 0.0)
        assert np.isclose(metrics['RMSE_dollar'], 0.0)
        assert np.isclose(metrics['MAPE_pct'], 0.0)

    def test_nonzero_residuals(self):
        y_true = np.array([5.0, 5.5])
        y_pred = np.array([5.1, 5.3])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics['MAE_log'] > 0.0
        assert metrics['RMSE_log'] > 0.0
        assert metrics['MAE_dollar'] > 0.0
        assert metrics['MAPE_pct'] > 0.0


# =====================================================================
# 2. Saved Model Artifact & Inference Integrity
# =====================================================================

class TestModelArtifactAndInference:
    """Test suite verifying saved XGBoost model bundle, preprocessing, and predictions."""

    @pytest.fixture(scope="class")
    def model_bundle(self):
        bundle_path = os.path.join("models", "price_model_xgboost.pkl")
        if not os.path.exists(bundle_path):
            pytest.skip(f"Model bundle not found at {bundle_path}")
        return joblib.load(bundle_path)

    def test_bundle_structure_and_keys(self, model_bundle):
        assert isinstance(model_bundle, dict)
        for key in ['model', 'preprocessor', 'feature_names', 'test_metrics']:
            assert key in model_bundle, f"Missing key '{key}' in model bundle"

    def test_persisted_performance_metrics(self, model_bundle):
        metrics = model_bundle['test_metrics']
        # Model must meet portfolio threshold: R2 >= 0.70, MedAE <= $50
        assert metrics['Test_R2'] >= 0.70, f"R2 {metrics['Test_R2']} below portfolio standard 0.70"
        assert metrics['Test_MedAE_USD'] <= 50.0, f"Median AE ${metrics['Test_MedAE_USD']} exceeds $50"

    def test_model_inference_pipeline(self, model_bundle, sample_cleaned_df):
        model = model_bundle['model']
        preprocessor = model_bundle['preprocessor']

        df_feat = engineer_features(sample_cleaned_df)
        X_trans = preprocessor.transform(df_feat)
        
        preds_log = model.predict(X_trans)
        assert len(preds_log) == len(df_feat)
        assert np.isfinite(preds_log).all(), "Predictions contain non-finite values (NaN or Inf)"

        # Exponentiated dollar prices
        preds_dollar = np.exp(preds_log)
        assert (preds_dollar > 20.0).all(), "All predicted prices should exceed $20"
        assert (preds_dollar < 5000.0).all(), "Predictions should be within realistic lodging boundaries"

    def test_capacity_sensitivity_check(self, model_bundle, sample_cleaned_df):
        """Sanity check: Larger capacity homes should predict higher rates."""
        model = model_bundle['model']
        preprocessor = model_bundle['preprocessor']

        base_listing = sample_cleaned_df.iloc[[0]].copy()
        small_unit = base_listing.copy()
        small_unit['accommodates'] = 2
        small_unit['bedrooms'] = 1.0
        small_unit['beds'] = 1.0

        large_unit = base_listing.copy()
        large_unit['accommodates'] = 8
        large_unit['bedrooms'] = 4.0
        large_unit['beds'] = 5.0

        small_feat = engineer_features(small_unit)
        large_feat = engineer_features(large_unit)

        pred_small = np.exp(model.predict(preprocessor.transform(small_feat))[0])
        pred_large = np.exp(model.predict(preprocessor.transform(large_feat))[0])

        assert pred_large > pred_small, (
            f"Sensitivity test failed: 4-bed unit (${pred_large:.2f}) was not priced higher than 1-bed unit (${pred_small:.2f})"
        )

    def test_pool_amenity_premium_sensitivity_check(self, model_bundle, sample_cleaned_df):
        """Sanity check: Adding a pool should increase or maintain listing rate."""
        model = model_bundle['model']
        preprocessor = model_bundle['preprocessor']

        no_pool = sample_cleaned_df.iloc[[1]].copy()
        no_pool['has_pool'] = 0

        with_pool = sample_cleaned_df.iloc[[1]].copy()
        with_pool['has_pool'] = 1

        pred_no_pool = np.exp(model.predict(preprocessor.transform(engineer_features(no_pool)))[0])
        pred_with_pool = np.exp(model.predict(preprocessor.transform(engineer_features(with_pool)))[0])

        assert pred_with_pool >= pred_no_pool, (
            f"Pool amenity premium violated: with pool (${pred_with_pool:.2f}) < without pool (${pred_no_pool:.2f})"
        )
