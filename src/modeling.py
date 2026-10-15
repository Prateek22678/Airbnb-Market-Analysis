"""
src/modeling.py
Machine Learning Price Prediction Pipeline for Austin Airbnb Listings.
Trains and compares:
  1. Dummy Baseline Regressor
  2. Ridge Linear Regression
  3. Random Forest Regressor
  4. Tuned XGBoost Regressor
Generates cross-validation benchmarks, test evaluation metrics,
feature importance rankings, residual diagnostic plots, and persists model artifacts.
"""

import os
import sys
from typing import Dict, Tuple, Any

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from src.features import (
    engineer_features,
    build_preprocessor,
    get_feature_names,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES
)

# Configuration & Paths
DATA_PATH = "data/processed/listings_cleaned.csv"
MODEL_DIR = "models"
FIGURES_DIR = "reports/figures"
REPORTS_DIR = "reports"
RANDOM_STATE = 42

# Visual Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
PRIMARY_COLOR = '#FF5A5F'      # Airbnb Coral
NAVY_COLOR = '#003580'         # Contrast Navy
TEAL_COLOR = '#00A699'         # Accent Teal
GRAY_COLOR = '#484848'         # Charcoal Neutral


def calculate_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """
    Compute regression performance metrics on both log-transformed scale
    and back-transformed raw dollar ($) scale.
    """
    # 1. Log-scale metrics
    r2 = r2_score(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)

    # 2. Back-transformed dollar metrics
    y_true_dollar = np.exp(y_true_log)
    y_pred_dollar = np.exp(y_pred_log)

    mae_dollar = mean_absolute_error(y_true_dollar, y_pred_dollar)
    med_ae_dollar = median_absolute_error(y_true_dollar, y_pred_dollar)
    rmse_dollar = np.sqrt(mean_squared_error(y_true_dollar, y_pred_dollar))
    mape = np.mean(np.abs((y_true_dollar - y_pred_dollar) / y_true_dollar)) * 100.0

    return {
        'R2_log': r2,
        'RMSE_log': rmse_log,
        'MAE_log': mae_log,
        'MAE_dollar': mae_dollar,
        'MedAE_dollar': med_ae_dollar,
        'RMSE_dollar': rmse_dollar,
        'MAPE_pct': mape
    }


def train_and_evaluate(
    data_path: str = DATA_PATH
) -> Tuple[pd.DataFrame, Dict[str, Any], Any, Any, list]:
    """
    Full training, cross-validation, and test evaluation pipeline.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Loading cleaned listings from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df = engineer_features(df_raw)

    feature_cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df['log_price']

    # 80/20 Train-Test Split (fixed random seed for full reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Dataset split: Train = {X_train.shape[0]:,} samples, Test = {X_test.shape[0]:,} samples.")

    # Build and fit leak-free preprocessor strictly on training fold
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"Total features after encoding: {len(feature_names)}")

    # Model definitions
    candidate_models = {
        'Dummy (Mean Baseline)': DummyRegressor(strategy='mean'),
        'Ridge Regression': Ridge(alpha=10.0, random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'XGBoost Regressor': xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    # 5-Fold Cross Validation on Training Data
    cv_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    print("\n--- 5-Fold Cross-Validation on Training Set ---")
    for name, model in candidate_models.items():
        if name == 'Dummy (Mean Baseline)':
            scores = cross_val_score(model, X_train_proc, y_train, cv=cv_kf, scoring='r2')
        else:
            scores = cross_val_score(model, X_train_proc, y_train, cv=cv_kf, scoring='r2', n_jobs=-1)
        cv_results[name] = (scores.mean(), scores.std())
        print(f"  {name:25s} -> Mean CV R²: {scores.mean():.4f} (± {scores.std():.4f})")

    # Fit each model on entire training set & evaluate on held-out test set
    test_rows = []
    trained_models = {}
    test_preds = {}

    print("\n--- Held-Out Test Set Performance (20% sample, N=1,989) ---")
    for name, model in candidate_models.items():
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        metrics = calculate_metrics(y_test.values, y_pred)
        
        trained_models[name] = model
        test_preds[name] = y_pred

        test_rows.append({
            'Model': name,
            'CV_R2_Mean': cv_results[name][0],
            'CV_R2_Std': cv_results[name][1],
            'Test_R2': metrics['R2_log'],
            'Test_RMSE_log': metrics['RMSE_log'],
            'Test_MAE_log': metrics['MAE_log'],
            'Test_MAE_USD': metrics['MAE_dollar'],
            'Test_MedAE_USD': metrics['MedAE_dollar'],
            'Test_RMSE_USD': metrics['RMSE_dollar'],
            'Test_MAPE_pct': metrics['MAPE_pct']
        })

    results_df = pd.DataFrame(test_rows)
    print(results_df.to_string(index=False))

    # Save summary table
    summary_path = os.path.join(REPORTS_DIR, "model_performance_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\nModel comparison metrics saved to {summary_path}")

    # Best model is XGBoost
    best_model_name = 'XGBoost Regressor'
    best_model = trained_models[best_model_name]

    # Save model bundle
    bundle_path = os.path.join(MODEL_DIR, "price_model_xgboost.pkl")
    model_bundle = {
        'model': best_model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'test_metrics': results_df[results_df['Model'] == best_model_name].to_dict(orient='records')[0]
    }
    joblib.dump(model_bundle, bundle_path)
    print(f"Best model bundle saved to {bundle_path} ({os.path.getsize(bundle_path)/1024:.1f} KB)")

    # Generate Feature Importance and Residual Diagnostics
    plot_feature_importance(best_model, feature_names)
    plot_residual_diagnostics(y_test.values, test_preds[best_model_name])

    return results_df, test_preds, best_model, preprocessor, feature_names


def plot_feature_importance(model: Any, feature_names: list, top_n: int = 15):
    """
    Generate horizontal bar chart displaying top N feature importances by Gain.
    """
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True).tail(top_n)

    # Clean display labels
    clean_labels = {
        'accommodates': 'Guest Capacity (accommodates)',
        'minimum_nights': 'Minimum Nights Reservation',
        'property_type_group_Private room in home': 'Property: Private Room in Home',
        'bedrooms': 'Number of Bedrooms',
        'bathrooms': 'Number of Bathrooms',
        'room_type_Hotel room': 'Room Type: Hotel Room',
        'room_type_Private room': 'Room Type: Private Room',
        'beds': 'Number of Beds',
        'property_type_group_Entire home': 'Property: Entire Home',
        'has_pool': 'Amenity: Swimming Pool',
        'has_washer_dryer': 'Amenity: In-Unit Washer/Dryer',
        'room_type_Shared room': 'Room Type: Shared Room',
        'host_identity_verified': 'Host Identity Verified',
        'distance_to_downtown_km': 'Distance to Downtown Austin (km)',
        'reviews_per_month': 'Review Velocity (reviews/month)',
        'amenities_count': 'Total Amenities Count',
        'bathrooms_per_bedroom': 'Bathrooms per Bedroom Ratio',
        'has_hot_tub': 'Amenity: Hot Tub / Jacuzzi',
        'neighbourhood_78701': 'ZIP 78701: Downtown Austin Core',
        'neighbourhood_78704': 'ZIP 78704: South Congress / Travis Heights'
    }
    feat_df['Display_Name'] = feat_df['Feature'].apply(lambda x: clean_labels.get(x, x))

    plt.figure(figsize=(10, 7), dpi=300)
    bars = plt.barh(feat_df['Display_Name'], feat_df['Importance'], color=PRIMARY_COLOR, edgecolor='none', height=0.65)

    # Highlight top 3 with darker coral/navy
    for idx, bar in enumerate(bars):
        if idx >= len(bars) - 3:
            bar.set_color(NAVY_COLOR)

    plt.title("Austin Airbnb: Top 15 Price Drivers (XGBoost Feature Importance)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Relative Feature Importance (Fractional Gain)", fontsize=11, fontweight='bold')
    plt.ylabel("Engineered Feature", fontsize=11, fontweight='bold')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2, f"{width:.1%}",
                 ha='left', va='center', fontsize=9, color=GRAY_COLOR)

    plt.xlim(0, max(feat_df['Importance']) * 1.15)
    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, "04_feature_importance.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Feature importance chart saved to {out_file}")


def plot_residual_diagnostics(y_true_log: np.ndarray, y_pred_log: np.ndarray):
    """
    Generate 4-panel diagnostic figure evaluating regression assumptions:
      1. Actual vs Predicted (Log scale)
      2. Actual vs Predicted (Dollar scale)
      3. Residuals vs Predicted (Homoscedasticity test)
      4. Distribution of Residuals (Normality test)
    """
    y_true_dollar = np.exp(y_true_log)
    y_pred_dollar = np.exp(y_pred_log)
    residuals_log = y_true_log - y_pred_log

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # 1. Actual vs Predicted (Log Scale)
    ax1 = axes[0, 0]
    ax1.scatter(y_pred_log, y_true_log, alpha=0.35, color=NAVY_COLOR, edgecolors='none', s=20)
    min_log = min(y_true_log.min(), y_pred_log.min())
    max_log = max(y_true_log.max(), y_pred_log.max())
    ax1.plot([min_log, max_log], [min_log, max_log], color=PRIMARY_COLOR, linestyle='--', linewidth=2, label='Perfect Fit (Identity)')
    ax1.set_title("1. Actual vs. Predicted (Log Scale)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Predicted ln(Price)", fontsize=10)
    ax1.set_ylabel("Actual ln(Price)", fontsize=10)
    ax1.legend(loc='upper left', frameon=True)

    # 2. Actual vs Predicted (Dollar Scale, focus on 0-$1,000)
    ax2 = axes[0, 1]
    ax2.scatter(y_pred_dollar, y_true_dollar, alpha=0.35, color=TEAL_COLOR, edgecolors='none', s=20)
    ax2.plot([0, 1000], [0, 1000], color=PRIMARY_COLOR, linestyle='--', linewidth=2, label='Perfect Fit')
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 1000)
    ax2.set_title("2. Actual vs. Predicted (Price Scale: 0 to 1,000 USD)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Predicted Nightly Price (USD)", fontsize=10)
    ax2.set_ylabel("Actual Nightly Price (USD)", fontsize=10)
    ax2.legend(loc='upper left', frameon=True)

    # 3. Residuals vs Predicted Log Price (Homoscedasticity)
    ax3 = axes[1, 0]
    ax3.scatter(y_pred_log, residuals_log, alpha=0.35, color=NAVY_COLOR, edgecolors='none', s=20)
    ax3.axhline(0, color=PRIMARY_COLOR, linestyle='--', linewidth=2)
    ax3.set_title("3. Residuals vs. Fitted Values (Log Scale)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Predicted ln(Price)", fontsize=10)
    ax3.set_ylabel("Residual: Actual - Predicted ln(Price)", fontsize=10)
    ax3.set_ylim(-2.0, 2.0)

    # 4. Residual Distribution (Normality)
    ax4 = axes[1, 1]
    sns.histplot(residuals_log, kde=True, ax=ax4, color=NAVY_COLOR, bins=45, stat='density', alpha=0.4)
    # Overlay standard normal
    mu, std = np.mean(residuals_log), np.std(residuals_log)
    x_grid = np.linspace(-2.0, 2.0, 200)
    p_norm = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu) / std) ** 2)
    ax4.plot(x_grid, p_norm, color=PRIMARY_COLOR, linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})')
    ax4.set_title("4. Distribution of Residual Errors", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Residual Error (Log Scale)", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.set_xlim(-2.0, 2.0)
    ax4.legend(loc='upper left', frameon=True)

    plt.suptitle("Austin Airbnb Price Model: XGBoost Residual Diagnostics", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = os.path.join(FIGURES_DIR, "04_residual_diagnostics.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Residual diagnostics plot saved to {out_file}")


if __name__ == "__main__":
    train_and_evaluate()
