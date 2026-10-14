# Airbnb Market Analysis: Pricing Drivers and Guest Sentiment in Austin, TX

A production-grade, end-to-end data analytics portfolio project examining the pricing dynamics, revenue drivers, and guest sentiment in Austin, Texas.

---

## 🎯 Executive Overview & Business Question
**Core Business Question:** *What drives listing prices and guest satisfaction in Austin, and what actionable operational steps should hosts and platform managers take to maximize revenue and guest retention?*

This project demonstrates core competencies across the modern data analyst lifecycle:
1. **Data Engineering & Cleaning**: Robust ETL on messy real-world structured and unstructured data.
2. **Relational Database & SQL Layer**: SQLite analytical database leveraging multi-table `JOIN`s, `GROUP BY`, CTEs, and Window Functions.
3. **Exploratory Data Analysis (EDA)**: Geospatial mapping, seasonal trends, and commercial distribution insights.
4. **Statistical Rigor**: Parametric and non-parametric hypothesis testing (ANOVA, Mann-Whitney U, Spearman correlation) with effect size calculations.
5. **Machine Learning**: Supervised log-price prediction (Linear Regression, Random Forest, XGBoost) with cross-validation and feature importance attribution.
6. **Natural Language Processing (NLP)**: Unstructured review sentiment analysis (VADER) and Latent Dirichlet Allocation (LDA) topic modeling.
7. **Interactive Web Dashboard**: Streamlit decision-support application for hosts and investors.

---

## 📊 Dataset Provenance
- **Source**: [Inside Airbnb](http://insideairbnb.com/get-the-data/)
- **Market**: Austin, Texas, United States
- **Data Snapshot Date**: September 17, 2026
- **Files Utilized**:
  - `data/raw/listings.csv.gz`: Detailed listing attributes, host metrics, and review scores.
  - `data/raw/calendar.csv.gz`: Nightly pricing and availability time-series (365 days forward).
  - `data/raw/reviews.csv.gz`: Guest review text and historical timestamps.

---

## 📁 Repository Structure
```
airbnb-market-analysis/
├── data/
│   ├── raw/                  # Compressed raw Inside Airbnb archives (untracked)
│   └── processed/            # Cleaned CSVs, parquet files, and SQLite database
├── sql/
│   ├── schema.sql            # DDL for analytical relational schema
│   └── queries.sql           # 10+ analytical queries with CTEs and Window Functions
├── notebooks/
│   ├── 01_cleaning.ipynb     # Interactive data audit and transformation
│   ├── 02_eda.ipynb          # Exploratory data analysis & visualizations
│   ├── 03_statistical_tests.ipynb # Hypothesis testing & effect size analysis
│   ├── 04_price_model.ipynb  # Machine learning pricing models
│   └── 05_review_nlp.ipynb   # Guest review sentiment & LDA topic modeling
├── src/
│   ├── cleaning.py           # Reusable data cleansing & validation utilities
│   ├── features.py           # Feature engineering pipeline
│   ├── modeling.py           # Model training, CV evaluation, and metrics
│   └── nlp.py                # Text preprocessing, VADER, and topic modeling
├── dashboard/
│   └── app.py                # Interactive Streamlit executive dashboard
├── reports/
│   ├── business_report.md    # 2-page executive summary & recommendations
│   └── figures/              # Exported production charts and plots
├── docs/
│   ├── data_dictionary.md    # Variable metadata and definitions
│   └── assumptions_log.md    # Audit trail of cleaning & modeling decisions
├── tests/
│   └── test_cleaning.py      # Automated pytest unit test suite
├── requirements.txt          # Pinned dependency manifest
├── CONTRIBUTING.md           # Branching strategy & commit standards
└── README.md                 # Primary project documentation
```

---

## 🛠️ Quickstart & Setup
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd "Airbnb Project"
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
