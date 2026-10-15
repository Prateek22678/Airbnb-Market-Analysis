# 🚀 Deployment & Collaboration Guide

This guide provides step-by-step instructions for:
1. Pushing the local repository to a remote GitHub account.
2. Deploying the interactive dashboard to **Streamlit Community Cloud** (free tier).
3. Continuous Integration (CI) configuration with GitHub Actions.
4. **5 Production GitHub Issues** demonstrating agile backlog grooming and collaborative development practices.

---

## 1. Pushing to GitHub

### Step 1.1: Create an Empty Repository on GitHub
1. Log in to [GitHub](https://github.com/).
2. Click **New Repository** (`+` icon at the top right).
3. Repository Name: `austin-airbnb-market-analysis`
4. Description: `Production Data Analytics & ML System for Austin Airbnb Pricing Drivers and Guest Sentiment`
5. Visibility: **Public** (recommended for portfolio visibility).
6. **Do NOT initialize** with a README, `.gitignore`, or license (we already created them locally).
7. Click **Create repository**.

### Step 1.2: Add Remote and Push Local Code
Open PowerShell in your project root directory (`c:\Users\Asus\Downloads\Airbnb Project`) and execute:

```powershell
# 1. Verify you are on the main branch
git checkout main

# 2. Add your GitHub remote URL (replace with your GitHub username)
git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/austin-airbnb-market-analysis.git

# 3. Rename branch to main if not already named main
git branch -M main

# 4. Push your code and commit history to GitHub
git push -u origin main
```

*(Note: When prompted, authenticate using your GitHub Personal Access Token or browser OAuth).*

---

## 2. Deploying to Streamlit Community Cloud

The Streamlit dashboard can be hosted live on the web for free using **Streamlit Community Cloud**, allowing recruiters, hiring managers, and stakeholders to test the interactive 3D map and live ML simulator directly in their browser.

### Step-by-Step Deployment:
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click the **"New app"** button.
3. Select your repository: `<YOUR-GITHUB-USERNAME>/austin-airbnb-market-analysis`
4. Specify branch: `main`
5. Main file path: `dashboard/app.py`
6. App URL (optional custom subdomain): e.g., `austin-airbnb-analytics.streamlit.app`
7. Click **Advanced settings**:
   - Python Version: Select **`3.12`**
8. Click **Deploy!**

Streamlit Cloud will provision a container, install dependencies from `requirements.txt`, and launch the web app. Within 2–3 minutes, your interactive dashboard will be live on a public URL.

---

## 3. Automated Continuous Integration (CI)

A production CI pipeline is pre-configured in `.github/workflows/ci.yml`.
Every time code is pushed or a pull request is opened against `main`, GitHub Actions will automatically:
- Provision Ubuntu runner environments for Python 3.11 and 3.12.
- Cache and install dependencies from `requirements.txt`.
- Execute `pytest -v` across all 91 unit and integration tests.
- Report a green build check badge on your pull requests.

---

## 4. Collaborative Development: 5 Production GitHub Issues

To demonstrate industry collaboration habits (issue tracking, prioritization, user stories, and acceptance criteria), you can create the following 5 issues on your GitHub repository.

---

### Issue #1: Feature: Integrate Real-Time Weather API to Quantify Heat Wave Impact on Review Sentiment
- **Labels**: `enhancement`, `nlp`, `data-engineering`
- **Priority**: Medium
- **Milestone**: v1.2 Release

#### Problem Statement
Austin experiences severe summer heat waves (often exceeding 100°F / 38°C for consecutive weeks). Topic modeling revealed that "Maintenance & Property Issues" (specifically HVAC failures) drive 82% of negative reviews. However, the current dataset does not link review dates directly with historical meteorological temperature anomalies.

#### User Story
*As an operations manager, I want to correlate daily historical temperature anomalies with review sentiment drops so that I can recommend proactive AC servicing schedules to hosts before high-heat weeks.*

#### Acceptance Criteria
- [ ] Ingest daily historical weather data for Austin (NOAA / Open-Meteo API) for 2024–2026.
- [ ] Join ambient high temperature onto review dates in `reviews_cleaned.csv`.
- [ ] Measure Pearson/Spearman correlation between temperature anomalies and daily mean sentiment.
- [ ] Add an interactive "Weather Shock" filter to the Streamlit Review Intelligence tab.

---

### Issue #2: Performance: Migrate Analytical Database from SQLite to DuckDB & Parquet
- **Labels**: `performance`, `database`, `refactor`
- **Priority**: High
- **Milestone**: v1.1 Release

#### Problem Statement
While SQLite (`airbnb_austin.db`, 227 MB) provides standard relational persistence, columnar OLAP operations (such as rolling 30-day aggregations across 4.05M calendar rows) require 3.2 seconds. Columnar storage via DuckDB and Parquet will accelerate query execution by 10x–50x and reduce disk footprint.

#### User Story
*As a data analyst, I want to query columnar Parquet partitions using DuckDB so that analytical queries and dashboard aggregations execute in under 100 milliseconds.*

#### Acceptance Criteria
- [ ] Export `listings`, `calendar`, and `reviews` to Snappy-compressed Parquet files in `data/processed/`.
- [ ] Implement DuckDB connection helper in `src/duckdb_layer.py`.
- [ ] Benchmark query execution times between SQLite and DuckDB on complex CTEs.
- [ ] Document performance speedup in `reports/benchmark_report.md`.

---

### Issue #3: ML: Implement SHAP TreeExplainer for Granular Local Price Attribution
- **Labels**: `machine-learning`, `explainability`, `ui/ux`
- **Priority**: Medium
- **Milestone**: v1.2 Release

#### Problem Statement
The current Streamlit ML price simulator provides global feature importance and a point prediction with confidence bounds, but does not display a waterfall breakdown explaining *why* a specific property was priced at that rate (e.g., "+$45 for pool, -$20 for distance").

#### User Story
*As a prospective host using the simulator, I want to view a personalized SHAP waterfall plot so that I understand exactly how each of my property's amenities and attributes adds or subtracts dollars from my suggested price.*

#### Acceptance Criteria
- [ ] Integrate `shap.TreeExplainer` into `src/modeling.py` for the XGBoost model.
- [ ] Generate waterfall plot visualization function using `matplotlib` or `plotly`.
- [ ] Embed the dynamic SHAP waterfall chart into the Streamlit "Live Price Predictor" tab.
- [ ] Ensure local SHAP inference adds less than 150ms to simulator interaction.

---

### Issue #4: DevOps: Add Pre-Commit Hooks & Flake8 Linting Pipeline
- **Labels**: `devops`, `code-quality`, `tooling`
- **Priority**: Low
- **Milestone**: v1.1 Release

#### Problem Statement
To maintain code quality across multiple contributors, code style enforcement (PEP 8, Black, Flake8) should be automated before commits are recorded.

#### User Story
*As a team contributor, I want automated git pre-commit hooks to format code and check for lint errors before pushing so that repo code remains uniform and error-free.*

#### Acceptance Criteria
- [ ] Add `.pre-commit-config.yaml` with Black, Flake8, and isort hooks.
- [ ] Configure `setup.cfg` or `pyproject.toml` with line length limit = 100.
- [ ] Add Flake8 step to `.github/workflows/ci.yml`.
- [ ] Update `CONTRIBUTING.md` with developer setup instructions (`pre-commit install`).

---

### Issue #5: Product: Add Forward Occupancy Probability Classification Model
- **Labels**: `machine-learning`, `product-feature`
- **Priority**: Medium
- **Milestone**: v1.3 Release

#### Problem Statement
Currently, pricing recommendations are modeled as regression on listing rates. However, hosts need to balance nightly price against booking probability (the demand elasticity curve) to maximize total Expected Revenue ($\text{Rate} \times P(\text{Booked})$).

#### User Story
*As a host, I want to see the estimated booking probability of my listing at different price points so that I do not overprice my property during off-peak weeks and risk 0% occupancy.*

#### Acceptance Criteria
- [ ] Construct binary classification target: `is_booked_next_30_days` from calendar data.
- [ ] Train a LightGBM/Logistic Regression booking probability model.
- [ ] Calibrate probabilities using Platt scaling or Isotonic Regression.
- [ ] Add an Expected Revenue Optimization curve to the Streamlit simulator.
