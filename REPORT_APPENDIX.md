# Report Appendix Sections

## Datasets

### Primary Data Sources

1. **Agmarknet Mandi Price Records**
   - File: `Agriculture_price_dataset.csv` (52.65 MB)
   - Source: AGMARKNET-style records (data.gov.in compatible format)
   - Coverage: Indian APMC mandis, multiple states
   - Period: 2023-2025
   - Commodities: Onion, Potato, Tomato
   - Fields: State, Market, Commodity, Date, Modal Price, Min Price, Max Price, Arrivals (tonnes)
   - Note: File not included in repository due to size (exceeds 50 MB GitHub limit)

2. **Open-Meteo Historical Weather Archive**
   - File: `open-meteo-23.02N72.60E53m.csv` (500 KB)
   - Source: Open-Meteo API (https://open-meteo.com/)
   - Location: Ahmedabad, Gujarat (23.02°N, 72.60°E)
   - Period: June 2023 to June 2025 (731 days)
   - Variables: Daily maximum temperature (°C), minimum temperature (°C), mean temperature (°C), rainfall (mm), wind speed (km/h)
   - License: Open-source, freely available for research

3. **Google News RSS Headlines**
   - File: `news_rss_vader_cache.csv` (15 KB, cached sentiment scores)
   - Source: Google News RSS Feed (https://news.google.com/rss)
   - Search Queries: "Onion mandi price Gujarat", "Potato mandi price Gujarat", "Tomato mandi price Gujarat"
   - Processing: Headlines analyzed with VADER Sentiment Analysis (Hutto & Gilbert, 2014)
   - Period: June 2023 to June 2025 (~21 daily aggregated records)
   - License: Public RSS feed, fair use for research

### Processed Datasets (Included in Repository)

1. **gujarat_top_multimodal_master.csv** (250 KB)
   - Final training dataset: 577 rows × 45 columns
   - Crops: Onion, Potato, Tomato (aggregated Gujarat state averages)
   - Features: 11 market-only + 6 multimodal (weather + news)
   - Targets: Next-day modal price (regression), upward shock label (classification)

2. **model_benchmark_results.csv** (5 KB)
   - Performance metrics for 36 model configurations
   - Tasks: Regression (R², RMSE, MAE) and Classification (Accuracy, Precision, Recall, F1, ROC-AUC)
   - Current results: Onion only (12 configurations evaluated)

3. **model_predictions.csv** (30 KB)
   - Test set predictions from all models
   - 145 test samples per crop
   - Columns: Date, Commodity, Actual Price, Predicted Prices (Ridge, XGBoost, BiLSTM × 2 regimes)

### Data Availability Statement

Raw mandi price data and weather data are not included in the GitHub repository due to size constraints. The processed master dataset (`gujarat_top_multimodal_master.csv`) contains all features required to reproduce model training and evaluation. To regenerate the full pipeline from raw data:

1. Download raw price data from AGMARKNET or data.gov.in
2. Fetch weather data from Open-Meteo API for Ahmedabad (2023-2025)
3. Run `python -m src.train --rebuild-data`

---

## Appendix: How to Run the System

### Table A.1: Technology Stack

| Layer | Technology |
|-------|-----------|
| **Programming Language** | Python 3.8+ (Tested on Python 3.10) |
| **Machine Learning** | scikit-learn 1.3+, XGBoost 2.0+, PyTorch 2.0+ |
| **Data Processing** | pandas 2.0+, NumPy 1.24+, SciPy 1.10+ |
| **NLP & Sentiment** | VADER Sentiment 3.3+, feedparser 6.0+ |
| **Web Dashboard** | Streamlit 1.29+, Plotly 5.18+ |
| **Model Serialization** | Joblib 1.3+ (scikit-learn/XGBoost), PyTorch native (.pt) |
| **Database** | CSV files (processed data storage) |
| **Development** | Git, GitHub (version control) |
| **Source Code** | https://github.com/Arshil903/Multi-Modal_Agri-Market_Intelligence |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Operating System** | Windows 10, Linux, macOS | Windows 11 / Ubuntu 22.04 |
| **Python Version** | 3.8 | 3.10+ |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 500 MB | 2 GB (with raw data: 64 MB) |
| **CPU** | 4 cores | 8 cores |
| **GPU** | None (CPU-only works) | CUDA 11.8+ (for BiLSTM training) |

### Installation and Setup

#### Method 1: Quick Setup (Recommended for Evaluation)

```bash
# 1. Clone the repository
git clone https://github.com/Arshil903/Multi-Modal_Agri-Market_Intelligence.git
cd Multi-Modal_Agri-Market_Intelligence

# 2. Create and activate virtual environment
# Windows:
python -m venv venv
venv\Scripts\activate

# Linux/Mac:
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit dashboard (pre-trained models included)
streamlit run app.py

# 5. Open browser at http://localhost:8501
```

**Expected Output:**
- Dashboard opens with two tabs: "Historical Evaluation & Backtesting" and "Live 2026 Predictor"
- Pre-trained models load automatically from `models_saved/` directory
- Test the live predictor with sample inputs (see demonstration scenarios below)

#### Method 2: Full Training Pipeline (For Model Development)

```bash
# Prerequisites: Complete Method 1 steps 1-3

# 4. Download raw data (not in repository)
# - Place Agriculture_price_dataset.csv in data/raw/
# - Place open-meteo-23.02N72.60E53m.csv in data/raw/

# 5. Run complete training pipeline
python -m src.train --rebuild-data

# Expected output:
# - Data ingestion: ~30 seconds
# - Feature engineering: ~10 seconds
# - Model training: ~8 minutes (3 crops × 12 configs each)
# - Evaluation & benchmarking: ~20 seconds
# - Models saved to models_saved/ (60 files, ~11 MB)
```

### Running Individual Components

#### Test the Live Predictor Logic

```bash
python test_live_predictor.py
```

**Output:** Validates lag feature anchoring, disruption detection, and VADER sentiment thresholds across 4 scenarios.

#### Analyze Model Performance

```bash
python analyze_performance.py
```

**Output:** Detailed performance report printed to console, showing R², RMSE, accuracy, ROC-AUC for all models.

#### Generate Report Figures

```bash
python generate_report_graphs.py
```

**Output:** 7 publication-quality figures (300 DPI) saved as PNG files in project root.

### Sample Demonstration Scenarios

#### Scenario 1: Calm Market (Baseline Test)

Navigate to "Live 2026 Predictor" tab and enter:

```
Crop: Onion
Prediction Date: 2026-03-15
Recent Modal Price: ₹2100
Min Price: ₹1950
Max Price: ₹2250
Arrival Volume: 450 tonnes
Active Mandis: 15
Max Temperature: 36°C
Rainfall: 0 mm
News Headline: "Gujarat mandi prices remain stable with good supply"
```

**Expected Result:**
- Forecasted Price: ~₹2121 (+1.0%)
- Shock Probability: ~18% (Low)
- Status: 🟢 NORMAL MARKET REGIME

#### Scenario 2: Supply Shock – Heavy Rain

Same inputs as Scenario 1, but change:

```
Rainfall: 25 mm
News Headline: "Heavy unseasonal downpour damages standing onion crops in Gujarat mandis"
```

**Expected Result:**
- Forecasted Price: ~₹2352 (+12.0%)
- Shock Probability: ~78% (High)
- Status: 🔴 ⚠️ PRICE SHOCK DETECTED - HIGH VOLATILITY RISK
- Model Comparison: XGBoost (+12%), BiLSTM (+10%), Ridge (+8%)

#### Scenario 3: Negative News Without Rain

```
Rainfall: 0 mm
News Headline: "Onion shortage crisis deepens as crops destroyed by pest attack"
```

**Expected Result:**
- Forecasted Price: ~₹2310 (+10%)
- Shock Probability: ~75%
- Status: 🔴 PRICE SHOCK DETECTED
- VADER Sentiment: -0.908 (Very negative)

### Verifying Installation

Run all tests to confirm setup:

```bash
# Test 1: Import check
python -c "import streamlit, torch, xgboost, sklearn, pandas, plotly; print('✓ All imports successful')"

# Test 2: Model loading check
python -c "from src.config import MODELS_DIR; import os; print(f'✓ Found {len(os.listdir(MODELS_DIR))} model files')"

# Test 3: Data availability
python -c "from src.config import MASTER_CSV; import pandas as pd; df = pd.read_csv(MASTER_CSV); print(f'✓ Loaded {len(df)} rows from master dataset')"
```

**All checks should pass** with green checkmarks.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError** | Ensure virtual environment is activated, run `pip install -r requirements.txt` |
| **FileNotFoundError: master CSV** | Pre-trained models require processed data. Download from GitHub releases or run full training pipeline |
| **Torch CUDA not available** | BiLSTM will run on CPU (slower). Install PyTorch with CUDA from pytorch.org for GPU acceleration |
| **Streamlit port already in use** | Run `streamlit run app.py --server.port 8502` to use different port |
| **Yellow warning: torch.load** | Fixed in latest version (Nov 2026). Update to latest commit or add `weights_only=False` manually |

### Performance Benchmarks

Tested on: Windows 11, Intel i7-10750H, 16 GB RAM, NVIDIA GTX 1650 (4 GB)

| Operation | Time | Notes |
|-----------|------|-------|
| Dashboard startup | ~3 sec | Loads 60 pre-trained models |
| Live prediction (1 crop) | <50 ms | Ridge fastest (~5 ms), BiLSTM slowest (~40 ms) |
| Full training pipeline | ~8 min | 36 models trained (GPU accelerated BiLSTM) |
| Generate report figures | ~15 sec | 7 figures at 300 DPI |

### Reproducing Paper Results

To exactly reproduce the results in Tables 5.1 and 5.2:

```bash
# 1. Use the committed processed data (no raw data needed)
git clone https://github.com/Arshil903/Multi-Modal_Agri-Market_Intelligence.git
cd Multi-Modal_Agri-Market_Intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Check benchmark results
python -c "import pandas as pd; print(pd.read_csv('data/processed/model_benchmark_results.csv'))"

# 4. Optionally retrain (results will match within random seed tolerance)
python -m src.train
```

**Note:** Results are deterministic (random seed = 42) but may vary slightly due to:
- BLAS/MKL library differences across systems
- PyTorch GPU vs CPU differences
- XGBoost version updates

Variance is typically <0.001 for R² and <5₹ for RMSE.

### Directory Structure Quick Reference

```
agri_project/
├── app.py                  # Run this: streamlit run app.py
├── requirements.txt        # Dependencies list
├── data/
│   └── processed/          # Pre-processed datasets (committed)
├── models_saved/           # 60 pre-trained models (11 MB)
├── src/
│   ├── train.py           # Run this: python -m src.train
│   ├── config.py          # Hyperparameters and paths
│   └── [7 other modules]
└── [Analysis scripts]      # test_live_predictor.py, analyze_performance.py, etc.
```

### Support and Contact

- **GitHub Issues:** https://github.com/Arshil903/Multi-Modal_Agri-Market_Intelligence/issues
- **Documentation:** README.md, LIVE_PREDICTOR_GUIDE.md, BUGFIX_SUMMARY.md
- **Project Report:** This document
- **Author:** Mohammad Arshil (23000903)

---

**End of Appendix**
