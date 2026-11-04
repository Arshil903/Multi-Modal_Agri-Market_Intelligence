# Project Structure

## Multimodal Agricultural Market Intelligence
**Gujarat Mandi Price Forecasting and Shock Detection System**

---

## 📁 Complete Directory Tree

```
agri_project/                              # Project root directory
│
├── 📄 app.py                              # Streamlit web application (650 LOC)
├── 📄 requirements.txt                    # Python dependencies (15 packages)
├── 📄 README.md                           # Project documentation (400 lines)
├── 📄 LICENSE                             # MIT License
├── 📄 .gitignore                         # Git ignore rules (excludes raw data, cache)
│
├── 📊 ANALYSIS & DOCUMENTATION FILES
│   ├── analyze_performance.py             # Performance analysis script
│   ├── MODEL_PERFORMANCE_REPORT.txt       # Detailed model comparison report
│   ├── BUGFIX_SUMMARY.md                  # Critical bug fixes documentation
│   ├── LIVE_PREDICTOR_GUIDE.md           # Viva demo scenarios guide
│   └── test_live_predictor.py            # Test suite for live predictor
│
├── 📂 src/                                # Source code modules (2,250 LOC total)
│   ├── __init__.py                        # Package initializer
│   ├── config.py                          # Configuration & hyperparameters (150 LOC)
│   ├── data_ingestion.py                  # API data fetching (250 LOC)
│   ├── preprocessing.py                   # Data cleaning & validation (200 LOC)
│   ├── feature_engineering.py             # Feature creation pipeline (300 LOC)
│   ├── models_regression.py               # Ridge, XGBoost, BiLSTM regressors (350 LOC)
│   ├── models_classification.py           # Logistic, RF, BiLSTM classifiers (350 LOC)
│   ├── train.py                           # Training orchestration (400 LOC)
│   └── evaluate.py                        # Model evaluation & metrics (250 LOC)
│
├── 📂 data/                               # Data directory
│   ├── processed/                         # Processed datasets (committed to Git)
│   │   ├── gujarat_top_multimodal_master.csv      # Final dataset (577 rows × 45 cols, 250 KB)
│   │   ├── news_rss_vader_cache.csv               # Cached sentiment scores (180 rows, 15 KB)
│   │   ├── model_benchmark_results.csv            # Performance metrics (36 models, 5 KB)
│   │   └── model_predictions.csv                  # Test predictions (145 rows, 30 KB)
│   │
│   └── raw/                               # Raw CSV files (gitignored, not on GitHub)
│       ├── Agriculture_price_dataset.csv          # Mandi prices (52.65 MB, 50K rows)
│       └── open-meteo-23.02N72.60E53m.csv        # Weather data (500 KB, 731 days)
│
└── 📂 models_saved/                       # Trained models (60 files, 11 MB compressed)
    │
    ├── 🧅 ONION MODELS (20 files)
    │   ├── Regression (Market-Only):
    │   │   ├── onion_reg_ridge_market_only.joblib              (100 KB)
    │   │   ├── onion_reg_xgboost_market_only.joblib            (500 KB)
    │   │   ├── onion_reg_bilstm_market_only.pt                 (300 KB)
    │   │   └── onion_reg_bilstm_market_only_scaler.joblib      (50 KB)
    │   │
    │   ├── Regression (Multimodal):
    │   │   ├── onion_reg_ridge_multimodal.joblib
    │   │   ├── onion_reg_xgboost_multimodal.joblib
    │   │   ├── onion_reg_bilstm_multimodal.pt
    │   │   └── onion_reg_bilstm_multimodal_scaler.joblib
    │   │
    │   ├── Classification (Market-Only):
    │   │   ├── onion_clf_logreg_market_only.joblib             (50 KB)
    │   │   ├── onion_clf_random_forest_market_only.joblib      (800 KB)
    │   │   ├── onion_clf_bilstm_market_only.pt                 (300 KB)
    │   │   └── onion_clf_bilstm_market_only_scaler.joblib      (50 KB)
    │   │
    │   ├── Classification (Multimodal):
    │   │   ├── onion_clf_logreg_multimodal.joblib
    │   │   ├── onion_clf_random_forest_multimodal.joblib
    │   │   ├── onion_clf_bilstm_multimodal.pt
    │   │   └── onion_clf_bilstm_multimodal_scaler.joblib
    │   │
    │   └── Feature Schemas:
    │       ├── onion_market_only_features.joblib
    │       └── onion_multimodal_features.joblib
    │
    ├── 🥔 POTATO MODELS (20 files)
    │   └── [Same structure as Onion]
    │
    ├── 🍅 TOMATO MODELS (20 files)
    │   └── [Same structure as Onion]
    │
    └── 🔧 LEGACY FILES (2 files)
        ├── onion_xgboost_regressor.joblib
        └── xgboost_regressor.joblib
```

---

## 📋 Detailed File Descriptions

### 1. Root Level Files

| File | Purpose | Size | Key Features |
|------|---------|------|--------------|
| **app.py** | Streamlit web dashboard | 40 KB | • 2-tab interface (Historical + Live 2026)<br>• Interactive Plotly charts<br>• Model comparison table<br>• Real-time what-if analysis |
| **requirements.txt** | Package dependencies | 1 KB | pandas, numpy, scikit-learn, xgboost, torch, streamlit, plotly, vaderSentiment, feedparser |
| **README.md** | Project documentation | 20 KB | • Architecture diagram<br>• Installation guide<br>• Model performance table<br>• Usage examples<br>• Research findings |
| **LICENSE** | MIT License | 1 KB | Permissive open-source license |
| **.gitignore** | Git exclusions | 1 KB | Excludes: __pycache__, *.zip, raw data (52 MB), .venv/, .env |

### 2. Source Code Modules (`src/`)

| Module | Responsibilities | Key Functions | Lines |
|--------|------------------|---------------|-------|
| **config.py** | • Project paths<br>• Hyperparameters<br>• Constants | `ensure_directories()`<br>`seed_everything()`<br>`get_device()` | 150 |
| **data_ingestion.py** | • Fetch Agmarknet prices<br>• Fetch weather data<br>• Fetch & analyze news | `fetch_agmarknet_prices()`<br>`fetch_weather_open_meteo()`<br>`fetch_news_rss_vader()` | 250 |
| **preprocessing.py** | • Remove outliers<br>• Handle missing data<br>• Validate data quality | `remove_outliers_iqr()`<br>`winsorize_prices()`<br>`forward_fill_missing()` | 200 |
| **feature_engineering.py** | • Create lag features<br>• Compute rolling stats<br>• Encode cyclical time | `compute_lag_features()`<br>`compute_rolling_stats()`<br>`encode_cyclical_time()` | 300 |
| **models_regression.py** | • BiLSTM architecture<br>• Train Ridge, XGBoost<br>• Train BiLSTM | `class BiLSTMRegressor(nn.Module)`<br>`train_ridge()`<br>`train_bilstm_regressor()` | 350 |
| **models_classification.py** | • BiLSTM classifier<br>• Train Logistic, RF<br>• Apply SMOTE | `class BiLSTMClassifier(nn.Module)`<br>`train_logreg()`<br>`apply_smote()` | 350 |
| **train.py** | • Orchestrate training<br>• Save models<br>• Generate predictions | `train_all_models()`<br>`save_model_artifacts()`<br>`save_predictions()` | 400 |
| **evaluate.py** | • Calculate metrics<br>• Generate reports<br>• Plot results | `evaluate_regression()`<br>`evaluate_classification()`<br>`plot_predictions()` | 250 |

**Total Source Code:** 2,250 lines

### 3. Data Files

#### Processed Data (`data/processed/`)

| File | Description | Dimensions | Size | Purpose |
|------|-------------|------------|------|---------|
| **gujarat_top_multimodal_master.csv** | Final training dataset | 577 rows × 45 cols | 250 KB | Master dataset with all features |
| **news_rss_vader_cache.csv** | Cached sentiment scores | 180 rows × 4 cols | 15 KB | Avoid API rate limits |
| **model_benchmark_results.csv** | Performance metrics | 36 rows × 14 cols | 5 KB | R², RMSE, Accuracy, ROC-AUC |
| **model_predictions.csv** | Test set predictions | 145 rows × 25 cols | 30 KB | All model predictions for comparison |

**Master Dataset Columns (45 total):**

1. **Market Features (11):**
   - `Price_Lag_1`, `Price_Lag_3`, `Price_Lag_7`, `Price_Lag_14`
   - `Price_Roll_Mean_7`, `Price_Roll_Std_7`
   - `Price_Roll_Mean_30`, `Price_Roll_Std_30`
   - `Intraday_Spread`, `Month_Sin`, `Month_Cos`

2. **Weather Features (6):**
   - `Max_Temp_C`, `Min_Temp_C`, `Mean_Temp_C`
   - `Rainfall_mm`, `Rainfall_Roll_7D`
   - `Wind_Speed_kmh`

3. **News Features (3):**
   - `News_Sentiment_Score` (VADER compound score)
   - `Sentiment_Roll_Mean_7D`
   - `Disruption_Count_7D`, `Event_Disruption_Flag`

4. **Target Variables (2):**
   - `Modal_Price` (continuous, ₹/quintal) → Regression
   - `Target_Shock` (binary, ±10% change) → Classification

5. **Metadata (23):**
   - `Date`, `Commodity`, `State`, `Market`
   - `Min_Price`, `Max_Price`, `Arrivals_Tonnes`
   - Additional market identifiers

#### Raw Data (`data/raw/`, gitignored)

| File | Source | Size | Rows | Excluded from Git? |
|------|--------|------|------|--------------------|
| **Agriculture_price_dataset.csv** | Agmarknet API | 52.65 MB | ~50,000 | ✅ Yes (exceeds 50 MB limit) |
| **open-meteo-23.02N72.60E53m.csv** | Open-Meteo API | 500 KB | 731 | ✅ Yes (can be regenerated) |

### 4. Model Files (`models_saved/`)

#### Model Naming Convention

```
{crop}_{task}_{model}_{regime}.{extension}

Components:
- crop:   onion | potato | tomato
- task:   reg (regression) | clf (classification)
- model:  ridge | xgboost | bilstm | logreg | random_forest
- regime: market_only | multimodal
- ext:    .joblib (scikit-learn/xgboost) | .pt (PyTorch)

Examples:
✓ onion_reg_ridge_market_only.joblib
✓ potato_clf_random_forest_multimodal.joblib
✓ tomato_reg_bilstm_multimodal.pt
✓ tomato_reg_bilstm_multimodal_scaler.joblib
```

#### Model Inventory Summary

| Model Type | Count per Crop | File Format | Size per Model | Total Size |
|------------|----------------|-------------|----------------|------------|
| Ridge Regression | 2 (MO + MM) | `.joblib` | ~100 KB | 600 KB |
| XGBoost Regressor | 2 (MO + MM) | `.joblib` | ~500 KB | 3 MB |
| BiLSTM Regressor | 2 + 2 scalers | `.pt` + `.joblib` | ~300 KB + 50 KB | 2.1 MB |
| Logistic Regression | 2 (MO + MM) | `.joblib` | ~50 KB | 300 KB |
| Random Forest | 2 (MO + MM) | `.joblib` | ~800 KB | 4.8 MB |
| BiLSTM Classifier | 2 + 2 scalers | `.pt` + `.joblib` | ~300 KB + 50 KB | 2.1 MB |
| Feature Schemas | 2 (MO + MM) | `.joblib` | ~10 KB | 60 KB |
| **Per Crop Total** | **20 files** | Mixed | | **~4 MB** |
| **3 Crops Total** | **60 files** | Mixed | | **~11 MB** |

**MO** = Market-Only, **MM** = Multimodal

#### Model Architecture Details

**1. Ridge Regression**
```python
Class: sklearn.linear_model.Ridge
Hyperparameters:
  - alpha: 1.0 (L2 regularization strength)
  - solver: 'auto'
Features: 11 (market_only) or 17 (multimodal)
Output: Continuous price (₹/quintal)
Training Time: ~1 second
Best R²: 0.9176 (market_only)
```

**2. XGBoost Regressor**
```python
Class: xgboost.XGBRegressor
Hyperparameters:
  - n_estimators: 300
  - max_depth: 5
  - learning_rate: 0.03
  - subsample: 0.9
  - colsample_bytree: 0.9
  - objective: 'reg:squarederror'
Features: 11 (market_only) or 17 (multimodal)
Output: Continuous price (₹/quintal)
Training Time: ~3 seconds
Best R²: 0.8380 (market_only)
```

**3. BiLSTM Regressor**
```python
Class: custom BiLSTMRegressor(nn.Module)
Architecture:
  - Input: (batch_size, lookback=14, n_features)
  - LSTM: 3 layers, 128 hidden units, 0.2 dropout
  - Output: 1 (continuous price)
Hyperparameters:
  - epochs: 100 (max), early stopping patience=4
  - batch_size: 32
  - learning_rate: 0.001
  - optimizer: Adam
  - loss: MSE
Features: 11 (market_only) or 17 (multimodal)
Training: GPU accelerated, actual epochs 12-30
Training Time: ~46 seconds (GPU)
Best R²: 0.7307 (multimodal)
```

**4. Logistic Regression Classifier**
```python
Class: sklearn.linear_model.LogisticRegression
Hyperparameters:
  - class_weight: 'balanced'
  - max_iter: 1000
  - solver: 'lbfgs'
Features: 11 (market_only) or 17 (multimodal)
Output: Binary (shock=1, no_shock=0)
Training Time: ~0.5 seconds
Best ROC-AUC: 0.90 (market_only)
```

**5. Random Forest Classifier**
```python
Class: sklearn.ensemble.RandomForestClassifier
Hyperparameters:
  - n_estimators: 200
  - class_weight: 'balanced_subsample'
  - min_samples_leaf: 3
Preprocessing: SMOTE applied for oversampling minority class
Features: 11 (market_only) or 17 (multimodal)
Output: Binary (shock=1, no_shock=0)
Training Time: ~1.5 seconds
Best Accuracy: 97.24% (but 0% precision!)
```

**6. BiLSTM Classifier**
```python
Class: custom BiLSTMClassifier(nn.Module)
Architecture:
  - Input: (batch_size, lookback=14, n_features)
  - LSTM: 3 layers, 128 hidden units, 0.2 dropout
  - Output: 1 (sigmoid activation for probability)
Hyperparameters:
  - epochs: 100 (max), early stopping patience=4
  - batch_size: 32
  - learning_rate: 0.001
  - optimizer: Adam
  - loss: BCELoss (Binary Cross Entropy)
Features: 11 (market_only) or 17 (multimodal)
Training: GPU accelerated, actual epochs 12-30
Training Time: ~46 seconds (GPU)
Best ROC-AUC: 0.82 (market_only)
```

---

## 🔄 Complete Data Pipeline

```
STAGE 1: DATA INGESTION (data_ingestion.py)
┌────────────────────────────────────────────────────────────┐
│ Agmarknet API    Open-Meteo API    Google News RSS        │
│ ↓                ↓                   ↓                     │
│ 50K price rows   731 weather days   180 news headlines    │
└────────────────────────────────────────────────────────────┘
                         ↓
STAGE 2: PREPROCESSING (preprocessing.py)
┌────────────────────────────────────────────────────────────┐
│ • Filter: Gujarat state + TOP crops (Onion, Potato, Tomato)│
│ • Remove outliers: IQR method (1.5 × IQR)                 │
│ • Winsorize prices: Clip extreme values                   │
│ • Forward fill: Handle missing values                     │
│ • Date filter: 2023-06-06 to 2025-06-11                  │
│ Output: 577 clean rows                                    │
└────────────────────────────────────────────────────────────┘
                         ↓
STAGE 3: FEATURE ENGINEERING (feature_engineering.py)
┌────────────────────────────────────────────────────────────┐
│ Market Features:    Weather Features:   News Features:     │
│ • Lags (1,3,7,14)  • Max/Min/Mean Temp  • VADER Sentiment │
│ • Roll Mean (7,30) • Rainfall (mm)      • Roll Sentiment  │
│ • Roll Std (7,30)  • Rainfall Roll 7D   • Disruption Flag │
│ • Intraday Spread  • Wind Speed         • Disruption Count│
│ • Cyclical (sin,cos)                                      │
│ Output: 45 columns total                                  │
└────────────────────────────────────────────────────────────┘
                         ↓
STAGE 4: TRAIN/TEST SPLIT
┌─────────────────────────────┬──────────────────────────────┐
│ Train Set (80%)             │ Test Set (20%)               │
│ 462 rows                    │ 115 rows                     │
│ Temporal split (no shuffle) │ Later dates for validation   │
└─────────────────────────────┴──────────────────────────────┘
                         ↓
STAGE 5: MODEL TRAINING (train.py)
┌────────────────────────────────────────────────────────────┐
│ For each crop (Onion, Potato, Tomato):                    │
│   For each regime (market_only, multimodal):              │
│     Regression:                                            │
│       • Ridge (1 sec)                                     │
│       • XGBoost (3 sec)                                   │
│       • BiLSTM (46 sec, GPU)                              │
│     Classification:                                        │
│       • Logistic (0.5 sec)                                │
│       • RandomForest + SMOTE (1.5 sec)                    │
│       • BiLSTM (46 sec, GPU)                              │
│                                                            │
│ Total: 3 crops × 2 regimes × 6 models = 36 models        │
│ Training Time: ~8 minutes (with GPU)                      │
└────────────────────────────────────────────────────────────┘
                         ↓
STAGE 6: EVALUATION (evaluate.py)
┌────────────────────────────────────────────────────────────┐
│ Regression Metrics:                                        │
│   R² Score, RMSE, MAE                                     │
│ Classification Metrics:                                    │
│   Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC       │
│ Output: model_benchmark_results.csv                       │
└────────────────────────────────────────────────────────────┘
                         ↓
STAGE 7: DEPLOYMENT (app.py)
┌────────────────────────────────────────────────────────────┐
│ Streamlit Web Dashboard                                    │
│ ┌──────────────────────────┬────────────────────────────┐ │
│ │ Tab 1: Historical        │ Tab 2: Live 2026 Predictor │ │
│ │ • Actual vs Predicted    │ • User input form          │ │
│ │ • Shock alerts           │ • Dynamic feature eng.     │ │
│ │ • Interactive charts     │ • 3-model comparison       │ │
│ │ • Date slider            │ • Shock probability        │ │
│ └──────────────────────────┴────────────────────────────┘ │
│ Command: streamlit run app.py                             │
└────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Dependencies (requirements.txt)

```txt
# Data Processing
pandas>=2.0.0              # DataFrame manipulation
numpy>=1.24.0              # Numerical operations

# Machine Learning
scikit-learn>=1.3.0        # Traditional ML models
xgboost>=2.0.0             # Gradient boosting
imbalanced-learn>=0.11.0   # SMOTE for class imbalance

# Deep Learning
torch>=2.0.0               # PyTorch for BiLSTM
torchvision>=0.15.0        # Vision utilities (optional)

# Web Dashboard
streamlit>=1.29.0          # Interactive web app
plotly>=5.18.0             # Interactive charts

# NLP & APIs
vaderSentiment>=3.3.2      # News sentiment analysis
feedparser>=6.0.10         # RSS feed parsing
requests>=2.31.0           # HTTP requests

# Utilities
joblib>=1.3.0              # Model serialization
python-dateutil>=2.8.2     # Date parsing
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python Version** | 3.8 | 3.10+ |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 500 MB | 2 GB |
| **CPU** | 4 cores | 8 cores |
| **GPU** | None (CPU-only works) | CUDA 11.8+ (for BiLSTM) |
| **OS** | Windows 10, Linux, macOS | Any 64-bit |

---

## 📊 Model Performance Summary

| Task | Winner | R²/Accuracy | RMSE/ROC-AUC | Training Time |
|------|--------|-------------|--------------|---------------|
| **Regression (Task A)** | Ridge (Market-Only) | **0.9176** | ₹201 | 1 sec |
| **Classification (Task B)** | LogisticReg (Market-Only) | 92.41% | **0.90** | 0.5 sec |

---

## 🚀 Quick Start Commands

```bash
# 1. Clone repository
git clone https://github.com/Arshil903/Multi-Modal_Agri-Market_Intelligence.git
cd Multi-Modal_Agri-Market_Intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (optional, pre-trained models included)
python -m src.train --rebuild-data

# 4. Run dashboard
streamlit run app.py

# 5. Test live predictor
python test_live_predictor.py

# 6. Analyze performance
python analyze_performance.py
```

---

## 📈 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Files** | 85 (25 source + 60 models) |
| **Lines of Code** | ~2,900 (Python only) |
| **Documentation Lines** | ~800 (MD files) |
| **Total Size (Versioned)** | 11.5 MB |
| **Total Size (With Raw Data)** | 64 MB |
| **Models Trained** | 60 files (36 unique models + 24 aux files) |
| **Training Time** | ~8 minutes (with GPU) |
| **Inference Time** | <20 ms per prediction |
| **Best R² Score** | 0.9176 (91.76%) |
| **Best ROC-AUC** | 0.90 |

---

## 📝 Maintenance & Versioning

**Version Control:** Git + GitHub  
**Branch:** main  
**Last Updated:** November 3, 2026  
**Total Commits:** 5+  
**Contributors:** 1 (Mohammad Arshil)

**Key Commits:**
1. Initial commit: Complete pipeline with 36 models
2. Bug fix: PyTorch 2.6 compatibility, lag feature anchoring, shock detection
3. Documentation: Performance analysis, demo guide, project structure

---

**End of Project Structure Documentation**
