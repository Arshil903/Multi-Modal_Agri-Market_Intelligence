# Multimodal Agricultural Market Intelligence

**A Deep Learning Framework for Mandi Price Forecasting and Shock Detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project implements a multimodal agricultural price forecasting system for Gujarat mandi markets (Onion, Potato, Tomato). It combines traditional market time-series data with external modalities (weather patterns and news sentiment) to predict price movements and detect market shocks.

### Research Hypothesis

**Can integrating multimodal data sources (weather + news sentiment) improve agricultural price forecasting accuracy compared to market-only time-series models?**

### Key Findings

Our empirical analysis reveals that **model complexity matters more than data modality**:

| Model | Regime | R² Score | RMSE (₹/quintal) | Training Time |
|-------|--------|----------|------------------|---------------|
| **Ridge Regression** | Market-Only | **0.918** ✅ | **201** | ~1 sec |
| Ridge Regression | Multimodal | 0.897 | 225 | ~1 sec |
| XGBoost | Market-Only | 0.911 | 209 | ~3 sec |
| XGBoost | Multimodal | 0.889 | 234 | ~5 sec |
| BiLSTM | Market-Only | 0.724 | 368 | ~46 sec (GPU) |
| BiLSTM | Multimodal | 0.731 | 363 | ~46 sec (GPU) |

**Insights:**
- ✅ Simple linear models (Ridge) achieve 91.8% accuracy with market data alone
- ⚠️ Multimodal data helps deep learning models (+1% for BiLSTM) but hurts traditional ML (-2% for Ridge/XGBoost)
- 📊 For small datasets (577 samples), simpler models generalize better
- 🎯 BiLSTM benefits slightly from multimodal features but remains undertrained

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                         │
├──────────────────┬──────────────────────┬──────────────────────┤
│  Agmarknet API   │  Open-Meteo Weather  │  Google News RSS +   │
│  (Mandi Prices)  │  (Temp, Precip, etc) │  VADER Sentiment     │
└────────┬─────────┴──────────┬───────────┴──────────┬───────────┘
         │                     │                       │
         ▼                     ▼                       ▼
┌────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING PIPELINE                       │
│  • Lag features (1-7 days)                                     │
│  • Rolling statistics (7/30 day windows)                       │
│  • Price volatility & momentum indicators                      │
│  • Seasonal encoding (month/day-of-week)                       │
│  • Weather aggregation (mean/max/min temps, rainfall)         │
│  • News sentiment scores (polarity, subjectivity)             │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                   MODEL ENSEMBLE                                │
├───────────────┬───────────────────┬───────────────────────────┤
│ Ridge/Logistic│  XGBoost/RF       │  BiLSTM (PyTorch)        │
│ Regression    │  (Tree-based)     │  (3 layers, 128 units)   │
└────────┬──────┴─────────┬─────────┴──────────┬────────────────┘
         │                 │                     │
         ▼                 ▼                     ▼
┌────────────────────────────────────────────────────────────────┐
│                    DUAL TASK OUTPUTS                            │
├────────────────────────────┬───────────────────────────────────┤
│  Regression Task           │  Classification Task              │
│  (Price Prediction)        │  (Shock Detection: ±10% change)  │
└────────────────────────────┴───────────────────────────────────┘
```

---

## 🚀 Features

- **Dual-Task Learning**: Simultaneous regression (price forecasting) and classification (shock detection)
- **Multimodal Data Fusion**: Market prices + weather patterns + news sentiment
- **Model Diversity**: Ridge, XGBoost, Random Forest, Logistic Regression, BiLSTM
- **Interactive Dashboard**: Streamlit app with historical backtesting and live 2026 predictions
- **Production-Ready**: Cached sentiment analysis, error handling, GPU acceleration support

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- (Optional) CUDA-capable GPU for BiLSTM training

### Setup

```bash
# Clone the repository
git clone https://github.com/Arshil903/Multi-Modal_Agri-Market_Intelligence.git
cd Multi-Modal_Agri-Market_Intelligence

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
torch>=2.0.0
streamlit>=1.29.0
plotly>=5.18.0
joblib>=1.3.0
feedparser>=6.0.10
vaderSentiment>=3.3.2
requests>=2.31.0
```

---

## 🎯 Usage

### 1. Run Streamlit Dashboard

```bash
streamlit run app.py
```

The app provides two modes:
- **Historical Backtesting**: Evaluate model performance on test data (2024-2025)
- **Live 2026 Predictions**: Forecast future prices with latest data

### 2. Train Models from Scratch

```bash
python src/train.py
```

This will:
1. Load and preprocess raw data
2. Engineer features for both market-only and multimodal regimes
3. Train all model variants (Ridge, XGBoost, BiLSTM × 3 crops × 2 regimes)
4. Save trained models to `models_saved/`
5. Generate benchmark results in `data/processed/model_benchmark_results.csv`

### 3. Evaluate Models

```bash
python src/evaluate.py
```

Generates comprehensive evaluation metrics:
- R² Score, RMSE, MAE (regression)
- Accuracy, Precision, Recall, F1 (classification)
- Per-crop performance breakdown

---

## 📊 Dataset

### Market Data
- **Source**: Agmarknet API (Gujarat APMC markets)
- **Crops**: Onion, Potato, Tomato
- **Period**: 2023-2025
- **Features**: Modal price, min/max price, arrivals (quintals)
- **Size**: 577 daily observations

### Weather Data
- **Source**: Open-Meteo Historical Weather API
- **Location**: Gujarat (23.02°N, 72.60°E)
- **Features**: Temperature (mean/max/min), precipitation, wind speed, humidity
- **Resolution**: Daily aggregates

### News Sentiment
- **Source**: Google News RSS Feed (agriculture queries)
- **Processing**: VADER Sentiment Analysis
- **Features**: Compound polarity score, positive/negative/neutral ratios
- **Caching**: Sentiment scores cached to avoid API rate limits

**Note**: Raw dataset (`Agriculture_price_dataset.csv`, 52.65 MB) is excluded from Git due to size. Download separately or use included processed datasets.

---

## 🗂️ Project Structure

```
agri_project/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                 # Git ignore rules
│
├── src/                       # Source code modules
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_ingestion.py      # API data fetching (Agmarknet, weather, news)
│   ├── preprocessing.py       # Data cleaning and validation
│   ├── feature_engineering.py # Feature creation pipeline
│   ├── models_regression.py   # Regression models (Ridge, XGBoost, BiLSTM)
│   ├── models_classification.py # Classification models (Logistic, RF, BiLSTM)
│   ├── train.py               # Training orchestration script
│   └── evaluate.py            # Model evaluation and benchmarking
│
├── data/
│   ├── raw/                   # (gitignored) Raw CSV files
│   └── processed/             # Processed datasets
│       ├── gujarat_top_multimodal_master.csv  # Final training dataset
│       ├── news_rss_vader_cache.csv           # Cached sentiment scores
│       ├── model_benchmark_results.csv        # Model performance metrics
│       └── model_predictions.csv              # Prediction outputs
│
└── models_saved/              # Trained model artifacts (56 files)
    ├── {crop}_reg_{model}_{regime}.joblib     # Regression models
    ├── {crop}_clf_{model}_{regime}.joblib     # Classification models
    ├── {crop}_{regime}_features.joblib        # Feature column names
    └── *_scaler.joblib                        # Data scalers for BiLSTM
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Handle missing values via forward-fill
- Remove outliers using IQR method
- Normalize prices for cross-crop comparison

### 2. Feature Engineering
- **Temporal Features**: Lag 1-7 days, rolling mean/std (7/30 day windows)
- **Market Indicators**: Price volatility, momentum, arrival trends
- **Seasonal Encoding**: Month and day-of-week cyclical features
- **Weather Aggregation**: Temperature ranges, cumulative rainfall
- **Sentiment Scores**: VADER polarity for agriculture news

### 3. Model Training
- **Train/Test Split**: 80/20 temporal split (no data leakage)
- **Target Variables**:
  - Regression: Next-day modal price (₹/quintal)
  - Classification: Binary shock label (±10% price change)
- **Hyperparameters**:
  - Ridge: α=1.0
  - XGBoost: 100 estimators, max depth 5
  - BiLSTM: 3 layers, 128 hidden units, 21-day lookback, early stopping

### 4. Evaluation
- **Metrics**: R², RMSE, MAE, Accuracy, Precision, Recall, F1
- **Cross-Validation**: Time-series aware (no shuffling)
- **Baseline**: Naive persistence model (yesterday's price)

---

## 📈 Results & Discussion

### Regression Performance (Price Forecasting)

**Best Model: Ridge Regression (Market-Only)**
- R² = 0.918 (explains 91.8% of price variance)
- RMSE = ₹201/quintal (~6.5% of mean price)
- Significantly outperforms naive baseline (R²=0.65)

**Why Simple Models Win:**
1. **Small Dataset Effect**: 577 samples insufficient for deep learning generalization
2. **Linear Price Dynamics**: Gujarat mandi prices exhibit strong autocorrelation
3. **Regularization Benefit**: Ridge's L2 penalty prevents overfitting better than complex models

### Classification Performance (Shock Detection)

**Best Model: Random Forest (Multimodal)**
- Accuracy = 79.3%
- F1 Score = 0.81 (balanced precision/recall)
- Successfully detects 8/10 actual price shocks

**Multimodal Advantage for Classification:**
- Weather extremes (heatwaves, heavy rain) correlate with supply disruptions
- Negative news sentiment precedes demand-side shocks

### Multimodal vs Market-Only Analysis

| Observation | Explanation |
|-------------|-------------|
| Multimodal helps BiLSTM (+1% R²) | Deep models can learn non-linear weather-price interactions |
| Multimodal hurts Ridge (-2% R²) | Linear models struggle with noisy external features |
| Weather > News sentiment | Temperature and rainfall have stronger predictive power |
| Future work needed | Larger datasets may unlock multimodal benefits |

---

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Deep Learning**: PyTorch (BiLSTM)
- **Visualization**: Plotly, Streamlit
- **APIs**: Agmarknet, Open-Meteo, Google News RSS
- **NLP**: VADER Sentiment Analysis

---

## 🔮 Future Work

1. **Expand Dataset**: Collect 2+ years of data for better deep learning performance
2. **Additional Modalities**: Satellite imagery (vegetation indices), social media trends
3. **Transformer Models**: Attention mechanisms for multi-variate time series
4. **Explainability**: SHAP values to interpret feature importance
5. **Deployment**: REST API with model serving (FastAPI + Docker)
6. **Real-Time Updates**: Automated daily predictions with scheduled jobs

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Mohammad Arshil**  
Data Science & Machine Learning Project  
Semester 7 (July-Dec 2026)

---

## 🙏 Acknowledgments

- **Data Sources**: Agmarknet (Govt. of India), Open-Meteo Weather API
- **Inspiration**: Agricultural price volatility challenges faced by Gujarat farmers
- **Academic Guidance**: DSML course instructors

---

## 📞 Contact

For questions or collaboration opportunities:
- GitHub: [@Arshil903](https://github.com/Arshil903)
- Repository: [Multi-Modal_Agri-Market_Intelligence](https://github.com/Arshil903/Multi-Modal_Agri-Market_Intelligence)

---

**⭐ If you find this project useful, please consider giving it a star!**
