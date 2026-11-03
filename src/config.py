"""Project-wide paths, seeds, commodity lists, and model hyperparameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models_saved"

PRICE_FILENAME = "Agriculture_price_dataset.csv"
WEATHER_FILENAME = "open-meteo-23.02N72.60E53m.csv"

COMMODITIES = ["Onion", "Potato", "Tomato"]
TARGET_STATE = "gujarat"

DATE_START = "2023-06-06"
DATE_END = "2025-06-11"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.80

MASTER_CSV = PROCESSED_DIR / "gujarat_top_multimodal_master.csv"
BENCHMARK_CSV = PROCESSED_DIR / "model_benchmark_results.csv"
PREDICTIONS_CSV = PROCESSED_DIR / "model_predictions.csv"
GDELT_CACHE_CSV = PROCESSED_DIR / "gdelt_tone_cache.csv"
NEWS_RSS_CACHE_CSV = PROCESSED_DIR / "news_rss_vader_cache.csv"

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMEOUT_SEC = 12
GDELT_MAX_RETRIES = 1

UNSEASONAL_RAIN_MM = 15.0
MONSOON_MONTHS = (7, 8)
HEATWAVE_TEMP_C = 42.0

IQR_FACTOR = 1.5
PRICE_WINSOR_COLS = ["Modal_Price", "Min_Price", "Max_Price"]

MARKET_ONLY_FEATURES = [
    "Price_Lag_1",
    "Price_Lag_3",
    "Price_Lag_7",
    "Price_Lag_14",
    "Price_Roll_Mean_7",
    "Price_Roll_Std_7",
    "Price_Roll_Mean_30",
    "Price_Roll_Std_30",
    "Intraday_Spread",
    "Month_Sin",
    "Month_Cos",
]

MULTIMODAL_EXTRA_FEATURES = [
    "Max_Temp_C",
    "Rainfall_mm",
    "Rainfall_Roll_7D",
    "News_Sentiment_Score",
    "Sentiment_Roll_Mean_7D",
    "Disruption_Count_7D",
]

MULTIMODAL_FEATURES = MARKET_ONLY_FEATURES + MULTIMODAL_EXTRA_FEATURES

FEATURE_REGIMES = {
    "market_only": MARKET_ONLY_FEATURES,
    "multimodal": MULTIMODAL_FEATURES,
}

RIDGE_ALPHA = 1.0

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.03,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
}

LOGREG_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

RF_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced_subsample",
    "min_samples_leaf": 3,
}

LSTM_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lookback": 14,
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "patience": 4,
    "weight_decay": 1e-5,
}

SHOCK_Z_THRESHOLD = 1.75


def resolve_raw_dir() -> Path:
    """Prefer agri_project/data/raw; fall back to the workspace data/raw folder."""
    local = DATA_DIR / "raw"
    if (local / PRICE_FILENAME).exists() and (local / WEATHER_FILENAME).exists():
        return local
    parent_raw = PROJECT_ROOT.parent / "data" / "raw"
    if (parent_raw / PRICE_FILENAME).exists() and (parent_raw / WEATHER_FILENAME).exists():
        return parent_raw
    return local


RAW_DIR = resolve_raw_dir()
PRICE_CSV = RAW_DIR / PRICE_FILENAME
WEATHER_CSV = RAW_DIR / WEATHER_FILENAME


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
