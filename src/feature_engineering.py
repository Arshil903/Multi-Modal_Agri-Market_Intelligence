"""Lag, rolling, cyclical, weather/news, and target construction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import MASTER_CSV, SHOCK_Z_THRESHOLD, ensure_directories
from src.data_ingestion import build_raw_multimodal
from src.preprocessing import preprocess

LOGGER = logging.getLogger(__name__)


def _add_price_memory(group: pd.DataFrame) -> pd.DataFrame:
    price = group["Modal_Price"]
    group["Price_Lag_1"] = price.shift(1)
    group["Price_Lag_3"] = price.shift(3)
    group["Price_Lag_7"] = price.shift(7)
    group["Price_Lag_14"] = price.shift(14)
    group["Price_Roll_Mean_7"] = price.rolling(window=7, min_periods=3).mean()
    group["Price_Roll_Std_7"] = price.rolling(window=7, min_periods=3).std()
    group["Price_Roll_Mean_30"] = price.rolling(window=30, min_periods=10).mean()
    group["Price_Roll_Std_30"] = price.rolling(window=30, min_periods=10).std()
    return group


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Intraday_Spread"] = out["Max_Price"] - out["Min_Price"]
    out["Rainfall_Roll_7D"] = (
        out.groupby("Commodity")["Rainfall_mm"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    )
    out["Sentiment_Roll_Mean_7D"] = (
        out.groupby("Commodity")["News_Sentiment_Score"].transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )
    )
    out["Disruption_Count_7D"] = (
        out.groupby("Commodity")["Event_Disruption_Flag"].transform(
            lambda s: s.rolling(7, min_periods=1).sum()
        )
    )

    month = out["Date"].dt.month
    dow = out["Date"].dt.dayofweek
    out["Month_Sin"] = np.sin(2 * np.pi * month / 12.0)
    out["Month_Cos"] = np.cos(2 * np.pi * month / 12.0)
    out["DayOfWeek_Sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["DayOfWeek_Cos"] = np.cos(2 * np.pi * dow / 7.0)

    pieces = [_add_price_memory(group.copy()) for _, group in out.groupby("Commodity", sort=False)]
    return pd.concat(pieces, ignore_index=True)


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Target_Price"] = out.groupby("Commodity")["Modal_Price"].shift(-1)
    z_denom = out["Price_Roll_Std_30"].replace(0.0, np.nan)
    shock_z = (out["Target_Price"] - out["Price_Roll_Mean_30"]) / z_denom
    out["Target_Shock"] = (shock_z > SHOCK_Z_THRESHOLD).astype("float")
    out.loc[out["Target_Price"].isna(), "Target_Shock"] = np.nan
    return out


def engineer_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    featured = add_temporal_features(cleaned)
    featured = add_targets(featured)
    featured["Price_Roll_Std_7"] = featured["Price_Roll_Std_7"].fillna(0.0)
    featured["Price_Roll_Std_30"] = featured["Price_Roll_Std_30"].fillna(0.0)
    LOGGER.info("Engineered feature frame: %s", featured.shape)
    return featured


def build_master_dataset(save: bool = True) -> pd.DataFrame:
    ensure_directories()
    raw = build_raw_multimodal()
    cleaned = preprocess(raw)
    master = engineer_features(cleaned)
    if save:
        master.to_csv(MASTER_CSV, index=False)
        LOGGER.info("Wrote master dataset to %s", MASTER_CSV)
    return master


def load_or_build_master(rebuild: bool = False) -> pd.DataFrame:
    if not rebuild and MASTER_CSV.exists():
        LOGGER.info("Loading existing master dataset from %s", MASTER_CSV)
        df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
        return df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
    return build_master_dataset(save=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_master_dataset()
    print(df.head())
    print(df.columns.tolist())
    print(df.groupby("Commodity")["Target_Shock"].mean())
