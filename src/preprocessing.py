"""Reindex daily calendars, fill Sunday closures, and winsorize price spikes."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import COMMODITIES, DATE_END, DATE_START, IQR_FACTOR, PRICE_WINSOR_COLS

LOGGER = logging.getLogger(__name__)


def study_calendar() -> pd.DatetimeIndex:
    return pd.date_range(start=DATE_START, end=DATE_END, freq="D")


def iqr_winsorize(series: pd.Series, factor: float = IQR_FACTOR) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return numeric
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return numeric.clip(lower, upper)


def reindex_commodity(frame: pd.DataFrame, commodity: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    subset = frame.loc[frame["Commodity"] == commodity].copy()
    subset = subset.drop_duplicates("Date").set_index("Date").sort_index()
    subset = subset.reindex(calendar)
    subset["Commodity"] = commodity
    subset.index.name = "Date"
    return subset.reset_index()


def fill_mandi_closures(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill prices on Sunday/holiday closures; closed days report 0 mandis."""
    out = frame.copy()
    price_cols = [c for c in ("Modal_Price", "Min_Price", "Max_Price") if c in out.columns]
    out[price_cols] = out.groupby("Commodity")[price_cols].ffill()
    out[price_cols] = out.groupby("Commodity")[price_cols].bfill()

    if "Active_Mandis_Reporting" in out.columns:
        out["Active_Mandis_Reporting"] = out["Active_Mandis_Reporting"].fillna(0).astype(int)

    weather_cols = [c for c in ("Max_Temp_C", "Min_Temp_C", "Rainfall_mm") if c in out.columns]
    for col in weather_cols:
        out[col] = out.groupby("Commodity")[col].transform(
            lambda s: s.interpolate(limit_direction="both").ffill().bfill()
        )

    if "News_Sentiment_Score" in out.columns:
        out["News_Sentiment_Score"] = (
            out.groupby("Commodity")["News_Sentiment_Score"]
            .transform(lambda s: s.interpolate(limit_direction="both"))
            .fillna(0.0)
        )
    if "Event_Disruption_Flag" in out.columns:
        out["Event_Disruption_Flag"] = out["Event_Disruption_Flag"].fillna(0).astype(int)
    if "GDELT_Tone" in out.columns:
        out["GDELT_Tone"] = out.groupby("Commodity")["GDELT_Tone"].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        out["GDELT_Tone"] = out["GDELT_Tone"].fillna(0.0)
    if "Arrival_Volume" in out.columns:
        out["Arrival_Volume"] = out.groupby("Commodity")["Arrival_Volume"].ffill()
        out["Arrival_Volume"] = out["Arrival_Volume"].fillna(0.0)
    return out


def winsorize_spikes(frame: pd.DataFrame) -> pd.DataFrame:
    """IQR-cap extreme recording spikes in prices and arrival volume if present."""
    out = frame.copy()
    cols = [c for c in PRICE_WINSOR_COLS if c in out.columns]
    if "Arrival_Volume" in out.columns:
        cols.append("Arrival_Volume")
    for commodity, idx in out.groupby("Commodity").groups.items():
        for col in cols:
            out.loc[idx, col] = iqr_winsorize(out.loc[idx, col])
        LOGGER.info("Winsorized %s columns %s", commodity, cols)
    return out


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    calendar = study_calendar()
    pieces = [reindex_commodity(raw, commodity, calendar) for commodity in COMMODITIES]
    aligned = pd.concat(pieces, ignore_index=True)
    filled = fill_mandi_closures(aligned)
    cleaned = winsorize_spikes(filled)
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    cleaned = cleaned.sort_values(["Commodity", "Date"]).reset_index(drop=True)
    LOGGER.info("Preprocessed shape: %s", cleaned.shape)
    return cleaned
