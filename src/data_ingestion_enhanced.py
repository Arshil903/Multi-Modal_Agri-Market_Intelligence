"""
Enhanced data ingestion with improved NLP pipeline.
Integrates relevance filtering, event classification, and deduplication.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import feedparser
import numpy as np
import pandas as pd

from src.config import (
    COMMODITIES,
    DATE_END,
    DATE_START,
    HEATWAVE_TEMP_C,
    MONSOON_MONTHS,
    NEWS_RSS_CACHE_CSV,
    PRICE_CSV,
    PROCESSED_DIR,
    TARGET_STATE,
    UNSEASONAL_RAIN_MM,
    WEATHER_CSV,
    ensure_directories,
)
from src.nlp_enhanced import process_news_with_enhanced_nlp

LOGGER = logging.getLogger(__name__)


# ========== PRICE DATA (UNCHANGED) ==========

def load_gujarat_top_prices(path: str | None = None) -> pd.DataFrame:
    """Load and aggregate Gujarat TOP mandi prices."""
    csv_path = path or PRICE_CSV
    LOGGER.info("Loading mandi prices from %s", csv_path)
    df = pd.read_csv(csv_path)
    required = {"STATE", "Commodity", "Min_Price", "Max_Price", "Modal_Price", "Price Date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Price CSV is missing columns: {sorted(missing)}")

    df["STATE"] = df["STATE"].astype(str).str.strip().str.lower()
    df = df.loc[df["STATE"] == TARGET_STATE].copy()
    df["Commodity"] = df["Commodity"].astype(str).str.strip()
    df = df.loc[df["Commodity"].isin(COMMODITIES)].copy()
    if df.empty:
        raise ValueError("No Gujarat Onion/Potato/Tomato rows found in the price file.")

    df["Date"] = pd.to_datetime(df["Price Date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["Date"])
    for col in ("Min_Price", "Max_Price", "Modal_Price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Modal_Price"])

    grouped = (
        df.groupby(["Date", "Commodity"], as_index=False)
        .agg(
            Modal_Price=("Modal_Price", "mean"),
            Min_Price=("Min_Price", "mean"),
            Max_Price=("Max_Price", "mean"),
            Active_Mandis_Reporting=("Modal_Price", "size"),
        )
    )
    if "Arrival_Volume" in df.columns:
        arrivals = (
            df.groupby(["Date", "Commodity"], as_index=False)["Arrival_Volume"]
            .mean()
        )
        grouped = grouped.merge(arrivals, on=["Date", "Commodity"], how="left")

    LOGGER.info("Gujarat TOP daily aggregates: %s rows", len(grouped))
    return grouped.sort_values(["Commodity", "Date"]).reset_index(drop=True)


# ========== WEATHER DATA (UNCHANGED) ==========

def load_ahmedabad_weather(path: str | None = None) -> pd.DataFrame:
    """Load Ahmedabad weather data."""
    csv_path = path or WEATHER_CSV
    LOGGER.info("Loading weather from %s", csv_path)
    weather = pd.read_csv(csv_path, skiprows=3)
    if weather.shape[1] < 4:
        raise ValueError("Weather CSV does not contain the expected 4 columns.")
    weather = weather.iloc[:, :4].copy()
    weather.columns = ["Date", "Max_Temp_C", "Min_Temp_C", "Rainfall_mm"]
    weather["Date"] = pd.to_datetime(weather["Date"], errors="coerce")
    weather = weather.dropna(subset=["Date"])
    for col in ("Max_Temp_C", "Min_Temp_C", "Rainfall_mm"):
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather = weather.drop_duplicates(subset=["Date"]).sort_values("Date")
    start = pd.Timestamp(DATE_START)
    end = pd.Timestamp(DATE_END)
    weather = weather.loc[(weather["Date"] >= start) & (weather["Date"] <= end)].copy()
    LOGGER.info("Weather rows in study window: %s", len(weather))
    return weather.reset_index(drop=True)


# ========== ENHANCED NEWS COLLECTION ==========

def _fetch_google_news_rss(commodity: str, max_retries: int = 2) -> list[dict[str, Any]]:
    """
    Fetch historical news headlines from Google News RSS.
    
    Args:
        commodity: Crop name (Onion, Potato, Tomato)
        max_retries: Number of retry attempts
    
    Returns:
        List of articles with title, published, link
    """
    query = f"{commodity}+mandi+price+gujarat+when:2y"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    for attempt in range(1, max_retries + 1):
        try:
            LOGGER.info(f"Fetching Google News RSS for {commodity} (attempt {attempt})")
            feed = feedparser.parse(url)
            
            if feed.bozo and not feed.entries:
                LOGGER.warning(f"Feed parse error for {commodity}: {feed.get('bozo_exception', 'Unknown')}")
                time.sleep(1.0 * attempt)
                continue
            
            articles = []
            for entry in feed.entries:
                title = entry.get('title', '')
                published_parsed = entry.get('published_parsed')
                link = entry.get('link', '')
                
                if not title or not published_parsed:
                    continue
                
                pub_date = datetime(*published_parsed[:6])
                
                articles.append({
                    'title': title,
                    'published': pub_date,
                    'link': link
                })
            
            LOGGER.info(f"Retrieved {len(articles)} articles for {commodity}")
            return articles
            
        except Exception as exc:
            LOGGER.warning(f"RSS fetch attempt {attempt} failed for {commodity}: {exc}")
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
    
    LOGGER.error(f"Failed to fetch RSS feed for {commodity} after {max_retries} attempts")
    return []


def fetch_enhanced_news_features(
    commodities: list[str] | None = None,
    use_cache: bool = True,
    relevance_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch and process news with enhanced NLP pipeline.
    
    Implements:
    - Relevance filtering (crop, region, market)
    - Deduplication (content hashing)
    - Event classification (11 categories)
    - Enhanced sentiment analysis
    - Temporal aggregation
    
    Args:
        commodities: List of crops to analyze
        use_cache: Whether to use cached results
        relevance_threshold: Minimum relevance score (0-1)
    
    Returns:
        DataFrame with daily NLP features per commodity
    """
    ensure_directories()
    commodities = commodities or COMMODITIES
    start = pd.Timestamp(DATE_START)
    end = pd.Timestamp(DATE_END)
    
    # Check cache
    enhanced_cache = PROCESSED_DIR / "news_enhanced_nlp_cache.csv"
    if use_cache and enhanced_cache.exists():
        cached = pd.read_csv(enhanced_cache, parse_dates=["Date"])
        LOGGER.info(f"Loaded enhanced NLP cache from {enhanced_cache} ({len(cached)} rows)")
        return cached
    
    all_features = []
    all_articles_meta = []
    
    for commodity in commodities:
        # Fetch raw articles
        raw_articles = _fetch_google_news_rss(commodity)
        
        if not raw_articles:
            LOGGER.warning(f"No articles found for {commodity}")
            continue
        
        # Apply enhanced NLP processing
        daily_features, enriched_articles = process_news_with_enhanced_nlp(
            raw_articles,
            commodity,
            relevance_threshold=relevance_threshold,
        )
        
        if not daily_features.empty:
            all_features.append(daily_features)
            all_articles_meta.extend(enriched_articles)
    
    if not all_features:
        LOGGER.warning("No NLP features extracted from news")
        empty = pd.DataFrame(columns=["Date", "Commodity", "News_Sentiment_Score"])
        empty.to_csv(enhanced_cache, index=False)
        return empty
    
    # Combine all commodities
    combined_features = pd.concat(all_features, ignore_index=True)
    
    # Filter date range
    combined_features = combined_features[
        (combined_features['Date'] >= start) & (combined_features['Date'] <= end)
    ].copy()
    
    # Save cache
    combined_features.to_csv(enhanced_cache, index=False)
    LOGGER.info(f"Cached {len(combined_features)} daily NLP feature rows to {enhanced_cache}")
    
    # Save article metadata for validation
    if all_articles_meta:
        meta_file = PROCESSED_DIR / "news_articles_metadata.csv"
        pd.DataFrame(all_articles_meta).to_csv(meta_file, index=False)
        LOGGER.info(f"Saved {len(all_articles_meta)} article metadata to {meta_file}")
    
    return combined_features


# ========== MULTIMODAL FUSION (ENHANCED) ==========

def build_news_signals(
    calendar: pd.DataFrame,
    weather: pd.DataFrame,
    news_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine enhanced news features with weather signals.
    
    Args:
        calendar: DataFrame with Date and Commodity
        weather: DataFrame with Date and weather columns
        news_features: DataFrame with enhanced NLP features
    
    Returns:
        DataFrame with fused multimodal features
    """
    frame = calendar.merge(weather, on="Date", how="left")
    
    # Merge enhanced news features
    if news_features is None or news_features.empty:
        # Create default features if no news
        frame["News_Sentiment_Score"] = 0.0
        frame["Sentiment_Roll_Mean_7D"] = 0.0
        frame["Event_Disruption_Flag"] = 0
        frame["Disruption_Count_7D"] = 0
        frame["Dominant_Event"] = "no_event"
        frame["Event_Diversity"] = 0
        frame["News_Intensity"] = 0
    else:
        # Merge news features
        news_cols = [
            'Date', 'Commodity', 'News_Sentiment_Score', 'Sentiment_Roll_Mean_7D',
            'Event_Disruption_Flag', 'Disruption_Count_7D', 'Dominant_Event',
            'Event_Diversity', 'News_Intensity',
        ]
        available_cols = [c for c in news_cols if c in news_features.columns]
        
        frame = frame.merge(
            news_features[available_cols],
            on=["Date", "Commodity"],
            how="left"
        )
        
        # Fill missing news data with defaults
        frame["News_Sentiment_Score"] = frame["News_Sentiment_Score"].fillna(0.0)
        frame["Sentiment_Roll_Mean_7D"] = frame["Sentiment_Roll_Mean_7D"].fillna(0.0)
        frame["Event_Disruption_Flag"] = frame["Event_Disruption_Flag"].fillna(0).astype(int)
        frame["Disruption_Count_7D"] = frame["Disruption_Count_7D"].fillna(0)
        frame["Dominant_Event"] = frame["Dominant_Event"].fillna("no_event")
        frame["Event_Diversity"] = frame["Event_Diversity"].fillna(0)
        frame["News_Intensity"] = frame["News_Intensity"].fillna(0)
    
    # Forward-fill missing sentiment values
    frame = frame.sort_values(['Commodity', 'Date'])
    frame["News_Sentiment_Score"] = frame.groupby("Commodity")["News_Sentiment_Score"].transform(
        lambda s: s.interpolate(method='linear', limit_direction='both').fillna(0.0)
    )
    
    # Weather-based sentiment (fallback/supplementary)
    month = frame["Date"].dt.month
    rainfall = frame["Rainfall_mm"].fillna(0.0)
    max_temp = frame["Max_Temp_C"].fillna(frame["Max_Temp_C"].median())
    
    unseasonal_rain = (rainfall > UNSEASONAL_RAIN_MM) & (~month.isin(MONSOON_MONTHS))
    heatwave = max_temp > HEATWAVE_TEMP_C
    extreme_rain = rainfall > 50.0
    
    weather_sentiment = np.zeros(len(frame), dtype=float)
    weather_sentiment = np.where(unseasonal_rain, weather_sentiment - 0.55, weather_sentiment)
    weather_sentiment = np.where(heatwave, weather_sentiment - 0.40, weather_sentiment)
    weather_sentiment = np.where(extreme_rain, weather_sentiment - 0.20, weather_sentiment)
    weather_sentiment = np.clip(weather_sentiment, -1.0, 1.0)
    
    # Blend news sentiment with weather sentiment (if news exists, use 70-30; else 100% weather)
    has_news = frame["News_Sentiment_Score"].abs() > 0.01
    frame["Combined_Sentiment"] = np.where(
        has_news,
        0.70 * frame["News_Sentiment_Score"] + 0.30 * weather_sentiment,
        weather_sentiment,
    )
    frame["Combined_Sentiment"] = np.clip(frame["Combined_Sentiment"], -1.0, 1.0)
    
    # Update disruption flag to include weather disruptions
    weather_disruption = unseasonal_rain | heatwave | (frame["Combined_Sentiment"] <= -0.50)
    frame["Event_Disruption_Flag"] = (
        (frame["Event_Disruption_Flag"] == 1) | weather_disruption
    ).astype(int)
    
    return frame


# ========== MAIN BUILD FUNCTION ==========

def build_raw_multimodal(use_enhanced_nlp: bool = True) -> pd.DataFrame:
    """
    Build raw multimodal dataset with optional enhanced NLP.
    
    Args:
        use_enhanced_nlp: If True, use enhanced NLP pipeline; else use basic VADER
    
    Returns:
        DataFrame with daily prices, weather, and news features
    """
    ensure_directories()
    
    # Load structured data
    prices = load_gujarat_top_prices()
    weather = load_ahmedabad_weather()
    calendar = prices[["Date", "Commodity"]].drop_duplicates()
    
    # Load news features (enhanced or basic)
    if use_enhanced_nlp:
        LOGGER.info("Using enhanced NLP pipeline")
        news_features = fetch_enhanced_news_features()
    else:
        LOGGER.info("Using basic VADER sentiment")
        # Fallback to old implementation (not shown here, would load NEWS_RSS_CACHE_CSV)
        news_features = pd.DataFrame()  # Placeholder
    
    # Fuse signals
    news_weather = build_news_signals(calendar, weather, news_features)
    
    # Merge with prices
    merged = prices.merge(
        news_weather.drop(columns=["Max_Temp_C", "Min_Temp_C", "Rainfall_mm"], errors="ignore"),
        on=["Date", "Commodity"],
        how="left",
    )
    merged = merged.merge(weather, on="Date", how="left")
    
    # Select columns
    keep = [
        "Date",
        "Commodity",
        "Modal_Price",
        "Min_Price",
        "Max_Price",
        "Active_Mandis_Reporting",
        "Max_Temp_C",
        "Min_Temp_C",
        "Rainfall_mm",
        "News_Sentiment_Score",
        "Sentiment_Roll_Mean_7D",
        "Event_Disruption_Flag",
        "Disruption_Count_7D",
        "Dominant_Event",
        "Event_Diversity",
        "News_Intensity",
        "Combined_Sentiment",
    ]
    
    if "Arrival_Volume" in merged.columns:
        keep.insert(5, "Arrival_Volume")
    
    existing = [c for c in keep if c in merged.columns]
    return merged[existing].sort_values(["Commodity", "Date"]).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    print("Building multimodal dataset with enhanced NLP...")
    df = build_raw_multimodal(use_enhanced_nlp=True)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDominant events distribution:")
    if 'Dominant_Event' in df.columns:
        print(df['Dominant_Event'].value_counts())
    
    print(f"\nNews intensity stats:")
    if 'News_Intensity' in df.columns:
        print(df['News_Intensity'].describe())
