"""Load Gujarat TOP mandi prices, Ahmedabad weather, and Google News RSS sentiment."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import feedparser
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import (
    COMMODITIES,
    DATE_END,
    DATE_START,
    HEATWAVE_TEMP_C,
    MONSOON_MONTHS,
    NEWS_RSS_CACHE_CSV,
    PRICE_CSV,
    TARGET_STATE,
    UNSEASONAL_RAIN_MM,
    WEATHER_CSV,
    ensure_directories,
)

LOGGER = logging.getLogger(__name__)
VADER_ANALYZER = SentimentIntensityAnalyzer()


def load_gujarat_top_prices(path: str | None = None) -> pd.DataFrame:
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


def load_ahmedabad_weather(path: str | None = None) -> pd.DataFrame:
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


def _gdelt_datetime(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


def _fetch_google_news_rss(commodity: str, max_retries: int = 2) -> list[dict[str, Any]]:
    """
    Fetch historical news headlines from Google News RSS for a given commodity.
    Returns list of dicts with 'title', 'published', and 'link'.
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
                
                # Convert time struct to datetime
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


def _compute_vader_sentiment(text: str) -> float:
    """
    Compute VADER compound sentiment score for a text.
    Returns value in [-1, 1] range.
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    scores = VADER_ANALYZER.polarity_scores(text)
    return scores['compound']


def fetch_news_sentiment_rss(commodities: list[str] | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch news sentiment from Google News RSS using VADER sentiment analysis.
    Returns DataFrame with Date, Commodity, News_Sentiment_Score columns.
    """
    ensure_directories()
    commodities = commodities or COMMODITIES
    start = pd.Timestamp(DATE_START)
    end = pd.Timestamp(DATE_END)
    
    if use_cache and NEWS_RSS_CACHE_CSV.exists():
        cached = pd.read_csv(NEWS_RSS_CACHE_CSV, parse_dates=["Date"])
        LOGGER.info(f"Loaded RSS sentiment cache from {NEWS_RSS_CACHE_CSV} ({len(cached)} rows)")
        return cached
    
    all_sentiment_data = []
    
    for commodity in commodities:
        articles = _fetch_google_news_rss(commodity)
        
        if not articles:
            LOGGER.warning(f"No articles found for {commodity}, will use weather-based fallback")
            continue
        
        # Compute sentiment for each article
        article_sentiments = []
        for article in articles:
            pub_date = pd.Timestamp(article['published'])
            
            # Filter to our date range
            if pub_date < start or pub_date > end:
                continue
            
            sentiment = _compute_vader_sentiment(article['title'])
            
            article_sentiments.append({
                'Date': pub_date.date(),
                'Commodity': commodity,
                'Sentiment': sentiment,
                'Title': article['title']
            })
        
        if article_sentiments:
            LOGGER.info(f"Processed {len(article_sentiments)} articles for {commodity} in date range")
            all_sentiment_data.extend(article_sentiments)
    
    if not all_sentiment_data:
        LOGGER.warning("No sentiment data collected from RSS, returning empty frame")
        empty = pd.DataFrame(columns=["Date", "Commodity", "News_Sentiment_Score"])
        empty.to_csv(NEWS_RSS_CACHE_CSV, index=False)
        return empty
    
    # Convert to DataFrame and aggregate by date and commodity
    sentiment_df = pd.DataFrame(all_sentiment_data)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    
    # Group by date and commodity, compute daily average sentiment
    daily_sentiment = (
        sentiment_df
        .groupby(['Date', 'Commodity'])['Sentiment']
        .mean()
        .reset_index()
        .rename(columns={'Sentiment': 'News_Sentiment_Score'})
    )
    
    # Apply 7-day rolling average to smooth the signal
    daily_sentiment = daily_sentiment.sort_values(['Commodity', 'Date'])
    daily_sentiment['News_Sentiment_Score'] = (
        daily_sentiment
        .groupby('Commodity')['News_Sentiment_Score']
        .transform(lambda x: x.rolling(window=7, min_periods=1, center=True).mean())
    )
    
    # Save to cache
    daily_sentiment.to_csv(NEWS_RSS_CACHE_CSV, index=False)
    LOGGER.info(f"Cached {len(daily_sentiment)} daily sentiment rows to {NEWS_RSS_CACHE_CSV}")
    
    return daily_sentiment


def build_news_signals(calendar: pd.DataFrame, weather: pd.DataFrame, news_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Combine RSS news sentiment with unseasonal rain and heatwave flags.

    `calendar` must contain Date and Commodity. Weather is merged on Date.
    News_sentiment contains Date, Commodity, News_Sentiment_Score from RSS+VADER.
    """
    frame = calendar.merge(weather, on="Date", how="left")
    
    # Merge news sentiment
    if news_sentiment is None or news_sentiment.empty:
        frame["News_Sentiment_Score"] = 0.0
    else:
        frame = frame.merge(
            news_sentiment[["Date", "Commodity", "News_Sentiment_Score"]], 
            on=["Date", "Commodity"], 
            how="left"
        )
    
    # Forward-fill and backward-fill missing sentiment values, then blend with weather
    frame["News_Sentiment_Score"] = frame.groupby("Commodity")["News_Sentiment_Score"].transform(
        lambda s: s.interpolate(method='linear', limit_direction='both').fillna(0.0)
    )

    month = frame["Date"].dt.month
    rainfall = frame["Rainfall_mm"].fillna(0.0)
    max_temp = frame["Max_Temp_C"].fillna(frame["Max_Temp_C"].median())
    
    # Weather-based sentiment adjustments
    unseasonal_rain = (rainfall > UNSEASONAL_RAIN_MM) & (~month.isin(MONSOON_MONTHS))
    heatwave = max_temp > HEATWAVE_TEMP_C
    extreme_rain = rainfall > 50.0

    weather_sentiment = np.zeros(len(frame), dtype=float)
    weather_sentiment = np.where(unseasonal_rain, weather_sentiment - 0.55, weather_sentiment)
    weather_sentiment = np.where(heatwave, weather_sentiment - 0.40, weather_sentiment)
    weather_sentiment = np.where(extreme_rain, weather_sentiment - 0.20, weather_sentiment)
    weather_sentiment = np.clip(weather_sentiment, -1.0, 1.0)

    # Blend news sentiment with weather sentiment (60% news, 40% weather)
    # For dates without news, rely more on weather
    has_news = frame["News_Sentiment_Score"].abs() > 0.01
    combined = np.where(
        has_news,
        0.60 * frame["News_Sentiment_Score"] + 0.40 * weather_sentiment,
        weather_sentiment,
    )
    
    frame["News_Sentiment_Score"] = np.clip(combined, -1.0, 1.0)
    frame["Event_Disruption_Flag"] = (
        unseasonal_rain | heatwave | (frame["News_Sentiment_Score"] <= -0.50)
    ).astype(int)
    
    return frame


def build_raw_multimodal() -> pd.DataFrame:
    """Return daily Gujarat TOP prices joined with weather and news signals."""
    ensure_directories()
    prices = load_gujarat_top_prices()
    weather = load_ahmedabad_weather()
    news_sentiment = fetch_news_sentiment_rss()
    calendar = prices[["Date", "Commodity"]].drop_duplicates()
    news_weather = build_news_signals(calendar, weather, news_sentiment)
    merged = prices.merge(
        news_weather.drop(columns=["Max_Temp_C", "Min_Temp_C", "Rainfall_mm"], errors="ignore"),
        on=["Date", "Commodity"],
        how="left",
    )
    merged = merged.merge(weather, on="Date", how="left")
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
        "Event_Disruption_Flag",
    ]
    if "Arrival_Volume" in merged.columns:
        keep.insert(5, "Arrival_Volume")
    existing = [c for c in keep if c in merged.columns]
    return merged[existing].sort_values(["Commodity", "Date"]).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = build_raw_multimodal()
    print(out.head())
    print("rows:", len(out), "commodities:", sorted(out["Commodity"].unique()))
