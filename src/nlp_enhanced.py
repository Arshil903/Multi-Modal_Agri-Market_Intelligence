"""
Enhanced NLP module for agricultural news analysis.
Implements improvements: relevance filtering, event classification, deduplication, and multilingual support.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LOGGER = logging.getLogger(__name__)
VADER_ANALYZER = SentimentIntensityAnalyzer()

# Event classification keywords (improved NLP feature)
EVENT_KEYWORDS = {
    "supply_shortage": ["shortage", "scarce", "unavailable", "limited supply", "stock out", "deficit"],
    "crop_damage": ["damage", "destroyed", "loss", "pest", "disease", "infestation", "rot", "spoil"],
    "heavy_rainfall": ["heavy rain", "flood", "waterlogg", "deluge", "downpour", "torrential"],
    "drought": ["drought", "dry spell", "water scarcity", "no rain", "scanty rain", "deficit rainfall"],
    "transport_disruption": ["transport strike", "road block", "supply chain", "logistics", "truckers strike"],
    "government_policy": ["government", "policy", "regulation", "subsidy", "export ban", "import duty", "tax"],
    "demand_increase": ["demand surge", "high demand", "consumer demand", "festive demand", "panic buying"],
    "market_closure": ["market closed", "mandi shut", "trading halt", "closure", "lockdown"],
    "good_harvest": ["bumper crop", "record harvest", "surplus", "abundant", "good yield", "high production"],
    "price_rise": ["price rise", "price surge", "expensive", "costly", "price hike", "inflation"],
    "price_fall": ["price fall", "price drop", "cheap", "price crash", "decline", "lower price"],
}

# Relevance keywords for filtering (crop, region, market)
RELEVANCE_KEYWORDS = {
    "crop": ["onion", "potato", "tomato", "vegetable", "crop", "produce", "agriculture", "farming"],
    "region": ["gujarat", "ahmedabad", "surat", "rajkot", "vadodara", "gandhinagar", "bhavnagar"],
    "market": ["mandi", "market", "apmc", "wholesale", "bazaar", "arrivals", "price", "trading"],
}

# Gujarati/Hindi detection (simple heuristic)
GUJARATI_CHARS = set("અઆઇઈઉઊઋએઐઓઔકખગઘચછજઝઞટઠડઢણતથદધનપફબભમયરલવશષસહળ")
HINDI_CHARS = set("अआइईउऊऋएऐओऔकखगघचछजझञटठडढणतथदधनपफबभमयरलवशषसह")


def detect_language(text: str) -> str:
    """Detect if text is English, Gujarati, Hindi, or unknown."""
    if not text:
        return "unknown"
    
    gujarati_count = sum(1 for c in text if c in GUJARATI_CHARS)
    hindi_count = sum(1 for c in text if c in HINDI_CHARS)
    
    if gujarati_count > 3:
        return "gujarati"
    elif hindi_count > 3:
        return "hindi"
    elif re.search(r'[a-zA-Z]', text):
        return "english"
    else:
        return "unknown"


def compute_relevance_score(text: str, commodity: str) -> float:
    """
    Compute relevance score [0, 1] based on keyword matching.
    
    Args:
        text: News headline or article text
        commodity: Target commodity (Onion, Potato, Tomato)
    
    Returns:
        Relevance score (0 = irrelevant, 1 = highly relevant)
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    commodity_lower = commodity.lower()
    
    # Must mention the specific commodity (strong requirement)
    if commodity_lower not in text_lower:
        return 0.0
    
    # Score based on category keywords
    scores = []
    
    # Crop relevance (required)
    crop_matches = sum(1 for kw in RELEVANCE_KEYWORDS["crop"] if kw in text_lower)
    scores.append(min(crop_matches / 2, 1.0))  # Normalize
    
    # Region relevance (optional but boosts score)
    region_matches = sum(1 for kw in RELEVANCE_KEYWORDS["region"] if kw in text_lower)
    scores.append(min(region_matches / 2, 1.0))
    
    # Market relevance (optional but boosts score)
    market_matches = sum(1 for kw in RELEVANCE_KEYWORDS["market"] if kw in text_lower)
    scores.append(min(market_matches / 2, 1.0))
    
    # Average score (all must have some match)
    relevance = np.mean(scores)
    
    return relevance


def classify_events(text: str) -> dict[str, int]:
    """
    Classify which agricultural events are mentioned in the text.
    
    Args:
        text: News headline or article text
    
    Returns:
        Dictionary with event categories as keys and binary flags (0/1) as values
    """
    if not text:
        return {event: 0 for event in EVENT_KEYWORDS}
    
    text_lower = text.lower()
    event_flags = {}
    
    for event_category, keywords in EVENT_KEYWORDS.items():
        # Check if any keyword from this category appears
        matched = any(kw in text_lower for kw in keywords)
        event_flags[event_category] = 1 if matched else 0
    
    return event_flags


def compute_content_hash(text: str) -> str:
    """Generate a hash of the article content for deduplication."""
    if not text:
        return ""
    # Normalize text: lowercase, remove extra spaces, punctuation
    normalized = re.sub(r'[^\w\s]', '', text.lower().strip())
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def deduplicate_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate or near-duplicate articles based on content hash.
    
    Args:
        articles: List of article dicts with 'title' key
    
    Returns:
        Deduplicated list of articles
    """
    if not articles:
        return []
    
    seen_hashes = set()
    unique_articles = []
    
    for article in articles:
        content = article.get('title', '')
        content_hash = compute_content_hash(content)
        
        if content_hash and content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_articles.append(article)
    
    LOGGER.info(f"Deduplication: {len(articles)} → {len(unique_articles)} articles")
    return unique_articles


def compute_enhanced_sentiment(text: str) -> dict[str, float]:
    """
    Compute sentiment with VADER and additional metadata.
    
    Args:
        text: News headline or article text
    
    Returns:
        Dictionary with compound, positive, negative, neutral scores
    """
    if not text:
        return {
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
        }
    
    scores = VADER_ANALYZER.polarity_scores(text)
    return scores


def analyze_article(
    article: dict[str, Any],
    commodity: str,
    relevance_threshold: float = 0.3,
) -> dict[str, Any] | None:
    """
    Comprehensive NLP analysis of a single news article.
    
    Args:
        article: Dict with 'title', 'published', 'link'
        commodity: Target commodity (Onion, Potato, Tomato)
        relevance_threshold: Minimum relevance score to keep article
    
    Returns:
        Enriched article dict with NLP features, or None if irrelevant
    """
    title = article.get('title', '')
    
    if not title:
        return None
    
    # 1. Language detection
    language = detect_language(title)
    
    # Currently only process English (future: add translation)
    if language != "english":
        LOGGER.debug(f"Skipping non-English article: {title[:50]}")
        return None
    
    # 2. Relevance filtering
    relevance = compute_relevance_score(title, commodity)
    if relevance < relevance_threshold:
        LOGGER.debug(f"Irrelevant article (score={relevance:.2f}): {title[:50]}")
        return None
    
    # 3. Event classification
    events = classify_events(title)
    
    # 4. Sentiment analysis
    sentiment = compute_enhanced_sentiment(title)
    
    # 5. Content hash for deduplication
    content_hash = compute_content_hash(title)
    
    # Build enriched article
    enriched = {
        'title': title,
        'published': article.get('published'),
        'link': article.get('link', ''),
        'commodity': commodity,
        'relevance_score': relevance,
        'language': language,
        'content_hash': content_hash,
        'sentiment_compound': sentiment['compound'],
        'sentiment_positive': sentiment['pos'],
        'sentiment_negative': sentiment['neg'],
        'sentiment_neutral': sentiment['neu'],
        **events,  # Add all event flags
    }
    
    return enriched


def aggregate_news_by_date(
    articles_df: pd.DataFrame,
    date_col: str = 'published',
    commodity_col: str = 'commodity',
) -> pd.DataFrame:
    """
    Aggregate multiple articles per day into daily features.
    
    Args:
        articles_df: DataFrame with enriched articles
        date_col: Column name for publication date
        commodity_col: Column name for commodity
    
    Returns:
        Daily aggregated DataFrame with NLP features
    """
    if articles_df.empty:
        return pd.DataFrame()
    
    # Convert published to date (remove time)
    articles_df['Date'] = pd.to_datetime(articles_df[date_col]).dt.date
    articles_df['Date'] = pd.to_datetime(articles_df['Date'])
    
    # Aggregate by Date and Commodity
    agg_dict = {
        'sentiment_compound': 'mean',
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean',
        'sentiment_neutral': 'mean',
        'relevance_score': 'mean',
        'title': 'count',  # Article count per day
    }
    
    # Add event counts (sum of binary flags)
    for event in EVENT_KEYWORDS.keys():
        if event in articles_df.columns:
            agg_dict[event] = 'sum'
    
    daily = articles_df.groupby(['Date', commodity_col]).agg(agg_dict).reset_index()
    daily.rename(columns={'title': 'article_count'}, inplace=True)
    
    # Normalize event counts to [0, 1] range (proportion of articles mentioning event)
    for event in EVENT_KEYWORDS.keys():
        if event in daily.columns:
            daily[event] = daily[event] / daily['article_count'].clip(lower=1)
    
    return daily


def get_dominant_event(row: pd.Series) -> str:
    """
    Identify the most prominent event category for a given day.
    
    Args:
        row: DataFrame row with event columns
    
    Returns:
        Name of dominant event, or "no_event" if none
    """
    event_scores = {}
    for event in EVENT_KEYWORDS.keys():
        if event in row.index:
            event_scores[event] = row[event]
    
    if not event_scores or max(event_scores.values()) == 0:
        return "no_event"
    
    return max(event_scores, key=event_scores.get)


def create_enhanced_nlp_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final NLP features for ML models.
    
    Args:
        daily_df: Daily aggregated news DataFrame
    
    Returns:
        DataFrame with engineered NLP features
    """
    if daily_df.empty:
        return daily_df
    
    df = daily_df.copy()
    
    # 1. Rename primary sentiment feature
    df['News_Sentiment_Score'] = df['sentiment_compound']
    
    # 2. 7-day rolling sentiment (smoothing)
    df = df.sort_values(['Commodity', 'Date'])
    df['Sentiment_Roll_Mean_7D'] = df.groupby('Commodity')['News_Sentiment_Score'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # 3. Disruption flag (sentiment < -0.5 OR negative events)
    negative_events = ['supply_shortage', 'crop_damage', 'heavy_rainfall', 'drought', 'transport_disruption']
    df['negative_event_score'] = df[[e for e in negative_events if e in df.columns]].sum(axis=1)
    df['Event_Disruption_Flag'] = (
        (df['News_Sentiment_Score'] < -0.5) | (df['negative_event_score'] > 0.3)
    ).astype(int)
    
    # 4. Disruption count (7-day window)
    df['Disruption_Count_7D'] = df.groupby('Commodity')['Event_Disruption_Flag'].transform(
        lambda x: x.rolling(window=7, min_periods=1).sum()
    )
    
    # 5. Dominant event category
    df['Dominant_Event'] = df.apply(get_dominant_event, axis=1)
    
    # 6. Event diversity (how many different events mentioned)
    event_cols = [e for e in EVENT_KEYWORDS.keys() if e in df.columns]
    df['Event_Diversity'] = df[event_cols].apply(lambda x: (x > 0).sum(), axis=1)
    
    # 7. News intensity (article count normalized)
    df['News_Intensity'] = np.log1p(df['article_count'])  # Log scale
    
    return df


def validate_sentiment_sample(
    articles: list[dict[str, Any]],
    sample_size: int = 50,
) -> pd.DataFrame:
    """
    Extract a random sample of articles for manual sentiment validation.
    
    Args:
        articles: List of enriched articles
        sample_size: Number of articles to sample
    
    Returns:
        DataFrame with articles for manual labeling
    """
    if not articles or len(articles) < sample_size:
        sample_size = len(articles)
    
    import random
    sampled = random.sample(articles, sample_size)
    
    validation_df = pd.DataFrame([
        {
            'title': a['title'],
            'vader_sentiment': a['sentiment_compound'],
            'vader_label': 'positive' if a['sentiment_compound'] > 0.05 else (
                'negative' if a['sentiment_compound'] < -0.05 else 'neutral'
            ),
            'manual_label': '',  # To be filled manually
            'dominant_event': get_dominant_event(pd.Series(a)),
        }
        for a in sampled
    ])
    
    return validation_df


# Main processing function
def process_news_with_enhanced_nlp(
    articles: list[dict[str, Any]],
    commodity: str,
    relevance_threshold: float = 0.3,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Complete NLP pipeline: filter → deduplicate → classify → aggregate.
    
    Args:
        articles: Raw articles from RSS feed
        commodity: Target commodity
        relevance_threshold: Minimum relevance score
    
    Returns:
        Tuple of (daily_features_df, enriched_articles_list)
    """
    LOGGER.info(f"Processing {len(articles)} articles for {commodity}")
    
    # Step 1: Analyze each article
    enriched = []
    for article in articles:
        analyzed = analyze_article(article, commodity, relevance_threshold)
        if analyzed:
            enriched.append(analyzed)
    
    LOGGER.info(f"After relevance filtering: {len(enriched)} articles")
    
    if not enriched:
        return pd.DataFrame(), []
    
    # Step 2: Deduplicate
    enriched = deduplicate_articles(enriched)
    
    # Step 3: Convert to DataFrame
    articles_df = pd.DataFrame(enriched)
    
    # Step 4: Aggregate by date
    daily_df = aggregate_news_by_date(articles_df)
    
    # Step 5: Create final features
    features_df = create_enhanced_nlp_features(daily_df)
    
    LOGGER.info(f"Created daily features: {len(features_df)} rows")
    
    return features_df, enriched


if __name__ == "__main__":
    # Test event classification
    test_headlines = [
        "Heavy rainfall destroys 40% of Gujarat onion crop",
        "Potato prices surge due to supply shortage in Ahmedabad mandi",
        "Government announces export ban on onions",
        "Record tomato harvest boosts farmer income",
        "Transport strike disrupts vegetable arrivals at Gujarat markets",
    ]
    
    print("Event Classification Test:")
    print("-" * 80)
    for headline in test_headlines:
        events = classify_events(headline)
        sentiment = compute_enhanced_sentiment(headline)
        relevance = compute_relevance_score(headline, "Onion")
        
        active_events = [e for e, v in events.items() if v == 1]
        
        print(f"\nHeadline: {headline}")
        print(f"  Relevance: {relevance:.2f}")
        print(f"  Sentiment: {sentiment['compound']:.2f}")
        print(f"  Events: {', '.join(active_events) if active_events else 'None'}")
