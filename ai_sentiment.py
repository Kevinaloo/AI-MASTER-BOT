import os
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Positive & negative keyword lists tuned for crypto/forex/Kenya markets
BULLISH_WORDS = [
    "surge", "rally", "breakout", "bullish", "gain", "rise", "soar",
    "jump", "climb", "uptrend", "record high", "buy", "growth",
    "adoption", "approval", "partnership", "upgrade", "positive",
    "profit", "boom", "recovery", "strong", "outperform"
]

BEARISH_WORDS = [
    "crash", "drop", "fall", "bearish", "decline", "plunge", "dump",
    "sell", "loss", "ban", "hack", "fraud", "lawsuit", "regulation",
    "warning", "risk", "weak", "fear", "panic", "crisis", "sanctions",
    "inflation", "recession", "debt", "default"
]

KENYA_POSITIVE = [
    "Kenya growth", "CBK rate cut", "KES strengthens", "NSE rally",
    "Safaricom profit", "Kenya investment", "IMF approval Kenya",
    "Kenya tech boom", "M-Pesa growth"
]

KENYA_NEGATIVE = [
    "Kenya drought", "CBK rate hike", "KES weakens", "Kenya inflation",
    "Kenya protests", "Kenya debt", "power outage Kenya", "fuel price Kenya"
]


def fetch_news(query, max_articles=10):
    """Fetches recent news articles for a given query."""
    if not NEWS_API_KEY:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "language": "en",
            "apiKey": NEWS_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"News fetch error: {e}")
        return []


def score_text(text):
    """
    Simple keyword-based sentiment scorer.
    Returns a score between -1.0 (very bearish) and +1.0 (very bullish).
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    bull_count = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear_count = sum(1 for w in BEARISH_WORDS if w in text_lower)
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return (bull_count - bear_count) / total


def get_sentiment(symbol, market="crypto"):
    """
    Fetches news for a symbol and returns:
    - sentiment_score: -1.0 to +1.0
    - sentiment_label: Positive / Negative / Neutral
    - headline: most relevant headline
    - article_count: how many articles found
    """
    # Build search query per market
    if market == "crypto":
        clean = symbol.replace("/USDT", "").replace("/USD", "")
        query = f"{clean} cryptocurrency price"
    elif market == "forex":
        query = f"{symbol} forex exchange rate"
    else:
        query = symbol

    articles = fetch_news(query, max_articles=15)

    if not articles:
        # Fallback: neutral sentiment
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "headline": "No recent news found",
            "article_count": 0
        }

    scores = []
    for article in articles:
        title = article.get("title", "")
        desc  = article.get("description", "")
        combined = f"{title} {desc}"
        scores.append(score_text(combined))

    avg_score = sum(scores) / len(scores)

    # Label
    if avg_score > 0.15:
        label = "Positive"
    elif avg_score < -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    # Best headline (highest absolute score)
    best_article = max(articles, key=lambda a: abs(score_text(
        f"{a.get('title','')} {a.get('description','')}"
    )))

    return {
        "sentiment_score": round(avg_score, 3),
        "sentiment_label": label,
        "headline": best_article.get("title", ""),
        "article_count": len(articles)
    }


def get_kenya_sentiment():
    """
    Special sentiment analysis focused on Kenyan economic news.
    Used to adjust forex (USD/KES) predictions.
    """
    articles = fetch_news("Kenya economy forex shilling", max_articles=10)
    if not articles:
        return 0.0

    scores = []
    for article in articles:
        text = f"{article.get('title','')} {article.get('description','')}".lower()
        score = score_text(text)
        # Extra weight for Kenya-specific signals
        for pos in KENYA_POSITIVE:
            if pos.lower() in text:
                score += 0.2
        for neg in KENYA_NEGATIVE:
            if neg.lower() in text:
                score -= 0.2
        scores.append(score)

    return round(sum(scores) / len(scores), 3) if scores else 0.0
