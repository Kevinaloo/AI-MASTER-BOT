from ai.technical import score_indicators
from ai.sentiment import get_sentiment, get_kenya_sentiment

# Weights for each signal type (must add to 100)
WEIGHTS = {
    "technical":  60,   # Chart indicators
    "sentiment":  25,   # News sentiment
    "kenya":      15,   # Kenya-specific context (for forex)
}

# Swahili reasoning templates
SWAHILI_BUY = [
    "{symbol} inaonyesha nguvu ya kupanda. Ishara za kiufundi ni chanya ({tech_score}/100). {news_line} Wakati mzuri wa kununua.",
    "Uchambuzi wetu wa AI unaonyesha {symbol} iko tayari kupanda. RSI iko chini ({rsi:.0f}) na MACD inasema BUY. {news_line}",
    "Soko la {symbol} linaonyesha dalili za kupanda. Confidence yetu ni {confidence}%. Nunua kwa bei ya {price}.",
]

SWAHILI_SELL = [
    "{symbol} inaonyesha dalili za kushuka. Ishara za kiufundi ni hasi ({tech_score}/100). {news_line} Fikiria kuuza.",
    "AI yetu inasema {symbol} inaweza kushuka. RSI iko juu ({rsi:.0f}) na trend ni hasi. {news_line}",
    "Hatari ya kushuka kwa {symbol}. Confidence yetu ni {confidence}%. Uza kwa bei ya {price}.",
]

SWAHILI_HOLD = [
    "{symbol} iko imara sasa. Subiri ishara wazi kabla ya kuingia au kutoka. Confidence ni {confidence}% tu.",
    "Soko la {symbol} halina mwelekeo wazi. AI inashauri usubiri. {news_line}",
]

import random

def build_swahili(action, symbol, confidence, price, rsi, tech_score, sentiment):
    news_line = f"Habari za hivi karibuni ni {sentiment.lower()}." if sentiment != "Neutral" else ""
    templates = SWAHILI_BUY if action == "BUY" else SWAHILI_SELL if action == "SELL" else SWAHILI_HOLD
    template = random.choice(templates)
    return template.format(
        symbol=symbol, confidence=confidence, price=price,
        rsi=rsi, tech_score=tech_score, news_line=news_line
    )


def build_english(action, signals, sentiment_data, tech_score, symbol):
    headline = sentiment_data.get("headline", "")
    count    = sentiment_data.get("article_count", 0)
    label    = sentiment_data.get("sentiment_label", "Neutral")
    parts    = []
    if signals:
        parts.append("Technical signals: " + "; ".join(signals[:3]) + ".")
    if headline and count > 0:
        parts.append(f"News sentiment is {label} ({count} articles analysed).")
    if action == "BUY":
        parts.append("Overall AI consensus: BULLISH — good entry opportunity.")
    elif action == "SELL":
        parts.append("Overall AI consensus: BEARISH — consider exiting or shorting.")
    else:
        parts.append("Market is consolidating. Wait for clearer directional signal.")
    return " ".join(parts)


def calculate_targets(action, price, market):
    """Calculate target price and stop loss based on market volatility."""
    if market == "crypto":
        move_pct  = 0.04   # 4% target
        stop_pct  = 0.025  # 2.5% stop
    elif market == "forex":
        move_pct  = 0.008
        stop_pct  = 0.005
    else:
        move_pct  = 0.03
        stop_pct  = 0.02

    if action == "BUY":
        target = round(price * (1 + move_pct), 6)
        stop   = round(price * (1 - stop_pct), 6)
    else:
        target = round(price * (1 - move_pct), 6)
        stop   = round(price * (1 + stop_pct), 6)

    return target, stop


def ensemble_decision(symbol, market, indicators, is_kenya_forex=False):
    """
    Main AI decision engine.
    Combines technical analysis + sentiment + Kenya context.
    Returns a complete signal dict ready to save to Supabase.
    """
    # 1. Technical score (-100 to +100)
    tech_score, signals = score_indicators(indicators)
    tech_normalised = max(-100, min(100, tech_score))

    # 2. Sentiment score (-100 to +100)
    sentiment_data  = get_sentiment(symbol, market)
    sent_score_raw  = sentiment_data["sentiment_score"]    # -1 to +1
    sent_normalised = sent_score_raw * 100                 # scale to -100..+100

    # 3. Kenya context (only for USD/KES or KES pairs)
    kenya_score = 0
    if is_kenya_forex:
        raw = get_kenya_sentiment()
        kenya_score = raw * 100

    # 4. Weighted ensemble
    weighted = (
        (tech_normalised  * WEIGHTS["technical"] / 100) +
        (sent_normalised  * WEIGHTS["sentiment"] / 100) +
        (kenya_score      * WEIGHTS["kenya"]     / 100)
    )

    # 5. Determine action
    if weighted >= 20:
        action = "BUY"
    elif weighted <= -20:
        action = "SELL"
    else:
        action = "HOLD"

    # 6. Confidence: map weighted score to 50-95% range
    raw_conf   = abs(weighted)
    confidence = int(50 + (raw_conf / 100) * 45)
    confidence = max(50, min(95, confidence))

    # 7. Price targets
    price          = indicators.get("current_price", 0)
    target, stop   = calculate_targets(action, price, market)

    # 8. Build reasoning
    rsi       = indicators.get("rsi", 50)
    reasoning_en = build_english(action, signals, sentiment_data, tech_score, symbol)
    reasoning_sw = build_swahili(
        action, symbol, confidence, price,
        rsi, tech_score, sentiment_data["sentiment_label"]
    )

    return {
        "symbol":              symbol,
        "market":              market,
        "action":              action,
        "confidence":          confidence,
        "entry_price":         round(price, 6),
        "target_price":        target,
        "stop_loss":           stop,
        "reasoning_english":   reasoning_en,
        "reasoning_swahili":   reasoning_sw,
        "tech_score":          round(tech_normalised, 1),
        "sentiment_score":     round(sent_normalised, 1),
        "sentiment_label":     sentiment_data["sentiment_label"],
    }
