from ai.technical import score_indicators
from ai.sentiment import get_sentiment, get_kenya_sentiment
from ai.regime import MarketRegimeDetector
from ai.position_sizer import PositionSizer
import random

regime_detector = MarketRegimeDetector()
position_sizer  = PositionSizer()

# Dynamic weights — adjusted based on market regime
WEIGHTS_TRENDING = {
    'technical':  65,
    'sentiment':  20,
    'kenya':      15,
}
WEIGHTS_SIDEWAYS = {
    'technical':  50,
    'sentiment':  35,
    'kenya':      15,
}

SWAHILI_BUY = [
    "{symbol} inaonyesha nguvu ya kupanda ({regime}). Ishara {tech_score}/100. {news}. Ingia kwa {price}.",
    "AI yetu inasema {symbol} ni wakati wa kununua. Confidence {confidence}%. Soko liko {regime}.",
    "Ishara nzuri kwa {symbol}! RSI={rsi:.0f}, trend ni chanya. {news}. Target: {target}.",
]
SWAHILI_SELL = [
    "{symbol} inaonyesha dalili za kushuka ({regime}). Confidence {confidence}%. Uza kwa {price}.",
    "AI inasema {symbol} itashuka. Ishara hasi {tech_score}/100. {news}. Stop loss: {stop}.",
    "Hatari! {symbol} iko katika trend hasi. Confidence {confidence}%. Jiepushe au uza.",
]
SWAHILI_HOLD = [
    "{symbol} haina mwelekeo wazi sasa ({regime}). Subiri ishara bora.",
    "Soko la {symbol} liko katika msongamano. AI inashauri usubiri. Confidence ni {confidence}% tu.",
]

def build_swahili(action, symbol, confidence, price, target, stop, rsi, tech_score, regime, sentiment_label):
    news = f"Habari ni {sentiment_label.lower()}" if sentiment_label != 'Neutral' else ""
    templates = SWAHILI_BUY if action == 'BUY' else SWAHILI_SELL if action == 'SELL' else SWAHILI_HOLD
    t = random.choice(templates)
    return t.format(
        symbol=symbol, confidence=confidence, price=price,
        target=target, stop=stop, rsi=rsi,
        tech_score=tech_score, regime=regime, news=news
    )

def build_english(action, signals, sentiment_data, regime_data, symbol):
    parts = []
    regime = regime_data.get('regime', 'UNKNOWN')
    label  = sentiment_data.get('sentiment_label', 'Neutral')
    count  = sentiment_data.get('article_count', 0)

    parts.append(f"Market regime: {regime} (ADX={regime_data.get('adx',0):.0f}).")
    if signals:
        parts.append("Technical: " + "; ".join(signals[:3]) + ".")
    if count > 0:
        parts.append(f"News sentiment is {label} ({count} articles).")
    if action == 'BUY':
        parts.append("AI consensus: BULLISH — entry opportunity detected.")
    elif action == 'SELL':
        parts.append("AI consensus: BEARISH — exit or short opportunity.")
    else:
        parts.append("AI consensus: NEUTRAL — wait for clearer signal.")
    return " ".join(parts)

def calculate_targets(action, price, market, atr_pct):
    """Dynamic targets based on current volatility (ATR)."""
    # Use ATR-based targets for accuracy
    if atr_pct > 0:
        move = atr_pct / 100 * 2.0   # 2× ATR target
        stop = atr_pct / 100 * 1.0   # 1× ATR stop (2:1 reward/risk)
    else:
        defaults = {'crypto': (0.04, 0.02), 'forex': (0.008, 0.004)}
        move, stop = defaults.get(market, (0.03, 0.015))

    if action == 'BUY':
        target = round(price * (1 + move), 6)
        stop_l = round(price * (1 - stop), 6)
    else:
        target = round(price * (1 - move), 6)
        stop_l = round(price * (1 + stop), 6)

    return target, stop_l

def ensemble_decision(symbol, market, indicators, wallet_kes=0, is_kenya_forex=False):
    """
    Full AI decision pipeline:
    1. Detect market regime
    2. If regime says skip → return HOLD immediately
    3. Score technical indicators
    4. Score sentiment
    5. Weighted ensemble vote
    6. Calculate position size
    7. Return complete signal
    """
    price = indicators.get('current_price', 0)
    atr_pct = indicators.get('atr_pct', 1.0)

    # ── STEP 1: REGIME CHECK ───────────────────────────────────
    # Build mini df for regime from indicators (simplified)
    import pandas as pd, numpy as np
    n = 50
    prices = [price * (1 + np.random.normal(0, 0.005)) for _ in range(n)]
    prices[-1] = price
    df_mini = pd.DataFrame({
        'close': prices,
        'high':  [p * 1.003 for p in prices],
        'low':   [p * 0.997 for p in prices],
        'volume':[1000] * n
    })
    regime_data = regime_detector.detect(df_mini)
    regime      = regime_data.get('regime', 'UNKNOWN')

    # Hard skip on high volatility
    if not regime_data.get('trade', True):
        return {
            'symbol': symbol, 'market': market,
            'action': 'HOLD', 'confidence': 30,
            'entry_price': price, 'target_price': price, 'stop_loss': price,
            'reasoning_english': f"Skipping — {regime_data.get('reason', 'regime filter')}",
            'reasoning_swahili': f"AI inashauri usubiri. Soko liko {regime}.",
            'regime': regime, 'regime_trade': False,
            'position_size_kes': 0
        }

    # ── STEP 2: TECHNICAL SCORE ────────────────────────────────
    from ai.technical import score_indicators as _score
    tech_score, signals = _score(indicators)
    tech_norm = max(-100, min(100, tech_score))

    # ── STEP 3: SENTIMENT ──────────────────────────────────────
    sentiment_data  = get_sentiment(symbol, market)
    sent_norm       = sentiment_data['sentiment_score'] * 100

    kenya_score = 0
    if is_kenya_forex:
        kenya_score = get_kenya_sentiment() * 100

    # ── STEP 4: WEIGHTED VOTE ──────────────────────────────────
    W = WEIGHTS_TRENDING if 'TREND' in regime else WEIGHTS_SIDEWAYS
    weighted = (
        tech_norm * W['technical'] / 100 +
        sent_norm * W['sentiment'] / 100 +
        kenya_score * W['kenya']   / 100
    )

    # ── STEP 5: REGIME BIAS ────────────────────────────────────
    # In a strong uptrend, require stronger bearish signal to sell
    if regime == 'STRONG_TREND_UP'   and weighted < 0: weighted *= 0.6
    if regime == 'STRONG_TREND_DOWN' and weighted > 0: weighted *= 0.6

    # ── STEP 6: DECISION ───────────────────────────────────────
    if weighted >= 20:
        action = 'BUY'
    elif weighted <= -20:
        action = 'SELL'
    else:
        action = 'HOLD'

    # Confidence: map to 50–95%
    confidence = int(50 + (abs(weighted) / 100) * 45)
    confidence = max(50, min(95, confidence))

    # Regime confidence bonus/penalty
    if regime in ('STRONG_TREND_UP', 'STRONG_TREND_DOWN'):
        confidence = min(95, confidence + 5)
    elif regime == 'SIDEWAYS':
        confidence = max(50, confidence - 10)

    # ── STEP 7: TARGETS ────────────────────────────────────────
    target, stop = calculate_targets(action, price, market, indicators.get('atr_pct', 1.0))

    # ── STEP 8: POSITION SIZE ──────────────────────────────────
    pos = position_sizer.calculate(wallet_kes, confidence, regime_data)

    rsi = indicators.get('rsi', 50)
    reasoning_en = build_english(action, signals, sentiment_data, regime_data, symbol)
    reasoning_sw = build_swahili(
        action, symbol, confidence, price, target, stop,
        rsi, tech_norm, regime, sentiment_data['sentiment_label']
    )

    return {
        'symbol':              symbol,
        'market':              market,
        'action':              action,
        'confidence':          confidence,
        'entry_price':         round(price, 6),
        'target_price':        target,
        'stop_loss':           stop,
        'reasoning_english':   reasoning_en,
        'reasoning_swahili':   reasoning_sw,
        'regime':              regime,
        'regime_trade':        True,
        'tech_score':          round(tech_norm, 1),
        'sentiment_score':     round(sent_norm, 1),
        'sentiment_label':     sentiment_data['sentiment_label'],
        'position_size_kes':   pos.get('trade_size_kes', 0),
        'risk_pct':            pos.get('risk_pct', 2.0),
    }
