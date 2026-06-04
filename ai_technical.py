import pandas as pd
import numpy as np
import ta

def calculate_indicators(df):
    """
    Takes a dataframe of OHLCV candles and returns
    all technical indicators the AI uses to make decisions.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    indicators = {}

    # ── TREND INDICATORS ──────────────────────────────────────
    # RSI: 0-30 = oversold (buy), 70-100 = overbought (sell)
    indicators['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    # MACD: when macd crosses above signal = buy signal
    macd = ta.trend.MACD(close)
    indicators['macd'] = macd.macd().iloc[-1]
    indicators['macd_signal'] = macd.macd_signal().iloc[-1]
    indicators['macd_diff'] = macd.macd_diff().iloc[-1]

    # Bollinger Bands: price near lower band = buy, upper band = sell
    bb = ta.volatility.BollingerBands(close, window=20)
    indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
    indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
    indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
    indicators['bb_pct'] = bb.bollinger_pband().iloc[-1]  # 0=lower, 1=upper

    # Moving Averages
    indicators['ema_9']  = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    indicators['ema_21'] = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    indicators['ema_50'] = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    indicators['sma_200'] = ta.trend.SMAIndicator(close, window=200).sma_indicator().iloc[-1]

    # ADX: trend strength. >25 = strong trend
    adx = ta.trend.ADXIndicator(high, low, close)
    indicators['adx'] = adx.adx().iloc[-1]
    indicators['adx_pos'] = adx.adx_pos().iloc[-1]  # +DI
    indicators['adx_neg'] = adx.adx_neg().iloc[-1]  # -DI

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    indicators['stoch_k'] = stoch.stoch().iloc[-1]
    indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]

    # ── VOLUME INDICATORS ──────────────────────────────────────
    indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]

    # ── CURRENT PRICE INFO ────────────────────────────────────
    indicators['current_price'] = close.iloc[-1]
    indicators['prev_price']    = close.iloc[-2]
    indicators['price_change']  = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
    indicators['volume_current'] = volume.iloc[-1]
    indicators['volume_avg']     = volume.rolling(20).mean().iloc[-1]

    return indicators


def score_indicators(ind):
    """
    Converts raw indicators into a BUY/SELL/HOLD score.
    Returns score (-100 to +100) where:
      positive = bullish (buy)
      negative = bearish (sell)
      near zero = neutral (hold)
    """
    score = 0
    signals = []

    rsi = ind.get('rsi', 50)
    price = ind.get('current_price', 0)

    # RSI signal (weight: 20)
    if rsi < 30:
        score += 20
        signals.append(f"RSI oversold ({rsi:.1f}) — bullish")
    elif rsi < 45:
        score += 10
        signals.append(f"RSI approaching oversold ({rsi:.1f})")
    elif rsi > 70:
        score -= 20
        signals.append(f"RSI overbought ({rsi:.1f}) — bearish")
    elif rsi > 55:
        score -= 10
        signals.append(f"RSI approaching overbought ({rsi:.1f})")

    # MACD signal (weight: 20)
    macd_diff = ind.get('macd_diff', 0)
    if macd_diff > 0:
        score += 20
        signals.append("MACD bullish crossover")
    elif macd_diff < 0:
        score -= 20
        signals.append("MACD bearish crossover")

    # EMA trend (weight: 20)
    ema9  = ind.get('ema_9', price)
    ema21 = ind.get('ema_21', price)
    ema50 = ind.get('ema_50', price)
    if ema9 > ema21 > ema50:
        score += 20
        signals.append("EMA bullish alignment (9 > 21 > 50)")
    elif ema9 < ema21 < ema50:
        score -= 20
        signals.append("EMA bearish alignment (9 < 21 < 50)")

    # Bollinger Bands (weight: 15)
    bb_pct = ind.get('bb_pct', 0.5)
    if bb_pct < 0.2:
        score += 15
        signals.append("Price near Bollinger lower band — oversold")
    elif bb_pct > 0.8:
        score -= 15
        signals.append("Price near Bollinger upper band — overbought")

    # ADX trend strength (weight: 15)
    adx     = ind.get('adx', 20)
    adx_pos = ind.get('adx_pos', 0)
    adx_neg = ind.get('adx_neg', 0)
    if adx > 25:
        if adx_pos > adx_neg:
            score += 15
            signals.append(f"Strong uptrend — ADX {adx:.1f}")
        else:
            score -= 15
            signals.append(f"Strong downtrend — ADX {adx:.1f}")

    # Volume confirmation (weight: 10)
    vol_curr = ind.get('volume_current', 0)
    vol_avg  = ind.get('volume_avg', 1)
    if vol_avg > 0 and vol_curr > vol_avg * 1.5:
        signals.append("High volume confirms move")
        score = int(score * 1.1)  # amplify signal on high volume

    return score, signals
