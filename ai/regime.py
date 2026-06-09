import numpy as np
import pandas as pd

class MarketRegimeDetector:
    """
    Detects what kind of market we are in.
    This is the #1 improvement to win rate —
    using the RIGHT strategy for the RIGHT market condition.

    Regimes:
      STRONG_TREND_UP   → Use trend following, buy dips
      STRONG_TREND_DOWN → Use trend following, sell rallies
      WEAK_TREND_UP     → Cautious buys only
      WEAK_TREND_DOWN   → Cautious sells only
      SIDEWAYS          → Mean reversion, fade extremes
      HIGH_VOLATILITY   → Reduce size or skip entirely
      LOW_VOLATILITY    → Range trading
    """

    def detect(self, df):
        close  = df['close']
        high   = df['high']
        low    = df['low']
        n      = len(close)

        if n < 50:
            return {'regime': 'UNKNOWN', 'confidence': 0, 'trade': False}

        # ── 1. TREND STRENGTH via ADX ──────────────────────────
        adx = self._adx(high, low, close, 14)
        adx_val = adx['adx'].iloc[-1]
        di_pos  = adx['di_pos'].iloc[-1]
        di_neg  = adx['di_neg'].iloc[-1]

        # ── 2. VOLATILITY via ATR ──────────────────────────────
        atr      = self._atr(high, low, close, 14).iloc[-1]
        atr_pct  = (atr / close.iloc[-1]) * 100
        atr_avg  = self._atr(high, low, close, 14).rolling(50).mean().iloc[-1]
        vol_ratio = atr / atr_avg if atr_avg > 0 else 1.0

        # ── 3. TREND DIRECTION via EMA ─────────────────────────
        ema20  = close.ewm(span=20).mean().iloc[-1]
        ema50  = close.ewm(span=50).mean().iloc[-1]
        ema200 = close.ewm(span=200).mean().iloc[-1]
        price  = close.iloc[-1]

        # ── 4. CHOPPINESS INDEX (0=trending, 100=choppy) ───────
        chop = self._choppiness(high, low, close, 14)

        # ── REGIME CLASSIFICATION ──────────────────────────────
        regime     = 'SIDEWAYS'
        confidence = 50
        should_trade = True

        # Skip if extremely volatile (news spike etc)
        if vol_ratio > 2.5 or atr_pct > 5:
            return {
                'regime': 'HIGH_VOLATILITY',
                'confidence': 85,
                'trade': False,
                'reason': f'ATR {atr_pct:.1f}% — too volatile, skip trade',
                'adx': adx_val, 'atr_pct': atr_pct, 'chop': chop
            }

        # Strong trend
        if adx_val > 30:
            if di_pos > di_neg and price > ema50:
                regime = 'STRONG_TREND_UP'
                confidence = min(95, 60 + adx_val)
            elif di_neg > di_pos and price < ema50:
                regime = 'STRONG_TREND_DOWN'
                confidence = min(95, 60 + adx_val)
            should_trade = True

        # Weak trend
        elif adx_val > 20:
            if di_pos > di_neg:
                regime = 'WEAK_TREND_UP'
                confidence = 55
            else:
                regime = 'WEAK_TREND_DOWN'
                confidence = 55
            should_trade = True

        # Sideways / choppy
        elif chop > 61.8:
            regime = 'SIDEWAYS'
            confidence = 65
            should_trade = False  # Mean reversion only — skip for now

        # Low volatility
        elif atr_pct < 0.3:
            regime = 'LOW_VOLATILITY'
            confidence = 60
            should_trade = True  # Range trades ok

        # Long-term trend alignment bonus
        above_200 = price > ema200
        trend_aligned = (
            (regime == 'STRONG_TREND_UP'   and above_200) or
            (regime == 'STRONG_TREND_DOWN' and not above_200)
        )
        if trend_aligned:
            confidence = min(95, confidence + 10)

        return {
            'regime':     regime,
            'confidence': confidence,
            'trade':      should_trade,
            'adx':        round(adx_val, 1),
            'atr_pct':    round(atr_pct, 2),
            'chop':       round(chop, 1),
            'above_200':  above_200,
            'vol_ratio':  round(vol_ratio, 2),
            'reason':     f'ADX={adx_val:.0f}, ATR={atr_pct:.1f}%, Chop={chop:.0f}'
        }

    def _adx(self, high, low, close, period=14):
        tr  = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)

        dm_pos = high.diff()
        dm_neg = -low.diff()
        dm_pos = dm_pos.where((dm_pos > dm_neg) & (dm_pos > 0), 0)
        dm_neg = dm_neg.where((dm_neg > dm_pos) & (dm_neg > 0), 0)

        tr_s   = tr.ewm(alpha=1/period, adjust=False).mean()
        dmp_s  = dm_pos.ewm(alpha=1/period, adjust=False).mean()
        dmn_s  = dm_neg.ewm(alpha=1/period, adjust=False).mean()

        di_pos = 100 * dmp_s / tr_s
        di_neg = 100 * dmn_s / tr_s
        dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg)
        adx    = dx.ewm(alpha=1/period, adjust=False).mean()

        return pd.DataFrame({'adx': adx, 'di_pos': di_pos, 'di_neg': di_neg})

    def _atr(self, high, low, close, period=14):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _choppiness(self, high, low, close, period=14):
        atr_sum = (pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)).rolling(period).sum()

        highest = high.rolling(period).max()
        lowest  = low.rolling(period).min()
        chop    = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
        return chop.iloc[-1]
