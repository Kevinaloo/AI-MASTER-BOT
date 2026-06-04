import os
import ccxt
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── CRYPTO via Binance ────────────────────────────────────────────────────────

exchange = ccxt.binance({
    "apiKey":  os.getenv("BINANCE_API_KEY", ""),
    "secret":  os.getenv("BINANCE_SECRET", ""),
    "options": {"defaultType": "spot"},
})

CRYPTO_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

def fetch_crypto_candles(symbol, timeframe="1h", limit=200):
    """
    Returns a pandas DataFrame of OHLCV candles for a crypto pair.
    timeframe options: 1m, 5m, 15m, 1h, 4h, 1d
    """
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df  = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def fetch_crypto_price(symbol):
    """Returns just the current price of a crypto pair."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker["last"]
    except Exception as e:
        print(f"Price fetch error {symbol}: {e}")
        return None


# ── FOREX via free API ────────────────────────────────────────────────────────

FOREX_PAIRS = [
    {"symbol": "EUR/USD", "base": "EUR", "quote": "USD"},
    {"symbol": "GBP/USD", "base": "GBP", "quote": "USD"},
    {"symbol": "USD/KES", "base": "USD", "quote": "KES"},
    {"symbol": "USD/JPY", "base": "USD", "quote": "JPY"},
    {"symbol": "XAU/USD", "base": "XAU", "quote": "USD"},  # Gold
]

def fetch_forex_rate(base, quote):
    """
    Fetches live forex rate using free exchangerate-api.
    Returns current rate as float.
    """
    try:
        url  = f"https://api.exchangerate-api.com/v4/latest/{base}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        rate = data.get("rates", {}).get(quote)
        return float(rate) if rate else None
    except Exception as e:
        print(f"Forex fetch error {base}/{quote}: {e}")
        return None


def fetch_forex_candles(base, quote, limit=200):
    """
    Builds synthetic OHLCV candles for forex using hourly snapshots.
    Uses Binance stablecoin pairs as proxy where available,
    otherwise builds from current rate with realistic noise.
    """
    try:
        # Try Binance for USD pairs (crypto-adjacent forex)
        forex_map = {
            "EURUSD": "EUR/USDT",
            "GBPUSD": "GBP/USDT",
        }
        key = f"{base}{quote}"
        if key in forex_map:
            return fetch_crypto_candles(forex_map[key], timeframe="1h", limit=limit)

        # Fallback: generate synthetic candles from current rate
        import numpy as np
        current_rate = fetch_forex_rate(base, quote)
        if not current_rate:
            return None

        # Simulate realistic historical prices
        np.random.seed(42)
        returns   = np.random.normal(0.0001, 0.003, limit)
        prices    = [current_rate]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        prices = prices[::-1]  # oldest first

        df = pd.DataFrame({
            "open":   prices,
            "high":   [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            "low":    [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            "close":  prices,
            "volume": [np.random.randint(1000, 10000) for _ in prices],
        })
        return df

    except Exception as e:
        print(f"Forex candle error {base}/{quote}: {e}")
        return None
