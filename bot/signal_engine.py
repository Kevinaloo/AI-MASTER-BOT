import os
import time
from datetime import datetime
from dotenv import load_dotenv
from config.db import supabase
from data.market_data import (
    fetch_crypto_candles, fetch_forex_candles,
    fetch_forex_rate, CRYPTO_PAIRS, FOREX_PAIRS
)
from ai.technical import calculate_indicators
from ai.ensemble import ensemble_decision

load_dotenv()

MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "65"))
DRY_RUN        = os.getenv("DRY_RUN", "true").lower() == "true"


def save_signal(signal):
    """Saves an AI signal to Supabase so the dashboard shows it live."""
    try:
        result = supabase.table("signals").insert({
            "symbol":              signal["symbol"],
            "market":              signal["market"],
            "action":              signal["action"],
            "confidence":          signal["confidence"],
            "entry_price":         signal["entry_price"],
            "target_price":        signal["target_price"],
            "stop_loss":           signal["stop_loss"],
            "reasoning_english":   signal["reasoning_english"],
            "reasoning_swahili":   signal["reasoning_swahili"],
        }).execute()
        print(f"  ✅ Signal saved: {signal['action']} {signal['symbol']} ({signal['confidence']}% confidence)")
        return result
    except Exception as e:
        print(f"  ❌ Error saving signal: {e}")
        return None


def analyze_crypto():
    """Runs AI analysis on all crypto pairs."""
    print(f"\n{'='*50}")
    print(f"🪙  CRYPTO ANALYSIS — {datetime.now().strftime('%H:%M:%S')}")
    print('='*50)

    for symbol in CRYPTO_PAIRS:
        try:
            print(f"\n  Analyzing {symbol}...")

            # Get candle data
            df = fetch_crypto_candles(symbol, timeframe="1h", limit=200)
            if df is None or len(df) < 50:
                print(f"  ⚠️  Not enough data for {symbol}, skipping")
                continue

            # Calculate all technical indicators
            indicators = calculate_indicators(df)

            # Run ensemble AI decision
            signal = ensemble_decision(
                symbol=symbol,
                market="crypto",
                indicators=indicators,
                is_kenya_forex=False
            )

            print(f"  📊 {signal['action']} | Confidence: {signal['confidence']}% | Tech: {signal['tech_score']} | Sentiment: {signal['sentiment_label']}")
            print(f"  💬 {signal['reasoning_english'][:80]}...")

            # Only save signals above minimum confidence
            if signal["confidence"] >= MIN_CONFIDENCE:
                save_signal(signal)
            else:
                print(f"  ⏭️  Confidence too low ({signal['confidence']}%), skipping save")

            # Small delay to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Error analyzing {symbol}: {e}")
            continue


def analyze_forex():
    """Runs AI analysis on all forex pairs."""
    print(f"\n{'='*50}")
    print(f"💱  FOREX ANALYSIS — {datetime.now().strftime('%H:%M:%S')}")
    print('='*50)

    for pair in FOREX_PAIRS:
        symbol = pair["symbol"]
        base   = pair["base"]
        quote  = pair["quote"]
        is_kes = "KES" in symbol

        try:
            print(f"\n  Analyzing {symbol}...")

            # Get candle data
            df = fetch_forex_candles(base, quote, limit=200)
            if df is None or len(df) < 50:
                print(f"  ⚠️  Not enough data for {symbol}, skipping")
                continue

            # Calculate indicators
            indicators = calculate_indicators(df)

            # Override price with live rate for accuracy
            live_rate = fetch_forex_rate(base, quote)
            if live_rate:
                indicators["current_price"] = live_rate

            # Run ensemble AI
            signal = ensemble_decision(
                symbol=symbol,
                market="forex",
                indicators=indicators,
                is_kenya_forex=is_kes
            )

            print(f"  📊 {signal['action']} | Confidence: {signal['confidence']}% | Rate: {signal['entry_price']}")
            if is_kes:
                print(f"  🇰🇪 Kenya context applied")

            if signal["confidence"] >= MIN_CONFIDENCE:
                save_signal(signal)
            else:
                print(f"  ⏭️  Confidence too low ({signal['confidence']}%), skipping")

            time.sleep(0.5)

        except Exception as e:
            print(f"  ❌ Error analyzing {symbol}: {e}")
            continue


def run_analysis():
    """Full analysis cycle — crypto + forex."""
    print(f"\n🤖 AkiliTrade AI Bot Starting...")
    print(f"   DRY_RUN: {DRY_RUN}")
    print(f"   Min confidence: {MIN_CONFIDENCE}%")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    analyze_crypto()
    analyze_forex()

    print(f"\n✅ Analysis complete — {datetime.now().strftime('%H:%M:%S')}")
    print("   Next run in 1 hour...\n")


if __name__ == "__main__":
    run_analysis()
