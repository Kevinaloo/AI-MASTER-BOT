"""
Run this BEFORE deploying the bot live.
Tests strategy on 1 year of real historical data.
Only deploy pairs that score A or B grade.
"""
import sys
sys.path.insert(0, '.')

from data.market_data import fetch_crypto_candles, CRYPTO_PAIRS
from ai.technical import calculate_indicators
from ai.ensemble import ensemble_decision
from backtest.backtester import Backtester

backtester = Backtester()

def signal_fn(df):
    """Wrapper that generates a signal from a candle window."""
    if len(df) < 50:
        return None
    indicators = calculate_indicators(df)
    return ensemble_decision(
        symbol='TEST',
        market='crypto',
        indicators=indicators,
        wallet_kes=10000
    )

def run_all_backtests():
    print("\n🚀 AKILITRADE BACKTEST SUITE")
    print("   Testing all pairs before live deployment\n")

    results = []
    deploy_list  = []
    caution_list = []
    skip_list    = []

    for symbol in CRYPTO_PAIRS:
        print(f"Fetching data for {symbol}...")
        df = fetch_crypto_candles(symbol, timeframe='1h', limit=1000)

        if df is None or len(df) < 200:
            print(f"  ⚠️  Insufficient data for {symbol}\n")
            continue

        result = backtester.run(df, signal_fn, symbol=symbol, verbose=True)
        results.append(result)

        grade = result.get('grade', 'F')
        if grade in ('A+', 'A'):
            deploy_list.append(symbol)
        elif grade == 'B':
            caution_list.append(symbol)
        else:
            skip_list.append(symbol)

    # Final summary
    print("\n" + "="*55)
    print("📋 DEPLOYMENT RECOMMENDATION SUMMARY")
    print("="*55)

    if deploy_list:
        print(f"\n✅ DEPLOY LIVE (Grade A/A+):")
        for s in deploy_list:
            print(f"   {s}")

    if caution_list:
        print(f"\n⚠️  DEPLOY WITH CAUTION (Grade B — 50% size):")
        for s in caution_list:
            print(f"   {s}")

    if skip_list:
        print(f"\n❌ DO NOT DEPLOY (Grade C/F):")
        for s in skip_list:
            print(f"   {s}")

    print(f"\n{'='*55}")
    print(f"Run this weekly to re-evaluate performance.")
    print(f"{'='*55}\n")

    return results

if __name__ == "__main__":
    run_all_backtests()
