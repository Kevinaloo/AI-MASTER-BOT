import schedule
import time
import os
from datetime import datetime
from bot.signal_engine import run_analysis, analyze_crypto
from ai.win_tracker import WinRateTracker

tracker = WinRateTracker()

print("🚀 AkiliTrade AI Bot v2.0 — Starting...")
print("   🧠 Market Regime Detection: ON")
print("   📐 Dynamic Position Sizing: ON")
print("   🔄 Self-Learning Win Tracker: ON")
print("   🛡️  2% Max Risk Rule: ON\n")

def run_with_optimization():
    """Run analysis + auto-adjust confidence threshold."""
    # Get optimal confidence from past trades
    opt_conf = tracker.get_optimal_min_confidence()
    os.environ['MIN_CONFIDENCE'] = str(opt_conf)
    print(f"🎯 Auto-adjusted MIN_CONFIDENCE to {opt_conf}%")
    run_analysis()

def weekly_report():
    """Print full performance report every week."""
    print("\n📊 WEEKLY PERFORMANCE REPORT")
    tracker.print_performance_report()

# Run immediately
run_with_optimization()

# Schedule
schedule.every(15).minutes.do(analyze_crypto)
schedule.every(1).hours.do(run_with_optimization)
schedule.every().monday.at("08:00").do(weekly_report)

print("⏰ Scheduler:")
print("   Every 15 min  → Crypto signals")
print("   Every 1 hour  → Full analysis + auto-optimize confidence")
print("   Every Monday  → Weekly performance report")
print("\n✅ Bot running 24/7...\n")

while True:
    schedule.run_pending()
    time.sleep(30)
