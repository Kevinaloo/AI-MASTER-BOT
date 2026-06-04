import schedule
import time
from bot.signal_engine import run_analysis

print("🚀 AkiliTrade AI Bot — Starting up...")
print("   Kenya's first AI trading signal engine")
print("   Crypto + Forex analysis every hour\n")

# Run immediately on startup
run_analysis()

# Then run every hour automatically
schedule.every(1).hours.do(run_analysis)

# Also run a quick crypto-only scan every 15 minutes
from bot.signal_engine import analyze_crypto
schedule.every(15).minutes.do(analyze_crypto)

print("⏰ Scheduler active:")
print("   Every 15 min → Crypto signals")
print("   Every 1 hour → Full analysis (Crypto + Forex)")
print("\nBot is running 24/7... Press Ctrl+C to stop\n")

while True:
    schedule.run_pending()
    time.sleep(30)
