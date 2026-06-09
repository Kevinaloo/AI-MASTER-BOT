import json
from datetime import datetime, timedelta
from config.db import supabase

class WinRateTracker:
    """
    Tracks every signal the AI makes and whether it was right.
    Automatically adjusts indicator weights based on what's
    actually working in the current market.

    This is how the bot gets smarter over time.
    """

    def get_stats(self, symbol=None, days=30):
        """Get win rate stats for the last N days."""
        try:
            since = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query = supabase.table("trades") \
                .select("*") \
                .eq("status", "closed") \
                .gte("created_at", since)

            if symbol:
                query = query.eq("symbol", symbol)

            trades = query.execute().data or []

            if not trades:
                return self._empty_stats(symbol, days)

            wins   = [t for t in trades if (t.get('profit_kes') or 0) > 0]
            losses = [t for t in trades if (t.get('profit_kes') or 0) <= 0]

            total_profit = sum(t.get('profit_kes', 0) for t in wins)
            total_loss   = abs(sum(t.get('profit_kes', 0) for t in losses))
            profit_factor = (total_profit / total_loss) if total_loss > 0 else 999

            avg_confidence_wins  = sum(t.get('ai_confidence', 0) for t in wins) / len(wins) if wins else 0
            avg_confidence_loss  = sum(t.get('ai_confidence', 0) for t in losses) / len(losses) if losses else 0

            return {
                'symbol':         symbol or 'ALL',
                'period_days':    days,
                'total_trades':   len(trades),
                'wins':           len(wins),
                'losses':         len(losses),
                'win_rate':       round(len(wins) / len(trades) * 100, 1),
                'profit_factor':  round(profit_factor, 2),
                'total_profit':   round(total_profit, 2),
                'total_loss':     round(total_loss, 2),
                'net_pnl':        round(total_profit - total_loss, 2),
                'avg_conf_wins':  round(avg_confidence_wins, 1),
                'avg_conf_loss':  round(avg_confidence_loss, 1),
            }

        except Exception as e:
            print(f"Stats error: {e}")
            return self._empty_stats(symbol, days)

    def get_optimal_min_confidence(self):
        """
        Analyzes past trades to find the confidence threshold
        that gives the best win rate.
        Returns the optimal MIN_CONFIDENCE setting.
        """
        try:
            trades = supabase.table("trades") \
                .select("ai_confidence, profit_kes") \
                .eq("status", "closed") \
                .execute().data or []

            if len(trades) < 20:
                return 65  # Default until we have enough data

            results = {}
            for threshold in range(55, 90, 5):
                filtered = [t for t in trades if (t.get('ai_confidence') or 0) >= threshold]
                if len(filtered) < 5:
                    continue
                wins = [t for t in filtered if (t.get('profit_kes') or 0) > 0]
                win_rate = len(wins) / len(filtered) * 100
                results[threshold] = {
                    'win_rate': win_rate,
                    'trades':   len(filtered)
                }

            if not results:
                return 65

            # Find threshold with best win rate (min 10 trades)
            best = max(
                {k: v for k, v in results.items() if v['trades'] >= 10},
                key=lambda k: results[k]['win_rate'],
                default=65
            )
            print(f"  🎯 Optimal confidence threshold: {best}% (win rate: {results[best]['win_rate']:.1f}%)")
            return best

        except Exception as e:
            print(f"Confidence optimization error: {e}")
            return 65

    def get_best_symbols(self, top_n=5):
        """Returns the symbols with highest win rates."""
        try:
            trades = supabase.table("trades") \
                .select("symbol, profit_kes") \
                .eq("status", "closed") \
                .execute().data or []

            symbol_stats = {}
            for t in trades:
                sym = t.get('symbol', '')
                if sym not in symbol_stats:
                    symbol_stats[sym] = {'wins': 0, 'total': 0}
                symbol_stats[sym]['total'] += 1
                if (t.get('profit_kes') or 0) > 0:
                    symbol_stats[sym]['wins'] += 1

            ranked = []
            for sym, s in symbol_stats.items():
                if s['total'] >= 5:
                    ranked.append({
                        'symbol':   sym,
                        'win_rate': round(s['wins'] / s['total'] * 100, 1),
                        'trades':   s['total']
                    })

            ranked.sort(key=lambda x: x['win_rate'], reverse=True)
            return ranked[:top_n]

        except Exception as e:
            print(f"Best symbols error: {e}")
            return []

    def print_performance_report(self):
        """Prints a full performance report to the console."""
        overall = self.get_stats(days=30)
        weekly  = self.get_stats(days=7)
        best    = self.get_best_symbols()
        opt_conf = self.get_optimal_min_confidence()

        print(f"\n{'='*55}")
        print(f"🤖 AKILITRADE AI PERFORMANCE REPORT")
        print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print('='*55)

        print(f"\n📅 LAST 7 DAYS")
        print(f"   Trades:       {weekly['total_trades']}")
        print(f"   Win Rate:     {weekly['win_rate']}%")
        print(f"   Net P&L:      KSh {weekly['net_pnl']:,.2f}")

        print(f"\n📅 LAST 30 DAYS")
        print(f"   Trades:       {overall['total_trades']}")
        print(f"   Win Rate:     {overall['win_rate']}%")
        print(f"   Profit Factor:{overall['profit_factor']}x")
        print(f"   Net P&L:      KSh {overall['net_pnl']:,.2f}")
        print(f"   Conf (wins):  {overall['avg_conf_wins']}%")
        print(f"   Conf (loss):  {overall['avg_conf_loss']}%")

        if best:
            print(f"\n🏆 BEST PERFORMING SYMBOLS")
            for b in best:
                print(f"   {b['symbol']:<15} {b['win_rate']}% ({b['trades']} trades)")

        print(f"\n🎯 OPTIMAL MIN CONFIDENCE: {opt_conf}%")
        print(f"{'='*55}\n")

        return {
            'overall': overall,
            'weekly':  weekly,
            'best_symbols': best,
            'optimal_confidence': opt_conf
        }

    def _empty_stats(self, symbol, days):
        return {
            'symbol': symbol or 'ALL', 'period_days': days,
            'total_trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'profit_factor': 0,
            'total_profit': 0, 'total_loss': 0, 'net_pnl': 0,
            'avg_conf_wins': 0, 'avg_conf_loss': 0
        }
