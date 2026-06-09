import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    """
    Tests our trading strategy on historical data.
    Tells us EXACTLY what win rate, profit factor, and drawdown
    our AI would have achieved in the past.

    If it doesn't work in the past → it won't work live.
    This is what separates professional bots from gambling.
    """

    COMMISSION_PCT = 0.001   # 0.1% per trade (Binance standard)
    INITIAL_CAPITAL = 10000  # KSh 10,000 starting capital

    def run(self, df, signals_fn, symbol="UNKNOWN", verbose=True):
        """
        df:         OHLCV DataFrame
        signals_fn: function that takes df slice → returns signal dict
        symbol:     pair name for reporting
        """
        if len(df) < 200:
            return {'error': 'Not enough data for backtest (need 200+ candles)'}

        capital      = self.INITIAL_CAPITAL
        peak_capital = capital
        trades       = []
        in_trade     = False
        entry        = {}

        if verbose:
            print(f"\n{'='*55}")
            print(f"📊 BACKTEST: {symbol}")
            print(f"   Candles: {len(df)} | Period: {df.index[0]} → {df.index[-1]}")
            print('='*55)

        # Walk forward through history
        for i in range(200, len(df) - 1):
            window = df.iloc[:i].copy()
            current_price = df['close'].iloc[i]
            current_time  = df.index[i]

            # ── MANAGE OPEN TRADE ──────────────────────────────
            if in_trade:
                hit_target    = (entry['action'] == 'BUY'  and current_price >= entry['target']) or \
                                (entry['action'] == 'SELL' and current_price <= entry['target'])
                hit_stop      = (entry['action'] == 'BUY'  and current_price <= entry['stop']) or \
                                (entry['action'] == 'SELL' and current_price >= entry['stop'])
                max_hold      = (i - entry['bar']) >= 48  # Max hold 48 candles

                if hit_target or hit_stop or max_hold:
                    exit_price = current_price
                    commission = entry['size'] * self.COMMISSION_PCT * 2

                    if entry['action'] == 'BUY':
                        pnl = (exit_price - entry['price']) / entry['price'] * entry['size']
                    else:
                        pnl = (entry['price'] - exit_price) / entry['price'] * entry['size']

                    pnl -= commission
                    capital += pnl
                    peak_capital = max(peak_capital, capital)

                    trades.append({
                        'symbol':      symbol,
                        'action':      entry['action'],
                        'entry_price': entry['price'],
                        'exit_price':  exit_price,
                        'entry_time':  entry['time'],
                        'exit_time':   current_time,
                        'pnl':         round(pnl, 2),
                        'pnl_pct':     round(pnl / entry['size'] * 100, 2),
                        'exit_reason': 'TARGET' if hit_target else 'STOP' if hit_stop else 'TIMEOUT',
                        'confidence':  entry['confidence'],
                        'regime':      entry['regime'],
                        'capital':     round(capital, 2),
                        'won':         pnl > 0
                    })
                    in_trade = False

                continue  # Don't look for new trades while in one

            # ── LOOK FOR NEW SIGNAL (every 4 candles) ─────────
            if i % 4 != 0:
                continue

            try:
                signal = signals_fn(window)
            except Exception:
                continue

            if not signal or signal.get('action') == 'HOLD':
                continue
            if signal.get('confidence', 0) < 60:
                continue
            if not signal.get('regime_trade', True):
                continue

            # Enter trade
            size = capital * 0.02  # 2% risk
            in_trade = True
            entry = {
                'action':     signal['action'],
                'price':      current_price,
                'target':     signal['target_price'],
                'stop':       signal['stop_loss'],
                'size':       size,
                'bar':        i,
                'time':       current_time,
                'confidence': signal.get('confidence', 0),
                'regime':     signal.get('regime', 'UNKNOWN'),
            }

        # ── COMPILE RESULTS ────────────────────────────────────
        return self._compile_results(trades, capital, peak_capital, symbol, verbose)

    def _compile_results(self, trades, final_capital, peak_capital, symbol, verbose):
        if not trades:
            return {
                'symbol': symbol, 'total_trades': 0,
                'win_rate': 0, 'profit_factor': 0,
                'total_return_pct': 0, 'max_drawdown_pct': 0,
                'sharpe': 0, 'grade': 'F',
                'recommendation': 'NO DATA — not enough signals generated'
            }

        df_t = pd.DataFrame(trades)
        wins  = df_t[df_t['won'] == True]
        losses = df_t[df_t['won'] == False]

        total_trades   = len(df_t)
        win_rate       = len(wins) / total_trades * 100
        avg_win        = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss       = losses['pnl'].mean() if len(losses) > 0 else 0
        profit_factor  = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else 999
        total_pnl      = df_t['pnl'].sum()
        total_return   = (final_capital - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL * 100

        # Max drawdown
        capitals       = df_t['capital'].values
        peak           = np.maximum.accumulate(capitals)
        drawdown       = (peak - capitals) / peak * 100
        max_drawdown   = drawdown.max()

        # Sharpe ratio (simplified)
        returns        = df_t['pnl_pct'].values
        sharpe         = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Exit analysis
        targets_hit    = len(df_t[df_t['exit_reason'] == 'TARGET'])
        stops_hit      = len(df_t[df_t['exit_reason'] == 'STOP'])
        timeouts       = len(df_t[df_t['exit_reason'] == 'TIMEOUT'])

        # Grade the strategy
        grade, rec = self._grade(win_rate, profit_factor, max_drawdown, sharpe)

        result = {
            'symbol':           symbol,
            'total_trades':     total_trades,
            'win_rate':         round(win_rate, 1),
            'avg_win_kes':      round(avg_win, 2),
            'avg_loss_kes':     round(avg_loss, 2),
            'profit_factor':    round(profit_factor, 2),
            'total_return_pct': round(total_return, 1),
            'max_drawdown_pct': round(max_drawdown, 1),
            'sharpe_ratio':     round(sharpe, 2),
            'targets_hit':      targets_hit,
            'stops_hit':        stops_hit,
            'timeouts':         timeouts,
            'final_capital':    round(final_capital, 2),
            'grade':            grade,
            'recommendation':   rec,
            'trades':           trades
        }

        if verbose:
            self._print_report(result)

        return result

    def _grade(self, win_rate, profit_factor, max_drawdown, sharpe):
        score = 0
        if win_rate >= 65:      score += 30
        elif win_rate >= 58:    score += 20
        elif win_rate >= 52:    score += 10

        if profit_factor >= 2.0: score += 30
        elif profit_factor >= 1.5: score += 20
        elif profit_factor >= 1.2: score += 10

        if max_drawdown <= 10:  score += 20
        elif max_drawdown <= 20: score += 10

        if sharpe >= 1.5:       score += 20
        elif sharpe >= 1.0:     score += 10

        if score >= 80:
            return 'A+', '🟢 EXCELLENT — Deploy live with full position sizing'
        elif score >= 65:
            return 'A',  '🟢 GOOD — Deploy live, monitor closely'
        elif score >= 50:
            return 'B',  '🟡 ACCEPTABLE — Deploy with reduced size (50%)'
        elif score >= 35:
            return 'C',  '🟠 MARGINAL — Paper trade 2 more weeks first'
        else:
            return 'F',  '🔴 POOR — Do NOT deploy. Needs strategy revision'

    def _print_report(self, r):
        print(f"\n{'─'*55}")
        print(f"  RESULTS: {r['symbol']}")
        print(f"{'─'*55}")
        print(f"  Total Trades:    {r['total_trades']}")
        print(f"  Win Rate:        {r['win_rate']}%")
        print(f"  Profit Factor:   {r['profit_factor']}x")
        print(f"  Total Return:    {r['total_return_pct']}%")
        print(f"  Max Drawdown:    {r['max_drawdown_pct']}%")
        print(f"  Sharpe Ratio:    {r['sharpe_ratio']}")
        print(f"  Targets Hit:     {r['targets_hit']}")
        print(f"  Stops Hit:       {r['stops_hit']}")
        print(f"  Final Capital:   KSh {r['final_capital']:,.2f}")
        print(f"  Grade:           {r['grade']}")
        print(f"  Verdict:         {r['recommendation']}")
        print(f"{'─'*55}\n")
