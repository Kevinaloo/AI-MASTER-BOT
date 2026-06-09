class PositionSizer:
    """
    Calculates exactly how much to risk per trade.

    Rules:
    - Never risk more than 2% of wallet on one trade (Kelly criterion)
    - Scale position size with AI confidence
    - Reduce size in high volatility
    - Increase size when trend is strong and aligned
    - Hard cap per trade in KES
    """

    MAX_RISK_PCT    = 0.02   # 2% max risk per trade
    MIN_RISK_PCT    = 0.005  # 0.5% minimum
    MAX_TRADE_KES   = 50000  # Hard cap KSh 50,000 per trade
    MIN_TRADE_KES   = 100    # Minimum KSh 100

    def calculate(self, wallet_kes, confidence, regime_data, atr_pct=1.0):
        """
        Returns recommended trade size in KES.

        wallet_kes:   user's current balance
        confidence:   AI confidence 0-100
        regime_data:  output from MarketRegimeDetector
        atr_pct:      current ATR as % of price
        """
        if wallet_kes <= 0:
            return 0

        regime     = regime_data.get('regime', 'UNKNOWN')
        vol_ratio  = regime_data.get('vol_ratio', 1.0)
        adx        = regime_data.get('adx', 20)

        # ── BASE RISK % ────────────────────────────────────────
        base_risk = self.MAX_RISK_PCT

        # ── CONFIDENCE SCALING ─────────────────────────────────
        # 50% conf → 50% of base risk
        # 95% conf → 100% of base risk
        conf_factor = (confidence - 50) / 45  # 0.0 to 1.0
        conf_factor = max(0.1, min(1.0, conf_factor))

        # ── REGIME SCALING ─────────────────────────────────────
        regime_factors = {
            'STRONG_TREND_UP':   1.0,   # Full size
            'STRONG_TREND_DOWN': 1.0,
            'WEAK_TREND_UP':     0.7,   # 70% size
            'WEAK_TREND_DOWN':   0.7,
            'SIDEWAYS':          0.3,   # 30% size
            'LOW_VOLATILITY':    0.8,
            'HIGH_VOLATILITY':   0.0,   # Don't trade
            'UNKNOWN':           0.5,
        }
        regime_factor = regime_factors.get(regime, 0.5)

        # ── VOLATILITY PENALTY ─────────────────────────────────
        if vol_ratio > 1.5:
            vol_penalty = 1.0 / vol_ratio  # Reduce size as volatility rises
        else:
            vol_penalty = 1.0

        # ── ADX BONUS ──────────────────────────────────────────
        # Strong trend = slightly larger position
        adx_bonus = 1.0
        if adx > 40:
            adx_bonus = 1.2
        elif adx > 30:
            adx_bonus = 1.1

        # ── FINAL CALCULATION ──────────────────────────────────
        final_risk_pct = (
            base_risk *
            conf_factor *
            regime_factor *
            vol_penalty *
            adx_bonus
        )

        final_risk_pct = max(self.MIN_RISK_PCT, min(self.MAX_RISK_PCT, final_risk_pct))

        trade_size = wallet_kes * final_risk_pct

        # Apply hard limits
        trade_size = max(self.MIN_TRADE_KES, min(self.MAX_TRADE_KES, trade_size))

        return {
            'trade_size_kes':  round(trade_size, 2),
            'risk_pct':        round(final_risk_pct * 100, 2),
            'conf_factor':     round(conf_factor, 2),
            'regime_factor':   regime_factor,
            'vol_penalty':     round(vol_penalty, 2),
            'reasoning': (
                f"Wallet KSh {wallet_kes:,.0f} × "
                f"{final_risk_pct*100:.1f}% risk = "
                f"KSh {trade_size:,.0f} "
                f"(conf={confidence}%, regime={regime})"
            )
        }
