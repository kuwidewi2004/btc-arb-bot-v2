
def strategy_pm_abs_deviation_high_regime(market, secs_left, tracker, position):
    """
    AUTO-GENERATED CANDIDATE — 2026-03-26 19:52:54
    ---------------------------------------------------------
    Source: ML regime analysis
    Feature:     pm_abs_deviation
    Regime:      HIGH (above median threshold)
    Threshold:   0.155000  (approximate median from training data)
    WF edge:     +0.1667 Brier improvement vs baseline
    Sample size: 680 test rows

    STATUS: OBSERVATION MODE — not validated for live trading
    ---------------------------------------------------------
    What this means:
    When pm_abs_deviation is HIGH (> 0.1550),
    the ML model predicts crowd direction more accurately (+0.1667 Brier).
    This suggests a systematic pattern worth investigating as a strategy filter.

    Next steps:
    1. Add this as a filter on top of an existing strategy (e.g. Liquidation Cascade)
    2. Log outcomes with/without filter for 50+ trades
    3. Compare win rates in regime vs out of regime
    4. If regime WR is 5%+ higher, hardcode the filter

    Implementation notes:
    - Replace threshold 0.1550 with data-driven value from SQL analysis
    - Feature pm_abs_deviation maps to: _regime / _vol_cache / _funding_cache
    - Consider combining with other regime filters for stronger signal
    """
    # TODO: Get the signal value for pm_abs_deviation
    # Example mapping (implement correctly for your codebase):
    # funding_zscore  → _regime.get("funding_zscore", 0.0)
    # vol_range_pct   → _vol_cache.get("range_pct", 0.0)
    # volatility_pct  → _regime.get("volatility_pct", 0.0)
    # liq_total       → sum(_liq_cache.values())
    # pm_abs_deviation → abs(get_poly_prices(market).get("up_mid", 0.5) - 0.5)

    signal_value = 0.0  # TODO: implement

    # Filter: only trade in HIGH regime
    if signal_value <= 0.1550:
        _log_signal("ML_pm_abs_deviation_HIGH", market, secs_left,
                    signal_value=signal_value,
                    threshold=0.1550,
                    reason="pm_abs_deviation_regime_filter")
        return position

    # Strategy fires in HIGH pm_abs_deviation regime
    # Add your existing strategy logic here, or use as a pre-filter
    # on top of Liquidation Cascade / Price Anchor / OB Pressure

    _log_signal("ML_pm_abs_deviation_HIGH", market, secs_left,
                signal_value=signal_value,
                threshold=0.1550,
                reason="observation_mode_no_bet")

    return position  # observation mode — remove when validated
