
def strategy_pm_uncertainty_low_regime(market, secs_left, tracker, position):
    """
    AUTO-GENERATED CANDIDATE — 2026-03-26 19:52:54
    ---------------------------------------------------------
    Source: ML regime analysis
    Feature:     pm_uncertainty
    Regime:      LOW (below median threshold)
    Threshold:   0.690000  (approximate median from training data)
    WF edge:     +0.1646 Brier improvement vs baseline
    Sample size: 722 test rows

    STATUS: OBSERVATION MODE — not validated for live trading
    ---------------------------------------------------------
    What this means:
    When pm_uncertainty is LOW (<= 0.6900),
    the ML model predicts crowd direction more accurately (+0.1646 Brier).
    This suggests a systematic pattern worth investigating as a strategy filter.

    Next steps:
    1. Add this as a filter on top of an existing strategy (e.g. Liquidation Cascade)
    2. Log outcomes with/without filter for 50+ trades
    3. Compare win rates in regime vs out of regime
    4. If regime WR is 5%+ higher, hardcode the filter

    Implementation notes:
    - Replace threshold 0.6900 with data-driven value from SQL analysis
    - Feature pm_uncertainty maps to: _regime / _vol_cache / _funding_cache
    - Consider combining with other regime filters for stronger signal
    """
    # TODO: Get the signal value for pm_uncertainty
    # Example mapping (implement correctly for your codebase):
    # funding_zscore  → _regime.get("funding_zscore", 0.0)
    # vol_range_pct   → _vol_cache.get("range_pct", 0.0)
    # volatility_pct  → _regime.get("volatility_pct", 0.0)
    # liq_total       → sum(_liq_cache.values())
    # pm_abs_deviation → abs(get_poly_prices(market).get("up_mid", 0.5) - 0.5)

    signal_value = 0.0  # TODO: implement

    # Filter: only trade in LOW regime
    if signal_value > 0.6900:
        _log_signal("ML_pm_uncertainty_LOW", market, secs_left,
                    signal_value=signal_value,
                    threshold=0.6900,
                    reason="pm_uncertainty_regime_filter")
        return position

    # Strategy fires in LOW pm_uncertainty regime
    # Add your existing strategy logic here, or use as a pre-filter
    # on top of Liquidation Cascade / Price Anchor / OB Pressure

    _log_signal("ML_pm_uncertainty_LOW", market, secs_left,
                signal_value=signal_value,
                threshold=0.6900,
                reason="observation_mode_no_bet")

    return position  # observation mode — remove when validated
