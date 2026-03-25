"""
ML Training Pipeline v3 — Execution-cost aware
================================================
Three models trained, each with TWO targets:

  target_direction  — outcome_binary (did UP win?)
  target_profitable — was the trade profitable after execution costs?

  target_profitable accounts for:
    - poly_fill_up / poly_fill_down  (taker fill price, not mid)
    - TAKER_FEE * 2                  (round trip: 0.005 total)
    - A snapshot is "profitable" only if:
        P(correct direction) - fill_price - round_trip_fee > 0

  Three models:
    1. Snapshot model  — market_snapshots -> direction + profitable
    2. Signal model    — signal_log -> direction + profitable
    3. Market model    — market_outcomes -> direction

  Baselines compared:
    - p_market Brier          (direction baseline)
    - fill_price Brier        (profitable baseline — is fill already fair?)

Usage:
  $env:DB_URL="postgresql://postgres.xxx:PASSWORD@pooler.supabase.com:6543/postgres"
  python train_model_v3.py

Output files:
  model_v3_snapshot_dir.pkl   — snapshot direction model
  model_v3_snapshot_pnl.pkl   — snapshot profitable model
  model_v3_signal_dir.pkl     — signal direction model
  model_v3_market.pkl         — market level model
  training_report_v3.txt

Install deps:
  python -m pip install psycopg2-binary scikit-learn lightgbm numpy
"""

import os
import sys
import json
import math
import logging
import pickle
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: python -m pip install psycopg2-binary")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: python -m pip install numpy")
    sys.exit(1)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.impute import SimpleImputer
except ImportError:
    print("ERROR: python -m pip install scikit-learn")
    sys.exit(1)

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: python -m pip install lightgbm")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────── CONFIG ───────

DB_URL       = os.environ.get("DB_URL", "")
TAKER_FEE    = 0.0025   # per side; round trip = 0.005
TRAIN_SPLIT  = 0.80
RANDOM_STATE = 42
LEAKAGE_SECS = 120  # raised from 60 — fill prices encode outcome at 60-120s

# ────────────────────────────────────────────────────────────── CONNECT ───────

def connect():
    if not DB_URL:
        log.error("Set DB_URL environment variable")
        sys.exit(1)
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=15)
        log.info("Connected to Postgres")
        return conn
    except Exception as e:
        log.error(f"Connection failed: {e}")
        sys.exit(1)


def query(conn, sql) -> list:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    return rows


# ─────────────────────────────────────────────────────────────── HELPERS ─────

def _f(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _split(cond_ids, train_frac=TRAIN_SPLIT):
    markets   = list(dict.fromkeys(cond_ids))
    split     = int(len(markets) * train_frac)
    train_set = set(markets[:split])
    test_set  = set(markets[split:])
    train_idx = [i for i, c in enumerate(cond_ids) if c in train_set]
    test_idx  = [i for i, c in enumerate(cond_ids) if c in test_set]
    return train_idx, test_idx, len(train_set), len(test_set)


def _brier_baseline(X_test, y_test, feat_names, col="p_market"):
    if col not in feat_names:
        return np.nan
    pm   = X_test[:, feat_names.index(col)]
    mask = ~np.isnan(pm)
    if mask.sum() == 0:
        return np.nan
    return brier_score_loss(y_test[mask], pm[mask])


def _net_edge(outcome_binary, fill_up, fill_down, p_market,
              taker_fee=TAKER_FEE):
    """
    Compute net edge after execution costs for a snapshot row.

    The model would bet UP if p_market >= 0.5, DOWN otherwise.
    Net edge = (payout if correct - fill_price - round_trip_fee).

    Returns (net_edge_float, profitable_int) or (nan, nan) if inputs missing.
    """
    if any(v is None or np.isnan(v) for v in [outcome_binary, p_market]):
        return np.nan, np.nan

    round_trip = taker_fee * 2

    if p_market >= 0.5:
        # Model bets UP — pays fill_up
        fill = fill_up if fill_up is not None and not np.isnan(fill_up) else p_market
        gross = outcome_binary - fill          # >0 if UP won and fill was cheap
    else:
        # Model bets DOWN — pays fill_down
        fill = fill_down if fill_down is not None and not np.isnan(fill_down) else (1 - p_market)
        gross = (1.0 - outcome_binary) - fill  # >0 if DOWN won and fill was cheap

    net = gross - round_trip
    return round(float(net), 6), int(net > 0)


def _brier_fill_baseline(X_test, y_profitable, feat_names):
    """
    Baseline Brier score for profitable target using fill_price as naive predictor.
    If fill_up < 0.5 the market thinks UP is cheap — naive model says profitable=1.
    This is a harder baseline than p_market because it already accounts for price.
    """
    if "p_market" not in feat_names:
        return np.nan
    pm_idx  = feat_names.index("p_market")
    pm      = X_test[:, pm_idx]
    # Naive prediction: profitable when p_market deviates enough from fill cost
    # Simplification: use p_market > 0.505 as proxy for "cheap UP entry"
    naive   = (pm > 0.505).astype(float)
    mask    = ~np.isnan(pm) & ~np.isnan(y_profitable)
    if mask.sum() == 0:
        return np.nan
    return brier_score_loss(y_profitable[mask].astype(int), naive[mask])


def _train_lgbm(X_train, y_train, n_est=150, leaves=20):
    imp = SimpleImputer(strategy="median")
    Xt  = imp.fit_transform(X_train)
    base = lgb.LGBMClassifier(
        n_estimators=n_est, learning_rate=0.05,
        num_leaves=leaves, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, verbose=-1,
    )
    cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
    cal.fit(Xt, y_train)
    return cal, imp


def _evaluate(name, y_true, y_pred, brier_pm, report):
    brier = brier_score_loss(y_true, y_pred)
    ll    = log_loss(y_true, y_pred)
    acc   = np.mean((y_pred >= 0.5) == y_true)
    imp   = brier_pm - brier if not np.isnan(brier_pm) else np.nan
    beat  = "[OK]" if imp > 0 else "[FAIL]"
    lines = [
        f"\n  {name}",
        f"    Brier      : {brier:.4f}  (p_market baseline: {brier_pm:.4f})",
        f"    Improvement: {imp:+.4f}  {beat}",
        f"    Log loss   : {ll:.4f}",
        f"    Accuracy   : {acc*100:.1f}%",
    ]
    for l in lines:
        print(l)
    report.extend(lines)
    return brier, imp


def _calibration(name, y_true, y_pred, report, n_bins=10):
    bins  = np.linspace(0, 1, n_bins + 1)
    lines = [f"\n  Calibration — {name}",
             f"  {'Range':<12} {'Actual%':>9}  {'N':>6}"]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        n    = mask.sum()
        if n == 0:
            continue
        actual = y_true[mask].mean()
        mid    = (lo + hi) / 2
        flag   = " <-" if abs(actual - mid) > 0.10 else ""
        lines.append(f"  {lo:.1f}-{hi:.1f}       {actual*100:>8.1f}%  {n:>6,}{flag}")
    for l in lines:
        print(l)
    report.extend(lines)


def _importance(model, feat_names, top_n=25, report=None):
    imps = None
    try:
        for est in model.calibrated_classifiers_:
            base = est.estimator
            if hasattr(base, "feature_importances_"):
                if imps is None:
                    imps = base.feature_importances_.copy()
                else:
                    imps += base.feature_importances_
        if imps is not None:
            imps /= len(model.calibrated_classifiers_)
    except Exception:
        return
    fi    = sorted(zip(feat_names, imps), key=lambda x: x[1], reverse=True)
    maxfi = fi[0][1] if fi else 1
    lines = [f"\n  Feature importance (top {top_n})",
             f"  {'Feature':<42} {'Score':>8}"]
    for fn, sc in fi[:top_n]:
        bar = "#" * int(sc / maxfi * 20)
        lines.append(f"  {fn:<42} {sc:>8.1f}  {bar}")
    for l in lines:
        print(l)
    if report is not None:
        report.extend(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — SNAPSHOT MODEL
# Features from market_snapshots (same as v2 but pulled fresh)
# ══════════════════════════════════════════════════════════════════════════════

REGIME_MAP    = {"TREND_UP":2,"TREND_DOWN":-2,"VOLATILE":1,"CALM":0,"DEAD":-1}
SESSION_MAP   = {"OVERLAP":3,"US":2,"LONDON":1,"ASIA":0,"OFFPEAK":-1}
ACTIVITY_MAP  = {"HIGH":2,"NORMAL":1,"LOW":0,"DEAD":-1}
DAY_MAP       = {"WEEKDAY":1,"WEEKEND":0}
BUCKET_MAP    = {"heavy_fav":3,"favourite":2,"underdog":1,"longshot":0}
MOM_LBL_MAP   = {"TREND_UP":1,"NEUTRAL":0,"TREND_DOWN":-1}
VOL_LBL_MAP   = {"VOLATILE":1,"NORMAL":0,"DEAD":-1}
FLOW_LBL_MAP  = {"LONG_CROWDED":1,"BALANCED":0,"SHORT_CROWDED":-1}


def fetch_snapshots(conn) -> list:
    log.info("Fetching market_snapshots...")
    rows = query(conn, """
        SELECT condition_id, secs_left, secs_to_resolution, market_progress,
               phase_early, phase_mid, phase_late, phase_final,
               hour_sin, hour_cos, dow_sin, dow_cos,
               price_vs_open_pct, price_vs_open_score,
               anchor_pct, anchor_score, anchor_progress,
               momentum_10s, momentum_30s, momentum_60s, momentum_120s,
               momentum_score,
               cl_divergence, cl_age, cl_vs_open_pct,
               liq_total, liq_imbalance, liq_delta, liq_accel,
               ob_imbalance, ob_spread_pct, ob_bid_delta, ob_ask_delta,
               vol_range_pct, volume_buy_ratio,
               p_market, poly_fill_up, poly_fill_down,
               poly_spread, poly_slip_up, poly_deviation,
               basis_pct, funding_rate, okx_funding, gate_funding,
               volatility_pct, flow_score, liquidity_score, funding_zscore,
               regime, session, activity, day_type,
               price_bucket, momentum_label, volatility_label, flow_label,
               interact_momentum_x_vol, interact_ob_x_spread,
               interact_liq_x_price_pos, interact_momentum_x_progress,
               outcome_binary
        FROM market_snapshots
        WHERE resolved_outcome IS NOT NULL
          AND outcome_binary   IS NOT NULL
        ORDER BY created_at ASC
    """)
    log.info(f"  {len(rows):,} rows")
    return rows


def build_snapshot_features(rows) -> tuple:
    records, y_vals, y_profitable, cond_ids, pm_raws = [], [], [], [], []
    skipped = 0
    for row in rows:
        ob   = _f(row.get("outcome_binary"))
        str_ = _f(row.get("secs_to_resolution")) or _f(row.get("secs_left")) or np.nan
        if np.isnan(ob) or (not np.isnan(str_) and str_ < LEAKAGE_SECS):
            skipped += 1
            continue

        # Filter rows where the market has already effectively decided.
        # When p_market > 0.75 or < 0.25 the crowd has priced the outcome —
        # training on these rows teaches the model to read p_market rather
        # than learn microstructure signals. These rows are also excluded
        # because fill prices at these extremes encode the outcome directly.
        pm_check = _f(row.get("p_market"))
        if (pm_check is None or np.isnan(pm_check) or
                pm_check > 0.75 or pm_check < 0.25):
            skipped += 1
            continue
        pm_raw_val = pm_check

        m10  = _f(row.get("momentum_10s"))
        m30  = _f(row.get("momentum_30s"))
        m60  = _f(row.get("momentum_60s"))
        m120 = _f(row.get("momentum_120s"))
        sl   = _f(row.get("secs_left")) or 0
        if m120 == 0.0 and sl > 120:
            m120 = np.nan

        mp   = _f(row.get("market_progress"))
        pvop = _f(row.get("price_vs_open_pct"))
        fr   = _f(row.get("funding_rate"))
        ps   = _f(row.get("poly_spread"))
        sl2  = _f(row.get("poly_slip_up"))
        lt   = _f(row.get("liq_total"))
        cl   = _f(row.get("cl_divergence"))
        vr   = _f(row.get("vol_range_pct"))
        obi  = _f(row.get("ob_imbalance"))
        li   = _f(row.get("liq_imbalance"))

        mom_vals = [v for v in [m10,m30,m60,m120] if not np.isnan(v)]

        f = {
            "secs_to_resolution":     str_,
            "log_secs_to_resolution": math.log1p(str_) if not np.isnan(str_) else np.nan,
            "market_progress":        mp,
            "anchor_progress":        _f(row.get("anchor_progress")),
            "phase_early":            float(row.get("phase_early") or 0),
            "phase_mid":              float(row.get("phase_mid")   or 0),
            "phase_late":             float(row.get("phase_late")  or 0),
            "phase_final":            float(row.get("phase_final") or 0),
            "hour_sin":               _f(row.get("hour_sin")),
            "hour_cos":               _f(row.get("hour_cos")),
            "dow_sin":                _f(row.get("dow_sin")),
            "dow_cos":                _f(row.get("dow_cos")),
            "price_vs_open_pct":      pvop,
            "price_vs_open_score":    _f(row.get("price_vs_open_score")),
            "anchor_pct":             _f(row.get("anchor_pct")),
            "anchor_score":           _f(row.get("anchor_score")),
            "momentum_10s":           m10,
            "momentum_30s":           m30,
            "momentum_60s":           m60,
            "momentum_120s":          m120,
            "momentum_score":         _f(row.get("momentum_score")),
            "mom_accel_short":        (m30-m10)  if not(np.isnan(m30) or np.isnan(m10))  else np.nan,
            "mom_accel_mid":          (m60-m30)  if not(np.isnan(m60) or np.isnan(m30))  else np.nan,
            "mom_accel_long":         (m120-m60) if not(np.isnan(m120)or np.isnan(m60))  else np.nan,
            "mom_windows_positive":   float(sum(1 for v in mom_vals if v>0)) if mom_vals else np.nan,
            "mom_all_agree":          1.0 if mom_vals and (all(v>0 for v in mom_vals) or all(v<0 for v in mom_vals)) else 0.0,
            "mom_mean":               sum(mom_vals)/len(mom_vals) if mom_vals else np.nan,
            "mom_anchor_div":         (m30-pvop) if not(np.isnan(m30) or np.isnan(pvop)) else np.nan,
            "mom30_x_progress":       (m30*mp)   if not(np.isnan(m30) or np.isnan(mp))   else np.nan,
            "mom60_x_progress":       (m60*mp)   if not(np.isnan(m60) or np.isnan(mp))   else np.nan,
            "cl_divergence":          cl,
            "cl_age":                 _f(row.get("cl_age")),
            "cl_vs_open_pct":         _f(row.get("cl_vs_open_pct")),
            "cl_mom_align":           (cl*(1.0 if not np.isnan(m30) and m30>0 else -1.0 if not np.isnan(m30) and m30<0 else 0.0)) if not np.isnan(cl) else np.nan,
            "liq_imbalance":          li,
            "liq_total":              lt,
            "liq_delta":              _f(row.get("liq_delta")),
            "liq_accel":              _f(row.get("liq_accel")),
            "log_liq_total":          math.log1p(lt) if not np.isnan(lt) and lt>=0 else np.nan,
            "ob_imbalance":           obi,
            "ob_spread_pct":          _f(row.get("ob_spread_pct")),
            "ob_bid_delta":           _f(row.get("ob_bid_delta")),
            "ob_ask_delta":           _f(row.get("ob_ask_delta")),
            "vol_range_pct":          vr,
            "volatility_pct":         _f(row.get("volatility_pct")),
            "volume_buy_ratio":       _f(row.get("volume_buy_ratio")),
            # p_market excluded from training features — it dominates all other
            # signals and prevents the model learning microstructure edge.
            # Used only as Brier baseline, not as an input feature.
            "poly_spread":            ps,
            "poly_slip_up":           sl2,
            # poly_deviation = p_market - 0.50 — excluded for same reason as p_market
            "effective_entry_cost":   (ps+sl2) if not(np.isnan(ps) or np.isnan(sl2)) else np.nan,
            "basis_pct":              _f(row.get("basis_pct")),
            "funding_rate":           fr,
            "funding_zscore":         _f(row.get("funding_zscore")),
            "okx_funding":            _f(row.get("okx_funding")),
            "gate_funding":           _f(row.get("gate_funding")),
            "funding_x_momentum":     (fr*m30) if not(np.isnan(fr) or np.isnan(m30)) else np.nan,
            "flow_score":             _f(row.get("flow_score")),
            "liquidity_score":        _f(row.get("liquidity_score")),
            "regime_enc":             REGIME_MAP.get(row.get("regime",""),0),
            "session_enc":            SESSION_MAP.get(row.get("session",""),0),
            "activity_enc":           ACTIVITY_MAP.get(row.get("activity",""),0),
            "day_type_enc":           DAY_MAP.get(row.get("day_type",""),0),
            "bucket_enc":             BUCKET_MAP.get(row.get("price_bucket",""),1),
            "mom_label_enc":          MOM_LBL_MAP.get(row.get("momentum_label",""),0),
            "vol_label_enc":          VOL_LBL_MAP.get(row.get("volatility_label",""),0),
            "flow_label_enc":         FLOW_LBL_MAP.get(row.get("flow_label",""),0),
        }

        def _rc(col, a, b):
            v = _f(row.get(col))
            return a*b if np.isnan(v) and not(np.isnan(a) or np.isnan(b)) else v

        f["interact_momentum_x_vol"]      = _rc("interact_momentum_x_vol",      m30, vr)
        f["interact_ob_x_spread"]         = _rc("interact_ob_x_spread",         obi, ps or np.nan)
        f["interact_liq_x_price_pos"]     = _rc("interact_liq_x_price_pos",     li,  pvop)
        f["interact_momentum_x_progress"] = _rc("interact_momentum_x_progress", m30, mp)

        # ── Execution cost features (NO label leakage) ───────────────────────
        # Only use features observable at snapshot time — fill prices and fees.
        # net_edge_if_up/down are NOT included because they use outcome_binary.
        # The model learns to predict whether conditions are good for profitability
        # from market state alone, not from the outcome itself.
        fill_up   = _f(row.get("poly_fill_up"))
        fill_down = _f(row.get("poly_fill_down"))
        pm        = _f(row.get("p_market"))
        net, prof = _net_edge(ob, fill_up, fill_down, pm)

        # Execution cost proxy features — observable at entry, no leakage.
        # poly_fill_up/down are NOT included in features because at late-market
        # snapshots they approach 0 or 1 and perfectly encode outcome.
        # We use poly_spread and poly_slip_up instead — these measure cost
        # without encoding direction.
        f["round_trip_cost"]         = (ps + sl2 + TAKER_FEE*2) if not(np.isnan(ps) or np.isnan(sl2)) else TAKER_FEE*2
        f["fill_spread_cost"]        = ps if not np.isnan(ps) else np.nan
        f["fill_slip_cost"]          = sl2 if not np.isnan(sl2) else np.nan

        records.append(f)
        pm_raws.append(pm_raw_val)
        y_vals.append(int(ob))
        y_profitable.append(int(prof) if isinstance(prof, (int, float)) and prof == prof else 0)
        cond_ids.append(row.get("condition_id",""))

    log.info(f"  Engineered {len(records):,} rows ({skipped:,} skipped)")
    log.info(f"  Profitable rows: {sum(1 for p in y_profitable if p==1):,} / {len(y_profitable):,} "
             f"({sum(1 for p in y_profitable if p==1)/max(len(y_profitable),1)*100:.1f}%)")

    feat_names = list(records[0].keys())
    X   = np.array([[r[k] for k in feat_names] for r in records], dtype=np.float32)
    y   = np.array(y_vals,  dtype=np.int32)
    yp  = np.array([int(p) if isinstance(p, (int, float)) and p == p else 0
                    for p in y_profitable], dtype=np.int32)
    pm  = np.array(pm_raws, dtype=np.float32)
    return X, y, yp, cond_ids, feat_names, pm


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — SIGNAL MODEL
# Features from signal_log_enriched — one row per strategy evaluation
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_ENC = {
    "Liquidation Cascade": 0,
    "OB Pressure":         1,
    "Price Anchor":        2,
    "Chainlink Arb":       3,
    "Odds Mispricing":     4,
    "Volume Clock":        5,
    "Funding Reversion":   6,
    "Basis Arb":           7,
}

REASON_ENC = {
    "not_in_ob_window_yet": 0, "not_in_volume_window_yet": 0,
    "too_early_in_market": 0, "too_close_to_resolution": 0,
    "market_closing_soon": 0, "secs_left_too_high": 0,
    "divergence_below_threshold": 1, "basis_too_small": 1,
    "displacement_too_small": 1, "liquidation_volume_too_low": 1,
    "imbalance_too_weak": 1, "momentum_too_weak": 1,
    "buy_ratio_neutral": 1, "exchanges_disagree_or_not_extreme": 1,
    "deviation_too_small": 1, "liquidation_imbalance_too_low": 2,
    "momentum_too_strong_to_fade": 2,
    "momentum_direction_mismatch": 3, "momentum_direction_conflict": 3,
    "direction_inconsistent": 3, "momentum_confirms_premium_skip_fade": 3,
    "btc_momentum_justifies_odds_deviation": 3,
    "momentum_not_confirming_crowding": 3,
    "spread_too_wide": 4, "polymarket_spread_too_wide": 4,
    "calm_regime_insufficient_size": 4, "offpeak_session_filtered": 4,
    "intensity_too_low": 4, "market_already_priced": 4,
}


def fetch_signal_enriched(conn) -> list:
    log.info("Fetching signal_log (inline join with market_snapshots)...")
    rows = query(conn, """
        SELECT
            sl.condition_id, sl.strategy, sl.secs_left,
            sl.signal_value, sl.threshold, sl.reason,
            sl.outcome_binary,
            sl.regime    AS sl_regime,
            sl.session   AS sl_session,
            sl.activity  AS sl_activity,
            -- signal ratio computed here if not stored on sl yet
            CASE WHEN sl.threshold > 0
                 THEN sl.signal_value / sl.threshold
                 ELSE NULL END                            AS signal_ratio,
            (sl.signal_value >= sl.threshold)            AS above_threshold,
            -- all microstructure context from nearest snapshot
            ms.p_market,
            ms.ob_imbalance,
            ms.vol_range_pct,
            ms.liq_total,
            ms.liq_imbalance,
            ms.funding_rate,
            ms.funding_zscore,
            ms.volatility_pct,
            ms.momentum_30s,
            ms.momentum_60s,
            ms.market_progress,
            ms.phase_early, ms.phase_mid, ms.phase_late, ms.phase_final,
            ms.price_vs_open_pct, ms.price_vs_open_score,
            ms.momentum_120s,
            ms.cl_divergence, ms.cl_vs_open_pct,
            ms.ob_spread_pct, ms.volume_buy_ratio,
            ms.flow_score, ms.poly_spread, ms.poly_deviation,
            ms.regime    AS snap_regime,
            ms.session   AS snap_session,
            ms.price_bucket,
            -- final columns (no coalesce needed — all from ms)
            ms.p_market          AS p_market_final,
            ms.ob_imbalance      AS ob_imbalance_final,
            ms.momentum_30s      AS momentum_30s_final,
            ms.momentum_60s      AS momentum_60s_final,
            ms.liq_total         AS liq_total_final,
            ms.liq_imbalance     AS liq_imbalance_final
        FROM signal_log sl
        LEFT JOIN market_snapshots ms
            ON  ms.condition_id = sl.condition_id
            AND ROUND(ms.secs_left::numeric / 5) = ROUND(sl.secs_left::numeric / 5)
            AND ms.resolved_outcome IS NOT NULL
        WHERE sl.outcome_binary IS NOT NULL
          AND sl.secs_left > 20
          AND ms.p_market BETWEEN 0.25 AND 0.75
          AND sl.strategy NOT IN ('Odds Mispricing', 'Volume Clock')
        ORDER BY sl.condition_id, sl.secs_left DESC
    """)
    log.info(f"  {len(rows):,} rows")
    return rows


def build_signal_features(rows) -> tuple:
    records, y_vals, cond_ids = [], [], []
    for row in rows:
        ob  = _f(row.get("outcome_binary"))
        if np.isnan(ob):
            continue
        sv  = _f(row.get("signal_value"))
        thr = _f(row.get("threshold"))
        sr  = _f(row.get("signal_ratio"))
        m30 = _f(row.get("momentum_30s_final") or row.get("momentum_30s"))
        m60 = _f(row.get("momentum_60s_final") or row.get("momentum_60s"))
        mp  = _f(row.get("market_progress"))
        pvop= _f(row.get("price_vs_open_pct"))
        lt  = _f(row.get("liq_total_final")    or row.get("liq_total"))
        li  = _f(row.get("liq_imbalance_final") or row.get("liq_imbalance"))
        vr  = _f(row.get("vol_range_pct"))
        pm  = _f(row.get("p_market_final")     or row.get("p_market"))
        fzs = _f(row.get("funding_zscore"))

        f = {
            # strategy identity
            "strategy_enc":           float(STRATEGY_ENC.get(row.get("strategy",""), -1)),
            # signal quality
            "signal_value":           sv,
            "log_signal_value":       math.log1p(sv) if not np.isnan(sv) and sv >= 0 else np.nan,
            "signal_ratio":           sr,
            "above_threshold":        _f(row.get("above_threshold")),
            "reason_enc":             float(REASON_ENC.get(row.get("reason",""), 0)),
            # time
            "secs_left":              _f(row.get("secs_left")),
            "log_secs_left":          math.log1p(_f(row.get("secs_left")) or 0),
            "market_progress":        mp,
            "phase_early":            float(row.get("phase_early") or 0),
            "phase_mid":              float(row.get("phase_mid")   or 0),
            "phase_late":             float(row.get("phase_late")  or 0),
            # market state
            "p_market":               pm,
            "poly_deviation":         _f(row.get("poly_deviation")),
            "price_vs_open_pct":      pvop,
            "price_vs_open_score":    _f(row.get("price_vs_open_score")),
            "momentum_30s":           m30,
            "momentum_60s":           m60,
            "liq_total":              lt,
            "log_liq_total":          math.log1p(lt) if not np.isnan(lt) and lt >= 0 else np.nan,
            "liq_imbalance":          li,
            "ob_imbalance":           _f(row.get("ob_imbalance")),
            "vol_range_pct":          vr,
            "funding_zscore":         fzs,
            "volatility_pct":         _f(row.get("volatility_pct")),
            # derived
            "signal_x_momentum":      (sr * m30) if not(np.isnan(sr) or np.isnan(m30)) else np.nan,
            "signal_x_progress":      (sr * mp)  if not(np.isnan(sr) or np.isnan(mp))  else np.nan,
            "liq_x_price":            (li * pvop) if not(np.isnan(li) or np.isnan(pvop)) else np.nan,
            "pm_deviation":           (pm - 0.5) if not np.isnan(pm) else np.nan,
            # categoricals
            "regime_enc":             REGIME_MAP.get(row.get("snap_regime") or row.get("sl_regime",""), 0),
            "session_enc":            SESSION_MAP.get(row.get("snap_session") or row.get("sl_session",""), 0),
            "activity_enc":           ACTIVITY_MAP.get(row.get("sl_activity",""), 0),
            "bucket_enc":             BUCKET_MAP.get(row.get("price_bucket",""), 1),
        }
        records.append(f)
        y_vals.append(int(ob))
        cond_ids.append(row.get("condition_id",""))

    log.info(f"  Engineered {len(records):,} signal rows")
    feat_names = list(records[0].keys())
    X = np.array([[r[k] for k in feat_names] for r in records], dtype=np.float32)
    y = np.array(y_vals, dtype=np.int32)
    return X, y, cond_ids, feat_names


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — MARKET LEVEL MODEL
# One row per market — 407 independent observations
# ══════════════════════════════════════════════════════════════════════════════

def fetch_market_outcomes(conn) -> list:
    log.info("Computing market_outcomes inline from market_snapshots + trades...")
    rows = query(conn, """
        SELECT
            ms.condition_id,
            ms.resolved_outcome,
            CASE WHEN ms.resolved_outcome='UP' THEN 1
                 WHEN ms.resolved_outcome='DOWN' THEN 0
                 ELSE NULL END                              AS outcome_binary,
            ms.market_end_time,
            ms.total_snapshots,
            ms.first_snap_secs_left,
            ms.p_market_open,
            ms.p_market_avg,
            ms.p_market_std,
            ms.p_market_max,
            ms.p_market_min,
            ms.btc_range_pct,
            ms.btc_move_pct,
            ms.avg_momentum_30s,
            ms.max_liq_total,
            ms.avg_liq_imbalance,
            ms.avg_ob_imbalance,
            ms.avg_cl_divergence,
            ms.avg_vol_range_pct,
            ms.avg_funding_rate,
            ms.avg_funding_zscore,
            ms.dominant_regime,
            ms.dominant_session,
            COALESCE(tf.n_strategies_fired, 0)  AS n_strategies_fired,
            COALESCE(tf.n_strategies_up,   0)   AS n_strategies_up,
            COALESCE(tf.n_strategies_down, 0)   AS n_strategies_down,
            COALESCE(tf.liq_fired, 0)            AS liq_fired,
            COALESCE(tf.pa_fired,  0)            AS pa_fired,
            COALESCE(tf.ob_fired,  0)            AS ob_fired,
            COALESCE(tf.strategies_agree, 0)     AS strategies_agree,
            COALESCE(sl.avg_liq_signal, 0)       AS avg_liq_signal,
            COALESCE(sl.max_liq_signal, 0)       AS max_liq_signal,
            COALESCE(sl.avg_pa_signal,  0)       AS avg_pa_signal,
            COALESCE(sl.max_pa_signal,  0)       AS max_pa_signal
        FROM (
            SELECT
                condition_id, resolved_outcome, market_end_time,
                COUNT(*)                                    AS total_snapshots,
                MAX(secs_left)                              AS first_snap_secs_left,
                AVG(p_market)                               AS p_market_avg,
                STDDEV(p_market)                            AS p_market_std,
                MAX(p_market)                               AS p_market_max,
                MIN(p_market)                               AS p_market_min,
                (ARRAY_AGG(p_market  ORDER BY secs_left DESC))[1] AS p_market_open,
                (ARRAY_AGG(btc_price ORDER BY secs_left DESC))[1] AS btc_open,
                (ARRAY_AGG(btc_price ORDER BY secs_left ASC))[1]  AS btc_close,
                CASE WHEN MIN(btc_price)>0
                     THEN (MAX(btc_price)-MIN(btc_price))/MIN(btc_price)*100
                     ELSE NULL END                          AS btc_range_pct,
                CASE WHEN (ARRAY_AGG(btc_price ORDER BY secs_left DESC))[1]>0
                     THEN ((ARRAY_AGG(btc_price ORDER BY secs_left ASC))[1]
                          -(ARRAY_AGG(btc_price ORDER BY secs_left DESC))[1])
                          /(ARRAY_AGG(btc_price ORDER BY secs_left DESC))[1]*100
                     ELSE NULL END                          AS btc_move_pct,
                AVG(momentum_30s)                           AS avg_momentum_30s,
                MAX(liq_total)                              AS max_liq_total,
                AVG(liq_imbalance)                          AS avg_liq_imbalance,
                AVG(ob_imbalance)                           AS avg_ob_imbalance,
                AVG(cl_divergence)                          AS avg_cl_divergence,
                AVG(vol_range_pct)                          AS avg_vol_range_pct,
                AVG(funding_rate)                           AS avg_funding_rate,
                AVG(funding_zscore)                         AS avg_funding_zscore,
                MODE() WITHIN GROUP (ORDER BY regime)       AS dominant_regime,
                MODE() WITHIN GROUP (ORDER BY session)      AS dominant_session
            FROM market_snapshots
            WHERE resolved_outcome IS NOT NULL
            GROUP BY condition_id, resolved_outcome, market_end_time
        ) ms
        LEFT JOIN (
            SELECT condition_id,
                COUNT(DISTINCT strategy)                                    AS n_strategies_fired,
                SUM(CASE WHEN side='UP'   THEN 1 ELSE 0 END)              AS n_strategies_up,
                SUM(CASE WHEN side='DOWN' THEN 1 ELSE 0 END)              AS n_strategies_down,
                MAX(CASE WHEN strategy='Liquidation Cascade' THEN 1 ELSE 0 END) AS liq_fired,
                MAX(CASE WHEN strategy='Price Anchor'        THEN 1 ELSE 0 END) AS pa_fired,
                MAX(CASE WHEN strategy='OB Pressure'         THEN 1 ELSE 0 END) AS ob_fired,
                CASE WHEN COUNT(DISTINCT side)=1 AND COUNT(DISTINCT strategy)>1
                     THEN 1 ELSE 0 END                                     AS strategies_agree
            FROM trades
            WHERE action='OPEN' AND resolved_outcome IS NOT NULL
            GROUP BY condition_id
        ) tf ON tf.condition_id = ms.condition_id
        LEFT JOIN (
            SELECT condition_id,
                AVG(CASE WHEN strategy='Liquidation Cascade' THEN signal_value END) AS avg_liq_signal,
                MAX(CASE WHEN strategy='Liquidation Cascade' THEN signal_value END) AS max_liq_signal,
                AVG(CASE WHEN strategy='Price Anchor'        THEN signal_value END) AS avg_pa_signal,
                MAX(CASE WHEN strategy='Price Anchor'        THEN signal_value END) AS max_pa_signal
            FROM signal_log
            WHERE resolved_outcome IS NOT NULL
            GROUP BY condition_id
        ) sl ON sl.condition_id = ms.condition_id
        WHERE ms.resolved_outcome IS NOT NULL
        ORDER BY ms.market_end_time ASC
    """)
    log.info(f"  {len(rows):,} markets")
    return rows


def build_market_features(rows) -> tuple:
    records, y_vals, cond_ids = [], [], []
    for row in rows:
        ob = _f(row.get("outcome_binary"))
        if np.isnan(ob):
            continue
        pm_avg = _f(row.get("p_market_avg"))
        pm_std = _f(row.get("p_market_std"))
        pm_max = _f(row.get("p_market_max"))
        pm_min = _f(row.get("p_market_min"))
        pm_opn = _f(row.get("p_market_open"))
        mlt    = _f(row.get("max_liq_total"))
        btcr   = _f(row.get("btc_range_pct"))
        btcm   = _f(row.get("btc_move_pct"))
        fzs    = _f(row.get("avg_funding_zscore"))
        nf     = _f(row.get("n_strategies_fired"))
        nu     = _f(row.get("n_strategies_up"))
        nd     = _f(row.get("n_strategies_down"))

        f = {
            # p_market features
            "p_market_avg":           pm_avg,
            "p_market_std":           pm_std,
            "p_market_max":           pm_max,
            "p_market_min":           pm_min,
            "p_market_open":          pm_opn,
            "p_market_range":         (pm_max - pm_min) if not(np.isnan(pm_max) or np.isnan(pm_min)) else np.nan,
            "p_market_deviation":     (pm_avg - 0.5) if not np.isnan(pm_avg) else np.nan,
            # BTC movement
            "btc_range_pct":          btcr,
            "btc_move_pct":           btcm,
            "log_btc_range":          math.log1p(btcr) if not np.isnan(btcr) and btcr >= 0 else np.nan,
            # signals
            "avg_momentum_30s":       _f(row.get("avg_momentum_30s")),
            "max_liq_total":          mlt,
            "log_max_liq":            math.log1p(mlt) if not np.isnan(mlt) and mlt >= 0 else np.nan,
            "avg_liq_imbalance":      _f(row.get("avg_liq_imbalance")),
            "avg_ob_imbalance":       _f(row.get("avg_ob_imbalance")),
            "avg_cl_divergence":      _f(row.get("avg_cl_divergence")),
            "avg_vol_range_pct":      _f(row.get("avg_vol_range_pct")),
            "avg_funding_rate":       _f(row.get("avg_funding_rate")),
            "avg_funding_zscore":     fzs,
            # strategy firing
            "n_strategies_fired":     nf,
            "n_strategies_up":        nu,
            "n_strategies_down":      nd,
            "strategies_agree":       _f(row.get("strategies_agree")),
            "liq_fired":              _f(row.get("liq_fired")),
            "pa_fired":               _f(row.get("pa_fired")),
            "ob_fired":               _f(row.get("ob_fired")),
            "net_direction":          (nu - nd) if not(np.isnan(nu) or np.isnan(nd)) else np.nan,
            # signal log
            "avg_liq_signal":         _f(row.get("avg_liq_signal")),
            "max_liq_signal":         mls if not np.isnan(mls := _f(row.get("max_liq_signal"))) else np.nan,
            "log_max_liq_signal":     math.log1p(mls) if not np.isnan(mls) and mls >= 0 else np.nan,
            "avg_pa_signal":          _f(row.get("avg_pa_signal")),
            # coverage
            "total_snapshots":        _f(row.get("total_snapshots")),
            # categoricals
            "regime_enc":             REGIME_MAP.get(row.get("dominant_regime",""), 0),
            "session_enc":            SESSION_MAP.get(row.get("dominant_session",""), 0),
        }
        records.append(f)
        y_vals.append(int(ob))
        cond_ids.append(row.get("condition_id",""))

    log.info(f"  Engineered {len(records):,} market rows")
    feat_names = list(records[0].keys())
    X = np.array([[r[k] for k in feat_names] for r in records], dtype=np.float32)
    y = np.array(y_vals, dtype=np.int32)
    return X, y, cond_ids, feat_names


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    report = []

    log.info("=" * 60)
    log.info("ML TRAINING PIPELINE v3")
    log.info("=" * 60)

    conn = connect()

    # ── Strategy trade summary ────────────────────────────────────────────────
    print("\n  Strategy summary (from trades):")
    print(f"  {'Strategy':<25} {'N':>5}  {'Wins':>5}  {'WR':>7}  {'PnL':>9}")
    print(f"  {'─'*25} {'─'*5}  {'─'*5}  {'─'*7}  {'─'*9}")
    strat_rows = query(conn, """
        SELECT strategy,
               COUNT(*) AS n,
               SUM(CASE WHEN actual_win THEN 1 ELSE 0 END) AS wins,
               ROUND(AVG(CASE WHEN actual_win THEN 1.0 ELSE 0.0 END)*100,1) AS wr,
               ROUND(SUM(pnl::numeric),2) AS pnl
        FROM trades
        WHERE action = 'OPEN'
          AND resolved_outcome IS NOT NULL
        GROUP BY strategy ORDER BY n DESC
    """)
    for r in strat_rows:
        print(f"  {r['strategy']:<25} {r['n']:>5}  {r['wins']:>5}  "
              f"{float(r['wr'] or 0):>6.1f}%  ${float(r['pnl'] or 0):>+8.2f}")
    report.extend([str(r) for r in strat_rows])

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 1: SNAPSHOT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL 1 — SNAPSHOT (market_snapshots)")
    print("=" * 60)
    report.append("\nMODEL 1: SNAPSHOT")

    snap_rows  = fetch_snapshots(conn)
    X1, y1, yp1, cids1, fn1, pm1 = build_snapshot_features(snap_rows)

    ti1, vi1, nt1, nv1 = _split(cids1)
    X1_tr, y1_tr, yp1_tr = X1[ti1], y1[ti1], yp1[ti1]
    X1_te, y1_te, yp1_te = X1[vi1], y1[vi1], yp1[vi1]
    log.info(f"  Train: {nt1} markets {len(y1_tr):,} rows | "
             f"Test: {nv1} markets {len(y1_te):,} rows")

    # 1a. Direction model — predicts outcome_binary
    pm1_te  = pm1[vi1]
    bpm1    = brier_score_loss(y1_te, pm1_te)
    log.info(f"  p_market baseline Brier (0.25-0.75 range): {bpm1:.4f}")
    log.info("  Training snapshot direction model...")
    m1d, imp1d = _train_lgbm(X1_tr, y1_tr, n_est=150, leaves=20)
    p1d = m1d.predict_proba(imp1d.transform(X1_te))[:, 1]
    b1d, i1d = _evaluate("Snapshot — direction", y1_te, p1d, bpm1, report)
    _calibration("Snapshot direction", y1_te, p1d, report)
    _importance(m1d, fn1, report=report)

    with open("model_v3_snapshot_dir.pkl", "wb") as f:
        pickle.dump({"model": m1d, "imputer": imp1d, "features": fn1,
                     "target": "direction"}, f)

    # 1b. Profitable model — predicts net_edge > 0 after execution costs
    # Baseline: what fraction of snapshots are profitable at fill price?
    pct_profitable = yp1_te.mean()
    # Brier of always predicting the base rate
    brier_fill_base = brier_score_loss(yp1_te,
                                        np.full(len(yp1_te), pct_profitable))
    log.info(f"  Profitable baseline: {pct_profitable*100:.1f}% of test rows "
             f"profitable — Brier={brier_fill_base:.4f}")
    log.info("  Training snapshot profitable model...")
    m1p, imp1p = _train_lgbm(X1_tr, yp1_tr, n_est=150, leaves=20)
    p1p = m1p.predict_proba(imp1p.transform(X1_te))[:, 1]
    b1p, i1p = _evaluate("Snapshot — profitable", yp1_te, p1p,
                          brier_fill_base, report)
    _calibration("Snapshot profitable", yp1_te, p1p, report)

    with open("model_v3_snapshot_pnl.pkl", "wb") as f:
        pickle.dump({"model": m1p, "imputer": imp1p, "features": fn1,
                     "target": "profitable", "taker_fee": TAKER_FEE}, f)

    # Combined edge: use direction model probability vs fill price
    print("\n  Combined edge analysis (direction model vs fill price):")
    print(f"  {'Pred P(UP)':>12}  {'Fill cost':>10}  {'Net edge':>10}  {'N':>6}  {'Actual profit%':>15}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*15}")
    for lo, hi in [(0.0,0.40),(0.40,0.45),(0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,1.01)]:
        mask = (p1d >= lo) & (p1d < hi)
        n    = mask.sum()
        if n == 0:
            continue
        avg_pm     = pm1_te[mask].mean()
        model_edge = p1d[mask].mean() - avg_pm
        act_prof   = yp1_te[mask].mean()
        print(f"  {lo:.2f}-{hi:.2f}      {avg_pm:>10.4f}  {model_edge:>+10.4f}  "
              f"{n:>6,}  {act_prof*100:>14.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 2: SIGNAL
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL 2 — SIGNAL (signal_log_enriched)")
    print("=" * 60)
    report.append("\nMODEL 2: SIGNAL")

    sig_rows   = fetch_signal_enriched(conn)
    X2, y2, cids2, fn2 = build_signal_features(sig_rows)

    ti2, vi2, nt2, nv2 = _split(cids2)
    X2_tr, y2_tr = X2[ti2], y2[ti2]
    X2_te, y2_te = X2[vi2], y2[vi2]
    log.info(f"  Train: {nt2} markets {len(y2_tr):,} rows | "
             f"Test: {nv2} markets {len(y2_te):,} rows")

    bpm2 = _brier_baseline(X2_te, y2_te, fn2)
    log.info("  Training signal direction model...")
    m2, imp2 = _train_lgbm(X2_tr, y2_tr, n_est=150, leaves=20)
    p2 = m2.predict_proba(imp2.transform(X2_te))[:, 1]
    b2, i2 = _evaluate("Signal — direction", y2_te, p2, bpm2, report)
    _calibration("Signal direction", y2_te, p2, report)
    _importance(m2, fn2, report=report)

    with open("model_v3_signal_dir.pkl", "wb") as f:
        pickle.dump({"model": m2, "imputer": imp2, "features": fn2,
                     "target": "direction"}, f)

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 3: MARKET LEVEL
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL 3 — MARKET LEVEL (market_outcomes)")
    print("=" * 60)
    report.append("\nMODEL 3: MARKET LEVEL")

    mkt_rows   = fetch_market_outcomes(conn)
    X3, y3, cids3, fn3 = build_market_features(mkt_rows)

    ti3, vi3, nt3, nv3 = _split(cids3)
    X3_tr, y3_tr = X3[ti3], y3[ti3]
    X3_te, y3_te = X3[vi3], y3[vi3]
    log.info(f"  Train: {nt3} markets | Test: {nv3} markets")

    bpm3 = _brier_baseline(X3_te, y3_te, fn3, col="p_market_avg")
    log.info("  Training market-level LightGBM...")
    m3, imp3 = _train_lgbm(X3_tr, y3_tr, n_est=100, leaves=15)
    p3 = m3.predict_proba(imp3.transform(X3_te))[:, 1]
    b3, i3 = _evaluate("Market-level LightGBM", y3_te, p3, bpm3, report)
    _calibration("Market-level", y3_te, p3, report)
    _importance(m3, fn3, report=report)

    with open("model_v3_market.pkl", "wb") as f:
        pickle.dump({"model": m3, "imputer": imp3, "features": fn3}, f)

    conn.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<35} {'Brier':>7}  {'vs baseline':>12}")
    print(f"  {'─'*35} {'─'*7}  {'─'*12}")
    for name, brier, imp in [
        ("Snapshot direction",      b1d, i1d),
        ("Snapshot profitable",     b1p, i1p),
        ("Signal direction",        b2,  i2),
        ("Market-level",            b3,  i3),
    ]:
        beat = "[OK]" if imp > 0 else "[FAIL]"
        print(f"  {name:<35} {brier:>7.4f}  {imp:>+8.4f}  {beat}")
    print()
    print("  Saved: model_v3_snapshot_dir.pkl")
    print("         model_v3_snapshot_pnl.pkl")
    print("         model_v3_signal_dir.pkl")
    print("         model_v3_market.pkl")
    print("=" * 60)

    with open("training_report_v3.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(str(l) for l in report))


if __name__ == "__main__":
    if not DB_URL:
        print("ERROR: set DB_URL environment variable")
        sys.exit(1)
    main()
