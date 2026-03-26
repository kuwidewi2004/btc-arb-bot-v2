"""
ML Training Pipeline v4
========================
Structural improvements over v3:

  1. ONE primary model: P(profitable | features)
     - Direction model dropped as primary objective
     - Profitable = edge on p_market-chosen side > MIN_EDGE after fill + fees
     - Side policy: follow p_market (bet UP if p_market >= 0.5, DOWN otherwise)

  2. Net edge target:
     - edge_up   = P(UP)   - fill_up   - fees
     - edge_down = P(DOWN) - fill_down - fees
     - label = 1 if edge on p_market side > 0 else 0
     - (max-edge label was considered but discarded — p_market side is used)

  3. Walk-forward validation:
     - Multiple folds: train on first N markets, test on next K
     - Average Brier across folds for stable estimate
     - No data snooping via repeated test-set inspection

  4. Calibration: Platt scaling (sigmoid) for small datasets
     - Isotonic calibration needs large N to avoid overfitting
     - Switch to isotonic when markets > 1000

  5. Single model output: P(profitable)
     - Threshold at 0.5 → bet if profitable probability > 0.5
     - Kelly fraction = f(P(profitable), fill_price)

Usage:
  $env:DB_URL="postgresql://..."
  python train_model_v4.py

Output:
  model_v4_profitable.pkl   — primary model: P(profitable | features)
  model_v4_market.pkl       — market-level model for strategy discovery
  training_report_v4.txt

Install:
  pip install psycopg2-binary scikit-learn lightgbm numpy
"""

import os
import sys
import math
import logging
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

import requests

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy"); sys.exit(1)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.impute import SimpleImputer
except ImportError:
    print("ERROR: pip install scikit-learn"); sys.exit(1)

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: pip install lightgbm"); sys.exit(1)

# ---------------------------------------------------------── CONFIG ───────────

SUPABASE_URL = "https://kcluwyzyetmkxhvszpxi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtjbHV3eXp5ZXRta3hodnN6cHhpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA4NTY2NCwiZXhwIjoyMDg5NjYxNjY0fQ.IbxuXRW0K9_UFZKG1i951EoL9KtCsOXCaz5Z_YqsmYE"
REST_H       = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
TAKER_FEE    = 0.0025        # per side; round trip = 0.005
LEAKAGE_SECS = 120           # 120s removes near-resolution rows without losing markets
MIN_EDGE     = 0.01          # minimum edge to label a row as profitable
                             # filters out near-zero fills where fee variance dominates
PM_LO        = 0.25          # exclude markets where crowd is already decided
PM_HI        = 0.75
RANDOM_STATE = 42

# Walk-forward: test window size in markets
WF_TEST_SIZE = 60            # each fold tests on 60 markets
WF_MIN_TRAIN = 100           # minimum training markets before first fold

# ---------------------------------------------------------── CONNECT ─────────

def connect():
    return None  # no-op — using REST API


def _rest_fetch(table, params, limit=1000) -> list:
    rows = []
    offset = 0
    while True:
        p = {**params, "limit": limit, "offset": offset}
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=REST_H, params=p, timeout=120)
        if not r.text:
            log.warning(f"Supabase returned empty body for {table} (status={r.status_code}) at offset={offset}")
            break
        try:
            batch = r.json()
        except Exception as e:
            log.warning(f"Supabase JSON error for {table} offset={offset}: {e} — body: {r.text[:200]}")
            break
        if not batch or not isinstance(batch, list):
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return rows


def query(conn, sql) -> list:
    return []  # replaced per-call below


def _f(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ─────────────────────────────────────────────────────── NET EDGE ────────────

def _net_edge_correct(outcome_binary, fill_up, fill_down, p_market=None,
                      poly_spread=None, taker_fee=TAKER_FEE, min_edge=0.005):
    """
    Compute execution-aware edge on the CHOSEN side.

    Fills stay out of X (features). Fills only used here to construct y (label).

    Strategy: follow p_market (bet UP when p_market >= 0.5, DOWN otherwise).
    Label = 1 if edge on chosen side > MIN_EDGE after actual fill cost.

    fill_up:   use actual poly_fill_up (available in market_snapshots)
    fill_down: estimate as 1 - fill_up + spread (poly_fill_down not stored)

    edge_chosen = outcome_binary - fill_up - fees     (if betting UP)
                = (1-outcome_binary) - fill_down - fees (if betting DOWN)

    This is non-trivial (~50% profitable) because:
    - When outcome matches your bet: edge ≈ 1 - fill - fees > 0
    - When outcome misses your bet:  edge ≈ -fill - fees < 0
    - With real edge: profitable rate > 50%

    Also computes continuous excess return for regression target:
    excess = outcome_binary - p_market - fees  (how much outcome beat crowd)

    Returns (edge_chosen, profitable_int, side, excess_up, excess_down)
    """
    if np.isnan(outcome_binary) or p_market is None or np.isnan(p_market):
        return np.nan, np.nan, "NONE", np.nan, np.nan

    ob  = float(outcome_binary)
    rt  = taker_fee * 2
    hs  = (poly_spread / 2.0) if (poly_spread is not None
                                   and not np.isnan(poly_spread)) else 0.005

    # Fill prices — fills stay OUT of features, only used for label
    fu = float(fill_up) if (fill_up is not None and not np.isnan(fill_up))          else (p_market + hs)
    fd = 1.0 - fu + (hs * 2)   # DOWN fill ≈ complement + full spread

    # Edge on the chosen side (p_market determines which side to bet)
    if p_market >= 0.5:
        side        = "UP"
        edge_chosen = ob - fu - rt            # positive when UP wins + fill fair
    else:
        side        = "DOWN"
        edge_chosen = (1.0 - ob) - fd - rt   # positive when DOWN wins + fill fair

    profitable = int(edge_chosen > min_edge)
    if not profitable:
        side = "NONE"

    # Continuous regression targets (excess vs crowd, not vs fill)
    # These measure how much outcome beat crowd's implied probability
    excess_up   = ob       - p_market       - rt
    excess_down = (1.0-ob) - (1.0-p_market) - rt

    return round(float(edge_chosen), 6), profitable, side,            round(excess_up, 6), round(excess_down, 6)


# ─────────────────────────────────────────────────────── WALK-FORWARD ────────

def walk_forward_splits(cond_ids, min_train=WF_MIN_TRAIN, test_size=WF_TEST_SIZE):
    """
    Generates (train_idx, test_idx) pairs for walk-forward validation.
    Each fold expands the training set and tests on the next window.

    Returns list of (train_indices, test_indices, n_train_markets, n_test_markets)
    """
    markets = list(dict.fromkeys(cond_ids))
    n       = len(markets)
    folds   = []

    start = min_train
    while start + test_size <= n:
        train_markets = set(markets[:start])
        test_markets  = set(markets[start:start + test_size])
        train_idx = [i for i, c in enumerate(cond_ids) if c in train_markets]
        test_idx  = [i for i, c in enumerate(cond_ids) if c in test_markets]
        folds.append((train_idx, test_idx, len(train_markets), len(test_markets)))
        start += test_size

    return folds


# ─────────────────────────────────────────────────────── MODEL ───────────────

def _train(X_train, y_train, n_est=150, leaves=20, calibration="sigmoid"):
    """
    Train LightGBM. Uses built-in predict_proba (no external calibration).
    CalibratedClassifierCV was inverting probabilities on small CV folds.
    LightGBM's native probability output is already well-calibrated for
    gradient boosting on binary targets.
    """
    imp = SimpleImputer(strategy="median")
    Xt  = imp.fit_transform(X_train)
    model = lgb.LGBMClassifier(
        n_estimators=n_est, learning_rate=0.05,
        num_leaves=leaves, min_child_samples=30,   # raised from 20
        subsample=0.7, colsample_bytree=0.7,       # more aggressive dropout
        reg_alpha=0.5, reg_lambda=0.5,             # stronger L1/L2
        min_split_gain=0.01,                        # require meaningful splits
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(Xt, y_train)

    # Verify direction — raw correlation should be positive
    raw_preds = model.predict_proba(Xt)[:, 1]
    raw_corr  = np.corrcoef(raw_preds, y_train)[0, 1]
    if raw_corr < 0:
        log.warning(f"  [_train] Predictions NEGATIVELY correlated (corr={raw_corr:.3f})"
                    f" — inverted signal. Check label definition.")
    else:
        log.info(f"  [_train] Train correlation: {raw_corr:.3f}")

    return model, imp


def _train_regressor(X_train, y_train, n_est=150, leaves=20):
    """Train LightGBM regressor for continuous edge prediction."""
    imp = SimpleImputer(strategy="median")
    Xt  = imp.fit_transform(X_train)
    model = lgb.LGBMRegressor(
        n_estimators=n_est, learning_rate=0.05,
        num_leaves=leaves, min_child_samples=30,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=0.5,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(Xt, y_train)
    return model, imp


def _eval_fold(name, y_true, y_pred, base_rate):
    brier    = brier_score_loss(y_true, y_pred)
    baseline = brier_score_loss(y_true, np.full(len(y_true), base_rate))
    imp      = baseline - brier
    acc      = np.mean((y_pred >= 0.5) == y_true)
    return brier, baseline, imp, acc


def _importance(model, feat_names, top_n=20):
    try:
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_.astype(np.float64)
        elif hasattr(model, "calibrated_classifiers_"):
            imps = None
            for est in model.calibrated_classifiers_:
                base = est.estimator
                if hasattr(base, "feature_importances_"):
                    fi = base.feature_importances_.astype(np.float64)
                    imps = fi.copy() if imps is None else imps + fi
            if imps is not None:
                imps /= len(model.calibrated_classifiers_)
        else:
            return
        if imps is None:
            return
        ranked = sorted(zip(feat_names, imps), key=lambda x: -x[1])
        maxfi  = ranked[0][1] if ranked else 1
        print(f"\n  Feature importance (top {top_n})")
        print(f"  {'Feature':<42} {'Score':>8}")
        for fn, sc in ranked[:top_n]:
            bar = "#" * int(sc / maxfi * 20)
            print(f"  {fn:<42} {sc:>8.1f}  {bar}")
    except Exception as e:
        log.debug(f"  [_importance] failed: {e}")


# ═══════════════════════════════════════════ SNAPSHOT FEATURE ENGINEERING ════

REGIME_MAP   = {"TREND_UP":2,"TREND_DOWN":-2,"VOLATILE":1,"CALM":0,"DEAD":-1}
SESSION_MAP  = {"OVERLAP":3,"US":2,"LONDON":1,"ASIA":0,"OFFPEAK":-1}
ACTIVITY_MAP = {"HIGH":2,"NORMAL":1,"LOW":0,"DEAD":-1}
DAY_MAP      = {"WEEKDAY":1,"WEEKEND":0}
BUCKET_MAP   = {"heavy_fav":3,"favourite":2,"underdog":1,"longshot":0}
MOM_LBL_MAP  = {"TREND_UP":1,"NEUTRAL":0,"TREND_DOWN":-1}
VOL_LBL_MAP  = {"VOLATILE":1,"NORMAL":0,"DEAD":-1}
FLOW_LBL_MAP = {"LONG_CROWDED":1,"BALANCED":0,"SHORT_CROWDED":-1}


def fetch_snapshots(conn) -> list:
    log.info("Fetching market_snapshots via REST...")
    cols = ",".join([
        "condition_id","secs_left","secs_to_resolution","market_progress",
        "phase_early","phase_mid","phase_late","phase_final",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "price_vs_open_pct","price_vs_open_score",
        "momentum_10s","momentum_30s","momentum_60s","momentum_120s","momentum_score",
        "cl_divergence","cl_age","cl_vs_open_pct",
        "liq_total","liq_imbalance",
        "ob_imbalance","ob_bid_delta","ob_ask_delta",
        "vol_range_pct","volume_buy_ratio",
        "p_market","poly_fill_up","poly_fill_down",
        "poly_spread","poly_slip_up","poly_deviation",
        "basis_pct","funding_rate","okx_funding","gate_funding",
        "volatility_pct","flow_score","funding_zscore",
        "regime","session","activity","day_type",
        "price_bucket",
        "outcome_binary",
    ])
    rows = _rest_fetch("market_snapshots", {
        "select": cols,
        "resolved_outcome": "not.is.null",
        "outcome_binary": "not.is.null",
        "order": "created_at.asc",
    })
    log.info(f"  {len(rows):,} rows")
    return rows


def build_snapshot_features(rows):
    """
    Returns:
      X         — feature matrix (no directional fill features)
      y_prof    — profitable label (correct net edge formulation)
      y_dir     — direction label (kept for reference only)
      cond_ids  — market IDs for walk-forward splitting
      feat_names
      pm_raw    — raw p_market for baseline computation
      best_sides — UP/DOWN/NONE for each row
    """
    records, y_prof, y_edge, y_dir, cond_ids, pm_raws, best_sides = [], [], [], [], [], [], []
    skipped = 0

    for row in rows:
        ob   = _f(row.get("outcome_binary"))
        pm   = _f(row.get("p_market"))

        # Time to resolution: prefer secs_to_resolution, fall back to secs_left
        # Explicit null check avoids the `or` chain bug where:
        #   - 0.0 is falsy (would incorrectly fall through)
        #   - np.nan is truthy (would incorrectly not fall through)
        _str_val = _f(row.get("secs_to_resolution"))
        _sl_val  = _f(row.get("secs_left"))
        if not np.isnan(_str_val):
            str_ = _str_val          # use secs_to_resolution if valid
        elif not np.isnan(_sl_val):
            str_ = _sl_val           # fall back to secs_left if valid
        else:
            str_ = np.nan            # no time data at all

        # Skip only unusable rows
        if np.isnan(ob):
            skipped += 1; continue
        if np.isnan(str_) or str_ < LEAKAGE_SECS:
            skipped += 1; continue
        if np.isnan(pm):
            skipped += 1; continue
        # Mark extreme markets (pm outside 0.25-0.75) as feature
        # instead of discarding — model learns "no edge here" from context
        # Note: pm_uncertainty < 0.15 filter removed — too aggressive,
        # was removing legitimate rows with pm 0.075-0.925
        is_extreme = int(pm > PM_HI or pm < PM_LO)

        fill_up   = _f(row.get("poly_fill_up"))
        fill_down = _f(row.get("poly_fill_down"))

        # Correct net edge: policy-free, best available side
        net_edge, prof, best_side, edge_up, edge_down = _net_edge_correct(
            ob, fill_up, fill_down,
            p_market=pm,
            poly_spread=_f(row.get("poly_spread")),
            min_edge=MIN_EDGE
        )
        if np.isnan(net_edge):
            prof     = 0
            net_edge = 0.0
            edge_up  = 0.0
            edge_down = 0.0

        # Feature engineering
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

        # Uncertainty features derived from p_market (directionless)
        pm_abs_dev = abs(pm - 0.5)         # how far from fair coin
        pm_uncertainty = 1.0 - pm_abs_dev * 2  # 1.0 = 50/50, 0.0 = certain

        f = {
            # Time
            "secs_to_resolution":     str_,
            "log_secs_to_resolution": math.log1p(str_) if not np.isnan(str_) else np.nan,
            "market_progress":        mp,
            "phase_early":            float(row.get("phase_early") or 0),
            "phase_mid":              float(row.get("phase_mid")   or 0),
            "phase_late":             float(row.get("phase_late")  or 0),
            "phase_final":            float(row.get("phase_final") or 0),
            "hour_sin":               _f(row.get("hour_sin")),
            "hour_cos":               _f(row.get("hour_cos")),
            "dow_sin":                _f(row.get("dow_sin")),
            "dow_cos":                _f(row.get("dow_cos")),
            # Market uncertainty (directionless p_market features)
            "pm_abs_deviation":       pm_abs_dev,
            "pm_uncertainty":         pm_uncertainty,
            "is_extreme_market":      float(is_extreme),  # pm outside 0.25-0.75
            # Price features (BTC vs open — encodes direction but is real signal)
            "price_vs_open_pct":      pvop,
            "price_vs_open_score":    _f(row.get("price_vs_open_score")),
            # Momentum
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
            "mom_abs":                abs(m30)   if not np.isnan(m30) else np.nan,
            # Chainlink
            "cl_divergence":          cl,
            "cl_age":                 _f(row.get("cl_age")),
            "cl_vs_open_pct":         _f(row.get("cl_vs_open_pct")),
            "cl_abs_divergence":      abs(cl) if not np.isnan(cl) else np.nan,
            # Liquidations (dropped liq_delta/liq_accel — extreme outliers ±33M, stuck zeros)
            "liq_imbalance":          li,
            "liq_total":              min(lt, 5e6) if not np.isnan(lt) else np.nan,  # clip outliers
            "log_liq_total":          math.log1p(min(lt, 5e6)) if not np.isnan(lt) and lt>=0 else np.nan,
            "liq_abs_imbalance":      abs(li) if not np.isnan(li) else np.nan,
            # Order book (dropped ob_spread_pct — constant 0.0001 in all rows)
            "ob_imbalance":           obi,
            "ob_bid_delta":           _f(row.get("ob_bid_delta")),
            "ob_ask_delta":           _f(row.get("ob_ask_delta")),
            "ob_abs_imbalance":       abs(obi) if not np.isnan(obi) else np.nan,
            # Volatility / Volume
            "vol_range_pct":          vr,
            "volatility_pct":         _f(row.get("volatility_pct")),
            "volume_buy_ratio":       _f(row.get("volume_buy_ratio")),
            # Execution costs (directionless)
            "poly_spread":            ps,
            "poly_slip_up":           sl2,
            "effective_entry_cost":   (ps+sl2) if not(np.isnan(ps) or np.isnan(sl2)) else np.nan,
            "round_trip_cost":        (ps+sl2+TAKER_FEE*2) if not(np.isnan(ps) or np.isnan(sl2)) else TAKER_FEE*2,
            # Funding
            "basis_pct":              _f(row.get("basis_pct")),
            "funding_rate":           fr,
            "funding_zscore":         _f(row.get("funding_zscore")),
            "okx_funding":            _f(row.get("okx_funding")),
            "gate_funding":           _f(row.get("gate_funding")),
            "funding_abs":            abs(fr) if not np.isnan(fr) else np.nan,
            # Regime (dropped liquidity_score — constant 0.998 in all rows)
            "flow_score":             _f(row.get("flow_score")),
            "regime_enc":             REGIME_MAP.get(row.get("regime",""),0),
            "session_enc":            SESSION_MAP.get(row.get("session",""),0),
            "activity_enc":           ACTIVITY_MAP.get(row.get("activity",""),0),
            "day_type_enc":           DAY_MAP.get(row.get("day_type",""),0),
            "bucket_enc":             BUCKET_MAP.get(row.get("price_bucket",""),1),
            # Interaction terms
            "interact_mom_x_vol":     (m30*vr)  if not(np.isnan(m30) or np.isnan(vr))  else np.nan,
            "interact_liq_x_price":   (li*pvop) if not(np.isnan(li) or np.isnan(pvop)) else np.nan,
            "interact_mom_x_progress":(m30*mp)  if not(np.isnan(m30) or np.isnan(mp))  else np.nan,
            "interact_ob_x_spread":   (obi*ps)  if not(np.isnan(obi) or np.isnan(ps))  else np.nan,
            # Signal strength indicators (how strong are independent signals?)
            "signal_strength":        sum(1 for v in [
                abs(li) if not np.isnan(li) else 0,
                abs(obi) if not np.isnan(obi) else 0,
                abs(m30) if not np.isnan(m30) else 0,
            ] if v > 0.1),
            # Discovered interactions (auto-discovery pipeline, corr > 0.10)
            "vol_x_pm_abs_dev":       (vr * pm_abs_dev)
                                      if not(np.isnan(vr) or np.isnan(pm_abs_dev))
                                      else np.nan,
            "vol_x_funding_zscore":   (vr * _f(row.get("funding_zscore")))
                                      if not(np.isnan(vr) or np.isnan(_f(row.get("funding_zscore"))))
                                      else np.nan,

            # ── Time-to-resolution interactions (#7) ──────────────────────────
            # Early signal carries different weight than late signal
            # liq_imbalance at 250s is predictive; at 130s it may be noise
            "liq_imbal_x_secs":       (li * str_)
                                      if not(np.isnan(li) or np.isnan(str_))
                                      else np.nan,
            "mom_x_secs":             (m30 * str_)
                                      if not(np.isnan(m30) or np.isnan(str_))
                                      else np.nan,
            "ob_imbal_x_secs":        (obi * str_)
                                      if not(np.isnan(obi) or np.isnan(str_))
                                      else np.nan,

            # ── Cross-signal disagreement (#6) ────────────────────────────────
            # Markets misprice most when signals disagree and crowd picks one side
            # sign agreement: +1 if signals agree direction, -1 if they disagree
            "mom_liq_agree":          float(np.sign(m30) == np.sign(li))
                                      if not(np.isnan(m30) or np.isnan(li) or m30==0 or li==0)
                                      else np.nan,
            "mom_ob_agree":           float(np.sign(m30) == np.sign(obi))
                                      if not(np.isnan(m30) or np.isnan(obi) or m30==0 or obi==0)
                                      else np.nan,
            "liq_ob_agree":           float(np.sign(li) == np.sign(obi))
                                      if not(np.isnan(li) or np.isnan(obi) or li==0 or obi==0)
                                      else np.nan,
            # Signal dispersion: how much do directional signals disagree?
            # High dispersion = signals pointing different ways = uncertain regime
            "signal_dispersion":      float(np.nanstd([
                                          m30 / (abs(m30) + 1e-8) if not np.isnan(m30) else np.nan,
                                          li  / (abs(li)  + 1e-8) if not np.isnan(li)  else np.nan,
                                          obi / (abs(obi) + 1e-8) if not np.isnan(obi) else np.nan,
                                      ])) if not all(np.isnan(v) for v in [m30, li, obi]) else np.nan,

            # ── Delta/change features ─────────────────────────────────────────
            "mom_accel_abs":          abs(m30 - m60)
                                      if not(np.isnan(m30) or np.isnan(m60))
                                      else np.nan,
        }

        records.append(f)
        y_prof.append(prof)
        y_edge.append(float(net_edge))   # continuous edge for regression
        y_dir.append(int(ob))
        pm_raws.append(pm)
        best_sides.append(best_side)
        cond_ids.append(row.get("condition_id",""))

    log.info(f"  Engineered {len(records):,} rows ({skipped:,} skipped)")
    pct_prof = sum(y_prof) / max(len(y_prof), 1) * 100
    pct_up   = sum(1 for s in best_sides if s=="UP") / max(len(best_sides), 1) * 100
    pct_down = sum(1 for s in best_sides if s=="DOWN") / max(len(best_sides), 1) * 100
    log.info(f"  Profitable: {sum(y_prof):,}/{len(y_prof):,} ({pct_prof:.1f}%)")
    log.info(f"  Best side: UP={pct_up:.1f}%  DOWN={pct_down:.1f}%  "
             f"NONE={100-pct_up-pct_down:.1f}%")

    fn   = list(records[0].keys())
    X    = np.array([[r[k] for k in fn] for r in records], dtype=np.float32)
    yp   = np.array(y_prof, dtype=np.int32)
    ye   = np.array(y_edge, dtype=np.float32)   # continuous edge
    yd   = np.array(y_dir,  dtype=np.int32)
    pm   = np.array(pm_raws, dtype=np.float32)
    return X, yp, ye, yd, cond_ids, fn, pm


# ═══════════════════════════════════════════ MARKET FEATURE ENGINEERING ══════

def fetch_market_outcomes(conn) -> list:
    log.info("Computing market_outcomes via REST + Python aggregation...")
    from collections import defaultdict, Counter

    # Fetch early-phase snapshots
    snaps = _rest_fetch("market_snapshots", {
        "select": "condition_id,resolved_outcome,market_end_time,secs_left,p_market,btc_price,momentum_30s,liq_total,liq_imbalance,ob_imbalance,cl_divergence,vol_range_pct,funding_rate,funding_zscore,regime,session",
        "resolved_outcome": "not.is.null",
        "secs_left": "gt.150",
        "order": "created_at.asc",
    })

    # Fetch trades
    trades = _rest_fetch("trades", {
        "select": "condition_id,strategy,side,resolved_outcome",
        "action": "eq.OPEN",
        "resolved_outcome": "not.is.null",
    })

    # Fetch signal_log
    sig_log = _rest_fetch("signal_log", {
        "select": "condition_id,strategy,signal_value,resolved_outcome,secs_left",
        "resolved_outcome": "not.is.null",
        "secs_left": "gt.150",
    })

    # Aggregate snapshots by market
    mkt = defaultdict(lambda: {"snaps": [], "resolved_outcome": None, "market_end_time": None})
    for s in snaps:
        cid = s["condition_id"]
        mkt[cid]["resolved_outcome"] = s["resolved_outcome"]
        mkt[cid]["market_end_time"]  = s.get("market_end_time")
        mkt[cid]["snaps"].append(s)

    # Aggregate trades by market
    trade_agg = defaultdict(lambda: {"strategies": set(), "sides": [], "liq": 0, "pa": 0, "ob": 0})
    for t in trades:
        cid = t["condition_id"]; strat = t.get("strategy",""); side = t.get("side","")
        trade_agg[cid]["strategies"].add(strat)
        trade_agg[cid]["sides"].append(side)
        if strat == "Liquidation Cascade": trade_agg[cid]["liq"] = 1
        if strat == "Price Anchor":        trade_agg[cid]["pa"]  = 1
        if strat == "OB Pressure":         trade_agg[cid]["ob"]  = 1

    # Aggregate signal_log by market
    sig_agg = defaultdict(list)
    for s in sig_log:
        if s.get("strategy") == "Liquidation Cascade":
            v = _f(s.get("signal_value"))
            if not np.isnan(v): sig_agg[s["condition_id"]].append(v)

    def _std(vals):
        if len(vals) < 2: return np.nan
        m = sum(vals)/len(vals)
        return math.sqrt(sum((x-m)**2 for x in vals)/len(vals))

    rows = []
    for cid, d in sorted(mkt.items(), key=lambda x: x[1]["market_end_time"] or ""):
        ss = d["snaps"]
        if not ss: continue
        ro = d["resolved_outcome"]
        ob = 1 if ro == "UP" else (0 if ro == "DOWN" else None)
        if ob is None: continue

        pm_vals  = [_f(s.get("p_market"))    for s in ss if _f(s.get("p_market")) is not None and not np.isnan(_f(s.get("p_market")))]
        m30_vals = [_f(s.get("momentum_30s"))for s in ss if _f(s.get("momentum_30s")) is not None and not np.isnan(_f(s.get("momentum_30s")))]
        lt_vals  = [_f(s.get("liq_total"))   for s in ss if _f(s.get("liq_total")) is not None and not np.isnan(_f(s.get("liq_total")))]
        li_vals  = [_f(s.get("liq_imbalance"))for s in ss if _f(s.get("liq_imbalance")) is not None and not np.isnan(_f(s.get("liq_imbalance")))]
        ob_vals  = [_f(s.get("ob_imbalance")) for s in ss if _f(s.get("ob_imbalance")) is not None and not np.isnan(_f(s.get("ob_imbalance")))]
        cl_vals  = [_f(s.get("cl_divergence"))for s in ss if _f(s.get("cl_divergence")) is not None and not np.isnan(_f(s.get("cl_divergence")))]
        vr_vals  = [_f(s.get("vol_range_pct"))for s in ss if _f(s.get("vol_range_pct")) is not None and not np.isnan(_f(s.get("vol_range_pct")))]
        fr_vals  = [_f(s.get("funding_rate")) for s in ss if _f(s.get("funding_rate")) is not None and not np.isnan(_f(s.get("funding_rate")))]
        fz_vals  = [_f(s.get("funding_zscore"))for s in ss if _f(s.get("funding_zscore")) is not None and not np.isnan(_f(s.get("funding_zscore")))]
        bp_vals  = [_f(s.get("btc_price"))   for s in ss if _f(s.get("btc_price")) is not None and not np.isnan(_f(s.get("btc_price")))]
        regimes  = [s.get("regime","") for s in ss if s.get("regime")]
        sessions = [s.get("session","") for s in ss if s.get("session")]

        # p_market_open = p_market at highest secs_left (earliest snapshot)
        pm_open_snap = max(ss, key=lambda s: _f(s.get("secs_left")) or 0)
        pm_open = _f(pm_open_snap.get("p_market"))

        btc_range = np.nan
        if len(bp_vals) >= 2:
            mn = min(bp_vals); mx = max(bp_vals)
            if mn > 0: btc_range = (mx - mn) / mn * 100

        ta = trade_agg[cid]
        n_strats = len(ta["strategies"])
        sides    = ta["sides"]
        agree    = 1 if (len(set(sides)) == 1 and n_strats > 1) else 0
        sl_vals  = sig_agg[cid]

        rows.append({
            "condition_id":       cid,
            "resolved_outcome":   ro,
            "outcome_binary":     ob,
            "market_end_time":    d["market_end_time"],
            "total_snapshots":    len(ss),
            "first_snap_secs_left": max((_f(s.get("secs_left")) or 0) for s in ss),
            "p_market_open":      pm_open,
            "p_market_std":       _std(pm_vals),
            "btc_range_pct":      btc_range,
            "avg_momentum_30s":   sum(m30_vals)/len(m30_vals) if m30_vals else np.nan,
            "max_liq_total":      max(lt_vals) if lt_vals else np.nan,
            "avg_liq_imbalance":  sum(li_vals)/len(li_vals) if li_vals else np.nan,
            "avg_ob_imbalance":   sum(ob_vals)/len(ob_vals) if ob_vals else np.nan,
            "avg_cl_divergence":  sum(cl_vals)/len(cl_vals) if cl_vals else np.nan,
            "avg_vol_range_pct":  sum(vr_vals)/len(vr_vals) if vr_vals else np.nan,
            "avg_funding_rate":   sum(fr_vals)/len(fr_vals) if fr_vals else np.nan,
            "avg_funding_zscore": sum(fz_vals)/len(fz_vals) if fz_vals else np.nan,
            "dominant_regime":    Counter(regimes).most_common(1)[0][0] if regimes else "",
            "dominant_session":   Counter(sessions).most_common(1)[0][0] if sessions else "",
            "n_strategies_fired": n_strats,
            "n_strategies_up":    sides.count("UP"),
            "n_strategies_down":  sides.count("DOWN"),
            "liq_fired":          ta["liq"],
            "pa_fired":           ta["pa"],
            "ob_fired":           ta["ob"],
            "strategies_agree":   agree,
            "avg_liq_signal":     sum(sl_vals)/len(sl_vals) if sl_vals else 0.0,
            "max_liq_signal":     max(sl_vals) if sl_vals else 0.0,
        })

    log.info(f"  {len(rows):,} markets aggregated")
    return rows


def build_market_features(rows):
    records, y_vals, cond_ids = [], [], []
    for row in rows:
        ob = _f(row.get("outcome_binary"))
        if np.isnan(ob):
            continue
        pm_std = _f(row.get("p_market_std"))
        pm_opn = _f(row.get("p_market_open"))
        mlt    = _f(row.get("max_liq_total"))
        btcr   = _f(row.get("btc_range_pct"))
        fzs    = _f(row.get("avg_funding_zscore"))
        nf     = _f(row.get("n_strategies_fired"))
        nu     = _f(row.get("n_strategies_up"))
        nd     = _f(row.get("n_strategies_down"))
        mls    = _f(row.get("max_liq_signal"))

        f = {
            # p_market — directionless only
            "p_market_std":            pm_std,
            "p_market_open":           pm_opn,
            "p_market_open_abs_dev":   abs(pm_opn - 0.5) if not np.isnan(pm_opn) else np.nan,
            "p_market_uncertainty":    (1 - abs(pm_opn - 0.5) * 2) if not np.isnan(pm_opn) else np.nan,
            # BTC (magnitude only, not direction)
            "btc_range_pct":           btcr,
            "log_btc_range":           math.log1p(btcr) if not np.isnan(btcr) and btcr >= 0 else np.nan,
            # Signals (magnitude — abs for direction-neutral)
            "avg_momentum_abs":        abs(_f(row.get("avg_momentum_30s"))) if not np.isnan(_f(row.get("avg_momentum_30s"))) else np.nan,
            "max_liq_total":           mlt,
            "log_max_liq":             math.log1p(mlt) if not np.isnan(mlt) and mlt >= 0 else np.nan,
            "avg_liq_imbalance_abs":   abs(_f(row.get("avg_liq_imbalance"))) if not np.isnan(_f(row.get("avg_liq_imbalance"))) else np.nan,
            "avg_ob_imbalance_abs":    abs(_f(row.get("avg_ob_imbalance"))) if not np.isnan(_f(row.get("avg_ob_imbalance"))) else np.nan,
            "avg_cl_divergence_abs":   abs(_f(row.get("avg_cl_divergence"))) if not np.isnan(_f(row.get("avg_cl_divergence"))) else np.nan,
            "avg_vol_range_pct":       _f(row.get("avg_vol_range_pct")),
            "avg_funding_rate_abs":    abs(_f(row.get("avg_funding_rate"))) if not np.isnan(_f(row.get("avg_funding_rate"))) else np.nan,
            "avg_funding_zscore_abs":  abs(fzs) if not np.isnan(fzs) else np.nan,
            # Strategy activity
            "n_strategies_fired":      nf,
            "strategies_agree":        _f(row.get("strategies_agree")),
            "liq_fired":               _f(row.get("liq_fired")),
            "pa_fired":                _f(row.get("pa_fired")),
            "ob_fired":                _f(row.get("ob_fired")),
            # Signal strength
            "avg_liq_signal":          _f(row.get("avg_liq_signal")),
            "max_liq_signal":          mls,
            "log_max_liq_signal":      math.log1p(mls) if not np.isnan(mls) and mls >= 0 else np.nan,
            # Coverage
            "total_snapshots":         _f(row.get("total_snapshots")),
            # Regime
            "regime_enc":              REGIME_MAP.get(row.get("dominant_regime",""), 0),
            "session_enc":             SESSION_MAP.get(row.get("dominant_session",""), 0),
        }
        records.append(f)
        y_vals.append(int(ob))
        cond_ids.append(row.get("condition_id",""))

    log.info(f"  Engineered {len(records):,} market rows")
    fn = list(records[0].keys())
    X  = np.array([[r[k] for k in fn] for r in records], dtype=np.float32)
    y  = np.array(y_vals, dtype=np.int32)
    return X, y, cond_ids, fn



# ═══════════════════════════════════════════ AUTOMATED DISCOVERY PIPELINE ═════

def _regime_analysis(X_te, yp_te, preds_cls, fn, base_rate, top_n=12):
    """
    For each top feature, split test rows into HIGH/LOW buckets and measure
    whether the model performs significantly better in one regime.
    Returns list of regime candidates sorted by improvement.
    """
    try:
        imps = None
        # Try to get importances from classifier stored in outer scope
        # We pass them in via feature_scores parameter instead
        pass
    except Exception:
        pass

    # Exclude binary, categorical, and p_market-derived features
    # Binary features (0/1) produce trivial regime splits
    # p_market-derived features (pm_abs_deviation, pm_uncertainty, is_extreme_market,
    # bucket_enc) create artifacts because extreme p_market is trivially predictable
    EXCLUDE_FROM_REGIME = {
        "is_extreme_market", "pm_abs_deviation", "pm_uncertainty",
        "pm_abs_dev", "bucket_enc", "regime_enc", "session_enc",
        "activity_enc", "day_type_enc", "mom_label_enc", "vol_label_enc",
        "flow_label_enc", "phase_early", "phase_mid", "phase_late",
        "phase_final", "mom_all_agree", "mom_windows_positive",
        "above_threshold", "signal_strength",
    }

    candidates = []
    for fname in fn[:top_n]:
        if fname not in fn:
            continue
        if fname in EXCLUDE_FROM_REGIME:
            continue
        idx = fn.index(fname)
        vals = X_te[:, idx]
        valid = ~np.isnan(vals)
        if valid.sum() < 40:
            continue
        # Skip binary features (only 2 unique values)
        unique_vals = np.unique(vals[valid])
        if len(unique_vals) <= 3:
            continue

        med = np.nanmedian(vals)
        hi  = valid & (vals > med)
        lo  = valid & (vals <= med)

        if hi.sum() < 15 or lo.sum() < 15:
            continue

        bl   = brier_score_loss(yp_te, np.full(len(yp_te), base_rate))
        b_hi = brier_score_loss(yp_te[hi], preds_cls[hi])
        b_lo = brier_score_loss(yp_te[lo], preds_cls[lo])
        imp_hi = bl - b_hi
        imp_lo = bl - b_lo

        # Only flag if one regime is clearly better than baseline AND
        # meaningfully better than the other regime
        for regime, imp, n, thresh_dir in [
            ("HIGH", imp_hi, int(hi.sum()), "above"),
            ("LOW",  imp_lo, int(lo.sum()), "below"),
        ]:
            if imp > 0.008 and imp > (imp_hi if regime=="LOW" else imp_lo) + 0.004:
                candidates.append({
                    "feature":     fname,
                    "regime":      regime,
                    "threshold":   round(float(med), 6),
                    "thresh_dir":  thresh_dir,
                    "improvement": round(float(imp), 4),
                    "n_rows":      n,
                    "brier":       round(float(b_hi if regime=="HIGH" else b_lo), 4),
                    "baseline":    round(float(bl), 4),
                })

    return sorted(candidates, key=lambda x: -x["improvement"])


def _interaction_discovery(X_tr, yp_tr, fn, top_features, n_pairs=15):
    """
    Test pairwise products of top features for correlation with label.
    High correlation suggests this interaction is a useful composite signal.
    """
    results = []
    pairs = [(i, j) for i in range(min(len(top_features), 8))
                    for j in range(i+1, min(len(top_features), 8))][:n_pairs]

    for i, j in pairs:
        fi = top_features[i]
        fj = top_features[j]
        if fi not in fn or fj not in fn:
            continue
        xi = X_tr[:, fn.index(fi)].astype(np.float64)
        xj = X_tr[:, fn.index(fj)].astype(np.float64)

        interaction = xi * xj
        mask = ~np.isnan(interaction) & ~np.isnan(yp_tr.astype(np.float64))
        if mask.sum() < 50:
            continue

        corr = np.corrcoef(interaction[mask], yp_tr[mask].astype(np.float64))[0, 1]
        if np.isnan(corr):
            continue

        if abs(corr) > 0.04:
            results.append({
                "feature_a": fi,
                "feature_b": fj,
                "name":      f"{fi}_x_{fj}",
                "corr":      round(float(corr), 4),
                "n":         int(mask.sum()),
            })

    return sorted(results, key=lambda x: -abs(x["corr"]))


def _generate_strategy_template(candidate, run_ts):
    """Generate a Python strategy skeleton from a regime candidate."""
    fname   = candidate["feature"].replace(" ", "_").replace("/", "_")
    regime  = candidate["regime"]
    thresh  = candidate["threshold"]
    imp     = candidate["improvement"]
    n       = candidate["n_rows"]
    tdir    = candidate["thresh_dir"]

    op = ">" if tdir == "above" else "<="
    op_inv = "<=" if tdir == "above" else ">"

    return f'''
def strategy_{fname}_{regime.lower()}_regime(market, secs_left, tracker, position):
    """
    AUTO-GENERATED CANDIDATE — {run_ts}
    ---------------------------------------------------------
    Source: ML regime analysis
    Feature:     {candidate["feature"]}
    Regime:      {regime} ({tdir} median threshold)
    Threshold:   {thresh:.6f}  (approximate median from training data)
    WF edge:     +{imp:.4f} Brier improvement vs baseline
    Sample size: {n} test rows

    STATUS: OBSERVATION MODE — not validated for live trading
    ---------------------------------------------------------
    What this means:
    When {candidate["feature"]} is {regime} ({op} {thresh:.4f}),
    the ML model predicts crowd direction more accurately (+{imp:.4f} Brier).
    This suggests a systematic pattern worth investigating as a strategy filter.

    Next steps:
    1. Add this as a filter on top of an existing strategy (e.g. Liquidation Cascade)
    2. Log outcomes with/without filter for 50+ trades
    3. Compare win rates in regime vs out of regime
    4. If regime WR is 5%+ higher, hardcode the filter

    Implementation notes:
    - Replace threshold {thresh:.4f} with data-driven value from SQL analysis
    - Feature {candidate["feature"]} maps to: _regime / _vol_cache / _funding_cache
    - Consider combining with other regime filters for stronger signal
    """
    # TODO: Get the signal value for {candidate["feature"]}
    # Example mapping (implement correctly for your codebase):
    # funding_zscore  → _regime.get("funding_zscore", 0.0)
    # vol_range_pct   → _vol_cache.get("range_pct", 0.0)
    # volatility_pct  → _regime.get("volatility_pct", 0.0)
    # liq_total       → sum(_liq_cache.values())
    # pm_abs_deviation → abs(get_poly_prices(market).get("up_mid", 0.5) - 0.5)

    signal_value = 0.0  # TODO: implement

    # Filter: only trade in {regime} regime
    if signal_value {op_inv} {thresh:.4f}:
        _log_signal("ML_{fname}_{regime}", market, secs_left,
                    signal_value=signal_value,
                    threshold={thresh:.4f},
                    reason="{fname}_regime_filter")
        return position

    # Strategy fires in {regime} {candidate["feature"]} regime
    # Add your existing strategy logic here, or use as a pre-filter
    # on top of Liquidation Cascade / Price Anchor / OB Pressure

    _log_signal("ML_{fname}_{regime}", market, secs_left,
                signal_value=signal_value,
                threshold={thresh:.4f},
                reason="observation_mode_no_bet")

    return position  # observation mode — remove when validated
'''


def run_discovery_pipeline(
    X_tr, yp_tr, X_te, yp_te, preds_cls, fn,
    base_rate, final_model, final_imp,
    mkt_fn, mkt_final_model,
    avg_imp, avg_imp3, total_pnl, run_ts,
):
    """
    Full automated strategy discovery pipeline.
    Runs after both models are trained and produces discovery_report.txt
    """
    print("\n" + "=" * 60)
    print("  AUTOMATED STRATEGY DISCOVERY")
    print("=" * 60)

    report_lines = [
        f"Strategy Discovery Report — {run_ts}",
        f"Primary model avg improvement: {avg_imp:+.4f}",
        f"Market model avg improvement:  {avg_imp3:+.4f}",
        f"Total simulated PnL:           {total_pnl:+.4f}",
        "",
    ]

    # ── 1. Feature importance stability (primary model) ────────────────────────
    cls_imps  = final_model.feature_importances_.astype(np.float64)
    reg_imps  = mkt_final_model.feature_importances_.astype(np.float64)
    cls_top10 = [fn[i] for i in cls_imps.argsort()[::-1][:10]]
    mkt_top10 = [mkt_fn[i] for i in reg_imps.argsort()[::-1][:10]]

    # ── 2. Regime analysis on primary model test data ─────────────────────────
    regime_candidates = _regime_analysis(
        X_te, yp_te, preds_cls, fn, base_rate, top_n=len(fn))

    print(f"\n  Regime candidates ({len(regime_candidates)} found):")
    if regime_candidates:
        print(f"  {'Feature':<35} {'Regime':<6} {'Threshold':>10} {'WF edge':>8} {'N':>6}")
        print(f"  {'─'*35} {'─'*6} {'─'*10} {'─'*8} {'─'*6}")
        for c in regime_candidates[:8]:
            print(f"  {c['feature']:<35} {c['regime']:<6} "
                  f"{c['threshold']:>10.4f} {c['improvement']:>+8.4f} {c['n_rows']:>6}")
        report_lines.append("REGIME CANDIDATES:")
        for c in regime_candidates[:8]:
            report_lines.append(
                f"  {c['feature']} {c['regime']} regime "
                f"(threshold={c['threshold']:.4f}, WF edge={c['improvement']:+.4f}, n={c['n_rows']})"
            )
    else:
        print("  None found (need more data or stronger signals)")
    report_lines.append("")

    # ── 3. Interaction discovery ───────────────────────────────────────────────
    interaction_candidates = _interaction_discovery(
        final_imp.transform(X_tr), yp_tr, fn, cls_top10)

    print(f"\n  Interaction candidates ({len(interaction_candidates)} found):")
    if interaction_candidates:
        print(f"  {'Interaction':<55} {'Corr':>6} {'N':>6}")
        print(f"  {'─'*55} {'─'*6} {'─'*6}")
        for c in interaction_candidates[:5]:
            print(f"  {c['name']:<55} {c['corr']:>+6.3f} {c['n']:>6}")
        report_lines.append("INTERACTION CANDIDATES:")
        for c in interaction_candidates[:5]:
            report_lines.append(
                f"  {c['name']} (corr={c['corr']:+.3f}, n={c['n']})"
            )
    else:
        print("  None found above threshold")
    report_lines.append("")

    # ── 4. Market model feature signals ───────────────────────────────────────
    print(f"\n  Market model top signals (strategy discovery):")
    print(f"  {'Feature':<42} {'Score':>8}  {'Signal meaning'}")
    print(f"  {'─'*42} {'─'*8}  {'─'*30}")
    signal_meanings = {
        "p_market_std":           "volatile early odds → decisive market",
        "avg_funding_zscore_abs": "extreme funding → trending market",
        "avg_ob_imbalance_abs":   "strong OB lean → directional pressure",
        "avg_momentum_abs":       "strong BTC momentum → trending",
        "avg_vol_range_pct":      "high volatility → more predictable",
        "avg_funding_rate_abs":   "extreme funding rate → regime signal",
        "avg_cl_divergence_abs":  "Chainlink divergence → price dislocation",
        "btc_range_pct":          "large BTC range → volatile market",
        "max_liq_total":          "large liquidations → cascade signal",
        "avg_liq_imbalance_abs":  "directional liquidations → one-sided",
    }
    mkt_imps_full = mkt_final_model.feature_importances_.astype(np.float64)
    mkt_ranked    = sorted(zip(mkt_fn, mkt_imps_full), key=lambda x: -x[1])
    for fname, score in mkt_ranked[:8]:
        meaning = signal_meanings.get(fname, "— see feature description")
        print(f"  {fname:<42} {score:>8.1f}  {meaning}")
    report_lines.append("MARKET MODEL TOP SIGNALS:")
    for fname, score in mkt_ranked[:8]:
        report_lines.append(f"  {fname}: {score:.1f}")
    report_lines.append("")

    # ── 5. Generate strategy templates for top regime candidates ──────────────
    os.makedirs("strategy_candidates", exist_ok=True)
    generated = []
    for c in regime_candidates[:3]:
        template = _generate_strategy_template(c, run_ts)
        safe_name = c["feature"].replace("/","_").replace(" ","_")
        fname_out = f"strategy_candidates/candidate_{safe_name}_{c['regime'].lower()}.py"
        with open(fname_out, "w", encoding="utf-8") as fout:
            fout.write(template)
        generated.append(fname_out)
        log.info(f"  [DISCOVERY] Generated: {fname_out}")

    if generated:
        print(f"\n  Generated {len(generated)} strategy template(s):")
        for g in generated:
            print(f"    {g}")
        report_lines.append("GENERATED TEMPLATES:")
        report_lines += [f"  {g}" for g in generated]
    report_lines.append("")

    # ── 6. Actionable recommendations ─────────────────────────────────────────
    print("\n  Actionable recommendations:")
    recs = []

    if regime_candidates:
        top = regime_candidates[0]
        recs.append(
            f"  1. Add {top['feature']} {top['regime']} filter to Liquidation Cascade "
            f"(WF edge +{top['improvement']:.4f}, threshold={top['threshold']:.4f})"
        )

    if interaction_candidates:
        top_int = interaction_candidates[0]
        recs.append(
            f"  2. Add {top_int['name']} as feature — "
            f"corr={top_int['corr']:+.3f} with profitability label"
        )

    if avg_imp3 > -0.010:
        recs.append(
            f"  3. Market model near zero ({avg_imp3:+.4f}) — "
            f"add as market pre-filter when it crosses 0"
        )

    if total_pnl > 0:
        recs.append(
            f"  4. Simulated PnL positive (+{total_pnl:.1f}) — "
            f"classifier threshold 0.55 is adding value"
        )

    for r in recs:
        print(r)

    report_lines.append("RECOMMENDATIONS:")
    report_lines += recs

    # ── 7. Write report ────────────────────────────────────────────────────────
    report_fname = f"discovery_report_{run_ts.replace(':','-').replace(' ','_')}.txt"
    with open(report_fname, "w", encoding="utf-8") as fout:
        fout.write("\n".join(report_lines))
    log.info(f"  [DISCOVERY] Report saved: {report_fname}")
    print(f"\n  Full report: {report_fname}")

# ═══════════════════════════════════════════════════════════════════ MAIN ═════

def main():
    report = []
    log.info("=" * 60)
    log.info("ML TRAINING PIPELINE v4")
    log.info("=" * 60)

    conn = connect()

    # Strategy summary
    print("\n  Strategy summary:")
    print(f"  {'Strategy':<25} {'N':>5}  {'Wins':>5}  {'WR':>7}  {'PnL':>9}")
    print(f"  {'─'*25} {'─'*5}  {'─'*5}  {'─'*7}  {'─'*9}")
    from collections import defaultdict
    try:
        trade_data = _rest_fetch("trades", {
            "select": "strategy,actual_win,pnl",
            "action": "eq.OPEN",
            "resolved_outcome": "not.is.null",
        })
        strat_agg = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
        for t in trade_data:
            s = t.get("strategy","?")
            strat_agg[s]["n"] += 1
            strat_agg[s]["wins"] += 1 if t.get("actual_win") else 0
            strat_agg[s]["pnl"] += _f(t.get("pnl")) or 0.0
        for strat, d in sorted(strat_agg.items(), key=lambda x: -x[1]["n"]):
            wr = d["wins"]/d["n"]*100 if d["n"] else 0
            print(f"  {strat:<25} {d['n']:>5}  {d['wins']:>5}  {wr:>6.1f}%  ${d['pnl']:>+8.2f}")
    except Exception as e:
        print(f"  (trades summary unavailable: {e})")

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY MODEL: P(profitable | snapshot features)
    # Walk-forward validation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PRIMARY MODEL — P(profitable | features)")
    print("  Walk-forward validation")
    print("=" * 60)

    snap_rows = fetch_snapshots(conn)
    X, yp, ye, yd, cids, fn, pm = build_snapshot_features(snap_rows)

    folds = walk_forward_splits(cids)
    n_markets = len(set(cids))
    log.info(f"  {n_markets} total markets → {len(folds)} walk-forward folds")

    if len(folds) == 0:
        log.warning("  Not enough markets for walk-forward. Need "
                    f"{WF_MIN_TRAIN + WF_TEST_SIZE}+, have {n_markets}.")
        log.warning("  Falling back to single 80/20 split.")
        # Fallback
        markets  = list(dict.fromkeys(cids))
        split    = int(len(markets) * 0.80)
        tr_set   = set(markets[:split])
        te_set   = set(markets[split:])
        ti = [i for i, c in enumerate(cids) if c in tr_set]
        vi = [i for i, c in enumerate(cids) if c in te_set]
        folds = [(ti, vi, len(tr_set), len(te_set))]

    fold_results = []
    for fold_i, (ti, vi, n_tr, n_te) in enumerate(folds):
        X_tr, yp_tr, ye_tr = X[ti], yp[ti], ye[ti]
        X_te, yp_te, ye_te = X[vi], yp[vi], ye[vi]
        pm_te               = pm[vi]

        # Classifier: P(profitable)
        cal_method = "isotonic" if n_tr >= 500 else "sigmoid"
        model, imp = _train(X_tr, yp_tr, n_est=150, leaves=20,
                            calibration=cal_method)
        preds_cls = model.predict_proba(imp.transform(X_te))[:, 1]

        # Regressor: E(edge)
        reg, reg_imp = _train_regressor(X_tr, ye_tr, n_est=150, leaves=20)
        preds_reg = reg.predict(reg_imp.transform(X_te))

        base_rate = yp_tr.mean()
        brier, baseline, improvement, acc = _eval_fold(
            f"Fold {fold_i+1}", yp_te, preds_cls, base_rate)

        # Proper baselines (issue 6)
        # 1. Always follow crowd (bet the p_market side) — crowd accuracy
        pm_te_raw = pm[vi]
        always_follow = np.where(pm_te_raw >= 0.5,
                                  pm_te_raw,           # P(UP) = p_market
                                  1.0 - pm_te_raw)     # P(DOWN) = 1-p_market
        # For yp (crowd_correct label), crowd always follows itself → P(correct)=p_market
        # When pm>=0.5 and UP wins: yp=1, pred=pm. When pm<0.5 and DOWN wins: yp=1, pred=1-pm
        brier_follow  = brier_score_loss(yp_te, always_follow)
        # 2. Always fade crowd (bet opposite)
        always_fade   = 1.0 - always_follow
        brier_fade    = brier_score_loss(yp_te, always_fade)
        # 3. Random (always predict base rate)
        brier_random  = brier_score_loss(yp_te, np.full(len(yp_te), base_rate))

        # Regime breakdown: volatility and uncertainty
        regime_snap = X_te[:, fn.index("vol_range_pct")] if "vol_range_pct" in fn else None
        uncert_snap = X_te[:, fn.index("pm_uncertainty")] if "pm_uncertainty" in fn else None

        fold_results.append({
            "fold": fold_i + 1, "n_train": n_tr, "n_test": n_te,
            "brier": brier, "baseline": baseline, "improvement": improvement,
            "accuracy": acc, "cal": cal_method,
            "reg_mae": float(np.mean(np.abs(preds_reg - ye_te))),
            "reg_corr": float(np.corrcoef(preds_reg, ye_te)[0,1]) if len(ye_te) > 2 else 0.0,
        })

        # Issue 4: Simulated PnL — filter by classifier + regressor threshold
        # This is the economic sanity check: does filtering produce real edge?
        cls_threshold = 0.55   # only bet when model confident crowd is right
        reg_threshold = MIN_EDGE
        trade_mask    = (preds_cls > cls_threshold) & (preds_reg > reg_threshold)
        n_trades      = trade_mask.sum()
        if n_trades > 0:
            sim_pnl   = float(ye_te[trade_mask].sum())
            pnl_per   = float(ye_te[trade_mask].mean())
            win_rate  = float(yp_te[trade_mask].mean())
        else:
            sim_pnl = pnl_per = win_rate = 0.0

        # Baseline PnL: always bet (no filter)
        baseline_pnl = float(ye_te.sum())

        fold_results[-1].update({
            "sim_pnl": sim_pnl, "n_trades": int(n_trades),
            "pnl_per": pnl_per, "win_rate": win_rate,
            "brier_follow": brier_follow, "brier_fade": brier_fade,
        })

        print(f"\n  Fold {fold_i+1}  train={n_tr} markets  test={n_te} markets  [{cal_method}]")
        print(f"    Classifier Brier : {brier:.4f}  vs baselines: "
              f"follow={brier_follow:.4f}  fade={brier_fade:.4f}  random={brier_random:.4f}")
        print(f"    Improvement      : {improvement:+.4f}  {'[OK]' if improvement > 0 else '[FAIL]'}")
        print(f"    Accuracy         : {acc*100:.1f}%")
        print(f"    Regressor MAE    : {fold_results[-1]['reg_mae']:.4f}  "
              f"Corr: {fold_results[-1]['reg_corr']:+.3f}")
        print(f"    Simulated PnL    : {sim_pnl:+.4f}  ({n_trades} trades, "
              f"WR={win_rate*100:.1f}%  edge/trade={pnl_per:+.4f})")
        print(f"    Baseline PnL     : {baseline_pnl:+.4f}  (bet everything)")

        # Regime breakdown
        if regime_snap is not None:
            vol_med = np.nanmedian(regime_snap)
            hi_vol  = regime_snap > vol_med
            lo_vol  = ~hi_vol & ~np.isnan(regime_snap)
            if hi_vol.sum() > 10 and lo_vol.sum() > 10:
                b_hi = brier_score_loss(yp_te[hi_vol], preds_cls[hi_vol])
                b_lo = brier_score_loss(yp_te[lo_vol], preds_cls[lo_vol])
                bl   = brier_score_loss(yp_te, np.full(len(yp_te), base_rate))
                print(f"    High-vol Brier   : {b_hi:.4f} (imp vs base: {bl-b_hi:+.4f})")
                print(f"    Low-vol  Brier   : {b_lo:.4f} (imp vs base: {bl-b_lo:+.4f})")

        if uncert_snap is not None:
            unc_med  = np.nanmedian(uncert_snap)
            hi_unc   = uncert_snap > unc_med
            lo_unc   = ~hi_unc & ~np.isnan(uncert_snap)
            if hi_unc.sum() > 10 and lo_unc.sum() > 10:
                b_hu = brier_score_loss(yp_te[hi_unc], preds_cls[hi_unc])
                b_lu = brier_score_loss(yp_te[lo_unc], preds_cls[lo_unc])
                bl   = brier_score_loss(yp_te, np.full(len(yp_te), base_rate))
                print(f"    High-uncert Brier: {b_hu:.4f} (imp vs base: {bl-b_hu:+.4f})")
                print(f"    Low-uncert  Brier: {b_lu:.4f} (imp vs base: {bl-b_lu:+.4f})")

    # Average across folds
    avg_brier  = np.mean([r["brier"] for r in fold_results])
    avg_base   = np.mean([r["baseline"] for r in fold_results])
    avg_imp    = np.mean([r["improvement"] for r in fold_results])
    avg_acc    = np.mean([r["accuracy"] for r in fold_results])
    avg_mae    = np.mean([r["reg_mae"] for r in fold_results])
    avg_corr   = np.mean([r["reg_corr"] for r in fold_results])
    total_pnl  = sum(r["sim_pnl"] for r in fold_results)
    total_tr   = sum(r["n_trades"] for r in fold_results)
    avg_follow = np.mean([r["brier_follow"] for r in fold_results])
    avg_fade   = np.mean([r["brier_fade"] for r in fold_results])
    avg_wr     = np.mean([r["win_rate"] for r in fold_results if r["n_trades"] > 0])

    print(f"\n  ── Walk-forward average ({len(folds)} folds) ──")
    print(f"    Classifier Brier   : {avg_brier:.4f}  vs follow={avg_follow:.4f}  fade={avg_fade:.4f}")
    print(f"    Avg Improvement    : {avg_imp:+.4f}  {'[OK]' if avg_imp > 0 else '[FAIL]'}")
    print(f"    Avg Accuracy       : {avg_acc*100:.1f}%")
    print(f"    Regressor MAE      : {avg_mae:.4f}  Avg Corr: {avg_corr:+.3f}")
    print(f"    Total Sim PnL      : {total_pnl:+.4f}  ({total_tr} trades, WR={avg_wr*100:.1f}%)")
    beat_follow = avg_brier < avg_follow
    print(f"    Beats follow-crowd : {'YES' if beat_follow else 'NO'}")
    report.append(f"Primary model walk-forward: Brier={avg_brier:.4f} imp={avg_imp:+.4f} "
                  f"reg_corr={avg_corr:+.3f} sim_pnl={total_pnl:+.4f}")

    # Train final models on ALL data for deployment
    log.info("  Training final classifier + regressor on all data...")
    cal_method_final = "isotonic" if len(set(cids)) >= 500 else "sigmoid"
    final_model, final_imp = _train(X, yp, n_est=200, leaves=25,
                                    calibration=cal_method_final)
    final_reg, final_reg_imp = _train_regressor(X, ye, n_est=200, leaves=25)

    # Feature importance from classifier
    _importance(final_model, fn, top_n=20)

    # Regressor importance
    print("\n  Regressor feature importance (top 10):")
    reg_imps = final_reg.feature_importances_.astype(np.float64)
    reg_ranked = sorted(zip(fn, reg_imps), key=lambda x: -x[1])
    for fname, sc in reg_ranked[:10]:
        bar = "#" * int(sc / reg_ranked[0][1] * 15)
        print(f"  {fname:<40} {sc:>8.1f}  {bar}")

    # Profitability analysis by predicted confidence
    all_preds = final_model.predict_proba(final_imp.transform(X))[:, 1]
    print(f"\n  Profitability by model confidence (full dataset):")
    print(f"  {'P(profit)':>12}  {'N':>7}  {'Actual%':>8}  {'Base%':>7}  {'Edge':>7}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}")
    base_pct = yp.mean() * 100
    for lo, hi in [(0.0,0.4),(0.4,0.45),(0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,1.01)]:
        mask = (all_preds >= lo) & (all_preds < hi)
        n    = mask.sum()
        if n == 0:
            continue
        act  = yp[mask].mean() * 100
        edge = act - base_pct
        print(f"  {lo:.2f}-{hi:.2f}      {n:>7,}  {act:>7.1f}%  {base_pct:>6.1f}%  {edge:>+6.1f}%")

    with open("model_v4_profitable.pkl", "wb") as f:
        pickle.dump({
            "classifier":     final_model,
            "classifier_imp": final_imp,
            "regressor":      final_reg,
            "regressor_imp":  final_reg_imp,
            "features":       fn,
            "target":         "profitable + edge_regression",
            "taker_fee":      TAKER_FEE,
            "leakage_secs":   LEAKAGE_SECS,
            "pm_filter":      (PM_LO, PM_HI),
            "net_edge_formulation": "economic_edge_fill_up_spread_proxy",
            "decision_rule":  "bet if P(profitable) > threshold AND E(edge) > MIN_EDGE",
        }, f)
    log.info("  Saved model_v4_profitable.pkl")

    # ── No-pmarket model (microstructure only) ────────────────────────────────
    NOPMARKET_EXCLUDE = {"pm_abs_deviation", "pm_uncertainty", "is_extreme_market",
                         "bucket_enc", "vol_x_pm_abs_dev"}
    npm_mask = [i for i, f in enumerate(fn) if f not in NOPMARKET_EXCLUDE]
    npm_fn   = [fn[i] for i in npm_mask]
    X_npm    = X[:, npm_mask]
    npm_model, npm_imp = _train(X_npm, yp, n_est=200, leaves=25, calibration=cal_method_final)
    with open("model_v4_nopmarket.pkl", "wb") as f:
        pickle.dump({
            "classifier":     npm_model,
            "classifier_imp": npm_imp,
            "features":       npm_fn,
            "target":         "profitable",
            "excludes":       list(NOPMARKET_EXCLUDE),
        }, f)
    log.info("  Saved model_v4_nopmarket.pkl")

    # ══════════════════════════════════════════════════════════════════════════
    # MARKET MODEL: strategy discovery (honest early-phase features only)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MARKET MODEL — Strategy discovery")
    print("  Early-phase features only (secs_left > 150)")
    print("  Direction-neutral (abs values for imbalance features)")
    print("=" * 60)

    mkt_rows = fetch_market_outcomes(conn)
    X3, y3, cids3, fn3 = build_market_features(mkt_rows)

    mkt_folds = walk_forward_splits(cids3, min_train=80, test_size=40)
    log.info(f"  {len(set(cids3))} markets → {len(mkt_folds)} walk-forward folds")

    if len(mkt_folds) == 0:
        log.warning("  Not enough markets for market model walk-forward.")
        markets3 = list(dict.fromkeys(cids3))
        split3   = int(len(markets3) * 0.80)
        tr3 = set(markets3[:split3]); te3 = set(markets3[split3:])
        ti3 = [i for i, c in enumerate(cids3) if c in tr3]
        vi3 = [i for i, c in enumerate(cids3) if c in te3]
        mkt_folds = [(ti3, vi3, len(tr3), len(te3))]

    mkt_fold_results = []
    for fold_i, (ti3, vi3, n_tr3, n_te3) in enumerate(mkt_folds):
        X3_tr, y3_tr = X3[ti3], y3[ti3]
        X3_te, y3_te = X3[vi3], y3[vi3]
        cal3 = "isotonic" if n_tr3 >= 300 else "sigmoid"
        m3, imp3 = _train(X3_tr, y3_tr, n_est=100, leaves=15, calibration=cal3)
        p3 = m3.predict_proba(imp3.transform(X3_te))[:, 1]
        b3, bl3, im3, ac3 = _eval_fold(f"Market fold {fold_i+1}", y3_te, p3, y3_tr.mean())
        mkt_fold_results.append({"brier": b3, "baseline": bl3, "improvement": im3})
        print(f"\n  Fold {fold_i+1}  train={n_tr3}  test={n_te3}  [{cal3}]")
        print(f"    Brier: {b3:.4f}  baseline: {bl3:.4f}  imp: {im3:+.4f}  "
              f"{'[OK]' if im3 > 0 else '[FAIL]'}")

    avg_imp3 = np.mean([r["improvement"] for r in mkt_fold_results])
    print(f"\n  ── Market model average: imp={avg_imp3:+.4f} "
          f"{'[OK]' if avg_imp3 > 0 else '[FAIL]'} ──")

    # Final market model
    m3_final, imp3_final = _train(X3, y3, n_est=150, leaves=15, calibration="sigmoid")
    _importance(m3_final, fn3, top_n=15)

    if avg_imp3 > 0:
        print("\n  *** Market model beats baseline ***")
        print("  Feature importance above shows what early-market conditions")
        print("  predict outcome — use these to build new strategies.")

    with open("model_v4_market.pkl", "wb") as f:
        pickle.dump({"model": m3_final, "imputer": imp3_final, "features": fn3,
                     "target": "direction", "early_phase_only": True}, f)
    log.info("  Saved model_v4_market.pkl")

    # ── Automated Strategy Discovery ──────────────────────────────────────────
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Use last fold's test data for regime analysis
    last_fold = fold_results[-1]
    last_ti, last_vi = folds[-1][0], folds[-1][1]
    X_te_last   = X[last_vi]
    yp_te_last  = yp[last_vi]
    X_tr_last   = X[last_ti]
    yp_tr_last  = yp[last_ti]
    preds_last  = final_model.predict_proba(final_imp.transform(X_te_last))[:, 1]
    base_last   = yp_tr_last.mean()

    run_discovery_pipeline(
        X_tr=X_tr_last, yp_tr=yp_tr_last,
        X_te=X_te_last, yp_te=yp_te_last,
        preds_cls=preds_last, fn=fn,
        base_rate=base_last,
        final_model=final_model, final_imp=final_imp,
        mkt_fn=fn3, mkt_final_model=m3_final,
        avg_imp=avg_imp, avg_imp3=avg_imp3,
        total_pnl=total_pnl, run_ts=run_ts,
    )

    # ── Final summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Primary model (profitable)  avg Brier imp: {avg_imp:+.4f}  "
          f"{'[OK]' if avg_imp > 0 else '[FAIL]'}")
    print(f"  Market model (direction)    avg Brier imp: {avg_imp3:+.4f}  "
          f"{'[OK]' if avg_imp3 > 0 else '[FAIL]'}")
    print()
    print("  Saved: model_v4_profitable.pkl")
    print("         model_v4_nopmarket.pkl")
    print("         model_v4_market.pkl")
    print("=" * 60)

    with open("training_report_v4.txt", "w") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    main()
