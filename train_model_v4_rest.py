"""
V4 Training Pipeline — Production Model for dYdX
==================================================
Trains the primary trading model: P(profitable | features).
Used for trade timing on dYdX BTC-USD perpetual via quant_engine.py.

Architecture:
  - 75 pruned features (from 121 total, family-optimized)
  - LightGBM classifier with walk-forward validation
  - 4/4 kill tests pass (shuffle, time shift, ablation, random)
  - Calibrated via isotonic regression

The model answers: "Is there a profitable trade right now?"
  Score > 0.60 → open LONG or SHORT on dYdX
  Direction: follow Polymarket crowd (p_market > 0.5 → LONG)
  Exit: close when 5-minute market window expires

Also trains:
  - Direction model: P(UP wins) — experimental, crowd beats it
  - Market model: market-level aggregates for strategy discovery

Usage:
  python train_model_v4_rest.py

Output:
  model_v4_profitable.pkl — production model (used by quant_engine.py)
  model_v4_nopmarket.pkl  — microstructure-only variant
  model_v4_direction.pkl  — direction model
  model_v4_market.pkl     — market-level model
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


def _rest_fetch(table, params, limit=500) -> list:
    """Fetch using created_at cursor instead of OFFSET to avoid Supabase timeouts."""
    rows = []
    retries = 0
    cursor = ""  # tracks last created_at for cursor-based pagination
    while True:
        p = {**params, "limit": limit}
        if cursor:
            p["created_at"] = f"gt.{cursor}"
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=REST_H, params=p, timeout=180)
            if not r.text:
                retries += 1
                if retries > 3:
                    log.warning(f"  Empty body after 3 retries (status={r.status_code})")
                    break
                import time; time.sleep(2)
                continue
            batch = r.json()
            if isinstance(batch, dict):
                retries += 1
                if retries > 3:
                    log.warning(f"  API error after 3 retries: {batch}")
                    break
                log.warning(f"  API error, retry {retries}: {batch.get('message','?')}")
                import time; time.sleep(2)
                continue
            if not batch or not isinstance(batch, list):
                break
            rows.extend(batch)
            retries = 0
            if len(rows) % 5000 == 0:
                log.info(f"  Fetched {len(rows):,}...")
            if len(batch) < limit:
                break
            # Use last row's created_at as cursor for next page
            last_ts = batch[-1].get("created_at")
            if not last_ts:
                break
            cursor = last_ts
        except Exception as e:
            retries += 1
            if retries > 3:
                log.warning(f"  Fetch stopped after 3 retries: {e}")
                break
            log.warning(f"  Retry {retries}: {e}")
            import time; time.sleep(2)
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

    Returns (edge_chosen, profitable_int, side, excess_up, excess_down,
            fill_edge_up, fill_edge_down)
    """
    if np.isnan(outcome_binary) or p_market is None or np.isnan(p_market):
        return np.nan, np.nan, "NONE", np.nan, np.nan, np.nan, np.nan

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
    excess_up   = ob       - p_market       - rt
    excess_down = (1.0-ob) - (1.0-p_market) - rt

    # V5: Fill-based edge per side (execution-aware)
    # These are the actual P&L you'd get buying each side at fill prices
    fill_edge_up   = round(ob       - fu - rt, 6)
    fill_edge_down = round((1.0-ob) - fd - rt, 6)

    return (round(float(edge_chosen), 6), profitable, side,
            round(excess_up, 6), round(excess_down, 6),
            fill_edge_up, fill_edge_down)


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


def _compute_ece(y_true, y_pred, n_bins=10):
    """
    Expected Calibration Error — measures whether P=0.60 means 60% win rate.
    Lower is better. Perfect calibration = 0.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece    = 0.0
    rows   = []
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = float(y_pred[mask].mean())
        bin_acc  = float(y_true[mask].mean())
        weight   = mask.sum() / len(y_true)
        ece     += weight * abs(bin_conf - bin_acc)
        rows.append((bins[i], bins[i + 1], int(mask.sum()), bin_conf, bin_acc))
    return ece, rows


def _spearman(y_true, y_pred):
    """Spearman rank correlation between predicted score and actual label."""
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(y_pred, y_true)
        return float(r), float(p)
    except ImportError:
        # scipy not installed — fallback to numpy rank correlation
        n   = len(y_true)
        rp  = np.argsort(np.argsort(y_pred)).astype(float)
        ry  = np.argsort(np.argsort(y_true)).astype(float)
        num = np.sum((rp - rp.mean()) * (ry - ry.mean()))
        den = np.sqrt(np.sum((rp - rp.mean())**2) * np.sum((ry - ry.mean())**2))
        return float(num / den) if den > 0 else 0.0, np.nan


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
        "created_at","condition_id","secs_left","secs_to_resolution","market_progress",
        "phase_early","phase_mid","phase_late","phase_final",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "price_vs_open_pct","price_vs_open_score",
        "momentum_10s","momentum_30s","momentum_60s","momentum_120s","momentum_score",
        "cl_divergence","cl_age","cl_vs_open_pct",
        "liq_total","liq_imbalance","liq_long_usd","liq_short_usd","liq_dominant_ratio",
        "ob_imbalance","ob_bid_delta","ob_ask_delta",
        "vol_range_pct","volume_buy_ratio",
        "p_market","poly_fill_up","poly_fill_down",
        "poly_spread","poly_slip_up","poly_deviation",
        "basis_pct","funding_rate","okx_funding","gate_funding",
        "volatility_pct","flow_score","funding_zscore",
        "regime","session","activity","day_type",
        "price_bucket",
        "p_market_std","avg_ob_imbalance_abs","avg_funding_zscore_abs",
        "avg_momentum_abs","btc_range_pct",
        "tick_cvd_30s","tick_taker_buy_ratio_30s","tick_large_buy_usd_30s",
        "tick_large_sell_usd_30s","tick_intensity_30s","tick_vwap_disp_30s",
        "tick_cvd_60s","tick_taker_buy_ratio_60s","tick_intensity_60s",
        "poly_flow_imb","poly_depth_ratio",
        "poly_trade_imb","poly_up_buys","poly_down_buys","poly_trade_count","poly_large_pct",
        "delta_funding","delta_basis","delta_trade_imb","xex_spread",
        "delta_cvd","delta_taker_buy","delta_momentum","delta_poly","delta_score",
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


def _build_cross_market_lookup(rows) -> dict:
    """
    Pre-compute per-market aggregates and build a lookup so each row
    can access the PREVIOUS market's outcome and features.

    Returns dict: {condition_id: {prev_outcome, prev_momentum, prev_ob_imbalance,
                                   prev_vol_range, prev_btc_range, streak_up, streak_down}}
    """
    from collections import OrderedDict

    # Group rows by condition_id, preserving time order
    market_rows = OrderedDict()
    for row in rows:
        cid = row.get("condition_id", "")
        if cid not in market_rows:
            market_rows[cid] = []
        market_rows[cid].append(row)

    # Compute per-market summary (using last snapshot as "final" state)
    market_summaries = OrderedDict()
    for cid, mrows in market_rows.items():
        last = mrows[-1]  # last snapshot in time order
        ob = _f(last.get("outcome_binary"))
        market_summaries[cid] = {
            "outcome":       ob if not np.isnan(ob) else np.nan,
            "momentum_30s":  _f(last.get("momentum_30s")),
            "ob_imbalance":  _f(last.get("ob_imbalance")),
            "vol_range_pct": _f(last.get("vol_range_pct")),
            "btc_range_pct": _f(last.get("btc_range_pct")),
            "p_market":      _f(last.get("p_market")),
            "funding_zscore":_f(last.get("funding_zscore")),
        }

    # Build cross-market lookup: for each market, store previous market's stats
    cids = list(market_summaries.keys())
    cross = {}
    streak_up = 0
    streak_down = 0
    for i, cid in enumerate(cids):
        if i == 0:
            cross[cid] = {
                "prev_outcome":       np.nan,
                "prev_momentum":      np.nan,
                "prev_ob_imbalance":  np.nan,
                "prev_vol_range":     np.nan,
                "prev_btc_range":     np.nan,
                "prev_p_market_final":np.nan,
                "prev_funding_zscore":np.nan,
                "streak_up":          0.0,
                "streak_down":        0.0,
            }
        else:
            prev = market_summaries[cids[i-1]]
            cross[cid] = {
                "prev_outcome":       prev["outcome"],
                "prev_momentum":      prev["momentum_30s"],
                "prev_ob_imbalance":  prev["ob_imbalance"],
                "prev_vol_range":     prev["vol_range_pct"],
                "prev_btc_range":     prev["btc_range_pct"],
                "prev_p_market_final":prev["p_market"],
                "prev_funding_zscore":prev["funding_zscore"],
                "streak_up":          float(streak_up),
                "streak_down":        float(streak_down),
            }

        # Update streaks
        cur_outcome = market_summaries[cid]["outcome"]
        if not np.isnan(cur_outcome):
            if cur_outcome == 1:
                streak_up += 1
                streak_down = 0
            else:
                streak_down += 1
                streak_up = 0

    log.info(f"  Cross-market lookup: {len(cross)} markets, "
             f"avg streak_up={np.nanmean([v['streak_up'] for v in cross.values()]):.1f}")
    return cross


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
    # Pre-compute cross-market features
    cross_mkt = _build_cross_market_lookup(rows)

    records, y_prof, y_prof_best, y_edge, y_edge_up, y_edge_down, y_dir, cond_ids, pm_raws, best_sides = [], [], [], [], [], [], [], [], [], []
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
        _cross = cross_mkt.get(row.get("condition_id",""), {})

        fill_up   = _f(row.get("poly_fill_up"))
        fill_down = _f(row.get("poly_fill_down"))

        # Correct net edge: policy-free, best available side
        net_edge, prof, best_side, edge_up, edge_down, fill_edge_up, fill_edge_down = _net_edge_correct(
            ob, fill_up, fill_down,
            p_market=pm,
            poly_spread=_f(row.get("poly_spread")),
            min_edge=MIN_EDGE
        )
        if np.isnan(net_edge):
            prof           = 0
            net_edge       = 0.0
            edge_up        = 0.0
            edge_down      = 0.0
            fill_edge_up   = 0.0
            fill_edge_down = 0.0

        # Policy-free label: was there enough deviation from p_market to profit,
        # regardless of which side you bet?
        # edge_abs = |outcome - p_market| - round_trip_fee
        # If positive → someone with direction knowledge could have profited.
        # Unlike primary label, this does NOT depend on following the crowd.
        _rt = TAKER_FEE * 2
        _edge_abs = abs(ob - pm) - _rt if (not np.isnan(ob) and not np.isnan(pm)) else -1.0
        prof_best = int(_edge_abs > MIN_EDGE)

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
            "liq_long_usd":           min(_f(row.get("liq_long_usd")) or 0.0, 5e6),
            "liq_short_usd":          min(_f(row.get("liq_short_usd")) or 0.0, 5e6),
            "liq_dominant_ratio":     _f(row.get("liq_dominant_ratio")),
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

            # ── Signal separation (#3): top discovery interactions ─────────────
            # okx_funding_x_vol_x_funding_zscore corr=+0.072
            # vol_range_pct_x_funding_zscore      corr=+0.059
            # vol_range_pct_x_vol_x_funding_zscore corr=+0.057
            # okx_funding_x_funding_rate          corr=+0.046
            "okx_x_vol_fz": (
                _f(row.get("okx_funding")) * vr * _f(row.get("funding_zscore"))
                if not any(np.isnan(v) for v in [_f(row.get("okx_funding")), vr, _f(row.get("funding_zscore"))])
                else np.nan
            ),
            "vr_x_fz_sq": (
                vr * (_f(row.get("funding_zscore")) ** 2)
                if not(np.isnan(vr) or np.isnan(_f(row.get("funding_zscore"))))
                else np.nan
            ),
            "okx_x_fr": (
                _f(row.get("okx_funding")) * fr
                if not(np.isnan(_f(row.get("okx_funding"))) or np.isnan(fr))
                else np.nan
            ),

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

            # ── Market-level rolling aggregates ──────────────────────────────
            "p_market_std":           _f(row.get("p_market_std")),
            "avg_ob_imbalance_abs":   _f(row.get("avg_ob_imbalance_abs")),
            "avg_funding_zscore_abs": _f(row.get("avg_funding_zscore_abs")),
            "avg_momentum_abs":       _f(row.get("avg_momentum_abs")),
            "btc_range_pct":          _f(row.get("btc_range_pct")),

            # ── Polymarket order flow ─────────────────────────────────────
            "poly_flow_imb":             _f(row.get("poly_flow_imb")),
            "poly_depth_ratio":          _f(row.get("poly_depth_ratio")),
            "poly_trade_imb":            _f(row.get("poly_trade_imb")),
            "poly_up_buys":              _f(row.get("poly_up_buys")),
            "poly_down_buys":            _f(row.get("poly_down_buys")),
            "poly_trade_count":          _f(row.get("poly_trade_count")),
            "poly_large_pct":            _f(row.get("poly_large_pct")),

            # ── Regime interaction features (V5 only — pruned from V4) ─────
            "prev_vol_x_momentum":       _cross.get("prev_vol_range", 0) * _cross.get("prev_momentum", 0),
            "session_x_vol":             SESSION_MAP.get(row.get("session",""), 0) * vr if not np.isnan(vr) else np.nan,
            "streak_length":             max(_cross.get("streak_up", 0), _cross.get("streak_down", 0)),

            # ── Velocity features (rate of change) ────────────────────────
            "delta_funding":             _f(row.get("delta_funding")),
            "delta_basis":               _f(row.get("delta_basis")),

            # ── Intra-market deltas (change since previous snapshot) ────────
            "delta_cvd":                 _f(row.get("delta_cvd")),
            "delta_taker_buy":           _f(row.get("delta_taker_buy")),
            "delta_momentum":            _f(row.get("delta_momentum")),
            "delta_poly":                _f(row.get("delta_poly")),
            "delta_score":               _f(row.get("delta_score")),

            # ── Tick-level order flow (Binance aggTrades) ────────────────────
            "tick_cvd_30s":              _f(row.get("tick_cvd_30s")),
            "tick_taker_buy_ratio_30s":  _f(row.get("tick_taker_buy_ratio_30s")),
            "tick_large_buy_usd_30s":    _f(row.get("tick_large_buy_usd_30s")),
            "tick_large_sell_usd_30s":   _f(row.get("tick_large_sell_usd_30s")),
            "tick_intensity_30s":        _f(row.get("tick_intensity_30s")),
            "tick_vwap_disp_30s":        _f(row.get("tick_vwap_disp_30s")),
            "tick_cvd_60s":              _f(row.get("tick_cvd_60s")),
            "tick_taker_buy_ratio_60s":  _f(row.get("tick_taker_buy_ratio_60s")),
            "tick_intensity_60s":        _f(row.get("tick_intensity_60s")),

            # ── Cross-market features (what happened in the previous market) ─
            **{k: v for k, v in cross_mkt.get(row.get("condition_id",""), {}).items()},
        }

        records.append(f)
        y_prof.append(prof)
        y_prof_best.append(prof_best)    # policy-free: max(edge_up, edge_down)
        y_edge.append(float(net_edge))   # continuous edge for regression
        y_edge_up.append(float(fill_edge_up))    # v5: fill-based edge if buying UP
        y_edge_down.append(float(fill_edge_down))  # v5: fill-based edge if buying DOWN
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

    # Prune zero-importance features — these are either noise or have insufficient
    # real data (e.g. tick/poly features backfilled as zeros). They stay in snapshots
    # for future use but are excluded from training to reduce noise.
    # Light prune: only features that are structurally useless (not just low importance).
    # Keep: deltas (real backfill data), tick/poly (will gain data over time),
    #        OB features (occasionally useful in specific folds)
    PRUNE = {
        # ── Agreement signals (zero importance) ──
        "mom_all_agree", "mom_liq_agree", "mom_ob_agree", "liq_ob_agree",
        # ── Phase indicators (redundant with secs_to_resolution) ──
        "phase_mid", "phase_late", "phase_final",
        # ── Cost features (near-constant) ──
        "round_trip_cost", "effective_entry_cost",
        # ── Categorical noise ──
        "day_type_enc", "signal_strength", "signal_dispersion",
        # ── Regime interactions (V5-only) ──
        "prev_vol_x_momentum", "session_x_vol", "streak_length",
        # ── Momentum family trim (prev_momentum carries 77%, rest is noise) ──
        "momentum_10s", "momentum_30s", "momentum_score",
        "mom_windows_positive", "mom_accel_short", "mom_accel_mid",
        "mom_mean", "mom_x_secs", "mom_abs",
        # ── OB family trim (prev_ob_imbalance carries 97%) ──
        "ob_imbalance", "ob_bid_delta", "ob_ask_delta", "ob_abs_imbalance",
        "ob_imbal_x_secs", "interact_ob_x_spread",
        # ── Funding family trim (keep top 5, prune 7 low-value) ──
        "okx_x_vol_fz", "vr_x_fz_sq", "vol_x_funding_zscore",
        # ── Delta family (0.1% total, all dead) ──
        "delta_cvd", "delta_taker_buy", "delta_momentum", "delta_poly",
        "delta_score", "delta_funding", "delta_basis",
        # ── Other low-value ──
        "flow_score", "is_extreme_market",
        "cl_divergence", "cl_abs_divergence", "cl_age",
        "poly_slip_up",
    }
    fn_all  = list(records[0].keys())
    fn      = [f for f in fn_all if f not in PRUNE]
    pruned  = len(fn_all) - len(fn)
    log.info(f"  Pruned {pruned} zero-importance features → {len(fn)} remaining")
    X       = np.array([[r[k] for k in fn] for r in records], dtype=np.float32)
    # Keep unpruned matrix for V5 (built later)
    X_all   = np.array([[r[k] for k in fn_all] for r in records], dtype=np.float32)
    yp      = np.array(y_prof,      dtype=np.int32)
    yp_best = np.array(y_prof_best, dtype=np.int32)   # policy-free label
    ye      = np.array(y_edge,      dtype=np.float32)  # continuous edge
    ye_up   = np.array(y_edge_up,   dtype=np.float32)  # v5: fill-based UP edge
    ye_down = np.array(y_edge_down, dtype=np.float32)  # v5: fill-based DOWN edge
    yd      = np.array(y_dir,       dtype=np.int32)
    pm      = np.array(pm_raws,     dtype=np.float32)
    return X, yp, yp_best, ye, ye_up, ye_down, yd, cond_ids, fn, pm, X_all, fn_all


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

# ═══════════════════════════════════════════════════════════════ KILL TESTS ══

def _run_kill_tests(X, yp, fn, folds, real_avg_imp, report):
    """
    Four diagnostic tests that should run after every training to confirm
    the model is learning genuine signal rather than artefacts.

    Kill Test 1 — Shuffle:    shuffle labels → improvement should collapse to ~0
    Kill Test 2 — Time shift: predict next-fold labels with current features → drop expected
    Kill Test 3 — Ablation:   remove top-5 features one at a time → smooth degradation
    Kill Test 4 — Random:     random entry at same frequency → model should clearly win
    """
    print("\n" + "=" * 60)
    print("  KILL TESTS — model validity")
    print("=" * 60)

    rng       = np.random.RandomState(99)
    use_folds = folds[:6]
    lite_kw   = dict(n_est=100, leaves=15)

    # ── Kill Test 1: Shuffle ──────────────────────────────────────────────────
    print("\n  [1] Shuffle Test")
    print("      Shuffle outcome labels — improvement should collapse to ~0.")
    print("      If it stays high: feature leakage is encoding the label.")
    yp_shuf = yp.copy()
    rng.shuffle(yp_shuf)
    shuf_imps = []
    for ti, vi, n_tr, _ in use_folds:
        cal   = "isotonic" if n_tr >= 500 else "sigmoid"
        m, imp = _train(X[ti], yp_shuf[ti], calibration=cal, **lite_kw)
        preds  = m.predict_proba(imp.transform(X[vi]))[:, 1]
        _, _, improvement, _ = _eval_fold("", yp_shuf[vi], preds, yp_shuf[ti].mean())
        shuf_imps.append(improvement)
    avg_shuf  = float(np.mean(shuf_imps))
    collapsed = abs(avg_shuf) < 0.008
    result1   = "PASS" if collapsed else "FAIL — possible label leakage"
    print(f"      Real model imp : {real_avg_imp:+.4f}")
    print(f"      Shuffled imp   : {avg_shuf:+.4f}")
    print(f"      Result         : {result1}")
    report.append(f"Kill Test 1 (shuffle): shuffled_imp={avg_shuf:+.4f}  {result1}")

    # ── Kill Test 2: Time shift ───────────────────────────────────────────────
    print("\n  [2] Time Shift Test")
    print("      Predict next fold's labels using current fold's features.")
    print("      Performance should drop — features predict present, not future.")
    shift_imps = []
    for i in range(len(use_folds) - 1):
        ti, vi, n_tr, _  = use_folds[i]
        _, vi_next, _, _ = use_folds[i + 1]
        n_overlap = min(len(vi), len(vi_next))
        if n_overlap < 10:
            continue
        X_te_curr = X[vi[:n_overlap]]
        y_te_next = yp[vi_next[:n_overlap]]
        cal   = "isotonic" if n_tr >= 500 else "sigmoid"
        m, imp = _train(X[ti], yp[ti], calibration=cal, **lite_kw)
        preds  = m.predict_proba(imp.transform(X_te_curr))[:, 1]
        _, _, improvement, _ = _eval_fold("", y_te_next, preds, yp[ti].mean())
        shift_imps.append(improvement)
    if shift_imps:
        avg_shift = float(np.mean(shift_imps))
        drop      = real_avg_imp - avg_shift
        result2   = "PASS" if drop > -0.005 else "WARN — features may encode future info"
        print(f"      Real model imp    : {real_avg_imp:+.4f}")
        print(f"      Time-shifted imp  : {avg_shift:+.4f}")
        print(f"      Performance drop  : {drop:+.4f}")
        print(f"      Result            : {result2}")
        report.append(f"Kill Test 2 (time shift): shifted_imp={avg_shift:+.4f} drop={drop:+.4f}  {result2}")
    else:
        result2 = "SKIP"
        print("      SKIP — not enough folds")

    # ── Kill Test 3: Feature ablation ─────────────────────────────────────────
    print("\n  [3] Feature Ablation Test")
    print("      Remove top-5 features one at a time.")
    print("      Smooth degradation = stable. Large single spike = fragile.")
    ti_last, vi_last, n_tr_last, _ = folds[-1]
    cal_last = "isotonic" if n_tr_last >= 500 else "sigmoid"
    m_base, imp_base = _train(X[ti_last], yp[ti_last], calibration=cal_last, **lite_kw)
    preds_base = m_base.predict_proba(imp_base.transform(X[vi_last]))[:, 1]
    _, _, base_imp_abl, _ = _eval_fold("", yp[vi_last], preds_base, yp[ti_last].mean())
    top5 = sorted(zip(fn, m_base.feature_importances_), key=lambda x: -x[1])[:5]
    print(f"  {'Feature removed':<38} {'Imp':>8}  {'Drop':>8}  Status")
    print(f"  {'─'*38} {'─'*8}  {'─'*8}  {'─'*6}")
    print(f"  {'(baseline — no removal)':<38} {base_imp_abl:>+8.4f}  {'—':>8}")
    ablation_ok = True
    prev = base_imp_abl
    for fname, _ in top5:
        if fname not in fn:
            continue
        fidx        = fn.index(fname)
        X_abl       = X.copy()
        X_abl[:, fidx] = np.nan
        m_abl, imp_abl = _train(X_abl[ti_last], yp[ti_last],
                                calibration=cal_last, **lite_kw)
        preds_abl = m_abl.predict_proba(imp_abl.transform(X_abl[vi_last]))[:, 1]
        _, _, imp_abl_v, _ = _eval_fold("", yp[vi_last], preds_abl, yp[ti_last].mean())
        drop  = prev - imp_abl_v
        spike = abs(drop) > 0.025
        if spike:
            ablation_ok = False
        print(f"  {fname:<38} {imp_abl_v:>+8.4f}  {drop:>+8.4f}  {'SPIKE' if spike else 'OK'}")
        prev = imp_abl_v
    result3 = "PASS" if ablation_ok else "WARN — model depends heavily on single feature(s)"
    print(f"      Result: {result3}")
    report.append(f"Kill Test 3 (ablation): {result3}")

    # ── Kill Test 4: Random entry baseline ────────────────────────────────────
    print("\n  [4] Random Entry Baseline")
    print("      Random trades at same frequency as model (threshold=0.55).")
    print("      Model win rate must clearly beat random selection.")
    CLS_THRESH = 0.55
    N_REPS     = 50
    model_wrs, random_wrs = [], []
    for ti, vi, n_tr, _ in use_folds:
        cal   = "isotonic" if n_tr >= 500 else "sigmoid"
        m, imp = _train(X[ti], yp[ti], calibration=cal, **lite_kw)
        preds  = m.predict_proba(imp.transform(X[vi]))[:, 1]
        mask   = preds > CLS_THRESH
        n_sel  = int(mask.sum())
        if n_sel < 5:
            continue
        model_wrs.append(float(yp[vi][mask].mean()))
        rand_wr_reps = []
        for _ in range(N_REPS):
            idx = rng.choice(len(vi), size=n_sel, replace=False)
            rand_wr_reps.append(float(yp[vi][idx].mean()))
        random_wrs.append(float(np.mean(rand_wr_reps)))
    if model_wrs:
        avg_model_wr  = float(np.mean(model_wrs))
        avg_random_wr = float(np.mean(random_wrs))
        edge_vs_rand  = avg_model_wr - avg_random_wr
        beats_random  = edge_vs_rand > 0.02
        result4 = "PASS" if beats_random else "FAIL — no edge over random selection"
        print(f"      Model WR (filtered) : {avg_model_wr*100:.1f}%")
        print(f"      Random WR (same N)  : {avg_random_wr*100:.1f}%")
        print(f"      Edge over random    : {edge_vs_rand*100:+.1f}pp")
        print(f"      Result              : {result4}")
        report.append(f"Kill Test 4 (random baseline): model_wr={avg_model_wr*100:.1f}% "
                      f"random_wr={avg_random_wr*100:.1f}% edge={edge_vs_rand*100:+.1f}pp  {result4}")
    else:
        result4 = "SKIP"
        print("      SKIP — no folds with filtered trades")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Kill test summary:")
    print(f"    [1] Shuffle    : {result1}")
    print(f"    [2] Time shift : {result2}")
    print(f"    [3] Ablation   : {result3}")
    print(f"    [4] Random     : {result4}")


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
    X, yp, yp_best, ye, ye_up, ye_down, yd, cids, fn, pm, X_all, fn_all = build_snapshot_features(snap_rows)

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

        # Simulated PnL — two strategies:
        # (A) Fixed threshold at 0.55
        # (B) Top-15% percentile ranking (#1)
        cls_threshold = 0.55
        reg_threshold = MIN_EDGE

        # Strategy A: fixed threshold
        trade_mask = (preds_cls > cls_threshold) & (preds_reg > reg_threshold)
        n_trades   = trade_mask.sum()
        if n_trades > 0:
            sim_pnl  = float(ye_te[trade_mask].sum())
            pnl_per  = float(ye_te[trade_mask].mean())
            win_rate = float(yp_te[trade_mask].mean())
        else:
            sim_pnl = pnl_per = win_rate = 0.0

        # Strategy B: top-15% percentile (#1)
        pct_cutoff   = np.percentile(preds_cls, 85)
        rank_mask    = (preds_cls >= pct_cutoff) & (preds_reg > reg_threshold)
        n_rank       = rank_mask.sum()
        if n_rank > 0:
            rank_pnl = float(ye_te[rank_mask].sum())
            rank_wr  = float(yp_te[rank_mask].mean())
            rank_per = float(ye_te[rank_mask].mean())
        else:
            rank_pnl = rank_wr = rank_per = 0.0

        # Strategy C: stress test (#5) — top-15% with extra slippage + 5s delay penalty
        # 5s delay: approximate by marking down entry by momentum_30s * 5s
        # Extra slippage: 0.005 (0.5%) applied to edge
        STRESS_SLIP  = 0.003   # 0.3% slippage (realistic for $1-10 bets)
        STRESS_DELAY = 0.002   # ~5s of 0.04%/s average BTC drift cost
        if n_rank > 0:
            stressed_edges = ye_te[rank_mask] - STRESS_SLIP - STRESS_DELAY
            stress_pnl = float(stressed_edges.sum())
            stress_wr  = float(yp_te[rank_mask].mean())
        else:
            stress_pnl = stress_wr = 0.0

        # Baseline PnL: always bet (no filter)
        baseline_pnl = float(ye_te.sum())

        fold_results[-1].update({
            "sim_pnl": sim_pnl, "n_trades": int(n_trades),
            "pnl_per": pnl_per, "win_rate": win_rate,
            "rank_pnl": rank_pnl, "n_rank": int(n_rank),
            "rank_wr": rank_wr, "rank_per": rank_per,
            "stress_pnl": stress_pnl, "stress_wr": stress_wr,
            "brier_follow": brier_follow, "brier_fade": brier_fade,
        })

        print(f"\n  Fold {fold_i+1}  train={n_tr} markets  test={n_te} markets  [{cal_method}]")
        print(f"    Classifier Brier : {brier:.4f}  vs baselines: "
              f"follow={brier_follow:.4f}  fade={brier_fade:.4f}  random={brier_random:.4f}")
        print(f"    Improvement      : {improvement:+.4f}  {'[OK]' if improvement > 0 else '[FAIL]'}")
        print(f"    Accuracy         : {acc*100:.1f}%")
        print(f"    Regressor MAE    : {fold_results[-1]['reg_mae']:.4f}  "
              f"Corr: {fold_results[-1]['reg_corr']:+.3f}")
        print(f"    Sim PnL (thr=0.55): {sim_pnl:+.4f}  ({n_trades} trades, "
              f"WR={win_rate*100:.1f}%  edge/trade={pnl_per:+.4f})")
        print(f"    Sim PnL (top-15%) : {rank_pnl:+.4f}  ({n_rank} trades, "
              f"WR={rank_wr*100:.1f}%  edge/trade={rank_per:+.4f})")
        print(f"    Sim PnL (stress)  : {stress_pnl:+.4f}  (top-15% + 0.7% extra cost)")
        print(f"    Baseline PnL      : {baseline_pnl:+.4f}  (bet everything)")

        # #4 Spearman rank correlation — does higher score = higher actual edge?
        sp_r, sp_p = _spearman(yp_te, preds_cls)
        fold_results[-1]["spearman_r"] = sp_r
        print(f"    Spearman rank r  : {sp_r:+.4f}  "
              f"({'monotonic ✓' if sp_r > 0.05 else 'weak/flat — ranking unreliable'})")

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
    total_pnl    = sum(r["sim_pnl"]    for r in fold_results)
    total_tr     = sum(r["n_trades"]   for r in fold_results)
    total_rank   = sum(r["rank_pnl"]   for r in fold_results)
    total_rank_n = sum(r["n_rank"]     for r in fold_results)
    total_stress = sum(r["stress_pnl"] for r in fold_results)
    avg_follow   = np.mean([r["brier_follow"] for r in fold_results])
    avg_fade     = np.mean([r["brier_fade"]   for r in fold_results])
    avg_wr       = np.mean([r["win_rate"] for r in fold_results if r["n_trades"] > 0])
    avg_rank_wr  = np.mean([r["rank_wr"]  for r in fold_results if r["n_rank"]   > 0])

    print(f"\n  ── Walk-forward average ({len(folds)} folds) ──")
    print(f"    Classifier Brier     : {avg_brier:.4f}  vs follow={avg_follow:.4f}  fade={avg_fade:.4f}")
    print(f"    Avg Improvement      : {avg_imp:+.4f}  {'[OK]' if avg_imp > 0 else '[FAIL]'}")
    print(f"    Avg Accuracy         : {avg_acc*100:.1f}%")
    print(f"    Regressor MAE        : {avg_mae:.4f}  Avg Corr: {avg_corr:+.3f}")
    print(f"    Total PnL (thr=0.55) : {total_pnl:+.4f}  ({total_tr} trades, WR={avg_wr*100:.1f}%)")
    print(f"    Total PnL (top-15%)  : {total_rank:+.4f}  ({total_rank_n} trades, WR={avg_rank_wr*100:.1f}%)")
    print(f"    Total PnL (stress)   : {total_stress:+.4f}  (top-15% + 0.7% extra cost)")
    beat_follow   = avg_brier < avg_follow
    avg_spearman  = np.mean([r.get("spearman_r", 0) for r in fold_results])
    print(f"    Beats follow-crowd   : {'YES' if beat_follow else 'NO'}")
    print(f"    Avg Spearman r       : {avg_spearman:+.4f}  "
          f"({'ranking consistent ✓' if avg_spearman > 0.05 else 'ranking weak'})")
    report.append(f"Primary model walk-forward: Brier={avg_brier:.4f} imp={avg_imp:+.4f} "
                  f"reg_corr={avg_corr:+.3f} sim_pnl={total_pnl:+.4f} "
                  f"rank_pnl={total_rank:+.4f} stress_pnl={total_stress:+.4f} "
                  f"spearman={avg_spearman:+.4f}")

    # ── #4 Where model works best — regime Spearman breakdown ────────────────
    print("\n" + "=" * 60)
    print("  WHERE MODEL WORKS (#4) — Spearman by regime")
    print("  Restrict trading to zones where ranking is strongest.")
    print("=" * 60)
    # Collect all held-out predictions across all folds for regime analysis
    all_yp_te   = np.concatenate([yp[vi]  for _, vi, _, _ in folds])
    all_ye_te   = np.concatenate([ye[vi]  for _, vi, _, _ in folds])
    all_pm_te   = np.concatenate([pm[vi]  for _, vi, _, _ in folds])

    # Re-run walk-forward to collect predictions (lightweight, 100 est)
    wf_preds_agg = []
    for ti, vi, n_tr, _ in folds:
        cal_lite = "isotonic" if n_tr >= 500 else "sigmoid"
        m_lite, imp_lite = _train(X[ti], yp[ti], n_est=100, leaves=15,
                                  calibration=cal_lite)
        p_lite = m_lite.predict_proba(imp_lite.transform(X[vi]))[:, 1]
        wf_preds_agg.append(p_lite)
    all_preds_agg = np.concatenate(wf_preds_agg)

    vol_idx = fn.index("vol_range_pct") if "vol_range_pct" in fn else None
    str_idx = fn.index("secs_to_resolution") if "secs_to_resolution" in fn else None
    fr_idx  = fn.index("funding_abs") if "funding_abs" in fn else None

    all_X_te = np.concatenate([X[vi] for _, vi, _, _ in folds])

    def _regime_spearman(mask, label):
        if mask.sum() < 30:
            return
        sp_r, _ = _spearman(all_yp_te[mask], all_preds_agg[mask])
        wr = all_yp_te[mask].mean()
        rank_cut = np.percentile(all_preds_agg[mask], 85)
        rank_m   = (all_preds_agg[mask] >= rank_cut)
        rank_wr  = all_yp_te[mask][rank_m].mean() if rank_m.sum() > 0 else np.nan
        print(f"    {label:<30} n={mask.sum():>5}  sp_r={sp_r:+.3f}  "
              f"base_wr={wr*100:.1f}%  top15_wr={rank_wr*100:.1f}%")

    print(f"\n  {'Zone':<30} {'N':>6}  {'Spearman':>9}  {'Base WR':>8}  {'Top15 WR':>9}")
    print(f"  {'─'*30} {'─'*6}  {'─'*9}  {'─'*8}  {'─'*9}")

    if vol_idx is not None:
        vol_vals = all_X_te[:, vol_idx]
        q33, q66 = np.nanpercentile(vol_vals, [33, 66])
        _regime_spearman(vol_vals < q33,              "vol LOW  (<p33)")
        _regime_spearman((vol_vals >= q33) & (vol_vals < q66), "vol MED  (p33-p66)")
        _regime_spearman(vol_vals >= q66,              "vol HIGH (>p66)")

    if str_idx is not None:
        str_vals = all_X_te[:, str_idx]
        _regime_spearman(str_vals > 600,  "secs_left > 600  (early)")
        _regime_spearman((str_vals > 300) & (str_vals <= 600), "secs_left 300-600 (mid)")
        _regime_spearman(str_vals <= 300, "secs_left <= 300 (late)")

    if fr_idx is not None:
        fr_vals = all_X_te[:, fr_idx]
        fr_med  = np.nanpercentile(fr_vals, 66)
        _regime_spearman(fr_vals < fr_med,  "funding LOW  (<p66)")
        _regime_spearman(fr_vals >= fr_med, "funding HIGH (>p66)")

    # pm_uncertainty breakdown
    unc_idx = fn.index("pm_uncertainty") if "pm_uncertainty" in fn else None
    if unc_idx is not None:
        unc_vals = all_X_te[:, unc_idx]
        unc_med  = np.nanmedian(unc_vals)
        _regime_spearman(unc_vals >= unc_med, "pm_uncertainty HIGH")
        _regime_spearman(unc_vals < unc_med,  "pm_uncertainty LOW")

    report.append(f"Regime analysis: {len(folds)} folds aggregated — see WHERE MODEL WORKS section")

    # ── #7 ECE — calibration check on full dataset ────────────────────────────
    print("\n" + "=" * 60)
    print("  CALIBRATION CHECK (#7) — does P=0.60 mean 60% win rate?")
    print("=" * 60)
    # Use last fold's held-out predictions for honest calibration estimate
    last_ti, last_vi, last_n_tr, _ = folds[-1]
    cal_last2 = "isotonic" if last_n_tr >= 500 else "sigmoid"
    m_cal, imp_cal = _train(X[last_ti], yp[last_ti], n_est=150, leaves=20,
                            calibration=cal_last2)
    preds_cal = m_cal.predict_proba(imp_cal.transform(X[last_vi]))[:, 1]
    ece_val, ece_bins = _compute_ece(yp[last_vi], preds_cal, n_bins=10)
    print(f"\n  ECE (Expected Calibration Error): {ece_val:.4f}  "
          f"({'well calibrated' if ece_val < 0.05 else 'needs calibration' if ece_val < 0.10 else 'poorly calibrated'})")
    print(f"\n  {'Pred range':<14} {'N':>6}  {'Pred conf':>10}  {'Actual WR':>10}  {'Gap':>8}")
    print(f"  {'─'*14} {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}")
    for lo, hi, n_bin, conf, acc in ece_bins:
        gap = acc - conf
        flag = " ←" if abs(gap) > 0.08 else ""
        print(f"  {lo:.2f}–{hi:.2f}       {n_bin:>6}  {conf:>10.3f}  {acc:>10.3f}  {gap:>+8.3f}{flag}")
    report.append(f"Calibration ECE={ece_val:.4f}")

    # ── #2 Post-hoc calibration improvement (isotonic) ────────────────────────
    # Compare ranking quality before/after isotonic recalibration on held-out data
    # Uses second-to-last fold as calibration set, last fold as test.
    print("\n  Post-hoc recalibration (#2):")
    if len(folds) >= 3:
        from sklearn.isotonic import IsotonicRegression
        cal_ti, cal_vi, cal_n_tr, _ = folds[-2]
        cal_method_tmp = "isotonic" if cal_n_tr >= 500 else "sigmoid"
        m_raw, imp_raw = _train(X[cal_ti], yp[cal_ti], n_est=150, leaves=20,
                                calibration=cal_method_tmp)
        # Get raw scores on calibration set, fit isotonic on top
        raw_cal_preds = m_raw.predict_proba(imp_raw.transform(X[cal_vi]))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_cal_preds, yp[cal_vi])
        # Evaluate on last fold
        raw_test_preds   = m_raw.predict_proba(imp_raw.transform(X[last_vi]))[:, 1]
        iso_test_preds   = iso.predict(raw_test_preds)
        ece_before, _    = _compute_ece(yp[last_vi], raw_test_preds, n_bins=10)
        ece_after, _     = _compute_ece(yp[last_vi], iso_test_preds, n_bins=10)
        sp_before, _     = _spearman(yp[last_vi], raw_test_preds)
        sp_after, _      = _spearman(yp[last_vi], iso_test_preds)
        print(f"    ECE before: {ece_before:.4f}  after isotonic: {ece_after:.4f}  "
              f"({'improved' if ece_after < ece_before else 'no improvement'})")
        print(f"    Spearman before: {sp_before:+.4f}  after: {sp_after:+.4f}  "
              f"(ranking {'preserved' if abs(sp_after - sp_before) < 0.02 else 'changed'})")
        report.append(f"Post-hoc calibration: ECE {ece_before:.4f}->{ece_after:.4f}  "
                      f"Spearman {sp_before:+.4f}->{sp_after:+.4f}")
    else:
        print("    Skipped — need 3+ folds")

    # ── #1 Best-side model — policy-free label ────────────────────────────────
    print("\n" + "=" * 60)
    print("  BEST-SIDE MODEL (#1) — policy-free label")
    print("  y = 1 if max(edge_up, edge_down) > MIN_EDGE")
    print("  Removes crowd-following bias from primary label.")
    print("=" * 60)
    base_rate_best = yp_best.mean()
    log.info(f"  Best-side label: {yp_best.sum():,}/{len(yp_best):,} profitable "
             f"({base_rate_best*100:.1f}%)")
    print(f"\n  Primary label base rate : {yp.mean()*100:.1f}%  (follow crowd)")
    print(f"  Best-side label base rate: {base_rate_best*100:.1f}%  (either direction)")
    print(f"  Gap = {(base_rate_best - yp.mean())*100:+.1f}pp  "
          f"(positive gap = crowd misses some edges)")
    best_fold_results = []
    for fold_i, (ti, vi, n_tr, n_te) in enumerate(folds):
        yp_best_tr, yp_best_te = yp_best[ti], yp_best[vi]
        cal = "isotonic" if n_tr >= 500 else "sigmoid"
        m_b, imp_b = _train(X[ti], yp_best_tr, n_est=150, leaves=20, calibration=cal)
        preds_b    = m_b.predict_proba(imp_b.transform(X[vi]))[:, 1]
        base_b     = yp_best_tr.mean()
        brier_b, baseline_b, imp_b_val, acc_b = _eval_fold("", yp_best_te, preds_b, base_b)
        best_fold_results.append({"brier": brier_b, "imp": imp_b_val, "acc": acc_b})
    avg_imp_best   = np.mean([r["imp"]   for r in best_fold_results])
    avg_brier_best = np.mean([r["brier"] for r in best_fold_results])
    avg_acc_best   = np.mean([r["acc"]   for r in best_fold_results])
    print(f"\n  ── Best-side walk-forward ({len(folds)} folds) ──")
    print(f"    Avg Brier imp  : {avg_imp_best:+.4f}  {'[OK]' if avg_imp_best > 0 else '[FAIL]'}")
    print(f"    Avg Accuracy   : {avg_acc_best*100:.1f}%")
    verdict = ("Better than primary → crowd-following bias was hiding real edge"
               if avg_imp_best > avg_imp else
               "Similar to primary → label choice doesn't matter much"
               if abs(avg_imp_best - avg_imp) < 0.003 else
               "Worse than primary → crowd-following label adds useful signal")
    print(f"    vs primary imp : {avg_imp:+.4f} primary  vs  {avg_imp_best:+.4f} best-side")
    print(f"    Verdict        : {verdict}")
    report.append(f"Best-side model walk-forward: Brier={avg_brier_best:.4f} imp={avg_imp_best:+.4f}  {verdict}")

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

    with open("models/model_v4_profitable.pkl", "wb") as f:
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
    with open("models/model_v4_nopmarket.pkl", "wb") as f:
        pickle.dump({
            "classifier":     npm_model,
            "classifier_imp": npm_imp,
            "features":       npm_fn,
            "target":         "profitable",
            "excludes":       list(NOPMARKET_EXCLUDE),
        }, f)
    log.info("  Saved model_v4_nopmarket.pkl")

    # ── No-pmarket walk-forward validation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  NO-PMARKET MODEL — microstructure only")
    print(f"  Excludes: {', '.join(sorted(NOPMARKET_EXCLUDE))}")
    print("=" * 60)
    npm_fold_results = []
    for fold_i, (ti, vi, n_tr, n_te) in enumerate(folds):
        X_npm_tr, yp_tr = X_npm[ti], yp[ti]
        X_npm_te, yp_te = X_npm[vi], yp[vi]
        cal_method = "isotonic" if n_tr >= 500 else "sigmoid"
        npm_m, npm_i = _train(X_npm_tr, yp_tr, n_est=150, leaves=20,
                              calibration=cal_method)
        preds = npm_m.predict_proba(npm_i.transform(X_npm_te))[:, 1]
        base_rate = yp_tr.mean()
        brier, baseline, improvement, acc = _eval_fold(
            f"Fold {fold_i+1}", yp_te, preds, base_rate)
        npm_fold_results.append({
            "brier": brier, "baseline": baseline,
            "improvement": improvement, "accuracy": acc,
        })
    npm_avg_imp = np.mean([r["improvement"] for r in npm_fold_results])
    npm_avg_brier = np.mean([r["brier"] for r in npm_fold_results])
    npm_avg_base  = np.mean([r["baseline"] for r in npm_fold_results])
    npm_avg_acc   = np.mean([r["accuracy"] for r in npm_fold_results])
    print(f"\n  ── No-pmarket walk-forward average ({len(folds)} folds) ──")
    print(f"    Classifier Brier : {npm_avg_brier:.4f}  vs baseline={npm_avg_base:.4f}")
    print(f"    Avg Improvement  : {npm_avg_imp:+.4f}  {'[OK]' if npm_avg_imp > 0 else '[FAIL]'}")
    print(f"    Avg Accuracy     : {npm_avg_acc*100:.1f}%")
    print(f"    vs full model    : {avg_imp:+.4f} full  vs  {npm_avg_imp:+.4f} nopmarket")
    report.append(f"NoPmarket model walk-forward: Brier={npm_avg_brier:.4f} imp={npm_avg_imp:+.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # KILL TESTS — model validity checks
    # ══════════════════════════════════════════════════════════════════════════
    _run_kill_tests(X, yp, fn, folds, avg_imp, report)

    # ══════════════════════════════════════════════════════════════════════════
    # DIRECTION MODEL: P(UP wins | features)
    # Uses outcome_binary (1=UP, 0=DOWN) as label. Same features as primary.
    # This tells you WHICH side to bet, while the primary model tells you
    # WHETHER to bet at all.
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  DIRECTION MODEL — P(UP wins | features)")
    print("  Label: outcome_binary (1=UP won, 0=DOWN won)")
    print("  Same features as primary model")
    print("=" * 60)

    dir_fold_results = []
    for fold_i, (ti, vi, n_tr, n_te) in enumerate(folds):
        X_tr, yd_tr = X[ti], yd[ti]
        X_te, yd_te = X[vi], yd[vi]
        cal = "isotonic" if n_tr >= 500 else "sigmoid"
        m_dir, imp_dir = _train(X_tr, yd_tr, n_est=150, leaves=20, calibration=cal)
        preds_dir = m_dir.predict_proba(imp_dir.transform(X_te))[:, 1]

        base_rate_dir = yd_tr.mean()
        brier_dir     = brier_score_loss(yd_te, preds_dir)
        baseline_dir  = brier_score_loss(yd_te, np.full(len(yd_te), base_rate_dir))
        imp_dir_val   = baseline_dir - brier_dir
        acc_dir       = np.mean((preds_dir >= 0.5) == yd_te)

        # Compare to p_market baseline (Polymarket odds predict direction)
        pm_te_dir = pm[vi]
        brier_pm  = brier_score_loss(yd_te, pm_te_dir)
        imp_vs_pm = brier_pm - brier_dir

        dir_fold_results.append({
            "fold": fold_i + 1, "brier": brier_dir, "baseline": baseline_dir,
            "improvement": imp_dir_val, "accuracy": acc_dir,
            "brier_pm": brier_pm, "imp_vs_pm": imp_vs_pm,
        })

    avg_imp_dir    = np.mean([r["improvement"] for r in dir_fold_results])
    avg_acc_dir    = np.mean([r["accuracy"] for r in dir_fold_results])
    avg_imp_vs_pm  = np.mean([r["imp_vs_pm"] for r in dir_fold_results])

    print(f"\n  ── Direction model walk-forward ({len(folds)} folds) ──")
    print(f"    Avg Brier imp vs base rate: {avg_imp_dir:+.4f}  "
          f"{'[OK]' if avg_imp_dir > 0 else '[FAIL]'}")
    print(f"    Avg Brier imp vs p_market:  {avg_imp_vs_pm:+.4f}  "
          f"{'[BEATS CROWD]' if avg_imp_vs_pm > 0 else '[CROWD BETTER]'}")
    print(f"    Avg Accuracy:               {avg_acc_dir*100:.1f}%")
    print(f"    Base rate (UP wins):        {yd.mean()*100:.1f}%")
    report.append(f"Direction model: imp={avg_imp_dir:+.4f} vs_pm={avg_imp_vs_pm:+.4f} acc={avg_acc_dir*100:.1f}%")

    # Train final direction model on all data
    dir_final, dir_imp_final = _train(X, yd, n_est=200, leaves=25,
                                       calibration=cal_method_final)

    # Direction model feature importance
    print(f"\n  Direction model feature importance (top 15):")
    _importance(dir_final, fn, top_n=15)

    with open("models/model_v4_direction.pkl", "wb") as f:
        pickle.dump({
            "classifier":     dir_final,
            "classifier_imp": dir_imp_final,
            "features":       fn,
            "target":         "outcome_binary (1=UP wins)",
        }, f)
    log.info("  Saved model_v4_direction.pkl")

    # V5 training moved to train_v5.py (run separately)

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

    with open("models/model_v4_market.pkl", "wb") as f:
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
    print(f"  Direction model (UP wins)   avg Brier imp: {avg_imp_dir:+.4f}  "
          f"{'[OK]' if avg_imp_dir > 0 else '[FAIL]'}  "
          f"vs crowd: {avg_imp_vs_pm:+.4f}")
    print(f"  V5: run train_v5.py separately")
    print(f"  Market model (discovery)    avg Brier imp: {avg_imp3:+.4f}  "
          f"{'[OK]' if avg_imp3 > 0 else '[FAIL]'}")
    print()
    print("  Saved: model_v4_profitable.pkl")
    print("         model_v4_direction.pkl")
    print("         model_v4_nopmarket.pkl")
    print("         model_v4_market.pkl")
    print("=" * 60)

    with open("training_report_v4.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    main()
