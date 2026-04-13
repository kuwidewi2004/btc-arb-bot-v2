"""
V5 Futures Edge Regressor Training Pipeline
=============================================
Trains dual regressors for dYdX BTC-USD perpetual futures:
  E(edge_long)  = (btc_price_3min_later - btc_price_now) / btc_price_now - fees
  E(edge_short) = (btc_price_now - btc_price_3min_later) / btc_price_now - fees

Labels: 3-minute lookahead BTC price movement (NOT market resolution),
  minus 0.02% maker round-trip fees (0.01% per side, dYdX Tier 1).
Features: 78 including Polymarket sentiment.
Data fetch: cursor-based pagination (not OFFSET) to avoid Supabase timeouts.
Spot OB features removed — permanently dead (Binance geo-block).

Walk-forward validation by condition_id grouping.
_compute_sequence_features() available but disabled — accumulating in
pipeline for future LSTM use.

Current best: corr ~+0.092, near break-even PnL with maker fees.

Usage:
  python train_v5_futures.py

Output:
  model_v5_edge.pkl — dual edge regressors for dYdX execution
"""

import sys
import logging
import pickle
import math
import warnings
import numpy as np
import requests
from collections import OrderedDict, defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import lightgbm as lgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Supabase config
SUPABASE_URL = "https://kcluwyzyetmkxhvszpxi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtjbHV3eXp5ZXRta3hodnN6cHhpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA4NTY2NCwiZXhwIjoyMDg5NjYxNjY0fQ.IbxuXRW0K9_UFZKG1i951EoL9KtCsOXCaz5Z_YqsmYE"
REST_H = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

LEAKAGE_SECS = 120
REGIME_MAP   = {"TREND_UP":2,"TREND_DOWN":-2,"VOLATILE":1,"CALM":0,"DEAD":-1}
SESSION_MAP  = {"OVERLAP":3,"US":2,"LONDON":1,"ASIA":0,"OFFPEAK":-1}
ACTIVITY_MAP = {"HIGH":2,"NORMAL":1,"LOW":0,"DEAD":-1}
WF_MIN_TRAIN = 100
WF_TEST_SIZE = 60

# ── V4 scoring for meta-feature ──────────────────────────────────────────
# Load V4 model to score each snapshot with P(profitable) and P(UP).
# These scores become features in V5 — proven +30% corr improvement.
import pickle as _pickle

_v4_clf = _v4_imp = _v4_features = None
_v4d_clf = _v4d_imp = None
try:
    with open("models/model_v4_profitable.pkl", "rb") as _f4:
        _v4 = _pickle.load(_f4)
    _v4_clf = _v4["classifier"]
    _v4_imp = _v4["classifier_imp"]
    _v4_features = _v4["features"]
    with open("models/model_v4_direction.pkl", "rb") as _f4d:
        _v4d = _pickle.load(_f4d)
    _v4d_clf = _v4d["classifier"]
    _v4d_imp = _v4d["classifier_imp"]
    log.info(f"V4 loaded for meta-feature: {len(_v4_features)} features")
except Exception as _e:
    log.warning(f"V4 NOT loaded (V5 will train without meta-feature): {_e}")

_V4_REGIME_MAP   = {"TREND_UP":2,"TREND_DOWN":-2,"VOLATILE":1,"CALM":0,"DEAD":-1}
_V4_SESSION_MAP  = {"OVERLAP":3,"US":2,"LONDON":1,"ASIA":0,"OFFPEAK":-1}
_V4_ACTIVITY_MAP = {"HIGH":2,"NORMAL":1,"LOW":0,"DEAD":-1}
_V4_DAY_MAP      = {"WEEKDAY":1,"WEEKEND":0}
_V4_BUCKET_MAP   = {"heavy_fav":3,"favourite":2,"underdog":1,"longshot":0}


def _score_v4(row):
    """Score a single row with V4. Returns (P(profitable), P(UP))."""
    if _v4_clf is None:
        return 0.5, 0.5
    try:
        pm = _f(row.get("p_market"))
        m30 = _f(row.get("momentum_30s"))
        m60 = _f(row.get("momentum_60s"))
        li = _f(row.get("liq_imbalance"))
        vr = _f(row.get("vol_range_pct"))
        fr = _f(row.get("funding_rate"))
        str_val = _f(row.get("secs_to_resolution"))
        sl_val = _f(row.get("secs_left"))
        str_ = str_val if not np.isnan(str_val) else (sl_val if not np.isnan(sl_val) else 300.0)
        if str_ < 0: str_ = 0.0
        mp = _f(row.get("market_progress"))
        if np.isnan(mp): mp = max(0.0, min(1.0, 1.0 - str_ / 300.0))

        f = {}
        for fname in _v4_features:
            val = _f(row.get(fname))
            if fname == "secs_to_resolution": val = str_
            elif fname == "log_secs_to_resolution": val = math.log1p(str_)
            elif fname == "market_progress": val = mp
            elif fname == "mom_accel_abs": val = abs(m30 - m60) if not (np.isnan(m30) or np.isnan(m60)) else np.nan
            elif fname == "pm_abs_deviation": val = abs(pm - 0.5) if not np.isnan(pm) else np.nan
            elif fname == "pm_uncertainty": val = 1.0 - abs(pm - 0.5) * 2 if not np.isnan(pm) else np.nan
            elif fname == "liq_imbal_x_secs": val = li * str_ if not np.isnan(li) else np.nan
            elif fname == "liq_abs_imbalance": val = abs(li) if not np.isnan(li) else np.nan
            elif fname == "funding_abs": val = abs(fr) if not np.isnan(fr) else np.nan
            elif fname == "interact_mom_x_vol": val = m30 * vr if not (np.isnan(m30) or np.isnan(vr)) else np.nan
            elif fname == "interact_liq_x_price": val = li * _f(row.get("price_vs_open_pct")) if not np.isnan(li) else np.nan
            elif fname == "interact_mom_x_progress": val = m30 * mp if not np.isnan(m30) else np.nan
            elif fname == "vol_x_pm_abs_dev": val = vr * abs(pm - 0.5) if not (np.isnan(vr) or np.isnan(pm)) else np.nan
            elif fname == "okx_x_fr": val = _f(row.get("okx_funding")) * fr if not np.isnan(fr) else np.nan
            elif fname == "regime_enc": val = _V4_REGIME_MAP.get(row.get("regime", ""), 0)
            elif fname == "session_enc": val = _V4_SESSION_MAP.get(row.get("session", ""), 0)
            elif fname == "activity_enc": val = _V4_ACTIVITY_MAP.get(row.get("activity", ""), 0)
            elif fname == "day_enc": val = _V4_DAY_MAP.get(row.get("day_type", ""), 0)
            elif fname == "bucket_enc": val = _V4_BUCKET_MAP.get(row.get("price_bucket", ""), 0)
            elif fname == "is_extreme_market": val = 1.0 if (not np.isnan(pm) and (pm > 0.85 or pm < 0.15)) else 0.0
            f[fname] = val if val is not None else np.nan

        X = np.array([[f.get(k, np.nan) for k in _v4_features]], dtype=np.float32)
        X = _v4_imp.transform(X)
        p_prof = float(_v4_clf.predict_proba(X)[0][1])
        X_dir = _v4d_imp.transform(X)
        p_up = float(_v4d_clf.predict_proba(X_dir)[0][1])
        return p_prof, p_up
    except:
        return 0.5, 0.5


def _f(val):
    if val is None:
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _rest_fetch(table, params, limit=500):
    """Fetch using created_at cursor — matches V4 fetch (handles 100k+ rows)."""
    rows = []
    retries = 0
    cursor = ""
    while True:
        p = {**params, "limit": limit}
        if cursor:
            p["created_at"] = f"gt.{cursor}"
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=REST_H, params=p, timeout=180)
            if not r.text:
                retries += 1
                if retries > 5:
                    break
                import time; time.sleep(2)
                continue
            batch = r.json()
            if isinstance(batch, dict):
                retries += 1
                if retries > 5:
                    log.warning(f"  API error after 5 retries at {len(rows):,} rows: {batch.get('message','?')}")
                    break
                log.warning(f"  API error, retry {retries}: {batch.get('message','?')[:60]}")
                import time; time.sleep(5)
                continue
            if not batch or not isinstance(batch, list):
                break
            rows.extend(batch)
            retries = 0
            if len(rows) % 5000 == 0:
                log.info(f"  Fetched {len(rows):,}...")
            if len(batch) < limit:
                break
            last_ts = batch[-1].get("created_at")
            if not last_ts:
                break
            cursor = last_ts
        except Exception as e:
            retries += 1
            if retries > 5:
                log.warning(f"  Fetch stopped at {len(rows):,} rows after 5 retries: {e}")
                break
            log.warning(f"  Retry {retries}: {e}")
            import time; time.sleep(5)
    return rows


def walk_forward_splits(cond_ids):
    markets = list(dict.fromkeys(cond_ids))
    folds = []
    for i in range(WF_MIN_TRAIN, len(markets) - WF_TEST_SIZE + 1, WF_TEST_SIZE):
        tr_set = set(markets[:i])
        te_set = set(markets[i:i + WF_TEST_SIZE])
        ti = [j for j, c in enumerate(cond_ids) if c in tr_set]
        vi = [j for j, c in enumerate(cond_ids) if c in te_set]
        folds.append((ti, vi, len(tr_set), len(te_set)))
    return folds


def _train_regressor(X, y, n_est=150, leaves=20):
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    model = lgb.LGBMRegressor(n_estimators=n_est, num_leaves=leaves,
                               learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, verbose=-1)
    model.fit(X_imp, y)
    return model, imp


def _importance(model, fn, top_n=10):
    imp = model.feature_importances_.astype(np.float64)
    ranked = sorted(zip(fn, imp), key=lambda x: -x[1])
    print(f"\n  Feature importance (top {top_n})")
    print(f"  {'Feature':<40} {'Score':>8}")
    for fname, sc in ranked[:top_n]:
        bar = "#" * int(sc / ranked[0][1] * 20) if ranked[0][1] > 0 else ""
        print(f"  {fname:<40} {sc:>8.1f}  {bar}")

# Unified cost model — matches quant_engine.py and execution.py
DYDX_MAKER_FEE = 0.0001   # 0.01% per side (1.0 bps) — Tier 1
DYDX_TAKER_FEE = 0.0005   # 0.05% per side (5.0 bps) — Tier 1
ROUND_TRIP_TAKER = DYDX_TAKER_FEE * 2   # 0.10% — worst case (market orders both sides)
ROUND_TRIP_MAKER = DYDX_MAKER_FEE * 2   # 0.02% — best case (limit orders both sides)
ROUND_TRIP       = ROUND_TRIP_MAKER      # train with maker fees — we'll use limit orders
MIN_EDGE         = 0.0001               # 0.01% minimum edge filter


def _build_cross_market_lookup(rows) -> dict:
    market_rows = OrderedDict()
    for row in rows:
        cid = row.get("condition_id", "")
        if cid not in market_rows:
            market_rows[cid] = []
        market_rows[cid].append(row)

    market_summaries = OrderedDict()
    for cid, mrows in market_rows.items():
        last = mrows[-1]
        ob = _f(last.get("outcome_binary"))
        market_summaries[cid] = {
            "outcome": ob if not np.isnan(ob) else np.nan,
            "momentum_30s": _f(last.get("momentum_30s")),
            "ob_imbalance": _f(last.get("ob_imbalance")),
            "vol_range_pct": _f(last.get("vol_range_pct")),
            "btc_range_pct": _f(last.get("btc_range_pct")),
            "p_market": _f(last.get("p_market")),
            "funding_zscore": _f(last.get("funding_zscore")),
        }

    cids = list(market_summaries.keys())
    cross = {}
    streak_up = streak_down = 0
    for i, cid in enumerate(cids):
        if i == 0:
            cross[cid] = {
                "prev_outcome": np.nan, "prev_momentum": np.nan,
                "prev_ob_imbalance": np.nan, "prev_vol_range": np.nan,
                "prev_btc_range": np.nan, "prev_p_market_final": np.nan,
                "prev_funding_zscore": np.nan,
                "streak_up": 0.0, "streak_down": 0.0,
            }
        else:
            prev = market_summaries[cids[i-1]]
            cross[cid] = {
                "prev_outcome": prev["outcome"],
                "prev_momentum": prev["momentum_30s"],
                "prev_ob_imbalance": prev["ob_imbalance"],
                "prev_vol_range": prev["vol_range_pct"],
                "prev_btc_range": prev["btc_range_pct"],
                "prev_p_market_final": prev["p_market"],
                "prev_funding_zscore": prev["funding_zscore"],
                "streak_up": float(streak_up),
                "streak_down": float(streak_down),
            }
        cur_outcome = market_summaries[cid]["outcome"]
        if not np.isnan(cur_outcome):
            if cur_outcome == 1:
                streak_up += 1; streak_down = 0
            else:
                streak_down += 1; streak_up = 0
    return cross


LOOKAHEAD_SECS = 180  # 3-minute lookahead for future BTC price
LOOKAHEAD_TOLERANCE = 15  # accept a match within ±15 seconds of target

def fetch_snapshots_v5() -> list:
    """Fetch ALL snapshots with btc_price, then compute 3-min lookahead labels."""
    log.info("Fetching market_snapshots for V5 (futures, 5-min lookahead)...")
    cols = ",".join([
        "created_at","condition_id","secs_left","secs_to_resolution","market_progress",
        "phase_early",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "price_vs_open_pct","price_vs_open_score",
        "momentum_30s","momentum_60s","momentum_120s",
        "cl_vs_open_pct",
        "liq_total","liq_imbalance","liq_long_usd","liq_short_usd","liq_dominant_ratio",
        "ob_imbalance","ob_bid_depth","ob_ask_depth",
        "vol_range_pct","volume_buy_ratio",
        "p_market","poly_fill_up","poly_fill_down","poly_spread",
        "basis_pct","funding_rate","okx_funding","gate_funding",
        "volatility_pct","flow_score","funding_zscore",
        "regime","session","activity","day_type",
        "avg_ob_imbalance_abs","avg_funding_zscore_abs",
        "avg_momentum_abs","btc_range_pct","p_market_std",
        "poly_flow_imb","poly_depth_ratio",
        "poly_trade_imb","poly_up_buys","poly_down_buys","poly_trade_count","poly_large_pct",
        "tick_cvd_30s","tick_taker_buy_ratio_30s","tick_large_buy_usd_30s",
        "tick_large_sell_usd_30s","tick_intensity_30s","tick_vwap_disp_30s",
        "tick_cvd_60s","tick_taker_buy_ratio_60s","tick_intensity_60s",
        "delta_cvd","delta_taker_buy","delta_momentum","delta_poly","delta_score",
        "delta_funding","delta_basis",
        "btc_price",
        "outcome_binary",
    ])
    from fetch_cache import cached_fetch
    all_rows = cached_fetch("v5_snapshots", cols, {
        "btc_price": "gt.0",
        "outcome_binary": "not.is.null",
    })
    log.info(f"  {len(all_rows):,} total rows with btc_price")

    # Parse timestamps and build a time-sorted price array for lookahead matching
    from datetime import datetime, timezone
    timestamps = []
    prices = []
    for row in all_rows:
        ts_str = row.get("created_at", "")
        btc = row.get("btc_price")
        if not ts_str or not btc:
            continue
        ts = datetime.fromisoformat(ts_str).timestamp()
        timestamps.append(ts)
        prices.append(float(btc))

    ts_arr = np.array(timestamps)
    px_arr = np.array(prices)
    log.info(f"  Built price timeline: {len(ts_arr):,} points, "
             f"span {(ts_arr[-1]-ts_arr[0])/3600:.1f} hours")

    # For each row, find the BTC price ~5 min later using binary search
    matched = 0
    no_match = 0
    for row in all_rows:
        ts_str = row.get("created_at", "")
        if not ts_str:
            row["btc_future_price"] = None
            no_match += 1
            continue
        ts = datetime.fromisoformat(ts_str).timestamp()
        target_ts = ts + LOOKAHEAD_SECS
        # Binary search for closest timestamp to target
        idx = np.searchsorted(ts_arr, target_ts)
        best_price = None
        best_gap = LOOKAHEAD_TOLERANCE + 1
        for candidate in [idx - 1, idx]:
            if 0 <= candidate < len(ts_arr):
                gap = abs(ts_arr[candidate] - target_ts)
                if gap < best_gap:
                    best_gap = gap
                    best_price = px_arr[candidate]
        if best_price is not None and best_gap <= LOOKAHEAD_TOLERANCE:
            row["btc_future_price"] = float(best_price)
            matched += 1
        else:
            row["btc_future_price"] = None
            no_match += 1

    log.info(f"  Lookahead matched: {matched:,}  no match: {no_match:,} "
             f"({matched/(matched+no_match)*100:.1f}% match rate)")
    return all_rows


def _compute_sequence_features(rows):
    """Pre-compute sequence features from consecutive snapshots within each condition_id."""
    from collections import defaultdict, deque as dq

    # Group row indices by condition_id
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.get("condition_id", "")].append(i)

    for cid, indices in groups.items():
        prev_delta_cvd = 0.0
        prev_delta_mom = 0.0
        mom_signs = dq(maxlen=10)
        ob_signs = dq(maxlen=10)

        for idx in indices:
            row = rows[idx]
            dc = row.get("delta_cvd") or 0.0
            dm = row.get("delta_momentum") or 0.0
            m30 = row.get("momentum_30s") or 0.0
            obi = row.get("ob_imbalance") or 0.0

            # Only compute from DB if not already backfilled
            if row.get("cvd_accel") is None:
                row["cvd_accel"] = round(dc - prev_delta_cvd, 2)
            if row.get("momentum_accel") is None:
                row["momentum_accel"] = round(dm - prev_delta_mom, 6)

            mom_signs.append(1 if m30 > 0 else (-1 if m30 < 0 else 0))
            if row.get("momentum_consistency_10") is None:
                if len(mom_signs) >= 3:
                    current = mom_signs[-1]
                    same = sum(1 for s in mom_signs if s == current)
                    row["momentum_consistency_10"] = round(same / len(mom_signs), 2)

            ob_signs.append(1 if obi > 0 else -1)
            if row.get("ob_flip_count_10") is None:
                if len(ob_signs) >= 3:
                    row["ob_flip_count_10"] = sum(1 for i in range(1, len(ob_signs))
                                                   if ob_signs[i] != ob_signs[i-1])

            prev_delta_cvd = dc
            prev_delta_mom = dm


def build_v5_features(rows):
    """
    Build feature matrix for V5 futures model.
    Labels: BTC price 3 minutes from now vs current price (minus fees).
    Every snapshot with a valid lookahead match is usable — no market dependency.
    """
    # _compute_sequence_features(rows)  # disabled — sequence features not in V5 yet (accumulating in pipeline)

    # Build V4 feature matrix for fold-specific scoring (avoids leakage)
    v4_feat_matrix = None
    v4_labels = None
    if _v4_features is not None:
        log.info("  Building V4 feature matrix (fold-specific scoring, no leakage)...")
        v4_feat_matrix = np.full((len(rows), len(_v4_features)), np.nan, dtype=np.float32)
        v4_labels = np.full(len(rows), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            pm = _f(row.get("p_market"))
            m30 = _f(row.get("momentum_30s"))
            m60 = _f(row.get("momentum_60s"))
            li = _f(row.get("liq_imbalance"))
            vr = _f(row.get("vol_range_pct"))
            fr = _f(row.get("funding_rate"))
            str_val = _f(row.get("secs_to_resolution"))
            sl_val = _f(row.get("secs_left"))
            str_ = str_val if not np.isnan(str_val) else (sl_val if not np.isnan(sl_val) else 300.0)
            if str_ < 0: str_ = 0.0
            mp = _f(row.get("market_progress"))
            if np.isnan(mp): mp = max(0.0, min(1.0, 1.0 - str_ / 300.0))
            for j, fname in enumerate(_v4_features):
                val = _f(row.get(fname))
                if fname == "secs_to_resolution": val = str_
                elif fname == "log_secs_to_resolution": val = math.log1p(str_)
                elif fname == "market_progress": val = mp
                elif fname == "mom_accel_abs": val = abs(m30 - m60) if not (np.isnan(m30) or np.isnan(m60)) else np.nan
                elif fname == "pm_abs_deviation": val = abs(pm - 0.5) if not np.isnan(pm) else np.nan
                elif fname == "pm_uncertainty": val = 1.0 - abs(pm - 0.5) * 2 if not np.isnan(pm) else np.nan
                elif fname == "liq_imbal_x_secs": val = li * str_ if not np.isnan(li) else np.nan
                elif fname == "liq_abs_imbalance": val = abs(li) if not np.isnan(li) else np.nan
                elif fname == "funding_abs": val = abs(fr) if not np.isnan(fr) else np.nan
                elif fname == "interact_mom_x_vol": val = m30 * vr if not (np.isnan(m30) or np.isnan(vr)) else np.nan
                elif fname == "interact_liq_x_price": val = li * _f(row.get("price_vs_open_pct")) if not np.isnan(li) else np.nan
                elif fname == "interact_mom_x_progress": val = m30 * mp if not np.isnan(m30) else np.nan
                elif fname == "vol_x_pm_abs_dev": val = vr * abs(pm - 0.5) if not (np.isnan(vr) or np.isnan(pm)) else np.nan
                elif fname == "okx_x_fr": val = _f(row.get("okx_funding")) * fr if not np.isnan(fr) else np.nan
                elif fname == "regime_enc": val = _V4_REGIME_MAP.get(row.get("regime", ""), 0)
                elif fname == "session_enc": val = _V4_SESSION_MAP.get(row.get("session", ""), 0)
                elif fname == "activity_enc": val = _V4_ACTIVITY_MAP.get(row.get("activity", ""), 0)
                elif fname == "day_enc": val = _V4_DAY_MAP.get(row.get("day_type", ""), 0)
                elif fname == "bucket_enc": val = _V4_BUCKET_MAP.get(row.get("price_bucket", ""), 0)
                elif fname == "is_extreme_market": val = 1.0 if (not np.isnan(pm) and (pm > 0.85 or pm < 0.15)) else 0.0
                v4_feat_matrix[i, j] = val if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan
            ob = _f(row.get("outcome_binary"))
            v4_labels[i] = ob
        log.info(f"    V4 feature matrix built: {v4_feat_matrix.shape}")

    cross_mkt = _build_cross_market_lookup(rows)

    records = []
    y_edge_long = []
    y_edge_short = []
    cond_ids = []
    row_indices = []  # maps to original row index (for V4 fold-specific scoring)
    skipped = 0

    for row_idx, row in enumerate(rows):
        pm = _f(row.get("p_market"))
        btc_price = _f(row.get("btc_price"))
        btc_future = _f(row.get("btc_future_price"))

        str_val = _f(row.get("secs_to_resolution"))
        sl_val  = _f(row.get("secs_left"))
        if not np.isnan(str_val):
            str_ = str_val
        elif not np.isnan(sl_val):
            str_ = sl_val
        else:
            str_ = np.nan

        if np.isnan(btc_price) or np.isnan(btc_future) or btc_price <= 0:
            skipped += 1; continue

        # Futures labels: BTC price 5 min from now minus fees
        edge_long  = (btc_future - btc_price) / btc_price - ROUND_TRIP
        edge_short = (btc_price - btc_future) / btc_price - ROUND_TRIP

        _cross = cross_mkt.get(row.get("condition_id", ""), {})
        mp = _f(row.get("market_progress"))
        if np.isnan(mp): mp = max(0.0, min(1.0, 1.0 - str_ / 300.0))
        vr = _f(row.get("vol_range_pct"))
        fr = _f(row.get("funding_rate"))
        fz = _f(row.get("funding_zscore"))
        m30 = _f(row.get("momentum_30s"))
        m60 = _f(row.get("momentum_60s"))
        m120 = _f(row.get("momentum_120s"))
        obi = _f(row.get("ob_imbalance"))
        li = _f(row.get("liq_imbalance"))

        f = {
            # secs_to_resolution / market_progress REMOVED — Polymarket lifecycle, leaky for BTC
            "hour_sin": _f(row.get("hour_sin")), "hour_cos": _f(row.get("hour_cos")),
            "dow_sin": _f(row.get("dow_sin")), "dow_cos": _f(row.get("dow_cos")),
            "price_vs_open_pct": _f(row.get("price_vs_open_pct")),
            "price_vs_open_score": _f(row.get("price_vs_open_score")),
            "momentum_30s": m30, "momentum_60s": m60, "momentum_120s": m120,
            "mom_accel_abs": abs(m30 - m60) if not (np.isnan(m30) or np.isnan(m60)) else np.nan,
            "cl_vs_open_pct": _f(row.get("cl_vs_open_pct")),
            "liq_total": min(_f(row.get("liq_total")), 5e6),
            "liq_imbalance": li,
            "liq_long_usd": min(_f(row.get("liq_long_usd")), 5e6),
            "liq_short_usd": min(_f(row.get("liq_short_usd")), 5e6),
            # liq_imbal_x_secs REMOVED — depended on secs_to_resolution
            "ob_imbalance": obi, "ob_bid_depth": _f(row.get("ob_bid_depth")),
            "ob_ask_depth": _f(row.get("ob_ask_depth")),
            "vol_range_pct": vr, "volatility_pct": _f(row.get("volatility_pct")),
            "volume_buy_ratio": _f(row.get("volume_buy_ratio")),
            # Polymarket features REMOVED — don't predict BTC price direction
            # Funding
            "basis_pct": _f(row.get("basis_pct")),
            "funding_rate": fr, "funding_zscore": fz,
            "okx_funding": _f(row.get("okx_funding")),
            "gate_funding": _f(row.get("gate_funding")),
            "funding_abs": abs(fr) if not np.isnan(fr) else np.nan,
            # Regime
            "regime_enc": REGIME_MAP.get(row.get("regime", ""), 0),
            "session_enc": SESSION_MAP.get(row.get("session", ""), 0),
            "activity_enc": ACTIVITY_MAP.get(row.get("activity", ""), 0),
            # Interactions
            "interact_mom_x_vol": m30 * vr if not (np.isnan(m30) or np.isnan(vr)) else np.nan,
            "interact_liq_x_price": li * _f(row.get("price_vs_open_pct")) if not np.isnan(li) else np.nan,
            "okx_x_fr": _f(row.get("okx_funding")) * fr if not np.isnan(fr) else np.nan,
            # Rolling
            "avg_ob_imbalance_abs": _f(row.get("avg_ob_imbalance_abs")),
            "avg_funding_zscore_abs": _f(row.get("avg_funding_zscore_abs")),
            "avg_momentum_abs": _f(row.get("avg_momentum_abs")),
            "btc_range_pct": _f(row.get("btc_range_pct")),
            # p_market_std REMOVED — Polymarket feature
            # Tick
            "tick_cvd_30s": _f(row.get("tick_cvd_30s")),
            "tick_taker_buy_ratio_30s": _f(row.get("tick_taker_buy_ratio_30s")),
            "tick_large_buy_usd_30s": _f(row.get("tick_large_buy_usd_30s")),
            "tick_large_sell_usd_30s": _f(row.get("tick_large_sell_usd_30s")),
            "tick_intensity_30s": _f(row.get("tick_intensity_30s")),
            "tick_vwap_disp_30s": _f(row.get("tick_vwap_disp_30s")),
            "tick_cvd_60s": _f(row.get("tick_cvd_60s")),
            "tick_taker_buy_ratio_60s": _f(row.get("tick_taker_buy_ratio_60s")),
            "tick_intensity_60s": _f(row.get("tick_intensity_60s")),
            # Spot OB features REMOVED — permanently dead (Binance geo-block)
            # Deltas
            "delta_cvd": _f(row.get("delta_cvd")),
            "delta_taker_buy": _f(row.get("delta_taker_buy")),
            "delta_momentum": _f(row.get("delta_momentum")),
            "delta_funding": _f(row.get("delta_funding")),
            "delta_basis": _f(row.get("delta_basis")),
            # Cross-market (BTC-relevant only, Polymarket features removed)
            "prev_momentum": _cross.get("prev_momentum", np.nan),
            "prev_ob_imbalance": _cross.get("prev_ob_imbalance", np.nan),
            "prev_vol_range": _cross.get("prev_vol_range", np.nan),
            "prev_btc_range": _cross.get("prev_btc_range", np.nan),
            "prev_funding_zscore": _cross.get("prev_funding_zscore", np.nan),
            "prev_vol_x_momentum": _cross.get("prev_vol_range", 0) * _cross.get("prev_momentum", 0),
            "session_x_vol": SESSION_MAP.get(row.get("session", ""), 0) * vr if not np.isnan(vr) else np.nan,
        }

        # V4 meta-features — placeholder, scored per-fold to avoid leakage
        if _v4_features is not None:
            f["v4_p_profitable"] = 0.5  # filled per-fold in walk-forward
            f["v4_p_up"] = 0.5          # filled per-fold in walk-forward

        records.append(f)
        y_edge_long.append(round(edge_long, 6))
        y_edge_short.append(round(edge_short, 6))
        cond_ids.append(row.get("condition_id", ""))
        row_indices.append(row_idx)

    log.info(f"  Engineered {len(records):,} rows ({skipped:,} skipped)")
    if not records:
        return None, None, None, None, None, None, None, None

    fn = list(records[0].keys())
    X  = np.array([[r[k] for k in fn] for r in records], dtype=np.float32)
    ye_long  = np.array(y_edge_long,  dtype=np.float32)
    ye_short = np.array(y_edge_short, dtype=np.float32)
    row_indices = np.array(row_indices, dtype=np.int64)

    # Subset V4 data to only the rows that survived feature engineering
    v4_X_subset = v4_feat_matrix[row_indices] if v4_feat_matrix is not None else None
    v4_y_subset = v4_labels[row_indices] if v4_labels is not None else None

    log.info(f"  Features: {len(fn)}  |  avg edge_long={ye_long.mean():.6f}  avg edge_short={ye_short.mean():.6f}")

    return X, ye_long, ye_short, cond_ids, fn, v4_X_subset, v4_y_subset, row_indices


def main():
    print("=" * 60)
    print("  V5 FUTURES EDGE REGRESSOR PIPELINE")
    print("  E(edge_long) and E(edge_short) per snapshot")
    print("  Labels: BTC price movement minus 0.04% round-trip fees")
    print("  Features: full set including Polymarket sentiment")
    print("=" * 60)

    rows = fetch_snapshots_v5()
    if not rows:
        log.error("No data with btc_resolution_price. Is resolver running?")
        return

    result = build_v5_features(rows)
    if result is None or result[0] is None:
        log.error("No usable rows after feature engineering.")
        return

    X, ye_long, ye_short, cids, fn, v4_X, v4_y, v5_row_idx = result

    # Find V4 meta-feature column indices in V5 feature matrix
    v4_prof_col = fn.index("v4_p_profitable") if "v4_p_profitable" in fn else None
    v4_up_col = fn.index("v4_p_up") if "v4_p_up" in fn else None

    folds = walk_forward_splits(cids)
    n_markets = len(set(cids))
    log.info(f"  {n_markets} markets, {len(X)} rows, {len(fn)} features, {len(folds)} folds")

    # Walk-forward validation
    fold_results = []
    all_fold_pred_edge = []   # for evaluation metrics
    all_fold_actual_edge = []
    all_fold_chosen_up = []
    all_fold_sessions = []
    all_fold_regimes = []

    for fold_i, (ti, vi, n_tr, n_te) in enumerate(folds):
        X_tr, X_te = X[ti].copy(), X[vi].copy()
        yl_tr, yl_te = ye_long[ti], ye_long[vi]
        ys_tr, ys_te = ye_short[ti], ye_short[vi]

        # Fold-specific V4 scoring — train V4 on fold's train data only
        if v4_X is not None and v4_prof_col is not None:
            v4_tr = v4_X[ti]
            v4_te = v4_X[vi]
            v4_y_tr = v4_y[ti]

            # Only train on rows with valid outcome_binary
            valid_v4 = ~np.isnan(v4_y_tr)
            if valid_v4.sum() >= 50:
                v4_imp_fold = SimpleImputer(strategy="median")
                v4_tr_imp = v4_imp_fold.fit_transform(v4_tr[valid_v4])
                v4_clf_fold = lgb.LGBMClassifier(n_estimators=80, num_leaves=15,
                                                  learning_rate=0.05, subsample=0.8,
                                                  colsample_bytree=0.8, verbose=-1)
                v4_clf_fold.fit(v4_tr_imp, v4_y_tr[valid_v4])
                # Score test rows with fold-specific V4
                v4_te_imp = v4_imp_fold.transform(v4_te)
                v4_probs = v4_clf_fold.predict_proba(v4_te_imp)[:, 1]
                X_te[:, v4_prof_col] = v4_probs
                # Score train rows too (for consistent training)
                v4_tr_all_imp = v4_imp_fold.transform(v4_tr)
                v4_tr_probs = v4_clf_fold.predict_proba(v4_tr_all_imp)[:, 1]
                X_tr[:, v4_prof_col] = v4_tr_probs
                # Direction model (predict UP vs DOWN among profitable)
                up_mask = valid_v4 & (v4_y_tr == 1)  # simplified: outcome=1 means UP won
                if up_mask.sum() >= 20:
                    X_te[:, v4_up_col] = v4_probs  # reuse P(profitable) as proxy
                    X_tr[:, v4_up_col] = v4_tr_probs

        reg_up, imp_up   = _train_regressor(X_tr, yl_tr, n_est=150, leaves=20)
        reg_down, imp_down = _train_regressor(X_tr, ys_tr, n_est=150, leaves=20)

        pred_up  = reg_up.predict(imp_up.transform(X_te))
        pred_down = reg_down.predict(imp_down.transform(X_te))

        mae_up  = float(np.mean(np.abs(pred_up  - yl_te)))
        mae_down = float(np.mean(np.abs(pred_down - ys_te)))
        corr_up  = float(np.corrcoef(pred_up,  yl_te)[0,1])  if len(yl_te) > 2 else 0.0
        corr_down = float(np.corrcoef(pred_down, ys_te)[0,1]) if len(ys_te) > 2 else 0.0

        # Decision: go long or short based on higher predicted edge
        chosen_up     = pred_up > pred_down
        pred_edge   = np.where(chosen_up, pred_up, pred_down)
        actual_edge = np.where(chosen_up, yl_te, ys_te)

        trade_mask = pred_edge > MIN_EDGE
        n_trades   = int(trade_mask.sum())

        if n_trades > 0:
            sim_pnl  = float(actual_edge[trade_mask].sum())
            pnl_per  = float(actual_edge[trade_mask].mean())
            win_rate = float((actual_edge[trade_mask] > 0).mean())
            actual_up = yl_te > ys_te
            side_acc = float((chosen_up[trade_mask] == actual_up[trade_mask]).mean())
        else:
            sim_pnl = pnl_per = win_rate = side_acc = 0.0

        fold_results.append({
            "fold": fold_i + 1, "n_train": n_tr, "n_test": n_te,
            "mae_up": mae_up, "mae_down": mae_down,
            "corr_up": corr_up, "corr_down": corr_down,
            "n_trades": n_trades, "sim_pnl": sim_pnl,
            "pnl_per": pnl_per, "win_rate": win_rate, "side_acc": side_acc,
            # Save for ensemble: test indices, predictions, actuals
            "test_row_indices": np.array([int(v5_row_idx[j]) for j in vi]),
            "pred_up": pred_up, "pred_down": pred_down,
            "actual_long": yl_te, "actual_short": ys_te,
        })

        # Collect data for extended evaluation
        all_fold_pred_edge.extend(pred_edge)
        all_fold_actual_edge.extend(actual_edge)
        all_fold_chosen_up.extend(chosen_up)
        for idx in vi:
            orig_idx = int(v5_row_idx[idx]) if v5_row_idx is not None else 0
            all_fold_sessions.append(rows[orig_idx].get("session", "UNKNOWN") if orig_idx < len(rows) else "UNKNOWN")
            all_fold_regimes.append(rows[orig_idx].get("regime", "UNKNOWN") if orig_idx < len(rows) else "UNKNOWN")

        print(f"\n  Fold {fold_i+1}  train={n_tr}  test={n_te}")
        print(f"    MAE  long={mae_up:.6f}  short={mae_down:.6f}")
        print(f"    Corr long={corr_up:+.3f}  short={corr_down:+.3f}")
        print(f"    Trades: {n_trades}  PnL={sim_pnl:+.6f}  "
              f"edge/trade={pnl_per:+.6f}  WR={win_rate*100:.1f}%")
        print(f"    Side accuracy: {side_acc*100:.1f}%")

    # Save fold predictions for ensemble eval
    import os
    os.makedirs("cache", exist_ok=True)
    save_dict = {}
    for i, r in enumerate(fold_results):
        for k in ["test_row_indices", "pred_up", "pred_down", "actual_long", "actual_short"]:
            if k in r and r[k] is not None:
                save_dict[f"fold_{i}_{k}"] = np.array(r[k])
    np.savez("cache/v5_fold_predictions.npz", **save_dict)
    log.info(f"  Saved V5 fold predictions to cache/v5_fold_predictions.npz")

    # Averages
    avg_corr_up  = np.mean([r["corr_up"]  for r in fold_results])
    avg_corr_down = np.mean([r["corr_down"] for r in fold_results])
    total_pnl      = sum(r["sim_pnl"]   for r in fold_results)
    total_trades   = sum(r["n_trades"]  for r in fold_results)
    folds_with     = [r for r in fold_results if r["n_trades"] > 0]
    avg_wr         = np.mean([r["win_rate"] for r in folds_with]) if folds_with else 0.0
    avg_side       = np.mean([r["side_acc"] for r in folds_with]) if folds_with else 0.0

    print(f"\n  ── V5 Futures walk-forward ({len(folds)} folds) ──")
    print(f"    Corr long={avg_corr_up:+.4f}  short={avg_corr_down:+.4f}")
    print(f"    Total PnL: {total_pnl:+.6f}  ({total_trades} trades)")
    print(f"    Avg WR: {avg_wr*100:.1f}%  Side acc: {avg_side*100:.1f}%")

    # ── EXTENDED EVALUATION ──────────────────────────────────────────
    pe = np.array(all_fold_pred_edge)
    ae = np.array(all_fold_actual_edge)
    cu = np.array(all_fold_chosen_up)

    if len(pe) > 100:
        print(f"\n  ── Extended Evaluation ({len(pe):,} test predictions) ──")

        # Top-decile analysis: when model is most confident, does it work?
        pct = np.percentile(pe, [90, 75, 50])
        for label, thresh in [("Top 10%", pct[0]), ("Top 25%", pct[1]), ("Top 50%", pct[2])]:
            mask = pe >= thresh
            n = int(mask.sum())
            if n > 0:
                edge = float(ae[mask].mean())
                wr = float((ae[mask] > 0).mean())
                pnl = float(ae[mask].sum())
                print(f"    {label} (n={n:,}):  avg edge={edge:+.6f}  WR={wr*100:.1f}%  PnL={pnl:+.4f}")

        # Long-only vs Short-only breakdown
        long_mask = cu.astype(bool)
        short_mask = ~long_mask
        for label, mask in [("Long-only", long_mask), ("Short-only", short_mask)]:
            n = int(mask.sum())
            if n > 0:
                edge = float(ae[mask].mean())
                wr = float((ae[mask] > 0).mean())
                pnl = float(ae[mask].sum())
                print(f"    {label} (n={n:,}):  avg edge={edge:+.6f}  WR={wr*100:.1f}%  PnL={pnl:+.4f}")

        # Session breakdown
        sessions = np.array(all_fold_sessions)
        print(f"\n    By session:")
        for sess in sorted(set(sessions)):
            mask = sessions == sess
            n = int(mask.sum())
            if n >= 20:
                edge = float(ae[mask].mean())
                wr = float((ae[mask] > 0).mean())
                print(f"      {sess:<10} (n={n:>5,}):  avg edge={edge:+.6f}  WR={wr*100:.1f}%")

        # Regime breakdown
        regimes = np.array(all_fold_regimes)
        print(f"\n    By regime:")
        for reg in sorted(set(regimes)):
            mask = regimes == reg
            n = int(mask.sum())
            if n >= 20:
                edge = float(ae[mask].mean())
                wr = float((ae[mask] > 0).mean())
                print(f"      {reg:<12} (n={n:>5,}):  avg edge={edge:+.6f}  WR={wr*100:.1f}%")

        # Confidence calibration: binned predicted edge vs actual
        print(f"\n    Confidence calibration:")
        bins = np.percentile(pe, [0, 20, 40, 60, 80, 100])
        for i in range(len(bins) - 1):
            mask = (pe >= bins[i]) & (pe < bins[i+1]) if i < len(bins)-2 else (pe >= bins[i])
            n = int(mask.sum())
            if n > 0:
                pred_avg = float(pe[mask].mean())
                actual_avg = float(ae[mask].mean())
                print(f"      Pred [{bins[i]:+.5f}, {bins[i+1]:+.5f}):  n={n:>5,}  pred={pred_avg:+.6f}  actual={actual_avg:+.6f}")

    # Train final models
    # ── KILL TESTS ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  V5 KILL TESTS — signal validity")
    print("=" * 60)

    last_ti, last_vi = folds[-1][0], folds[-1][1]
    X_last_tr, X_last_te = X[last_ti], X[last_vi]
    yl_last_tr, yl_last_te = ye_long[last_ti], ye_long[last_vi]
    ys_last_tr, ys_last_te = ye_short[last_ti], ye_short[last_vi]

    # [1] Shuffle Test
    print("\n  [1] Shuffle Test")
    shuf_idx = np.random.permutation(len(yl_last_tr))
    shuf_reg, shuf_imp = _train_regressor(X_last_tr, yl_last_tr[shuf_idx], n_est=100, leaves=15)
    shuf_pred = shuf_reg.predict(shuf_imp.transform(X_last_te))
    shuf_corr = float(np.corrcoef(shuf_pred, yl_last_te)[0,1]) if len(yl_last_te) > 2 else 0
    real_corr = fold_results[-1]["corr_up"]
    shuf_pass = abs(shuf_corr) < abs(real_corr) * 0.5
    print(f"      Real corr:     {real_corr:+.4f}")
    print(f"      Shuffled corr: {shuf_corr:+.4f}")
    print(f"      Result:        {'PASS' if shuf_pass else 'FAIL'}")

    # [2] Time Shift Test
    print("\n  [2] Time Shift Test")
    if len(folds) >= 3:
        prev_ti = folds[-2][0]
        ts_reg, ts_imp = _train_regressor(X[prev_ti], ye_long[prev_ti], n_est=100, leaves=15)
        ts_pred = ts_reg.predict(ts_imp.transform(X_last_te))
        ts_corr = float(np.corrcoef(ts_pred, yl_last_te)[0,1]) if len(yl_last_te) > 2 else 0
        ts_pass = (real_corr - ts_corr) > -0.05
        print(f"      Real corr:         {real_corr:+.4f}")
        print(f"      Time-shifted corr: {ts_corr:+.4f}")
        print(f"      Result:            {'PASS' if ts_pass else 'WARN'}")
    else:
        print("      Skipped — need 3+ folds")
        ts_pass = True

    # [3] Sign Test
    print("\n  [3] Sign Test (Direction Accuracy)")
    reg_up_t, imp_up_t = _train_regressor(X_last_tr, yl_last_tr, n_est=100, leaves=15)
    reg_dn_t, imp_dn_t = _train_regressor(X_last_tr, ys_last_tr, n_est=100, leaves=15)
    p_up_t = reg_up_t.predict(imp_up_t.transform(X_last_te))
    p_dn_t = reg_dn_t.predict(imp_dn_t.transform(X_last_te))
    model_picks_up = p_up_t > p_dn_t
    actual_up_better = yl_last_te > ys_last_te
    sign_acc = float((model_picks_up == actual_up_better).mean())
    sign_pass = sign_acc > 0.52
    print(f"      Side accuracy: {sign_acc*100:.1f}%")
    print(f"      Edge vs random: {(sign_acc-0.5)*100:+.1f}pp")
    print(f"      Result:        {'PASS' if sign_pass else 'FAIL'}")

    # [4] Edge Magnitude Test
    print("\n  [4] Edge Magnitude Test")
    best_edge = np.maximum(p_up_t, p_dn_t)
    actual_best = np.where(model_picks_up, yl_last_te, ys_last_te)
    q75 = np.percentile(best_edge, 75)
    q25 = np.percentile(best_edge, 25)
    top_actual = actual_best[best_edge >= q75].mean() if (best_edge >= q75).sum() > 0 else 0
    bot_actual = actual_best[best_edge <= q25].mean() if (best_edge <= q25).sum() > 0 else 0
    mag_pass = top_actual > bot_actual
    print(f"      Top-25% pred → actual: {top_actual:+.6f}")
    print(f"      Bot-25% pred → actual: {bot_actual:+.6f}")
    print(f"      Result:        {'PASS' if mag_pass else 'FAIL'}")

    print(f"\n  Kill test summary:")
    print(f"    [1] Shuffle:   {'PASS' if shuf_pass else 'FAIL'}")
    print(f"    [2] Time shift:{'PASS' if ts_pass else 'WARN'}")
    print(f"    [3] Sign:      {'PASS' if sign_pass else 'FAIL'}")
    print(f"    [4] Magnitude: {'PASS' if mag_pass else 'FAIL'}")

    log.info("  Training final V5 futures regressors...")
    final_up,  imp_up  = _train_regressor(X, ye_long,  n_est=200, leaves=25)
    final_down, imp_down = _train_regressor(X, ye_short, n_est=200, leaves=25)

    print("\n  Edge UP importance (top 15):")
    _importance(final_up, fn, top_n=15)
    print("\n  Edge DOWN importance (top 15):")
    _importance(final_down, fn, top_n=15)

    # Save
    with open("models/model_v5_edge.pkl", "wb") as f:
        pickle.dump({
            "regressor_up":      final_up,
            "regressor_up_imp":  imp_up,
            "regressor_down":     final_down,
            "regressor_down_imp": imp_down,
            "features":            fn,
            "target_long":         "edge_long = (btc_res - btc_price) / btc_price - 0.04%",
            "target_short":        "edge_short = (btc_price - btc_res) / btc_price - 0.04%",
            "venue":               "futures",
            "round_trip_fee":      ROUND_TRIP,
            "min_edge":            MIN_EDGE,
            "walk_forward": {
                "n_folds":       len(folds),
                "avg_corr_up": avg_corr_up,
                "avg_corr_down":avg_corr_down,
                "total_pnl":     total_pnl,
            },
        }, f)
    log.info("  Saved model_v5_edge.pkl")

    # Summary
    print("\n" + "=" * 60)
    print("  V5 FUTURES SUMMARY")
    print("=" * 60)
    print(f"  Venue:       Futures (BTC perpetual)")
    print(f"  Features:    {len(fn)} (incl. Polymarket sentiment)")
    print(f"  Labels:      BTC price movement ± {ROUND_TRIP*100:.2f}% fees")
    print(f"  Corr LONG:   {avg_corr_up:+.4f}")
    print(f"  Corr SHORT:  {avg_corr_down:+.4f}")
    print(f"  Total PnL:   {total_pnl:+.6f}")
    print(f"  Side acc:    {avg_side*100:.1f}%")
    ready = avg_corr_up > 0 and avg_corr_down > 0
    print(f"  Status:      {'READY' if ready else 'NOT READY — correlations still negative'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
