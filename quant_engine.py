"""
Quant Engine v5.0
==================
BTC perpetual futures trading engine. Executes on dYdX v4 (Cosmos chain).
Uses Polymarket 5-minute BTC markets as timing signal and sentiment source.

Models:
  v4 Primary   — P(profitable) classifier. Production model for trade timing.
                  Scores every tick, trades when score > 0.60.
  v5 Futures   — E(edge_long), E(edge_short) regressors. Future upgrade.
                  Predicts BTC price movement directly. Not yet production.

Execution:
  - Venue: dYdX v4 BTC-USD perpetual (decentralized, no geo-blocking)
  - Entry: open LONG or SHORT when V4 score > threshold
  - Exit: close position when Polymarket market window expires (~5 min)
  - Sizing: flat $10 per trade (V5 will enable proportional sizing)

Data sources (7 real-time WebSockets):
  - BTC spot:       Coinbase WebSocket (sub-second)
  - Order book:     Binance futures depth WebSocket (100ms updates)
  - Tick flow:      Binance aggTrades WebSocket (every trade)
  - Liquidations:   Binance WebSocket (real-time)
  - Chainlink:      Polymarket RTDS WebSocket (resolution price)
  - Poly flow:      Polymarket CLOB WebSocket (trade imbalance, sentiment)
  - REST:           OKX funding/basis/volume, Deribit IV (slower refresh)

Pipeline:
  1. Collect 75+ features every ~2.7s via hybrid snapshot trigger
  2. Score with LightGBM V4 (walk-forward validated, 4/4 kill tests pass)
  3. Execute on dYdX when score > threshold (paper or live mode)
  4. Write market_snapshots to Supabase (resolved by resolver.py)
  5. Retrain daily — V4 for production, V5 futures in background
"""

import os
import time
import logging
import json
import sys
import io
import uuid
import threading
import smtplib
import email.mime.text
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import math
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import websocket
from execution import DydxExecutor

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multi_strategy.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ CONFIG ---

POLYMARKET_HOST  = "https://clob.polymarket.com"
GAMMA_API        = "https://gamma-api.polymarket.com"
OKX_BASE         = "https://www.okx.com"
BTC_SYMBOL       = "BTCUSDT"
ETH_SYMBOL       = "ETHUSDT"
OKX_BTC          = "BTC-USDT-SWAP"

# ── Unified cost model ──
# All fee calculations use these constants. One source of truth.
DYDX_TAKER_FEE   = 0.0005   # 0.05% per side on dYdX
DYDX_ROUND_TRIP   = DYDX_TAKER_FEE * 2  # 0.10% round trip
POLY_TAKER_FEE   = 0.0025   # 0.25% per side on Polymarket (kept for training labels)
TAKER_FEE        = POLY_TAKER_FEE  # backwards compat for training script
STARTING_BALANCE = 100.0
MIN_BET          = 3.0
POLL_SEC         = 2

# Kelly criterion sizing config
SKIP_HOURS_UTC   = {}

# ── ML Quant-only trading config ──
# The model outputs P(profitable on best side) — a RANKING signal, not a
# calibrated probability. Higher score = more likely there exists an edge.
# It does NOT predict direction or EV. Direction comes from a separate heuristic.
ML_QUANT_SCORE_THRESHOLD = 0.60   # minimum score to consider (ranking cutoff)
ML_QUANT_LOG_INTERVAL    = 30     # seconds between signal_log writes per market
ML_QUANT_MAX_CONCURRENT  = 1      # max trades at once (take top-N by score)



SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


def _make_session() -> requests.Session:
    """Shared session with retry + exponential backoff for all HTTP calls."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,          # 0.5s, 1s, 2s between retries
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET", "POST", "PATCH"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

_session = _make_session()


# --------------------------------------------------------- SUPABASE HELPERS --

_SB_PAGE_SIZE = 1000


def _sb_fetch_all(table: str, params: dict) -> list:
    """
    Paginate through Supabase results using Content-Range header.
    Supabase returns 'Content-Range: 0-999/2345' — we iterate until all
    rows are fetched rather than silently truncating at page size.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    rows   = []
    offset = 0
    while True:
        range_header = f"{offset}-{offset + _SB_PAGE_SIZE - 1}"
        try:
            resp = _session.get(
                f"{SUPABASE_URL}/rest/v1/{table}",
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type":  "application/json",
                    "Range":         range_header,
                },
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            rows.extend(page)
            content_range = resp.headers.get("Content-Range", "")
            if "/" in content_range:
                try:
                    total = int(content_range.split("/")[1])
                    if offset + _SB_PAGE_SIZE >= total:
                        break
                except ValueError:
                    break
            else:
                if len(page) < _SB_PAGE_SIZE:
                    break
            offset += _SB_PAGE_SIZE
        except Exception as e:
            log.warning(f"Supabase paginated fetch failed ({table}, offset={offset}): {e}")
            break
    return rows


def supabase_market_snapshot(snapshot: dict):
    """Insert one snapshot row — the primary ML training table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        r = _session.post(
            f"{SUPABASE_URL}/rest/v1/market_snapshots",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=snapshot,
            timeout=5,
        )
        if r.status_code >= 400:
            log.warning(f"market_snapshot insert HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.warning(f"market_snapshot insert failed: {e}")


# --------------------------------------------------------------- DATACLASSES --

@dataclass
class Market:
    condition_id:  str
    up_token_id:   str
    down_token_id: str
    question:      str
    end_time:      datetime


# ── Shared data caches (updated by refresh_shared_data / WebSockets) ─────
_price_cache:    dict  = {"btc": 0.0, "eth": 0.0, "fetched_at": 0.0}
_poly_cache:     dict  = {"up_mid": 0.5, "fill_up": 0.5, "fill_down": 0.5, "spread": 0.02, "fetched_at": 0.0}
_funding_cache:  dict  = {"okx": 0.0, "binance": 0.0, "rate": 0.0, "fetched_at": 0.0}
_iv_cache:       dict  = {
    "atm_iv": 0.0, "skew_25d": 0.0, "iv_rank": 0.5,
    "iv_history": [],       # rolling 30d IV for percentile
    "iv_30d_high": 0.0, "iv_30d_low": 999.0,
    "fetched_at": 0.0,
}
_basis_cache:    dict  = {"spot": 0.0, "futures": 0.0, "fetched_at": 0.0}
_liq_cache:      dict  = {"long": 0.0, "short": 0.0, "fetched_at": 0.0}
_vol_cache:      dict  = {"range_pct": 0.0, "fetched_at": 0.0}
_ob_cache:       dict  = {"imbalance": 0.0, "bid_depth": 0.0, "ask_depth": 0.0,
                           "fetched_at": 0.0}
_oi_cache:       dict  = {"open_interest": 0.0, "oi_change_5m": 0.0, "fetched_at": 0.0}
_lsr_cache:      dict  = {"long_short_ratio": 1.0, "long_account_pct": 0.5,
                           "short_account_pct": 0.5, "fetched_at": 0.0}


def _cache_ages() -> dict:
    """Return age-in-seconds for each shared data cache at the moment of call.
    Used to annotate snapshots and decisions so the model can learn which
    data was fresh vs stale when a given signal fired or was skipped."""
    now = time.time()
    return {
        "age_btc_secs":     round(now - _price_cache.get("fetched_at",   0.0), 1),
        "age_ob_secs":      round(now - _ob_cache.get("fetched_at",      0.0), 1),
        "age_funding_secs": round(now - _funding_cache.get("fetched_at", 0.0), 1),
        "age_liq_secs":     round(now - _liq_cache.get("fetched_at",     0.0), 1),
        "age_poly_secs":    round(now - _poly_cache.get("fetched_at",    0.0), 1),
        "age_vol_secs":     round(now - _vol_cache.get("fetched_at",     0.0), 1),
    }


# ─────────────────────────────────────────── ML GATE (Option A) ────────────
# Model scores every market each tick. If P(profitable) < threshold, open()
# is blocked. Uses 1-tick lag (score computed after snapshot variables ready).
import pickle as _pickle

# Pre-check: can lightgbm load? Test in subprocess to avoid segfault killing main process.
_lgbm_available = False
try:
    import subprocess as _sp
    _rc = _sp.call([sys.executable, "-c", "import lightgbm; print('ok')"],
                   timeout=10, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    if _rc == 0:
        import lightgbm
        _lgbm_available = True
        log.info(f"lightgbm {lightgbm.__version__} available")
    else:
        log.warning(f"lightgbm subprocess test failed (rc={_rc}) — ML scoring disabled")
except Exception as _e:
    log.warning(f"lightgbm check failed: {_e} — ML scoring disabled")

_ml_scores: dict  = {}    # condition_id -> latest P(profitable)
_ml_bundle        = None

if _lgbm_available:
    try:
        with open("model_v4_profitable.pkl", "rb") as _mf:
            _ml_bundle = _pickle.load(_mf)
        log.info(f"ML bundle loaded: {len(_ml_bundle['features'])} features")
    except Exception as _e:
        log.warning(f"ML bundle NOT loaded: {_e}  — gate disabled")
else:
    log.warning("ML bundle NOT loaded: lightgbm unavailable — gate disabled")

# Encoding maps — must match train_model_v4_rest.py exactly
_ML_REGIME_ENC   = {"TREND_UP": 2, "TREND_DOWN": -2, "VOLATILE": 1, "CALM": 0, "DEAD": -1}
_ML_SESSION_ENC  = {"OVERLAP": 3, "US": 2, "LONDON": 1, "ASIA": 0, "OFFPEAK": -1}
_ML_ACTIVITY_ENC = {"HIGH": 2, "NORMAL": 1, "LOW": 0, "DEAD": -1}
_ML_DAY_ENC      = {"WEEKDAY": 1, "WEEKEND": 0}
_ML_BUCKET_ENC   = {"heavy_fav": 3, "favourite": 2, "underdog": 1, "longshot": 0}

# No-pmarket variant (microstructure-only — p_market features excluded)
_ml_bundle_npm = None
if _lgbm_available:
    try:
        with open("model_v4_nopmarket.pkl", "rb") as _mf2:
            _ml_bundle_npm = _pickle.load(_mf2)
        log.info(f"ML nopmarket bundle loaded: {len(_ml_bundle_npm['features'])} features")
    except Exception as _e:
        log.warning(f"ML nopmarket bundle NOT loaded: {_e}")

_ml_scores_npm: dict = {}  # condition_id -> P(profitable) from nopmarket model

# Direction model: P(UP wins | features) — tells you WHICH side to bet
_ml_bundle_dir = None
_ml_scores_dir: dict = {}  # condition_id -> P(UP wins)

# Cross-market state: tracks previous market's outcome and features
_prev_market: dict = {
    "prev_outcome":       0.0,   # NaN until first market resolves
    "prev_momentum":      0.0,
    "prev_ob_imbalance":  0.0,
    "prev_vol_range":     0.0,
    "prev_btc_range":     0.0,
    "prev_p_market_final":0.5,
    "prev_funding_zscore":0.0,
    "streak_up":          0.0,
    "streak_down":        0.0,
}
if _lgbm_available:
    try:
        with open("model_v4_direction.pkl", "rb") as _mf3:
            _ml_bundle_dir = _pickle.load(_mf3)
        log.info(f"ML direction bundle loaded: {len(_ml_bundle_dir['features'])} features")
    except Exception as _e:
        log.warning(f"ML direction bundle NOT loaded: {_e}")


def _update_ml_score(condition_id: str, secs_to_res: float, market_progress: float,
                     phase_early: int, phase_mid: int, phase_late: int, phase_final: int,
                     momentum_10s: float, momentum_30s: float, momentum_60s: float,
                     momentum_120s: float, liq_imbalance: float, liq_total: float,
                     liq_long: float, liq_short: float,
                     price_vs_open_pct: float, price_vs_open_score: float,
                     poly_up_mid: float, poly_spread: float, poly_slip_up: float,
                     price_bucket: str, cl_div: float, cl_age: float,
                     cl_vs_open_pct: float, basis: float,
                     ob_bid_delta: float, ob_ask_delta: float, buy_ratio: float,
                     p_market_std: float = 0.0,
                     avg_ob_imbalance_abs: float = 0.0,
                     avg_funding_zscore_abs: float = 0.0,
                     avg_momentum_abs: float = 0.0,
                     btc_range_pct: float = 0.0,
                     tick_cvd_30s: float = 0.0,
                     tick_taker_buy_30s: float = 0.5,
                     tick_large_buy_30s: float = 0.0,
                     tick_large_sell_30s: float = 0.0,
                     tick_intensity_30s: float = 0.0,
                     tick_vwap_disp_30s: float = 0.0,
                     tick_cvd_60s: float = 0.0,
                     tick_taker_buy_60s: float = 0.5,
                     tick_intensity_60s: float = 0.0,
                     prev_vol_x_momentum: float = 0.0,
                     session_x_vol: float = 0.0,
                     streak_length: float = 0.0,
                     delta_funding: float = 0.0,
                     delta_basis: float = 0.0,
                     delta_trade_imb: float = 0.0,
                     xex_spread: float = 0.0,
                     delta_cvd: float = 0.0,
                     delta_taker_buy: float = 0.0,
                     delta_momentum: float = 0.0,
                     delta_poly: float = 0.0,
                     spot_ob_imbalance: float = 0.0,
                     ob_divergence: float = 0.0,
                     poly_flow_imb: float = 0.0,
                     poly_depth_ratio: float = 0.5,
                     poly_trade_imb: float = 0.0,
                     poly_up_buys: float = 0.0,
                     poly_down_buys: float = 0.0,
                     poly_trade_count: float = 0.0,
                     poly_large_pct: float = 0.0) -> float:
    """
    Build feature vector matching train_model_v4_rest.py exactly, impute, and score.

    Feature parity rules:
    - mom_accel_abs = abs(m30 - m60)  [NOT m30-m10 — matches training]
    - All NaNs imputed via saved classifier_imp before predict_proba
    - Encoding maps must match REGIME_MAP/SESSION_MAP/etc in training script

    Stores result in _ml_scores[condition_id] and _ml_scores_npm[condition_id].
    Returns P(profitable) from primary model, or -1.0 if model unavailable.
    """
    global _ml_scores
    if _ml_bundle is None:
        return -1.0
    try:
        pm = poly_up_mid or 0.5
        pm_abs_dev    = abs(pm - 0.5)
        pm_uncertainty = 1.0 - pm_abs_dev * 2
        is_extreme    = float(pm < 0.25 or pm > 0.75)
        m10, m30, m60, m120 = momentum_10s, momentum_30s, momentum_60s, momentum_120s
        str_   = float(secs_to_res)
        mp     = float(market_progress)
        pvop   = float(price_vs_open_pct)
        li     = float(liq_imbalance)
        lt     = min(float(liq_total), 5e6)
        obi    = float(_ob_cache.get("imbalance", 0.0))
        vr     = float(_vol_cache.get("range_pct", 0.0))
        ps     = float(poly_spread)
        sl2    = float(poly_slip_up)
        fr     = float(_funding_cache.get("rate", 0.0))
        fz     = float(_regime.get("funding_zscore", 0.0) or 0.0)
        sig_vals = [abs(li), abs(obi), min(abs(m30) * 10, 1.0)]
        f = {
            "secs_to_resolution":      str_,
            "log_secs_to_resolution":  math.log1p(str_),
            "market_progress":         mp,
            "phase_early":             float(phase_early),
            "phase_mid":               float(phase_mid),
            "phase_late":              float(phase_late),
            "phase_final":             float(phase_final),
            "hour_sin":                float(_regime.get("hour_sin", 0.0) or 0.0),
            "hour_cos":                float(_regime.get("hour_cos", 0.0) or 0.0),
            "dow_sin":                 float(_regime.get("dow_sin", 0.0) or 0.0),
            "dow_cos":                 float(_regime.get("dow_cos", 0.0) or 0.0),
            "pm_abs_deviation":        pm_abs_dev,
            "pm_uncertainty":          pm_uncertainty,
            "is_extreme_market":       is_extreme,
            "price_vs_open_pct":       pvop,
            "price_vs_open_score":     float(price_vs_open_score or 0.0),
            "momentum_10s":            m10,
            "momentum_30s":            m30,
            "momentum_60s":            m60,
            "momentum_120s":           m120,
            "momentum_score":          float(_regime.get("momentum_score", 0.0) or 0.0),
            "mom_accel_short":         m30 - m10,
            "mom_accel_mid":           m60 - m30,
            "mom_accel_long":          m120 - m60,
            "mom_windows_positive":    float(sum(1 for v in [m10, m30, m60, m120] if v > 0)),
            "mom_all_agree":           1.0 if (all(v > 0 for v in [m10,m30,m60,m120]) or
                                               all(v < 0 for v in [m10,m30,m60,m120])) else 0.0,
            "mom_mean":                (m10 + m30 + m60 + m120) / 4,
            "mom_anchor_div":          m30 - pvop,
            "mom_abs":                 abs(m30),
            "cl_divergence":           float(cl_div or 0.0),
            "cl_age":                  float(cl_age or 0.0),
            "cl_vs_open_pct":          float(cl_vs_open_pct or 0.0),
            "cl_abs_divergence":       abs(float(cl_div or 0.0)),
            "liq_imbalance":           li,
            "liq_total":               lt,
            "log_liq_total":           math.log1p(lt) if lt >= 0 else 0.0,
            "liq_abs_imbalance":       abs(li),
            "liq_long_usd":            min(float(liq_long), 5e6),
            "liq_short_usd":           min(float(liq_short), 5e6),
            "liq_dominant_ratio":      max(float(liq_long), float(liq_short)) / max(float(liq_long) + float(liq_short), 1.0),
            "ob_imbalance":            obi,
            "ob_bid_delta":            float(ob_bid_delta),
            "ob_ask_delta":            float(ob_ask_delta),
            "ob_abs_imbalance":        abs(obi),
            "vol_range_pct":           vr,
            "volatility_pct":          float(_regime.get("volatility_pct", 0.0) or 0.0),
            "volume_buy_ratio":        float(buy_ratio or 0.5),
            "poly_spread":             ps,
            "poly_slip_up":            sl2,
            "effective_entry_cost":    ps + sl2,
            "round_trip_cost":         ps + sl2 + TAKER_FEE * 2,
            "basis_pct":               float(basis or 0.0),
            "funding_rate":            fr,
            "funding_zscore":          fz,
            "okx_funding":             float(_funding_cache.get("okx", 0.0)),
            "gate_funding":            float(_funding_cache.get("binance", 0.0)),
            "funding_abs":             abs(fr),
            "funding_divergence":      float(_funding_cache.get("okx", 0.0)) - float(_funding_cache.get("binance", 0.0)),
            "prev_market_error":       abs(float(_prev_market.get("prev_p_market_final", 0.5)) - float(_prev_market.get("prev_outcome", 0.5))),
            "flow_score":              float(_regime.get("flow_score", 0.0) or 0.0),
            "regime_enc":              _ML_REGIME_ENC.get(_regime.get("composite", ""), 0),
            "session_enc":             _ML_SESSION_ENC.get(_regime.get("session", ""), 0),
            "activity_enc":            _ML_ACTIVITY_ENC.get(_regime.get("activity", ""), 0),
            "day_type_enc":            _ML_DAY_ENC.get(_regime.get("day_type", ""), 0),
            "bucket_enc":              _ML_BUCKET_ENC.get(price_bucket, 1),
            "interact_mom_x_vol":      m30 * vr,
            "interact_liq_x_price":    li * pvop,
            "interact_mom_x_progress": m30 * mp,
            "interact_ob_x_spread":    obi * ps,
            "signal_strength":         float(sum(1 for v in sig_vals if v > 0.1)),
            "vol_x_pm_abs_dev":        vr * pm_abs_dev,
            "vol_x_funding_zscore":    vr * fz,
            "okx_x_vol_fz":            float(_funding_cache.get("okx", 0.0)) * vr * fz,
            "vr_x_fz_sq":              vr * (fz ** 2),
            "okx_x_fr":                float(_funding_cache.get("okx", 0.0)) * fr,
            "liq_imbal_x_secs":        li * str_,
            "mom_x_secs":              m30 * str_,
            "ob_imbal_x_secs":         obi * str_,
            "mom_liq_agree":           (1.0 if (m30 > 0 and li > 0) or (m30 < 0 and li < 0)
                                        else (-1.0 if m30 != 0 and li != 0 else 0.0)),
            "mom_ob_agree":            (1.0 if (m30 > 0 and obi > 0) or (m30 < 0 and obi < 0)
                                        else (-1.0 if m30 != 0 and obi != 0 else 0.0)),
            "liq_ob_agree":            (1.0 if (li > 0 and obi > 0) or (li < 0 and obi < 0)
                                        else (-1.0 if li != 0 and obi != 0 else 0.0)),
            "signal_dispersion":       float(np.std(sig_vals)),
            "mom_accel_abs":           abs(m30 - m60),  # matches training: abs(m30 - m60)
            "p_market_std":            float(p_market_std or 0.0),
            "avg_ob_imbalance_abs":    float(avg_ob_imbalance_abs),
            "avg_funding_zscore_abs":  float(avg_funding_zscore_abs),
            "avg_momentum_abs":        float(avg_momentum_abs),
            "btc_range_pct":           float(btc_range_pct),
            # Cross-market features
            **{k: float(v) for k, v in _prev_market.items()},
            # Spot vs futures divergence
            "spot_ob_imbalance":      float(spot_ob_imbalance),
            "ob_divergence":          float(ob_divergence),
            # Polymarket order flow
            "poly_flow_imb":          float(poly_flow_imb),
            "poly_depth_ratio":       float(poly_depth_ratio),
            "poly_trade_imb":         float(poly_trade_imb),
            "poly_up_buys":           float(poly_up_buys),
            "poly_down_buys":         float(poly_down_buys),
            "poly_trade_count":       float(poly_trade_count),
            "poly_large_pct":         float(poly_large_pct),
            # Regime interaction features
            "prev_vol_x_momentum":    float(_prev_market.get("prev_vol_range", 0)) * float(_prev_market.get("prev_momentum", 0)),
            "session_x_vol":          float(_ML_SESSION_ENC.get(_regime.get("session",""), 0)) * vr,
            "streak_length":          max(float(_prev_market.get("streak_up", 0)), float(_prev_market.get("streak_down", 0))),
            # Velocity features (rate of change)
            "delta_funding":          float(delta_funding),
            "delta_basis":            float(delta_basis),
            "delta_trade_imb":        float(delta_trade_imb),
            "xex_spread":             float(xex_spread),
            # Intra-market deltas (how features changed since last snapshot)
            "delta_cvd":              float(delta_cvd),
            "delta_taker_buy":        float(delta_taker_buy),
            "delta_momentum":         float(delta_momentum),
            "delta_poly":             float(delta_poly),
            "delta_score":            0.0,  # computed after scoring, always 0 for current tick
            # Tick-level order flow (from Binance aggTrades WebSocket)
            "tick_cvd_30s":           float(tick_cvd_30s),
            "tick_taker_buy_ratio_30s": float(tick_taker_buy_30s),
            "tick_large_buy_usd_30s": float(tick_large_buy_30s),
            "tick_large_sell_usd_30s":float(tick_large_sell_30s),
            "tick_intensity_30s":     float(tick_intensity_30s),
            "tick_vwap_disp_30s":     float(tick_vwap_disp_30s),
            "tick_cvd_60s":           float(tick_cvd_60s),
            "tick_taker_buy_ratio_60s": float(tick_taker_buy_60s),
            "tick_intensity_60s":     float(tick_intensity_60s),
        }
        fn = _ml_bundle["features"]
        missing = [k for k in fn if k not in f]
        if missing:
            log.warning(f"ML gate: {len(missing)} missing features {missing[:5]} — score unreliable")
        X  = np.array([[f.get(k, float('nan')) for k in fn]], dtype=np.float32)
        # Apply saved imputer before scoring — must match training pipeline exactly
        X  = _ml_bundle["classifier_imp"].transform(X)
        p  = float(_ml_bundle["classifier"].predict_proba(X)[0][1])
        _ml_scores[condition_id] = p

        # Score direction model: P(UP wins) — same features as primary
        if _ml_bundle_dir is not None:
            try:
                fn_dir = _ml_bundle_dir["features"]
                X_dir  = np.array([[f.get(k, float('nan')) for k in fn_dir]], dtype=np.float32)
                X_dir  = _ml_bundle_dir["classifier_imp"].transform(X_dir)
                p_up   = float(_ml_bundle_dir["classifier"].predict_proba(X_dir)[0][1])
                _ml_scores_dir[condition_id] = p_up
            except Exception as e2:
                log.debug(f"ML direction score error: {e2}")

        return p
    except Exception as e:
        log.warning(f"[ML] SCORING FAILED: {e}")
        return -1.0


_regime:         dict  = {
    # Market microstructure regime — recomputed every poll cycle
    "momentum_label":   "NEUTRAL",  # TREND_UP | TREND_DOWN | NEUTRAL
    "volatility_label": "NORMAL",   # VOLATILE | NORMAL | DEAD
    "flow_label":       "BALANCED", # LONG_CROWDED | SHORT_CROWDED | BALANCED
    "composite":        "CALM",     # TREND_UP | TREND_DOWN | CALM | VOLATILE | DEAD
    # Continuous scores — smooth versions for ML (no information loss at boundaries)
    "momentum_score":   0.0,        # -1 to +1, tanh-scaled 30s BTC momentum
    "volatility_pct":   0.5,        # 0 to 1, percentile of vol vs recent 24h history
    "flow_score":       0.0,        # -1 to +1, combined directional flow pressure
    "liquidity_score":  1.0,        # 0 to 1, higher = more liquid / tighter spread
    "funding_zscore":   0.0,        # z-score of funding vs recent 24h history
    # Cyclical time encoding — preserves continuity across midnight/week boundaries
    "hour_sin":         0.0,        # sin(2π * hour/24)
    "hour_cos":         1.0,        # cos(2π * hour/24)
    "dow_sin":          0.0,        # sin(2π * weekday/7)
    "dow_cos":          1.0,        # cos(2π * weekday/7)
    # Calendar labels (kept for human readability and categorical ML features)
    "session":          "UNKNOWN",  # ASIA | LONDON | OVERLAP | US | OFFPEAK
    "day_type":         "WEEKDAY",  # WEEKDAY | WEEKEND
    "activity":         "NORMAL",   # HIGH | NORMAL | LOW | DEAD
}
# Rolling 24h histories for percentile / z-score computation (~288 samples at 5min intervals)
_vol_history_pct:     deque = deque(maxlen=288)
_funding_history_pct: deque = deque(maxlen=288)
_btc_history:    deque = deque(maxlen=1000)  # ~120s+ of BTC prices (Coinbase WS ~5-10 ticks/s)
                                           # 30 entries needed for momentum_120s lookback
_volume_history: deque = deque(maxlen=25)
_volume_fetched: float = 0.0


def _fetch_deribit_iv():
    """
    Fetch BTC implied volatility from Deribit public API.
    No API key required. Runs every 5 minutes inside refresh_shared_data().

    Stores in _iv_cache:
      atm_iv   — ATM IV for nearest weekly expiry (annualised %)
      skew_25d — 25-delta skew: put_iv - call_iv
                 negative = puts expensive = market fears downside
      iv_rank  — IV percentile vs rolling 30d range (0=low IV, 1=high IV)

    High iv_rank → volatile regime likely → Liquidation Cascade fires better
    Negative skew → sophisticated money pricing downside risk
    Low iv_rank  → calm market → reduce position sizes
    """
    global _iv_cache
    try:
        # BTC index price
        r = _session.get(
            "https://www.deribit.com/api/v2/public/get_index_price",
            params={"index_name": "btc_usd"}, timeout=5)
        rj = r.json()
        if "result" not in rj:
            log.warning(f"[Deribit IV] unexpected response: {str(rj)[:200]}")
            return
        btc_price = rj["result"]["index_price"]

        # All active BTC option instruments
        r2 = _session.get(
            "https://www.deribit.com/api/v2/public/get_instruments",
            params={"currency": "BTC", "kind": "option", "expired": "false"},
            timeout=5)
        rj2 = r2.json()
        if "result" not in rj2:
            log.warning(f"[Deribit IV] instruments response: {str(rj2)[:200]}")
            return
        instruments = rj2["result"]

        # Nearest expiry at least 24h away
        now_ts = time.time()
        expiries = sorted(set(
            i["expiration_timestamp"] / 1000 for i in instruments))
        future = [e for e in expiries if e > now_ts + 86400]
        if not future:
            return
        exp = future[0]

        calls = [i for i in instruments
                 if i["expiration_timestamp"] / 1000 == exp
                 and i["option_type"] == "call"]
        puts  = [i for i in instruments
                 if i["expiration_timestamp"] / 1000 == exp
                 and i["option_type"] == "put"]
        if not calls or not puts:
            return

        # ATM strike = closest to spot
        atm_call = min(calls, key=lambda x: abs(x["strike"] - btc_price))
        atm_put  = min(puts,  key=lambda x: abs(x["strike"] - btc_price))
        # 25-delta strikes ≈ ±8% from ATM
        c25 = min(calls, key=lambda x: abs(x["strike"] - btc_price * 1.08))
        p25 = min(puts,  key=lambda x: abs(x["strike"] - btc_price * 0.92))

        def _mark_iv(name):
            try:
                r = _session.get(
                    "https://www.deribit.com/api/v2/public/ticker",
                    params={"instrument_name": name}, timeout=5)
                return float(r.json()["result"].get("mark_iv", 0.0))
            except Exception:
                return 0.0

        iv_ac = _mark_iv(atm_call["instrument_name"])
        iv_ap = _mark_iv(atm_put["instrument_name"])
        iv_c25 = _mark_iv(c25["instrument_name"])
        iv_p25 = _mark_iv(p25["instrument_name"])

        if iv_ac == 0 or iv_ap == 0:
            return

        atm_iv   = (iv_ac + iv_ap) / 2.0
        skew_25d = iv_p25 - iv_c25  # negative = bearish fear premium

        # Rolling 30d range for IV rank
        if atm_iv > _iv_cache["iv_30d_high"]:
            _iv_cache["iv_30d_high"] = atm_iv
        if 0 < atm_iv < _iv_cache["iv_30d_low"]:
            _iv_cache["iv_30d_low"] = atm_iv
        iv_range = _iv_cache["iv_30d_high"] - _iv_cache["iv_30d_low"]
        iv_rank  = ((atm_iv - _iv_cache["iv_30d_low"]) / iv_range
                    if iv_range > 1.0 else 0.5)

        _iv_cache["atm_iv"]     = round(float(atm_iv), 2)
        _iv_cache["skew_25d"]   = round(float(skew_25d), 2)
        _iv_cache["iv_rank"]    = round(float(min(max(iv_rank, 0.0), 1.0)), 4)
        _iv_cache["fetched_at"] = now_ts

        log.info(f"Deribit IV — ATM: {atm_iv:.1f}%  "
                 f"skew: {skew_25d:+.1f}%  rank: {iv_rank:.2f}")

    except Exception as e:
        log.warning(f"[Deribit IV] fetch failed: {e}")


# ── Coinbase WebSocket — real-time BTC spot price ────────────────────────
_COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
_coinbase_ws_connected = False


def _on_coinbase_message(ws, message):
    try:
        d = json.loads(message)
        if d.get("type") != "ticker":
            return
        price = float(d["price"])
        now = time.time()
        _price_cache["btc"] = price
        _price_cache["fetched_at"] = now
        _btc_history.append({"price": price, "ts": now})
    except Exception:
        pass


def _on_coinbase_open(ws):
    global _coinbase_ws_connected
    _coinbase_ws_connected = True
    sub = json.dumps({
        "type": "subscribe",
        "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}]
    })
    ws.send(sub)
    log.info("[Coinbase WS] Connected — subscribed to BTC-USD ticker")


def _on_coinbase_error(ws, error):
    log.warning(f"[Coinbase WS] error: {error}")


def _on_coinbase_close(ws, close_status_code, close_msg):
    global _coinbase_ws_connected
    _coinbase_ws_connected = False
    log.warning(f"[Coinbase WS] closed ({close_status_code}) — will reconnect")


def _coinbase_ws_thread():
    while True:
        try:
            ws = websocket.WebSocketApp(
                _COINBASE_WS,
                on_open    = _on_coinbase_open,
                on_message = _on_coinbase_message,
                on_error   = _on_coinbase_error,
                on_close   = _on_coinbase_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.warning(f"[Coinbase WS] thread exception: {e}")
        log.info("[Coinbase WS] Reconnecting in 5s...")
        time.sleep(5)


def start_coinbase_ws():
    t = threading.Thread(target=_coinbase_ws_thread, daemon=True)
    t.start()
    log.info("[Coinbase WS] Background thread started")


def fetch_spot(symbol: str = BTC_SYMBOL) -> Optional[float]:
    """Coinbase WS primary, REST fallback, Kraken last resort."""
    # If WebSocket is feeding real-time data, use it
    if symbol == BTC_SYMBOL and _coinbase_ws_connected and _price_cache["btc"] > 0:
        return _price_cache["btc"]
    coin_map   = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}
    kraken_map = {"BTCUSDT": "XBTUSD",  "ETHUSDT": "ETHUSD"}
    try:
        r = _session.get(
            f"https://api.coinbase.com/v2/prices/{coin_map.get(symbol,'BTC-USD')}/spot",
            timeout=5)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception:
        pass
    try:
        r = _session.get(
            f"https://api.kraken.com/0/public/Ticker?pair={kraken_map.get(symbol,'XBTUSD')}",
            timeout=5)
        r.raise_for_status()
        res = r.json()["result"]
        return float(res[list(res.keys())[0]]["c"][0])
    except Exception:
        return None


def refresh_shared_data():
    """Refresh all shared data using geo-unblocked sources."""
    now = time.time()

    # Spot prices
    if now - _price_cache["fetched_at"] >= POLL_SEC:
        btc = fetch_spot(BTC_SYMBOL)
        eth = fetch_spot(ETH_SYMBOL)
        if btc:
            _price_cache["btc"] = btc
            _btc_history.append({"price": btc, "ts": now})
        if eth: _price_cache["eth"] = eth
        _price_cache["fetched_at"] = now

    # Funding rate — OKX + Binance, both required for signal agreement
    if now - _funding_cache["fetched_at"] >= 60:
        try:
            r = _session.get(
                f"{OKX_BASE}/api/v5/public/funding-rate",
                params={"instId": OKX_BTC},
                timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [{}])[0]
            okx_rate = float(data.get("fundingRate", 0))
            _funding_cache["okx"] = okx_rate
        except Exception as e:
            log.warning(f"OKX funding fetch failed: {e}")

        try:
            r = _session.get(
                "https://api.gateio.ws/api/v4/futures/usdt/contracts/BTC_USDT",
                timeout=5)
            r.raise_for_status()
            gate_rate = float(r.json().get("funding_rate", 0))
            _funding_cache["binance"] = gate_rate  # reusing key for compatibility
        except Exception as e:
            log.warning(f"Gate.io funding fetch failed: {e}")

        # Combined rate = average of OKX + Gate.io
        _funding_cache["rate"]       = (_funding_cache["okx"] + _funding_cache["binance"]) / 2
        _funding_cache["fetched_at"] = now
        log.info(f"Funding — OKX: {_funding_cache['okx']:+.6f} "
                 f"Gate.io: {_funding_cache['binance']:+.6f} "
                 f"avg: {_funding_cache['rate']:+.6f}")

    # Deribit IV — forward-looking vol signal, refresh every 5 min
    if time.time() - _iv_cache["fetched_at"] >= 300:
        _fetch_deribit_iv()

    # OKX open interest — how crowded is the BTC futures market?
    # Rapid OI increase + price stall = potential liquidation cascade
    # (Binance fapi REST is geo-blocked from Railway US; OKX works fine)
    if now - _oi_cache["fetched_at"] >= 30:
        try:
            r = _session.get(
                f"{OKX_BASE}/api/v5/public/open-interest",
                params={"instType": "SWAP", "instId": OKX_BTC}, timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [{}])[0]
            new_oi = float(data.get("oiCcy", 0))  # OI in BTC
            prev_oi = _oi_cache["open_interest"]
            _oi_cache["oi_change_5m"] = round((new_oi - prev_oi) / max(prev_oi, 1.0), 6) if prev_oi > 0 else 0.0
            _oi_cache["open_interest"] = new_oi
            _oi_cache["fetched_at"] = now
        except Exception as e:
            log.warning(f"OKX OI fetch failed: {e}")

    # OKX long/short ratio — contract-level positioning
    # Extreme readings (>2.0 or <0.5) = contrarian signal
    if now - _lsr_cache["fetched_at"] >= 60:
        try:
            r = _session.get(
                f"{OKX_BASE}/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
                params={"instId": OKX_BTC, "period": "5m"}, timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                ratio = float(data[0][1])  # [timestamp, ratio]
                long_pct = round(ratio / (1.0 + ratio), 4)
                short_pct = round(1.0 - long_pct, 4)
                _lsr_cache["long_short_ratio"] = round(ratio, 4)
                _lsr_cache["long_account_pct"] = long_pct
                _lsr_cache["short_account_pct"] = short_pct
                _lsr_cache["fetched_at"] = now
        except Exception as e:
            log.warning(f"OKX LSR fetch failed: {e}")

    # Basis via OKX mark price vs spot
    if now - _basis_cache["fetched_at"] >= 10:
        try:
            r = _session.get(
                f"{OKX_BASE}/api/v5/public/mark-price",
                params={"instType": "SWAP", "instId": OKX_BTC},
                timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [{}])[0]
            _basis_cache["futures"]    = float(data.get("markPx", 0))
            _basis_cache["spot"]       = _price_cache["btc"]
            _basis_cache["fetched_at"] = now
        except Exception as e:
            log.warning(f"Basis fetch failed: {e}")

    # Liquidations via OKX — 2-minute window, refreshed every 60s
    # _liq_cache holds OKX-only values. Binance WebSocket data is combined
    # inside strategy_liquidation_cascade() with a matching 2-minute window.
    if now - _liq_cache["fetched_at"] >= 60:
        try:
            r = _session.get(
                f"{OKX_BASE}/api/v5/public/liquidation-orders",
                params={"instType": "SWAP", "uly": "BTC-USDT",
                        "state": "filled", "limit": "20"},
                timeout=5)
            r.raise_for_status()
            orders    = r.json().get("data", [{}])[0].get("details", [])
            cutoff_ms = (now - 120) * 1000   # 2-minute window matches Binance
            long_liqs = short_liqs = 0.0
            for o in orders:
                if float(o.get("ts", 0)) < cutoff_ms:
                    continue
                usd = float(o.get("sz", 0)) * float(o.get("bkPx", 0))
                if o.get("side") == "buy":  short_liqs += usd
                else:                       long_liqs  += usd
            _liq_cache["long"]       = long_liqs
            _liq_cache["short"]      = short_liqs
            _liq_cache["fetched_at"] = now
        except Exception as e:
            log.warning(f"Liquidation fetch failed: {e}")

    # 5-minute BTC price range for volatility-adjusted liquidation signal
    # Uses OKX 1m klines — high/low over last 5 candles
    global _vol_cache
    if now - _vol_cache["fetched_at"] >= 60:
        try:
            r = _session.get(
                f"{OKX_BASE}/api/v5/market/candles",
                params={"instId": OKX_BTC, "bar": "1m", "limit": "5"},
                timeout=5)
            r.raise_for_status()
            candles = r.json().get("data", [])
            if candles:
                highs = [float(c[2]) for c in candles]
                lows  = [float(c[3]) for c in candles]
                price_range = max(highs) - min(lows)
                mid         = (max(highs) + min(lows)) / 2
                _vol_cache["range_pct"]  = (price_range / mid * 100) if mid > 0 else 0.0
                _vol_cache["fetched_at"] = now
                log.debug(f"5min BTC range: {_vol_cache['range_pct']:.3f}%")
        except Exception as e:
            log.warning(f"Volatility range fetch failed: {e}")

    # Volume via OKX klines
    global _volume_fetched
    if now - _volume_fetched >= 60:
        try:
            r = _session.get(
                f"{OKX_BASE}/api/v5/market/candles",
                params={"instId": OKX_BTC, "bar": "1m", "limit": "22"},
                timeout=8)
            r.raise_for_status()
            candles = r.json().get("data", [])
            _volume_history.clear()
            for c in candles:
                close = float(c[4])
                open_ = float(c[1])
                vol   = float(c[5])
                _volume_history.append({
                    "volume":  vol,
                    "close":   close,
                    "open":    open_,
                    "buy_vol": vol if close >= open_ else 0.0,
                })
            _volume_fetched = now
        except Exception as e:
            log.warning(f"Volume fetch failed: {e}")

    # OB now handled by Binance depth WebSocket (real-time, not polled)

def compute_regime():
    """
    Compute market regime labels once per poll cycle.
    Writes to _regime dict so all strategies read a consistent snapshot
    rather than each computing their own momentum independently.

    Two axes:
      1. Microstructure regime  — what the market is doing RIGHT NOW
      2. Calendar regime        — what session/day we're in (structural baseline)
    """
    now = datetime.now(timezone.utc)

    # ── 1. CALENDAR REGIME ────────────────────────────────────────────────────
    hour    = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    is_weekend = weekday >= 5

    # Trading sessions (UTC)
    if 0 <= hour < 6:
        session = "ASIA" if hour >= 1 else "OFFPEAK"
    elif 6 <= hour < 8:
        session = "LONDON"
    elif 8 <= hour < 12:
        session = "OVERLAP"   # London + early US
    elif 12 <= hour < 17:
        session = "US"
    elif 17 <= hour < 21:
        session = "OFFPEAK"
    else:
        session = "ASIA"      # late UTC = early Asia

    day_type = "WEEKEND" if is_weekend else "WEEKDAY"

    # Activity level based on session + day
    if is_weekend:
        activity = "LOW"
    elif session in ("US", "OVERLAP"):
        activity = "HIGH"
    elif session == "LONDON":
        activity = "NORMAL"
    elif session == "ASIA":
        activity = "NORMAL"
    else:
        activity = "LOW"      # OFFPEAK

    _regime["session"]  = session
    _regime["day_type"] = day_type
    _regime["activity"] = activity

    # ── 2. MICROSTRUCTURE REGIME ──────────────────────────────────────────────
    momentum   = btc_momentum_pct(lookback_secs=30) or 0.0
    vol_range  = _vol_cache.get("range_pct", 0.0)
    ob_imbal   = _ob_cache.get("imbalance", 0.0)
    funding    = _funding_cache.get("rate", 0.0)

    # Momentum label
    if momentum > 0.08 and ob_imbal > 0.10:
        momentum_label = "TREND_UP"
    elif momentum < -0.08 and ob_imbal < -0.10:
        momentum_label = "TREND_DOWN"
    else:
        momentum_label = "NEUTRAL"

    # Volatility label
    if vol_range > 0.25:
        volatility_label = "VOLATILE"
    elif vol_range < 0.04:
        volatility_label = "DEAD"
    else:
        volatility_label = "NORMAL"

    # Flow/crowding label
    if funding > 0.0003 and momentum > 0:
        flow_label = "LONG_CROWDED"
    elif funding < -0.0003 and momentum < 0:
        flow_label = "SHORT_CROWDED"
    else:
        flow_label = "BALANCED"

    # Composite label — single string for logging and strategy gating
    if volatility_label == "DEAD":
        composite = "DEAD"
    elif volatility_label == "VOLATILE" and momentum_label == "NEUTRAL":
        composite = "VOLATILE"
    elif momentum_label == "TREND_UP":
        composite = "TREND_UP"
    elif momentum_label == "TREND_DOWN":
        composite = "TREND_DOWN"
    else:
        composite = "CALM"

    _regime["momentum_label"]   = momentum_label
    _regime["volatility_label"] = volatility_label
    _regime["flow_label"]       = flow_label
    _regime["composite"]        = composite

    # ── 3. CONTINUOUS SCORES ──────────────────────────────────────────────────

    # Momentum score: tanh-scaled so ±0.08% maps to ±0.66, ±0.20% maps to ±0.97
    # Smooth gradient — model sees "how much" not just "which side of threshold"
    momentum_score = round(math.tanh(momentum / 0.10), 4)

    # Volatility percentile: where does today's vol_range sit vs recent 24h history
    _vol_history_pct.append(vol_range)
    if len(_vol_history_pct) >= 10:
        sorted_vols = sorted(_vol_history_pct)
        rank = sum(1 for v in sorted_vols if v <= vol_range)
        volatility_pct = round(rank / len(sorted_vols), 4)
    else:
        volatility_pct = 0.5  # not enough history yet, use neutral prior

    # Flow score: weighted combination of OB imbalance, liquidation imbalance, volume
    binance_liq  = get_binance_liq_2min()
    liq_long     = _liq_cache.get("long", 0.0)  + binance_liq["long"]
    liq_short    = _liq_cache.get("short", 0.0) + binance_liq["short"]
    liq_total    = liq_long + liq_short
    liq_imbal    = (liq_long - liq_short) / liq_total if liq_total > 0 else 0.0

    candles      = list(_volume_history)
    recent       = candles[-3:] if len(candles) >= 3 else []
    total_vol    = sum(c["volume"] for c in recent)
    buy_vol      = sum(c["buy_vol"] for c in recent)
    buy_ratio    = (buy_vol / total_vol) if total_vol > 0 else 0.5
    vol_flow     = (buy_ratio - 0.5) * 2  # -1 to +1

    # Weighted combination: OB imbalance most responsive, liq most impactful, vol confirmatory
    flow_score = round(ob_imbal * 0.40 + liq_imbal * 0.40 + vol_flow * 0.20, 4)

    # Liquidity score: inverse of OB spread — tight spread = liquid market
    ob_spread    = _ob_cache.get("spread_pct", 1.0)
    MAX_SPREAD   = 0.05  # above this = very illiquid
    liquidity_score = round(max(0.0, 1.0 - (ob_spread / MAX_SPREAD)), 4)

    # Funding z-score: how extreme is current funding vs recent history
    _funding_history_pct.append(funding)
    if len(_funding_history_pct) >= 10:
        f_mean = sum(_funding_history_pct) / len(_funding_history_pct)
        f_std  = (sum((x - f_mean) ** 2 for x in _funding_history_pct) / len(_funding_history_pct)) ** 0.5
        funding_zscore = round((funding - f_mean) / f_std, 4) if f_std > 0 else 0.0
    else:
        funding_zscore = 0.0

    # Cyclical time encoding — preserves continuity at midnight and end of week
    hour_sin = round(math.sin(2 * math.pi * now.hour / 24), 4)
    hour_cos = round(math.cos(2 * math.pi * now.hour / 24), 4)
    dow_sin  = round(math.sin(2 * math.pi * now.weekday() / 7), 4)
    dow_cos  = round(math.cos(2 * math.pi * now.weekday() / 7), 4)

    _regime["momentum_score"]  = momentum_score
    _regime["volatility_pct"]  = volatility_pct
    _regime["flow_score"]      = flow_score
    _regime["liquidity_score"] = liquidity_score
    _regime["funding_zscore"]  = funding_zscore
    _regime["hour_sin"]        = hour_sin
    _regime["hour_cos"]        = hour_cos
    _regime["dow_sin"]         = dow_sin
    _regime["dow_cos"]         = dow_cos

    log.debug(
        f"Regime: {composite} | session={session} ({day_type}/{activity}) | "
        f"momentum={momentum:+.3f}% (score={momentum_score:+.2f}) | "
        f"vol={vol_range:.3f}% (pct={volatility_pct:.2f}) | "
        f"flow={flow_score:+.3f} | liq={liquidity_score:.2f} | "
        f"funding_z={funding_zscore:+.2f}"
    )


# ------------------------------------------------- CHAINLINK WEBSOCKET CACHE --
# Polymarket broadcasts the exact Chainlink price it uses for resolution
# via wss://ws-subscriptions-clob.polymarket.com/ws/ on topic
# crypto_prices_chainlink. We subscribe in a background thread and keep
# the latest price + timestamp in a shared dict.
# fetch_chainlink_price() reads from this cache instead of making HTTP calls.

_chainlink_cache: dict = {
    "price":      None,   # latest BTC/USD from Chainlink via Polymarket RTDS
    "updated_at": None,   # datetime when last updated
}
_POLYMARKET_WS = "wss://ws-live-data.polymarket.com"
_SUBSCRIBE_MSG = json.dumps({
    "action": "subscribe",
    "subscriptions": [
        {
            "topic":   "crypto_prices_chainlink",
            "type":    "*",
            "filters": "{\"symbol\":\"btc/usd\"}"
        }
    ]
})


def _on_ws_message(ws, message):
    global _chainlink_cache
    if not message:
        return
    try:
        data = json.loads(message)
        if data.get("topic") != "crypto_prices_chainlink":
            return
        payload = data.get("payload", {})
        if payload.get("symbol", "").lower() != "btc/usd":
            return
        price = float(payload["value"])
        ts_ms = payload.get("timestamp")
        if ts_ms:
            updated = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        else:
            updated = datetime.now(timezone.utc)
        _chainlink_cache["price"]      = price
        _chainlink_cache["updated_at"] = updated
        log.debug(f"[Chainlink WS] BTC/USD={price} updated_at={updated.isoformat()}")
    except Exception as e:
        log.warning(f"[Chainlink WS] message parse error: {e}")


def _on_ws_open(ws):
    log.info("[Chainlink WS] Connected to Polymarket RTDS — subscribing to btc/usd")
    ws.send(_SUBSCRIBE_MSG)


def _on_ws_error(ws, error):
    log.warning(f"[Chainlink WS] error: {error}")


def _on_ws_close(ws, close_status_code, close_msg):
    log.warning(f"[Chainlink WS] closed ({close_status_code}) — will reconnect")


def _chainlink_ws_thread():
    """Background thread: maintains persistent WebSocket to Polymarket RTDS."""
    while True:
        try:
            ws = websocket.WebSocketApp(
                _POLYMARKET_WS,
                on_open    = _on_ws_open,
                on_message = _on_ws_message,
                on_error   = _on_ws_error,
                on_close   = _on_ws_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.warning(f"[Chainlink WS] thread exception: {e}")
        log.info("[Chainlink WS] Reconnecting in 5s...")
        time.sleep(5)


def start_chainlink_ws():
    """Start the Chainlink WebSocket listener in a daemon background thread."""
    t = threading.Thread(target=_chainlink_ws_thread, daemon=True)
    t.start()
    log.info("[Chainlink WS] Background thread started")


def fetch_chainlink_price() -> tuple:
    """
    Returns (price, age_seconds) from the Polymarket RTDS Chainlink cache.
    This is the exact same BTC/USD feed Polymarket uses to resolve markets.
    Falls back to (None, None) if the WebSocket hasn't received a price yet
    or the last update is stale (> 60s).
    """
    price      = _chainlink_cache.get("price")
    updated_at = _chainlink_cache.get("updated_at")

    if price is None or updated_at is None:
        log.debug("[Chainlink WS] No price yet — WebSocket still connecting")
        return None, None

    age_sec = (datetime.now(timezone.utc) - updated_at).total_seconds()

    if age_sec > 60:
        log.warning(f"[Chainlink WS] Price stale ({age_sec:.0f}s) — possible disconnect")
        return None, None

    return price, age_sec


# ------------------------------------------------ BINANCE LIQUIDATION CACHE --
# Binance processes 3-5x more BTC liquidation volume than OKX.
# We subscribe to the Binance USDM futures forced liquidation stream
# in a background thread and accumulate liq USD over a rolling 5-minute
# window. The OKX REST poll and this WebSocket feed are combined in
# _liq_cache so the liquidation cascade strategy sees the full market picture.

_binance_liq_buffer: deque = deque()   # raw events: {"side": "BUY"|"SELL", "usd": float, "ts": ms}
_binance_liq_lock = threading.Lock()
_BINANCE_LIQ_WS   = "wss://fstream.binance.com/ws/!forceOrder@arr"


def _on_binance_liq_message(ws, message):
    try:
        data  = json.loads(message)
        order = data.get("o", {})
        # Only care about BTCUSDT perpetual
        if order.get("s", "") != "BTCUSDT":
            return
        side   = order.get("S", "")   # "BUY" = short liq, "SELL" = long liq
        qty    = float(order.get("q", 0))
        price  = float(order.get("ap", 0)) or float(order.get("p", 0))
        usd    = qty * price
        ts_ms  = int(order.get("T", time.time() * 1000))
        with _binance_liq_lock:
            _binance_liq_buffer.append({"side": side, "usd": usd, "ts": ts_ms})
            # Keep only last 10 minutes of events to bound memory
            cutoff = (time.time() - 600) * 1000
            while _binance_liq_buffer and _binance_liq_buffer[0]["ts"] < cutoff:
                _binance_liq_buffer.popleft()
        liq_type = "SHORT_LIQ" if side == "BUY" else "LONG_LIQ"
        log.info(f"[Binance Liq] {liq_type} {side} ${usd:,.0f} BTCUSDT")
    except Exception as e:
        log.warning(f"[Binance Liq] message parse error: {e}")


def _on_binance_liq_open(ws):
    log.info("[Binance Liq] Connected to Binance USDM liquidation stream")


def _on_binance_liq_error(ws, error):
    log.warning(f"[Binance Liq] error: {error}")


def _on_binance_liq_close(ws, close_status_code, close_msg):
    log.warning(f"[Binance Liq] closed ({close_status_code}) — will reconnect")


def _binance_liq_ws_thread():
    """Background thread: maintains persistent WebSocket to Binance liquidation feed."""
    while True:
        try:
            ws = websocket.WebSocketApp(
                _BINANCE_LIQ_WS,
                on_open    = _on_binance_liq_open,
                on_message = _on_binance_liq_message,
                on_error   = _on_binance_liq_error,
                on_close   = _on_binance_liq_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.warning(f"[Binance Liq] thread exception: {e}")
        log.info("[Binance Liq] Reconnecting in 5s...")
        time.sleep(5)


def start_binance_liq_ws():
    """Start the Binance liquidation WebSocket listener in a daemon background thread."""
    t = threading.Thread(target=_binance_liq_ws_thread, daemon=True)
    t.start()
    log.info("[Binance Liq] Background thread started")


def get_binance_liq_5min() -> dict:
    """
    Returns combined long/short liquidation USD over the last 5 minutes
    from the Binance feed.
      BUY  side = short squeeze (shorts liquidated) → bullish
      SELL side = long liquidation (longs liquidated) → bearish
    """
    cutoff = (time.time() - 300) * 1000
    long_liqs = short_liqs = 0.0
    with _binance_liq_lock:
        for e in _binance_liq_buffer:
            if e["ts"] < cutoff:
                continue
            if e["side"] == "SELL":   # long liquidated
                long_liqs  += e["usd"]
            else:                     # short liquidated (BUY)
                short_liqs += e["usd"]
    return {"long": long_liqs, "short": short_liqs}


def get_binance_liq_2min() -> dict:
    """
    Returns combined long/short liquidation USD over the last 2 minutes.
    Shorter lookback = more recent signal, price has had less time to react.
    Used by Liquidation Cascade strategy for tighter timing.
    """
    cutoff = (time.time() - 120) * 1000
    long_liqs = short_liqs = 0.0
    with _binance_liq_lock:
        for e in _binance_liq_buffer:
            if e["ts"] < cutoff:
                continue
            if e["side"] == "SELL":
                long_liqs  += e["usd"]
            else:
                short_liqs += e["usd"]
    return {"long": long_liqs, "short": short_liqs}


# ── Binance aggTrades WebSocket — tick-level order flow ──────────────────
# Streams every BTC-USDT trade in real-time. Each trade has:
#   price, quantity, isBuyerMaker (False = taker bought = aggressive buyer)
# We accumulate in a deque and compute flow features every poll cycle.

_BINANCE_TRADES_WS = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
_trades_buffer: deque = deque(maxlen=30000)  # ~2 min at peak activity
_trades_lock = threading.Lock()


def _on_trades_message(ws, message):
    """Handle each aggTrade tick from Binance USDT-M futures."""
    try:
        d = json.loads(message)
        price = float(d["p"])
        qty   = float(d["q"])
        usd   = price * qty
        # isBuyerMaker=True means taker was SELLER (aggressive sell)
        # isBuyerMaker=False means taker was BUYER (aggressive buy)
        is_buy = not d["m"]
        ts_ms  = int(d["T"])

        with _trades_lock:
            _trades_buffer.append({
                "ts": ts_ms, "price": price, "qty": qty,
                "usd": usd, "is_buy": is_buy,
            })
            # Trim to last 2 minutes
            cutoff = (time.time() - 120) * 1000
            while _trades_buffer and _trades_buffer[0]["ts"] < cutoff:
                _trades_buffer.popleft()
    except Exception:
        pass  # don't log every parse error on high-freq stream


def _on_trades_open(ws):
    log.info("[Binance Trades] Connected to aggTrade stream")


def _on_trades_error(ws, error):
    log.warning(f"[Binance Trades] error: {error}")


def _on_trades_close(ws, close_status_code, close_msg):
    log.warning(f"[Binance Trades] closed ({close_status_code}) — will reconnect")


def _binance_trades_ws_thread():
    """Background thread: persistent WebSocket to Binance aggTrade stream."""
    while True:
        try:
            ws = websocket.WebSocketApp(
                _BINANCE_TRADES_WS,
                on_open    = _on_trades_open,
                on_message = _on_trades_message,
                on_error   = _on_trades_error,
                on_close   = _on_trades_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.warning(f"[Binance Trades] thread exception: {e}")
        log.info("[Binance Trades] Reconnecting in 5s...")
        time.sleep(5)


def start_binance_trades_ws():
    """Start the Binance aggTrade WebSocket in a daemon background thread."""
    t = threading.Thread(target=_binance_trades_ws_thread, daemon=True)
    t.start()
    log.info("[Binance Trades] Background thread started")


# ── Binance Depth WebSocket — real-time order book ───────────────────────
# Replaces REST polling for OB data. Updates _ob_cache in real-time.
# Uses the @depth20@100ms stream — top 20 levels, 100ms updates.

_BINANCE_DEPTH_WS = "wss://fstream.binance.com/ws/btcusdt@depth20@100ms"
_depth_lock = threading.Lock()


def _on_depth_message(ws, message):
    """Handle depth snapshot from Binance USDT-M futures."""
    try:
        d = json.loads(message)
        bids = d.get("b", [])  # [[price, qty], ...]
        asks = d.get("a", [])

        bid_depth = sum(float(b[1]) for b in bids)
        ask_depth = sum(float(a[1]) for a in asks)
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
        spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0.0

        with _depth_lock:
            _ob_cache["imbalance"]  = round(imbalance, 4)
            _ob_cache["bid_depth"]  = round(bid_depth, 2)
            _ob_cache["ask_depth"]  = round(ask_depth, 2)
            _ob_cache["spread_pct"] = round(spread_pct, 4)
            _ob_cache["fetched_at"] = time.time()
    except Exception:
        pass  # don't log on high-freq stream


def _on_depth_open(ws):
    log.info("[Binance Depth] Connected to depth20@100ms stream")


def _on_depth_error(ws, error):
    log.warning(f"[Binance Depth] error: {error}")


def _on_depth_close(ws, close_status_code, close_msg):
    log.warning(f"[Binance Depth] closed ({close_status_code}) — will reconnect")


def _binance_depth_ws_thread():
    """Background thread: persistent WebSocket to Binance depth stream."""
    while True:
        try:
            ws = websocket.WebSocketApp(
                _BINANCE_DEPTH_WS,
                on_open    = _on_depth_open,
                on_message = _on_depth_message,
                on_error   = _on_depth_error,
                on_close   = _on_depth_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.warning(f"[Binance Depth] thread exception: {e}")
        log.info("[Binance Depth] Reconnecting in 5s...")
        time.sleep(5)


def start_binance_depth_ws():
    """Start the Binance depth WebSocket in a daemon background thread."""
    t = threading.Thread(target=_binance_depth_ws_thread, daemon=True)
    t.start()
    log.info("[Binance Depth] Background thread started")


# ── Binance SPOT Depth WebSocket — spot order book for divergence ────────
# Compares spot vs futures OB to detect arbitrage pressure.
# Spot leads futures = directional signal.

_BINANCE_SPOT_DEPTH_WS = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
_spot_ob_cache: dict = {"imbalance": 0.0, "bid_depth": 0.0, "ask_depth": 0.0,
                         "best_bid": 0.0, "best_ask": 0.0, "fetched_at": 0.0}
_spot_depth_lock = threading.Lock()


def _on_spot_depth_message(ws, message):
    try:
        d = json.loads(message)
        # Spot uses "bids"/"asks", futures uses "b"/"a"
        bids = d.get("bids", d.get("b", []))
        asks = d.get("asks", d.get("a", []))

        bid_depth = sum(float(b[1]) for b in bids)
        ask_depth = sum(float(a[1]) for a in asks)
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0

        with _spot_depth_lock:
            _spot_ob_cache["imbalance"]  = round(imbalance, 4)
            _spot_ob_cache["bid_depth"]  = round(bid_depth, 2)
            _spot_ob_cache["ask_depth"]  = round(ask_depth, 2)
            _spot_ob_cache["best_bid"]   = best_bid
            _spot_ob_cache["best_ask"]   = best_ask
            _spot_ob_cache["fetched_at"] = time.time()
    except Exception:
        pass


def _on_spot_depth_open(ws):
    log.info("[Binance Spot Depth] Connected to spot depth20@100ms stream")

def _on_spot_depth_first_msg(ws, message):
    """Log first message to debug format, then switch to normal handler."""
    log.info(f"[Binance Spot Depth] First msg keys: {list(json.loads(message).keys())[:10]}")
    ws.on_message = _on_spot_depth_message
    _on_spot_depth_message(ws, message)


def _on_spot_depth_error(ws, error):
    log.warning(f"[Binance Spot Depth] error: {error}")


def _on_spot_depth_close(ws, close_status_code, close_msg):
    log.warning(f"[Binance Spot Depth] closed ({close_status_code}) — will reconnect")


def _binance_spot_depth_ws_thread():
    while True:
        try:
            ws = websocket.WebSocketApp(
                _BINANCE_SPOT_DEPTH_WS,
                on_open    = _on_spot_depth_open,
                on_message = _on_spot_depth_first_msg,
                on_error   = _on_spot_depth_error,
                on_close   = _on_spot_depth_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.warning(f"[Binance Spot Depth] thread exception: {e}")
        log.info("[Binance Spot Depth] Reconnecting in 5s...")
        time.sleep(5)


def start_binance_spot_depth_ws():
    t = threading.Thread(target=_binance_spot_depth_ws_thread, daemon=True)
    t.start()
    log.info("[Binance Spot Depth] Background thread started")


def get_spot_futures_divergence() -> dict:
    """
    Compute spot vs futures order book divergence.
    Returns:
      spot_ob_imbalance  — spot book imbalance (-1 to +1)
      ob_divergence      — futures_imb - spot_imb (positive = futures more bullish than spot)
      spot_futures_spread — (futures_mid - spot_mid) / spot_mid * 100 (basis from OB)
    """
    futures_imb = _ob_cache.get("imbalance", 0.0)
    spot_imb    = _spot_ob_cache.get("imbalance", 0.0)
    divergence  = round(futures_imb - spot_imb, 4)

    # Spot vs futures mid price spread
    spot_mid = (_spot_ob_cache.get("best_bid", 0) + _spot_ob_cache.get("best_ask", 0)) / 2
    futures_bid = _ob_cache.get("bid_depth", 0)  # these are depths not prices
    # Use price_cache for actual price comparison
    spot_price = _spot_ob_cache.get("best_bid", 0)
    futures_price = _price_cache.get("btc", 0)  # Coinbase price as proxy

    return {
        "spot_ob_imbalance":   round(spot_imb, 4),
        "ob_divergence":       divergence,
        "spot_bid_depth":      round(_spot_ob_cache.get("bid_depth", 0), 2),
        "spot_ask_depth":      round(_spot_ob_cache.get("ask_depth", 0), 2),
    }


def get_tick_features(lookback_secs: float = 30.0) -> dict:
    """
    Compute tick-level order flow features from the aggTrade buffer.

    Returns dict with:
      cvd_30s          — cumulative volume delta (buy_vol - sell_vol) in USD
      taker_buy_ratio  — fraction of volume that was aggressive buying (0-1)
      large_buy_usd    — USD volume from large trades (>$50k) on buy side
      large_sell_usd   — USD volume from large trades (>$50k) on sell side
      trade_intensity  — trades per second
      vwap_displacement — (current_price - VWAP) / current_price * 100
    """
    cutoff = (time.time() - lookback_secs) * 1000
    buy_vol = sell_vol = 0.0
    large_buy = large_sell = 0.0
    price_vol_sum = 0.0
    total_qty = 0.0
    n_trades = 0
    last_price = 0.0

    with _trades_lock:
        for t in _trades_buffer:
            if t["ts"] < cutoff:
                continue
            n_trades += 1
            last_price = t["price"]
            total_qty += t["qty"]
            price_vol_sum += t["price"] * t["qty"]
            if t["is_buy"]:
                buy_vol += t["usd"]
                if t["usd"] > 50000:
                    large_buy += t["usd"]
            else:
                sell_vol += t["usd"]
                if t["usd"] > 50000:
                    large_sell += t["usd"]

    total_vol = buy_vol + sell_vol
    vwap = price_vol_sum / total_qty if total_qty > 0 else last_price
    vwap_disp = (last_price - vwap) / vwap * 100 if vwap > 0 else 0.0

    return {
        "cvd":              round(buy_vol - sell_vol, 2),
        "taker_buy_ratio":  round(buy_vol / total_vol, 4) if total_vol > 0 else 0.5,
        "large_buy_usd":    round(large_buy, 2),
        "large_sell_usd":   round(large_sell, 2),
        "trade_intensity":  round(n_trades / max(lookback_secs, 1), 2),
        "vwap_displacement": round(vwap_disp, 6),
    }


# ── Polymarket CLOB Trade WebSocket — live trade stream ─────────────────
# Streams every trade on the current BTC market's UP and DOWN tokens.
# Gives us: who's buying UP vs DOWN, trade sizes, trade frequency.
# This is the most direct signal — it's the actual Polymarket order flow.

_POLY_TRADES_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_poly_trades_buffer: deque = deque(maxlen=5000)
_poly_trades_lock = threading.Lock()
_poly_ws_instance = None  # track current WS for resubscription
_poly_ws_token_ids: list = []  # current subscribed token IDs


def _on_poly_trade_message(ws, message):
    """Handle trade messages from Polymarket CLOB WebSocket."""
    global _poly_last_trade_ts
    if message == "PONG":
        return
    try:
        raw = json.loads(message)

        # Polymarket sends trades as JSON array OR single object
        if isinstance(raw, list):
            trades = raw
        elif isinstance(raw, dict):
            # Single event object (e.g. last_trade_price, subscription confirmation)
            evt = raw.get("event_type", "")
            if evt and evt != "last_trade_price":
                log.debug(f"[Poly Trades] event: {evt}")
                return
            if not evt:
                log.info(f"[Poly Trades] msg: {str(message)[:200]}")
                return
            trades = [raw]
        else:
            return

        for d in trades:
            asset_id = d.get("asset_id", "")
            price    = float(d.get("price", 0))
            size     = float(d.get("size", 0))
            side     = d.get("side", "")  # BUY or SELL
            ts_ms    = int(d.get("timestamp", time.time() * 1000))

            # Determine if this is an UP or DOWN token trade
            is_up_token = (asset_id == _poly_ws_token_ids[0]) if _poly_ws_token_ids else None

            _poly_last_trade_ts = time.time()

            with _poly_trades_lock:
                _poly_trades_buffer.append({
                    "ts": ts_ms, "price": price, "size": size,
                    "usd": price * size, "side": side,
                    "is_up_token": is_up_token,
                })
                # Trim to last 2 minutes
                cutoff = (time.time() - 120) * 1000
                while _poly_trades_buffer and _poly_trades_buffer[0]["ts"] < cutoff:
                    _poly_trades_buffer.popleft()
    except Exception as e:
        log.warning(f"[Poly Trades] parse error: {e} | msg={str(message)[:200]}")


def _on_poly_trade_open(ws):
    log.info("[Poly Trades] Connected to CLOB market channel")
    # Subscribe to current token IDs
    if _poly_ws_token_ids:
        sub_msg = json.dumps({
            "assets_ids": _poly_ws_token_ids,
            "type": "market",
        })
        ws.send(sub_msg)
        log.info(f"[Poly Trades] Subscribed to {len(_poly_ws_token_ids)} tokens: "
                 f"{_poly_ws_token_ids[0][:16]}...")
    else:
        log.warning("[Poly Trades] WS connected but NO token IDs yet — will subscribe on market fetch")


def _on_poly_trade_error(ws, error):
    log.warning(f"[Poly Trades] error: {error}")


def _on_poly_trade_close(ws, close_status_code, close_msg):
    log.warning(f"[Poly Trades] closed ({close_status_code}) — will reconnect")


_poly_last_trade_ts = 0.0  # track when we last received a trade

def _poly_ping_thread():
    """Send PING string every 10s — Polymarket requires this, not WS protocol ping.
    Also monitors for stale connections: if no trades for 5 min, force reconnect."""
    global _poly_last_trade_ts
    while True:
        time.sleep(10)
        ws = _poly_ws_instance
        if ws and ws.sock and ws.sock.connected:
            try:
                ws.send("PING")
            except Exception:
                pass
            # Force reconnect if no trades for 5 minutes and we have token IDs
            if (_poly_ws_token_ids and _poly_last_trade_ts > 0
                    and time.time() - _poly_last_trade_ts > 300):
                log.warning("[Poly Trades] No trades for 5 min — forcing reconnect")
                _poly_last_trade_ts = time.time()  # reset to avoid spam
                try:
                    ws.close()
                except Exception:
                    pass


def _poly_trades_ws_thread():
    """Background thread: persistent WebSocket to Polymarket CLOB trade stream."""
    global _poly_ws_instance
    # Start ping thread once
    threading.Thread(target=_poly_ping_thread, daemon=True).start()
    while True:
        try:
            ws = websocket.WebSocketApp(
                _POLY_TRADES_WS,
                on_open    = _on_poly_trade_open,
                on_message = _on_poly_trade_message,
                on_error   = _on_poly_trade_error,
                on_close   = _on_poly_trade_close,
            )
            _poly_ws_instance = ws
            ws.run_forever()  # no built-in ping — we send PING string manually
        except Exception as e:
            log.warning(f"[Poly Trades] thread exception: {e}")
        _poly_ws_instance = None
        log.info("[Poly Trades] Reconnecting in 5s...")
        time.sleep(5)


def start_poly_trades_ws():
    """Start the Polymarket CLOB trade WebSocket in a daemon thread."""
    t = threading.Thread(target=_poly_trades_ws_thread, daemon=True)
    t.start()
    log.info("[Poly Trades] Background thread started")


def update_poly_trade_subscription(market: "Market"):
    """Resubscribe to new token IDs when market changes."""
    global _poly_ws_token_ids
    _poly_ws_token_ids = [market.up_token_id, market.down_token_id]
    log.info(f"[Poly Trades] Token IDs: UP={market.up_token_id[:16]}... DOWN={market.down_token_id[:16]}...")
    # Clear buffer on market change
    with _poly_trades_lock:
        _poly_trades_buffer.clear()
    # Force reconnect to subscribe with new token IDs
    # (Polymarket WS doesn't support resubscription on an existing connection)
    if _poly_ws_instance and _poly_ws_instance.sock and _poly_ws_instance.sock.connected:
        log.info("[Poly Trades] Closing WS to reconnect with new tokens...")
        try:
            _poly_ws_instance.close()
        except Exception:
            pass
    else:
        log.info("[Poly Trades] WS not connected — will subscribe on reconnect")


def get_poly_trade_flow(lookback_secs: float = 30.0) -> dict:
    """
    Compute Polymarket-specific trade flow features from the CLOB trade stream.

    Returns:
      poly_trade_imb  — net buy/sell imbalance across UP+DOWN tokens (-1 to +1)
      poly_up_buys    — USD volume of BUY orders on UP token (bullish)
      poly_down_buys  — USD volume of BUY orders on DOWN token (bearish)
      poly_trade_count — number of trades in window
      poly_avg_size   — average trade size in USD
      poly_large_pct  — fraction of volume from trades > $100
    """
    cutoff = (time.time() - lookback_secs) * 1000
    up_buy_vol = up_sell_vol = 0.0
    down_buy_vol = down_sell_vol = 0.0
    n_trades = 0
    large_vol = 0.0
    total_vol = 0.0

    with _poly_trades_lock:
        for t in _poly_trades_buffer:
            if t["ts"] < cutoff:
                continue
            n_trades += 1
            usd = t["usd"]
            total_vol += usd
            if usd > 100:
                large_vol += usd

            if t["is_up_token"] is True:
                if t["side"] == "BUY":
                    up_buy_vol += usd
                else:
                    up_sell_vol += usd
            elif t["is_up_token"] is False:
                if t["side"] == "BUY":
                    down_buy_vol += usd
                else:
                    down_sell_vol += usd

    # Net imbalance: positive = more buying UP / selling DOWN = bullish
    bullish  = up_buy_vol + down_sell_vol
    bearish  = up_sell_vol + down_buy_vol
    total    = bullish + bearish
    trade_imb = round((bullish - bearish) / max(total, 1.0), 4)

    return {
        "poly_trade_imb":    trade_imb,
        "poly_up_buys":      round(up_buy_vol, 2),
        "poly_down_buys":    round(down_buy_vol, 2),
        "poly_trade_count":  n_trades,
        "poly_avg_size":     round(total_vol / max(n_trades, 1), 2),
        "poly_large_pct":    round(large_vol / max(total_vol, 1.0), 4),
    }


def btc_momentum_pct(lookback_secs: float = 30.0) -> Optional[float]:
    """
    Returns BTC price change % over the last `lookback_secs` seconds.
    Uses the rolling _btc_history deque (sampled every POLL_SEC).
    Returns None if not enough history.
    Positive = BTC moved up, Negative = BTC moved down.

    Guard: if the oldest available sample is younger than lookback_secs,
    returns None rather than silently computing a shorter window.
    """
    now = time.time()
    cutoff = now - lookback_secs
    history = list(_btc_history)
    if not history:
        return None
    if history[0]["ts"] > cutoff:
        return None
    old = next((h for h in history if h["ts"] >= cutoff), None)
    current = history[-1]
    if not old or old["price"] == 0:
        return None
    return (current["price"] - old["price"]) / old["price"] * 100

def fetch_current_market(exclude: str = "") -> Optional[Market]:
    """
    Fetch the next active BTC up/down market from Polymarket.

    exclude: condition_id to skip — used when pre-fetching the next market
             while the current one is still running. Prevents the function
             from returning the market we're already tracking.

    Iterates candidate windows in ascending end-time order so we always
    land on the soonest market that isn't excluded and hasn't expired.
    """
    try:
        now        = datetime.now(timezone.utc)
        ts         = int(now.timestamp())
        current_5m = (ts // 300) * 300
        for window_ts in [current_5m + 300, current_5m + 600, current_5m + 900]:
            slug = f"btc-updown-5m-{window_ts}"
            try:
                r = _session.get(f"{GAMMA_API}/markets",
                                 params={"slug": slug}, timeout=10)
                r.raise_for_status()
                markets = r.json()
                if not markets:
                    continue
                m = markets[0]
                if m.get("closed"):
                    continue
                end_time = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
                if end_time <= now:
                    continue
                # Skip the market we're already tracking
                if exclude and m["conditionId"] == exclude:
                    continue
                token_ids = json.loads(m["clobTokenIds"])
                outcomes  = m.get("outcomes")
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if outcomes and len(outcomes) == len(token_ids):
                    outcome_map   = {o.upper(): tid for o, tid in zip(outcomes, token_ids)}
                    up_token_id   = outcome_map.get("UP")
                    down_token_id = outcome_map.get("DOWN")
                else:
                    log.warning("No outcomes field in market response — falling back to index order")
                    up_token_id   = token_ids[0]
                    down_token_id = token_ids[1]
                if not up_token_id or not down_token_id:
                    log.warning(f"Could not map UP/DOWN tokens for {m['conditionId'][:12]}... — skipping")
                    continue
                return Market(
                    condition_id  = m["conditionId"],
                    up_token_id   = up_token_id,
                    down_token_id = down_token_id,
                    question      = m["question"],
                    end_time      = end_time,
                )
            except Exception:
                continue
        return None
    except Exception as e:
        log.error(f"Market fetch failed: {e}")
        return None


def _book_fill_price(book: dict, size_usdc: float) -> tuple:
    """
    Walk the ask ladder to estimate a size-weighted taker fill price.
    Returns (fill_price, slippage_vs_best_ask).
    """
    asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
    bids = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)

    if not asks:
        return 0.5, 0.0

    best_ask = float(asks[0]["price"])

    remaining  = size_usdc
    total_cost = 0.0
    total_qty  = 0.0
    for level in asks:
        px       = float(level["price"])
        qty      = float(level["size"])
        fill_qty = min(qty, remaining / px)
        total_cost += fill_qty * px
        total_qty  += fill_qty
        remaining  -= fill_qty * px
        if remaining <= 0:
            break

    if total_qty == 0:
        return best_ask, 0.0

    fill_price = total_cost / total_qty
    slippage   = fill_price - best_ask
    return round(fill_price, 6), round(slippage, 6)


def fetch_poly_prices(market: Market, size_usdc: float = 50.0) -> dict:
    """
    Returns executable fill prices (ask-side) for both tokens, plus spread.
    fill_up / fill_down are estimated taker prices for a buy of size_usdc.
    up_mid / down_mid are kept for reference/logging (e.g. snapshots) only —
    do not use as entry price.
    """
    try:
        up_book   = _session.get(f"{POLYMARKET_HOST}/book",
                                 params={"token_id": market.up_token_id},   timeout=8).json()
        down_book = _session.get(f"{POLYMARKET_HOST}/book",
                                 params={"token_id": market.down_token_id}, timeout=8).json()

        up_asks   = sorted(up_book.get("asks",   []), key=lambda x: float(x["price"]))
        up_bids   = sorted(up_book.get("bids",   []), key=lambda x: float(x["price"]), reverse=True)
        down_asks = sorted(down_book.get("asks", []), key=lambda x: float(x["price"]))
        down_bids = sorted(down_book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)

        spread = (float(up_asks[0]["price"]) - float(up_bids[0]["price"])
                  ) if up_asks and up_bids else 0.1

        fill_up,   slip_up   = _book_fill_price(up_book,   size_usdc)
        fill_down, slip_down = _book_fill_price(down_book, size_usdc)

        up_mid   = (float(up_asks[0]["price"])   + float(up_bids[0]["price"]))   / 2 \
                   if up_asks and up_bids else 0.5
        down_mid = (float(down_asks[0]["price"]) + float(down_bids[0]["price"])) / 2 \
                   if down_asks and down_bids else 0.5

        if slip_up > 0.005 or slip_down > 0.005:
            log.warning(f"High slippage estimate: UP +{slip_up:.4f} DOWN +{slip_down:.4f}")

        # Poly order book depth — measure directional interest on Polymarket itself
        up_bid_depth  = sum(float(o["size"]) * float(o["price"]) for o in up_bids)
        up_ask_depth  = sum(float(o["size"]) * float(o["price"]) for o in up_asks)
        down_bid_depth = sum(float(o["size"]) * float(o["price"]) for o in down_bids)
        down_ask_depth = sum(float(o["size"]) * float(o["price"]) for o in down_asks)

        # Poly flow features:
        # up_buy_pressure: how much $ is resting to buy UP tokens (bullish interest)
        # down_buy_pressure: how much $ is resting to buy DOWN tokens (bearish interest)
        # poly_flow_imbalance: net directional flow on Polymarket
        #   positive = more $ wants to buy UP than DOWN = bullish
        up_buy_pressure   = round(up_bid_depth, 2)
        down_buy_pressure = round(down_bid_depth, 2)
        total_pressure    = up_buy_pressure + down_buy_pressure
        poly_flow_imb     = round((up_buy_pressure - down_buy_pressure) /
                                   max(total_pressure, 1.0), 4)

        # Poly depth ratio — which side has more liquidity?
        up_total   = up_bid_depth + up_ask_depth
        down_total = down_bid_depth + down_ask_depth
        poly_depth_ratio = round(up_total / max(up_total + down_total, 1.0), 4)

        # Update shared cache
        _poly_cache["up_mid"]     = up_mid
        _poly_cache["fill_up"]    = fill_up
        _poly_cache["fill_down"]  = fill_down
        _poly_cache["spread"]     = spread
        _poly_cache["fetched_at"] = time.time()
        _poly_cache["poly_flow_imb"]     = poly_flow_imb
        _poly_cache["poly_depth_ratio"]  = poly_depth_ratio
        _poly_cache["up_buy_pressure"]   = up_buy_pressure
        _poly_cache["down_buy_pressure"] = down_buy_pressure

        return {
            "up_mid":    up_mid,
            "down_mid":  down_mid,
            "fill_up":   fill_up,
            "fill_down": fill_down,
            "slip_up":   slip_up,
            "slip_down": slip_down,
            "spread":    spread,
            "poly_flow_imb":     poly_flow_imb,
            "poly_depth_ratio":  poly_depth_ratio,
            "up_buy_pressure":   up_buy_pressure,
            "down_buy_pressure": down_buy_pressure,
        }
    except Exception as e:
        log.warning(f"fetch_poly_prices failed: {e}")
        return {
            "up_mid": 0.5, "down_mid": 0.5,
            "fill_up": 0.5, "fill_down": 0.5,
            "slip_up": 0.0, "slip_down": 0.0,
            "spread": 0.1,
        }


# --------------------------------------------------------- RESOLUTION --------

def fetch_market_winner(condition_id: str) -> Optional[str]:
    """
    Polls Gamma API every 5 seconds until a winner is found.
    Gives up after 10 minutes and returns "VOID".
    """
    POLL_INTERVAL    = 5
    MAX_WAIT         = 600
    VOID_THRESHOLD   = 120

    start_time       = time.time()
    zero_price_since = None

    attempt = 0
    while True:
        elapsed = time.time() - start_time
        attempt += 1

        if elapsed > MAX_WAIT:
            log.warning(f"[{condition_id}] No resolution after {MAX_WAIT}s — declaring VOID")
            return "VOID"

        try:
            r = _session.get(
                f"{GAMMA_API}/markets",
                params={"conditionId": condition_id},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()

            if not data:
                log.warning(f"[{condition_id}] Gamma returned empty (attempt {attempt}, "
                            f"{elapsed:.0f}s elapsed) — retrying")
                time.sleep(POLL_INTERVAL)
                continue

            m          = data[0]
            raw_prices = m.get("outcomePrices", '["0","0"]')
            prices     = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            up_price   = float(prices[0])
            down_price = float(prices[1])

            log.info(f"[{condition_id}] attempt={attempt} elapsed={elapsed:.0f}s | "
                     f"closed={m.get('closed')} up={up_price} down={down_price}")

            if up_price > 0.9:
                log.info(f"[{condition_id}] Resolved UP after {elapsed:.0f}s")
                return "UP"
            elif down_price > 0.9:
                log.info(f"[{condition_id}] Resolved DOWN after {elapsed:.0f}s")
                return "DOWN"

            if m.get("closed") and up_price == 0.0 and down_price == 0.0:
                if zero_price_since is None:
                    zero_price_since = time.time()
                zero_duration = time.time() - zero_price_since
                log.info(f"[{condition_id}] Closed with zero prices for {zero_duration:.0f}s")
                if zero_duration >= VOID_THRESHOLD:
                    log.warning(f"[{condition_id}] Prices stayed zero for {zero_duration:.0f}s "
                                f"— declaring VOID")
                    return "VOID"
            else:
                zero_price_since = None

        except Exception as e:
            log.warning(f"[{condition_id}] fetch error (attempt {attempt}, "
                        f"{elapsed:.0f}s elapsed): {e}")

        time.sleep(POLL_INTERVAL)


def _make_market_state() -> dict:
    """
    Returns a fresh per-market state dict.
    Called once on startup and once each time we switch to a new market.
    Keeping state in a dict makes the switch atomic — one assignment replaces all fields.
    """
    return {
        "market":            None,
        # positions dict removed in v4.0 — no strategy-based trading
        "cl_history":        deque(maxlen=6),
        "market_open_price": 0.0,
        "cl_open_price":     0.0,
        "prev_liq_total":    0.0,
        "prev_ob_bid_depth": 0.0,
        "prev_ob_ask_depth": 0.0,
        "max_secs_left":     0.0,
        "liq_total_history": deque(maxlen=6),
        "p_market_history":  deque(maxlen=30),  # for p_market_open, std, vs_open_delta
        "p_market_open":     None,              # first p_market seen this market
        "ml_quant_fired":    False,             # ML Quant dry run — only fire trade once per market
        "ml_quant_last_log": 0.0,              # wall time of last ML_Quant signal_log write
        # Rolling market-level aggregates (mirror of market model features)
        "ob_history":        deque(maxlen=30),  # abs OB imbalance per poll
        "fz_history":        deque(maxlen=30),  # abs funding zscore per poll
        "mom_history":       deque(maxlen=30),  # abs momentum_30s per poll
        "btc_high":          0.0,               # BTC high since market open
        "btc_low":           float("inf"),      # BTC low since market open
        # Snapshot trigger tracking — previous values for change detection
        "snap_prev_btc":     0.0,
        "snap_prev_poly":    0.5,
        "snap_prev_ob":      0.0,
        "snap_prev_liq_imb": 0.0,
        "snap_last_ts":      0.0,   # wall time of last snapshot write
        # Intra-market delta tracking — previous values for computing deltas
        "prev_cvd_30s":      0.0,
        "prev_taker_buy_30s":0.5,
        "prev_momentum_30s": 0.0,
        "prev_ml_score":     0.5,
        "prev_poly_up_mid":  0.5,
        "prev_funding_rate": 0.0,
        "prev_basis_pct":    0.0,
        "prev_trade_imb":    0.0,
        # Sequence tracking — for LSTM-like features
        "prev_delta_cvd":    0.0,
        "prev_delta_mom":    0.0,
        "mom_sign_history":  deque(maxlen=10),   # momentum sign per snapshot
        "ob_sign_history":   deque(maxlen=10),   # OB imbalance sign per snapshot
        # Intra-market memory — backward-looking summary of this market so far
        "snapshots_seen":    0,
        "max_score":         0.0,        # highest ML score seen this market
        "cvd_history":       deque(maxlen=30),  # CVD per snapshot for trend
        "pm_history_vel":    deque(maxlen=10),   # p_market for velocity
    }


# Snapshot trigger thresholds — what counts as a "meaningful change"
_SNAP_BTC_BPS      = 5.0    # BTC move >= 5 bps  (~$5 on $100k BTC)
_SNAP_POLY_TICK    = 0.005  # Poly mid moves >= 0.5 cents
_SNAP_OB_DELTA     = 0.08   # OB imbalance shifts >= 0.08
_SNAP_LIQ_DELTA    = 0.10   # Liq imbalance shifts >= 0.10
_SNAP_TIMER_SECS   = 2      # Write every poll cycle for maximum data accumulation


def _snapshot_trigger(cur: dict, spot: float, poly_mid: float,
                      ob_imb: float, liq_imb: float,
                      force: bool = False) -> tuple:
    """
    Decide whether to write a snapshot this cycle.
    Returns (should_write: bool, reason: str, trigger_feature: str, trigger_magnitude: float).

    Policy (hybrid):
      - force=True        → always write (decision_eval, market_open, market_close)
      - timer             → write if >= _SNAP_TIMER_SECS since last snapshot
      - btc_move          → BTC moved >= _SNAP_BTC_BPS basis points
      - poly_move         → Poly mid moved >= _SNAP_POLY_TICK
      - ob_change         → OB imbalance shifted >= _SNAP_OB_DELTA
      - liq_imb_change    → Liq imbalance shifted >= _SNAP_LIQ_DELTA
    """
    if force:
        return True, "decision_eval", "", 0.0

    now = time.time()
    elapsed = now - cur["snap_last_ts"]

    prev_btc  = cur["snap_prev_btc"]
    prev_poly = cur["snap_prev_poly"]
    prev_ob   = cur["snap_prev_ob"]
    prev_liq  = cur["snap_prev_liq_imb"]

    # BTC move in basis points
    btc_bps = abs(spot - prev_btc) / prev_btc * 10000 if prev_btc > 0 else 0.0
    if btc_bps >= _SNAP_BTC_BPS:
        return True, "btc_move", "btc_price", round(btc_bps, 2)

    poly_delta = abs(poly_mid - prev_poly)
    if poly_delta >= _SNAP_POLY_TICK:
        return True, "poly_move", "p_market", round(poly_delta, 4)

    ob_delta = abs(ob_imb - prev_ob)
    if ob_delta >= _SNAP_OB_DELTA:
        return True, "ob_change", "ob_imbalance", round(ob_delta, 4)

    liq_delta = abs(liq_imb - prev_liq)
    if liq_delta >= _SNAP_LIQ_DELTA:
        return True, "liq_imb_change", "liq_imbalance", round(liq_delta, 4)

    if elapsed >= _SNAP_TIMER_SECS:
        return True, "timer", "", round(elapsed, 1)

    return False, "skip", "", 0.0


_REPORT_EMAIL    = "kingkuon2004@gmail.com"
_SMTP_USER       = os.environ.get("SMTP_USER", "")   # set in Railway env vars
_SMTP_PASS       = os.environ.get("SMTP_PASS", "")   # Gmail app password


def _send_hourly_report() -> None:
    """Send hourly ML quant status email. Silently skips if SMTP creds not set."""
    if not _SMTP_USER or not _SMTP_PASS:
        log.debug("Hourly email skipped — SMTP_USER/SMTP_PASS not set")
        return
    try:
        now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"ML Quant Engine — Hourly Report ({now})", ""]

        # ML model scores
        lines.append("=== ML QUANT SCORES ===")
        lines.append(f"  Score threshold:  {ML_QUANT_SCORE_THRESHOLD}")
        lines.append(f"  Score threshold:  {ML_QUANT_SCORE_THRESHOLD}")
        lines.append(f"  Active markets:   {len(_ml_scores)}")
        if _ml_scores:
            scores = list(_ml_scores.values())
            lines.append(f"  Score range:  min={min(scores):.3f}  max={max(scores):.3f}  avg={sum(scores)/len(scores):.3f}")
        lines.append("")

        body = "\n".join(lines)
        msg  = email.mime.text.MIMEText(body, "plain")
        msg["Subject"] = f"[BTC Bot] Hourly Report — {now}"
        msg["From"]    = _SMTP_USER
        msg["To"]      = _REPORT_EMAIL

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(_SMTP_USER, _SMTP_PASS)
            srv.sendmail(_SMTP_USER, _REPORT_EMAIL, msg.as_string())
        log.info(f"Hourly report sent to {_REPORT_EMAIL}")
    except Exception as e:
        log.warning(f"Hourly email failed: {e}")


def run():
    log.info("Quant Engine v5.0")
    log.info("Data: Coinbase + Kraken + OKX + Binance + Deribit + Polymarket")

    executor = DydxExecutor()

    # Start data feed WebSockets
    start_coinbase_ws()
    start_chainlink_ws()
    start_binance_liq_ws()
    start_binance_trades_ws()
    start_binance_depth_ws()
    # start_binance_spot_depth_ws()  # disabled — Binance spot API geo-blocked on Railway US
    start_poly_trades_ws()
    log.info("Waiting 3s for WebSocket connections to establish...")
    time.sleep(3)

    # ── Dual-market state ─────────────────────────────────────────────────────
    # cur: the market we are actively trading and snapshotting right now.
    # nxt: pre-fetched next market, ready to swap in the instant cur expires.
    #
    # Timeline for a 5-minute market:
    #   T+0s    cur market starts, nxt = None
    #   T+240s  secs_left < PRE_FETCH_SECS → start fetching nxt in background
    #   T+300s  cur expires → swap nxt → cur, reset per-market state, nxt = None
    #
    # This gives us complete phase coverage on both markets:
    #   cur is polled all the way to 0s  (captures phase_late + phase_final)
    #   nxt is ready at T=0 of new market (captures phase_early from poll 1)
    #
    PRE_FETCH_SECS   = 90    # start pre-fetching when this many seconds remain
    next_fetch_tried = False  # guard: only attempt one pre-fetch per market

    cur = _make_market_state()
    nxt: Optional[dict] = None   # pre-fetched next market state (market field only)

    last_summary       = 0
    last_email         = time.time() - 3300  # fire ~5 min after startup

    # Initial market fetch
    cur["market"] = fetch_current_market()
    if cur["market"]:
        log.info(f"Initial market: {cur['market'].question}")
        update_poly_trade_subscription(cur["market"])

    while True:
        try:
            now  = datetime.now(timezone.utc)
            hour = now.hour

            if hour in SKIP_HOURS_UTC:
                log.info(f"Skipping hour {hour:02d}:00 UTC")
                time.sleep(30)
                continue

            refresh_shared_data()
            compute_regime()

            # ── No current market — fetch one ─────────────────────────────────
            if cur["market"] is None:
                if nxt and nxt["market"]:
                    cur  = nxt
                    nxt  = None
                    next_fetch_tried = False
                    log.info(f"Switched to pre-fetched market: {cur['market'].question}")
                    update_poly_trade_subscription(cur["market"])
                else:
                    fetched = fetch_current_market()
                    if fetched:
                        cur["market"] = fetched
                        log.info(f"Fetched market: {fetched.question}")
                        update_poly_trade_subscription(fetched)
                    else:
                        log.warning("No market available — retrying in 15s")
                        time.sleep(15)
                        continue

            secs_left = (cur["market"].end_time - now).total_seconds()

            # ── Pre-fetch next market when approaching end ────────────────────
            # Runs once per market (next_fetch_tried guard) so we don't hammer
            # the API. Fetches in the main thread — the call is fast (<200ms)
            # and we have POLL_SEC budget anyway.
            if (not next_fetch_tried
                    and secs_left <= PRE_FETCH_SECS
                    and secs_left > 0):
                next_fetch_tried = True
                fetched_next = fetch_current_market(exclude=cur["market"].condition_id)
                if fetched_next:
                    nxt = _make_market_state()
                    nxt["market"] = fetched_next
                    log.info(f"Pre-fetched next market: {fetched_next.question} "
                             f"(current has {secs_left:.0f}s left)")
                else:
                    log.warning(f"Pre-fetch failed with {secs_left:.0f}s left — "
                                f"will retry at expiry")

            # ── Current market expired — switch ───────────────────────────────
            if secs_left <= 0:
                log.info(f"Market expired: {cur['market'].question}")

                # Close any open position on market expiry
                if executor.position is not None:
                    executor.close_position()
                    log.info(executor.summary())

                # Update cross-market state for next market's features
                _prev_market["prev_momentum"]       = float(btc_momentum_pct(30) or 0.0)
                _prev_market["prev_ob_imbalance"]   = float(_ob_cache.get("imbalance", 0.0))
                _prev_market["prev_vol_range"]      = float(_vol_cache.get("range_pct", 0.0))
                _prev_market["prev_btc_range"]      = float(cur.get("btc_high", 0) - cur.get("btc_low", 0)) / max(cur.get("btc_low", 1), 1) * 100 if cur.get("btc_low", 0) > 0 and cur.get("btc_low") != float("inf") else 0.0
                _prev_market["prev_p_market_final"] = float(_poly_cache.get("up_mid", 0.5))
                _prev_market["prev_funding_zscore"] = float(_regime.get("funding_zscore", 0.0) or 0.0)
                # outcome will be patched by resolver later — use p_market as proxy
                pm_final = float(_poly_cache.get("up_mid", 0.5))
                if pm_final >= 0.5:
                    _prev_market["prev_outcome"] = 1.0
                    _prev_market["streak_up"]   += 1
                    _prev_market["streak_down"]  = 0.0
                else:
                    _prev_market["prev_outcome"] = 0.0
                    _prev_market["streak_down"] += 1
                    _prev_market["streak_up"]    = 0.0

                if nxt and nxt["market"]:
                    cur  = nxt
                    nxt  = None
                    next_fetch_tried = False
                    log.info(f"Switched to pre-fetched market: {cur['market'].question}")
                    update_poly_trade_subscription(cur["market"])
                else:
                    log.warning("No pre-fetched market ready at expiry — fetching next cycle")
                    cur = _make_market_state()
                    next_fetch_tried = False
                time.sleep(2)
                continue

            # ── Normal poll cycle ─────────────────────────────────────────────
            spot    = _price_cache["btc"]
            futures = _basis_cache["futures"]
            basis   = (futures - spot) / spot * 100 if spot > 0 and futures > 0 else 0.0

            log.info(f"BTC=${spot:.2f} | {secs_left:.0f}s left | "
                     f"funding={_funding_cache['rate']:+.6f} | basis={basis:+.3f}% | "
                     f"regime={_regime['composite']} | session={_regime['session']} | "
                     f"activity={_regime['activity']}"
                     + (f" | next=ready" if nxt else ""))

            # Capture opening price on first poll of each market
            if cur["market_open_price"] == 0.0 and spot > 0:
                cur["market_open_price"] = spot
                cur["snap_last_ts"] = 0.0   # force snapshot on market open
                log.info(f"Market open price captured: ${spot:.2f}")

            cur["max_secs_left"] = max(cur["max_secs_left"], secs_left)

            # ── Pre-fetch Polymarket prices once per cycle ────────────────────
            # Ensures _poly_cache is populated before ANY strategy calls
            # Poly prices fetched before ML scoring
            try:
                _prefetch_prices = fetch_poly_prices(cur["market"])
            except Exception:
                pass  # _poly_cache retains last good values

            mkt = cur["market"]

            # ── ML TRAINING SNAPSHOT ─────────────────────────────────────────
            try:
                cl_price, cl_age = fetch_chainlink_price()
                cl_div = round((spot - cl_price) / cl_price * 100, 6) if cl_price else 0.0

                binance_liq   = get_binance_liq_2min()
                liq_long      = _liq_cache["long"]  + binance_liq["long"]
                liq_short     = _liq_cache["short"] + binance_liq["short"]
                liq_total     = liq_long + liq_short
                liq_imbalance = round((liq_long - liq_short) / liq_total, 4) if liq_total > 0 else 0.0

                candles   = list(_volume_history)
                recent    = candles[-3:] if len(candles) >= 3 else []
                total_vol = sum(c["volume"] for c in recent)
                buy_vol   = sum(c["buy_vol"] for c in recent)
                buy_ratio = round(buy_vol / total_vol, 4) if total_vol > 0 else 0.5

                if cur["cl_open_price"] == 0.0 and cl_price:
                    cur["cl_open_price"] = cl_price

                market_open_price   = cur["market_open_price"]
                price_vs_open_pct   = round((spot - market_open_price) / market_open_price * 100, 6) \
                                      if market_open_price > 0 else 0.0

                # Fetch poly prices first — used in p_market tracking below
                poly_prices    = fetch_poly_prices(mkt, 20.0)
                poly_up_mid    = round(poly_prices.get("up_mid", 0.5), 4)
                poly_spread    = round(poly_prices.get("spread", 0.1), 4)
                poly_fill_up   = round(poly_prices.get("fill_up",  0.5), 4)
                poly_fill_down = round(poly_prices.get("fill_down", 0.5), 4)
                poly_slip_up   = round(poly_prices.get("slip_up", 0.0), 4)
                poly_deviation = round(poly_up_mid - 0.50, 4)

                # p_market tracking — open, std, delta for ML features
                if poly_up_mid and poly_up_mid > 0:
                    cur["p_market_history"].append(float(poly_up_mid))
                    if cur["p_market_open"] is None:
                        cur["p_market_open"] = float(poly_up_mid)
                p_market_open      = cur["p_market_open"]
                p_market_hist      = list(cur["p_market_history"])
                p_market_std       = round(float(np.std(p_market_hist)), 4) \
                                     if len(p_market_hist) >= 3 else None
                p_market_vs_delta  = round(float(poly_up_mid) - p_market_open, 4) \
                                     if p_market_open and poly_up_mid else None

                # Rolling market-level aggregates — mirror market model features
                # (ob/fz appended here; mom appended after momentum_30s is computed below)

                # secs_to_resolution — same as secs_left but explicit name for ML
                secs_to_res        = round(secs_left, 1)
                price_vs_open_score = round(math.tanh(price_vs_open_pct / 0.10), 4)

                cl_open_price  = cur["cl_open_price"]
                cl_vs_open_pct = round((cl_price - cl_open_price) / cl_open_price * 100, 6) \
                                 if cl_price and cl_open_price > 0 else 0.0

                effective_duration = cur["max_secs_left"] if cur["max_secs_left"] > 0 else 300.0
                market_progress    = round(max(0.0, min(1.0, 1.0 - secs_left / effective_duration)), 4)

                momentum_10s  = round(btc_momentum_pct(lookback_secs=10)  or 0.0, 6)
                momentum_30s  = round(btc_momentum_pct(lookback_secs=30)  or 0.0, 6)
                momentum_60s  = round(btc_momentum_pct(lookback_secs=60)  or 0.0, 6)
                momentum_120s = round(btc_momentum_pct(lookback_secs=120) or 0.0, 6)

                # Tick-level order flow features
                tick_30  = get_tick_features(lookback_secs=30)
                tick_60  = get_tick_features(lookback_secs=60)

                # Polymarket CLOB trade flow
                poly_flow = get_poly_trade_flow(lookback_secs=30)
                spot_div  = get_spot_futures_divergence()

                # Intra-market deltas — how features changed since last snapshot
                delta_cvd       = round(tick_30["cvd"] - cur["prev_cvd_30s"], 2)
                delta_taker_buy = round(tick_30["taker_buy_ratio"] - cur["prev_taker_buy_30s"], 4)
                delta_momentum  = round(momentum_30s - cur["prev_momentum_30s"], 6)
                delta_poly      = round(poly_up_mid - cur["prev_poly_up_mid"], 4)
                # ML score delta computed after scoring below

                # Velocity features — rate of change of key signals
                _fr_now = float(_funding_cache.get("rate", 0.0))
                _basis_now = round((futures - spot) / spot * 100, 6) if spot > 0 and futures > 0 else 0.0
                _pti_now = poly_flow["poly_trade_imb"]
                delta_funding  = round(_fr_now - cur.get("prev_funding_rate", 0.0), 8)
                delta_basis    = round(_basis_now - cur.get("prev_basis_pct", 0.0), 6)
                delta_trade_imb = round(_pti_now - cur.get("prev_trade_imb", 0.0), 4)

                # Cross-exchange spread: Coinbase spot vs Binance tick VWAP
                binance_vwap = tick_30.get("vwap_displacement", 0.0)
                xex_spread   = round(binance_vwap, 6)  # VWAP displacement IS the cross-exchange signal

                # Sequence-aware features (LSTM-like for LightGBM)
                cvd_accel = round(delta_cvd - cur["prev_delta_cvd"], 2)
                momentum_accel = round(delta_momentum - cur["prev_delta_mom"], 6)

                mom_sign = 1 if momentum_30s > 0 else (-1 if momentum_30s < 0 else 0)
                cur["mom_sign_history"].append(mom_sign)
                if len(cur["mom_sign_history"]) >= 3:
                    same = sum(1 for s in cur["mom_sign_history"] if s == mom_sign)
                    momentum_consistency_10 = round(same / len(cur["mom_sign_history"]), 2)
                else:
                    momentum_consistency_10 = 0.5

                ob_sign = 1 if _ob_cache.get("imbalance", 0.0) > 0 else -1
                cur["ob_sign_history"].append(ob_sign)
                if len(cur["ob_sign_history"]) >= 3:
                    ob_flip_count_10 = sum(1 for i in range(1, len(cur["ob_sign_history"]))
                                           if cur["ob_sign_history"][i] != cur["ob_sign_history"][i-1])
                else:
                    ob_flip_count_10 = 0

                # Update prev values for next cycle
                cur["prev_delta_cvd"]     = delta_cvd
                cur["prev_delta_mom"]     = delta_momentum
                cur["prev_cvd_30s"]       = tick_30["cvd"]
                cur["prev_taker_buy_30s"] = tick_30["taker_buy_ratio"]
                cur["prev_momentum_30s"]  = momentum_30s
                cur["prev_poly_up_mid"]   = poly_up_mid
                cur["prev_funding_rate"]  = _fr_now
                cur["prev_basis_pct"]     = _basis_now
                cur["prev_trade_imb"]     = _pti_now

                # Rolling market-level aggregates (now momentum_30s is available)
                _obi_now = float(_ob_cache.get("imbalance", 0.0))
                _fz_now  = float(_regime.get("funding_zscore", 0.0) or 0.0)
                cur["ob_history"].append(abs(_obi_now))
                cur["fz_history"].append(abs(_fz_now))
                cur["mom_history"].append(abs(momentum_30s))
                if spot > 0:
                    if cur["btc_high"] < spot: cur["btc_high"] = spot
                    if cur["btc_low"]  > spot: cur["btc_low"]  = spot
                _ob_hist  = list(cur["ob_history"])
                _fz_hist  = list(cur["fz_history"])
                _mom_hist = list(cur["mom_history"])
                avg_ob_imbalance_abs   = round(float(np.mean(_ob_hist)),  4) if _ob_hist  else 0.0
                avg_funding_zscore_abs = round(float(np.mean(_fz_hist)),  4) if _fz_hist  else 0.0
                avg_momentum_abs       = round(float(np.mean(_mom_hist)), 4) if _mom_hist else 0.0
                btc_range_pct          = round(
                    (cur["btc_high"] - cur["btc_low"]) / cur["btc_low"] * 100, 4
                ) if cur["btc_low"] > 0 and cur["btc_low"] != float("inf") else 0.0

                phase_early = 1 if secs_left > 200 else 0
                phase_mid   = 1 if 100 < secs_left <= 200 else 0
                phase_late  = 1 if 30 < secs_left <= 100 else 0
                phase_final = 1 if secs_left <= 30 else 0

                liq_total_history = cur["liq_total_history"]
                liq_total_history.append(liq_total)
                prev_liq = cur["prev_liq_total"]
                liq_delta = round(liq_total - prev_liq, 2)
                liq_accel = round(liq_delta - (prev_liq - (liq_total_history[0]
                            if len(liq_total_history) >= 3 else prev_liq)), 2)
                cur["prev_liq_total"] = liq_total

                cur_bid_depth = _ob_cache.get("bid_depth", 0.0)
                cur_ask_depth = _ob_cache.get("ask_depth", 0.0)
                ob_bid_delta  = round(cur_bid_depth - cur["prev_ob_bid_depth"], 2)
                ob_ask_delta  = round(cur_ask_depth - cur["prev_ob_ask_depth"], 2)
                cur["prev_ob_bid_depth"] = cur_bid_depth
                cur["prev_ob_ask_depth"] = cur_ask_depth

                interact_momentum_x_vol      = round(momentum_30s * _vol_cache.get("range_pct", 0.0), 6)
                interact_ob_x_spread         = round(_ob_cache.get("imbalance", 0.0) * poly_spread, 6)
                interact_liq_x_price_pos     = round(liq_imbalance * price_vs_open_pct, 6)
                interact_momentum_x_progress = round(momentum_30s * market_progress, 6)

                # Price bucket — captures the market's current confidence level.
                # Used as a categorical feature at training time so the ML can learn
                # that signal behaviour differs by price regime (e.g. contrarian signals
                # work near fair value but not at extreme longshot prices).
                # Also used post-hoc to diagnose per-bucket strategy performance.
                price_bucket = (
                    "longshot"   if poly_up_mid < 0.40 else
                    "underdog"   if poly_up_mid < 0.50 else
                    "favourite"  if poly_up_mid < 0.60 else
                    "heavy_fav"
                )

                # Intra-market memory — derived values for ML and snapshots
                snapshots_seen = float(cur["snapshots_seen"])
                max_score      = float(cur["max_score"])
                _cvd_hist = list(cur["cvd_history"])
                cvd_trend = round((_cvd_hist[-1] - _cvd_hist[0]) / len(_cvd_hist), 2) if len(_cvd_hist) >= 3 else 0.0
                _pm_hist = list(cur["pm_history_vel"])
                pm_velocity = round((_pm_hist[-1] - _pm_hist[0]) / len(_pm_hist), 6) if len(_pm_hist) >= 3 else 0.0

                # ── ML scoring ──────────────────────────────────────────────────
                _ml_p = _update_ml_score(
                    mkt.condition_id, secs_to_res, market_progress,
                    phase_early, phase_mid, phase_late, phase_final,
                    momentum_10s, momentum_30s, momentum_60s, momentum_120s,
                    liq_imbalance, liq_total, liq_long, liq_short,
                    price_vs_open_pct, price_vs_open_score,
                    poly_up_mid, poly_spread, poly_slip_up,
                    price_bucket, cl_div, cl_age, cl_vs_open_pct, basis,
                    ob_bid_delta, ob_ask_delta, buy_ratio,
                    p_market_std=p_market_std,
                    avg_ob_imbalance_abs=avg_ob_imbalance_abs,
                    avg_funding_zscore_abs=avg_funding_zscore_abs,
                    avg_momentum_abs=avg_momentum_abs,
                    btc_range_pct=btc_range_pct,
                    tick_cvd_30s=tick_30["cvd"],
                    tick_taker_buy_30s=tick_30["taker_buy_ratio"],
                    tick_large_buy_30s=tick_30["large_buy_usd"],
                    tick_large_sell_30s=tick_30["large_sell_usd"],
                    tick_intensity_30s=tick_30["trade_intensity"],
                    tick_vwap_disp_30s=tick_30["vwap_displacement"],
                    tick_cvd_60s=tick_60["cvd"],
                    tick_taker_buy_60s=tick_60["taker_buy_ratio"],
                    tick_intensity_60s=tick_60["trade_intensity"],
                    prev_vol_x_momentum=float(_prev_market.get("prev_vol_range", 0)) * float(_prev_market.get("prev_momentum", 0)),
                    session_x_vol=float(_ML_SESSION_ENC.get(_regime.get("session",""), 0)) * float(_vol_cache.get("range_pct", 0.0)),
                    streak_length=max(float(_prev_market.get("streak_up", 0)), float(_prev_market.get("streak_down", 0))),
                    delta_funding=delta_funding,
                    delta_basis=delta_basis,
                    delta_trade_imb=delta_trade_imb,
                    xex_spread=xex_spread,
                    delta_cvd=delta_cvd,
                    delta_taker_buy=delta_taker_buy,
                    delta_momentum=delta_momentum,
                    delta_poly=delta_poly,
                    spot_ob_imbalance=spot_div["spot_ob_imbalance"],
                    ob_divergence=spot_div["ob_divergence"],
                    poly_flow_imb=float(_poly_cache.get("poly_flow_imb", 0.0)),
                    poly_depth_ratio=float(_poly_cache.get("poly_depth_ratio", 0.5)),
                    poly_trade_imb=poly_flow["poly_trade_imb"],
                    poly_up_buys=poly_flow["poly_up_buys"],
                    poly_down_buys=poly_flow["poly_down_buys"],
                    poly_trade_count=poly_flow["poly_trade_count"],
                    poly_large_pct=poly_flow["poly_large_pct"],
                )
                if _ml_p >= 0:
                    delta_score = round(_ml_p - cur["prev_ml_score"], 4)
                    cur["prev_ml_score"] = _ml_p
                    # Log score every 30 seconds so we can see what the model is producing
                    if time.time() - cur.get("last_score_log", 0) > 30:
                        log.info(f"[ML] score={_ml_p:.3f} threshold={ML_QUANT_SCORE_THRESHOLD} "
                                 f"{'>>> WOULD TRADE' if _ml_p > ML_QUANT_SCORE_THRESHOLD else '(below threshold)'}")
                        cur["last_score_log"] = time.time()
                else:
                    delta_score = 0.0
                    if time.time() - cur.get("last_score_log", 0) > 30:
                        log.warning(f"[ML] score=-1 (model failed to score)")
                        cur["last_score_log"] = time.time()

                # Intra-market memory features (computed before snapshot write)
                cur["snapshots_seen"] += 1
                if _ml_p >= 0 and _ml_p > cur["max_score"]:
                    cur["max_score"] = _ml_p
                cur["cvd_history"].append(tick_30["cvd"])
                cur["pm_history_vel"].append(poly_up_mid)

                # ── Execution Engine — score, decide, log ────────────────────
                if _ml_p >= 0:
                    quant_score = _ml_p
                    decision_ts = time.time()

                    # Direction from Model B (P(UP wins)), fallback to p_market
                    p_up = _ml_scores_dir.get(mkt.condition_id, -1.0)
                    if p_up >= 0:
                        quant_side = "UP" if p_up >= 0.50 else "DOWN"
                        direction_confidence = abs(p_up - 0.50) * 2
                    else:
                        quant_side = "UP" if poly_up_mid >= 0.50 else "DOWN"
                        direction_confidence = abs(poly_up_mid - 0.50) * 2

                    quant_fill = poly_fill_up if quant_side == "UP" else poly_fill_down
                    passes_score = quant_score > ML_QUANT_SCORE_THRESHOLD
                    would_trade  = passes_score

                    now_ts = time.time()
                    should_log_quant = (
                        (would_trade and not cur["ml_quant_fired"])
                        or (now_ts - cur["ml_quant_last_log"] >= ML_QUANT_LOG_INTERVAL)
                    )

                    if should_log_quant:
                        if would_trade and not cur["ml_quant_fired"]:
                            cur["ml_quant_fired"] = True
                            cur["snap_last_ts"] = 0.0

                        cur["ml_quant_last_log"] = now_ts

                        # Execution log — full trade decision record
                        exec_record = {
                            "timestamp":          datetime.now(timezone.utc).isoformat(),
                            "decision_latency_ms": round((time.time() - decision_ts) * 1000, 1),
                            "condition_id":       mkt.condition_id,
                            "secs_left":          round(secs_left, 1),
                            "action":             "TRADE" if would_trade else "SKIP",
                            "side":               quant_side,
                            "score":              round(quant_score, 4),
                            "direction_confidence": round(direction_confidence, 4),
                            "expected_fill":      round(quant_fill, 4),
                            "poly_spread":        round(poly_spread, 4),
                            "p_market":           round(poly_up_mid, 4),
                            "btc_price":          round(spot, 2),
                            "ob_imbalance":       round(float(_ob_cache.get("imbalance", 0.0)), 4),
                            "tick_cvd_30s":       round(tick_30["cvd"], 2),
                            "tick_taker_buy":     round(tick_30["taker_buy_ratio"], 4),
                            "poly_trade_imb":     round(poly_flow["poly_trade_imb"], 4),
                            # Execution fields — filled when live trading is enabled
                            "order_submitted":    False,
                            "actual_fill":        None,
                            "slippage":           None,
                            "fill_latency_ms":    None,
                        }

                        # Append to local execution log
                        try:
                            with open("execution_log.jsonl", "a") as ef:
                                ef.write(json.dumps(exec_record) + "\n")
                        except Exception:
                            pass

                        if would_trade and executor.position is None:
                            # Data quality gate — don't trade on stale or missing data
                            ages = _cache_ages()
                            stale = []
                            if ages["age_btc_secs"] > 5: stale.append(f"btc={ages['age_btc_secs']:.1f}s")
                            if ages["age_ob_secs"] > 5: stale.append(f"ob={ages['age_ob_secs']:.1f}s")
                            if ages["age_funding_secs"] > 120: stale.append(f"funding={ages['age_funding_secs']:.0f}s")
                            if spot <= 0: stale.append("btc_price=0")
                            if poly_spread > 0.08: stale.append(f"spread={poly_spread:.3f}")

                            if stale:
                                log.warning(f"[QUALITY] Trade blocked — stale data: {', '.join(stale)}")
                            else:
                                trade_side = "LONG" if quant_side == "UP" else "SHORT"
                                executor.open_position(
                                    side=trade_side,
                                    size_usd=float(os.environ.get("TRADE_SIZE", "10.0")),
                                    score=quant_score,
                                    reason=f"score={quant_score:.3f} secs={secs_left:.0f}",
                                )

                ob_imb_now = round(_ob_cache.get("imbalance", 0.0), 4)
                should_snap, snap_reason, snap_feature, snap_magnitude = _snapshot_trigger(
                    cur, spot, poly_up_mid, ob_imb_now, liq_imbalance
                )

                if should_snap:
                    supabase_market_snapshot({
                        "condition_id":        mkt.condition_id,
                        "market_question":     mkt.question,
                        "market_end_time":     mkt.end_time.isoformat(),
                        "secs_left":           round(secs_left, 1),
                        "market_progress":     market_progress,
                        "phase_early":         phase_early,
                        "phase_mid":           phase_mid,
                        "phase_late":          phase_late,
                        "phase_final":         phase_final,
                        "btc_price":           round(spot, 2),
                        "market_open_price":   round(market_open_price, 2),
                        "price_vs_open_pct":   price_vs_open_pct,
                        "price_vs_open_score": price_vs_open_score,
                        "basis_pct":           round(basis, 6),
                        "funding_rate":        round(_funding_cache["rate"], 6),
                        "okx_funding":         round(_funding_cache["okx"], 6),
                        "gate_funding":        round(_funding_cache["binance"], 6),
                        "momentum_10s":        momentum_10s,
                        "momentum_30s":        momentum_30s,
                        "momentum_60s":        momentum_60s,
                        "momentum_120s":       momentum_120s,
                        "cl_divergence":       round(cl_div, 6),
                        "cl_age":              round(cl_age, 1) if cl_age else 0.0,
                        "cl_open_price":       round(cl_open_price, 2),
                        "cl_vs_open_pct":      cl_vs_open_pct,
                        "liq_total":           round(liq_total, 2),
                        "liq_long":            round(liq_long, 2),
                        "liq_short":           round(liq_short, 2),
                        "liq_imbalance":       liq_imbalance,
                        "liq_delta":           liq_delta,
                        "liq_accel":           liq_accel,
                        "vol_range_pct":       round(_vol_cache.get("range_pct", 0.0), 6),
                        "ob_imbalance":        ob_imb_now,
                        "ob_bid_depth":        round(cur_bid_depth, 2),
                        "ob_ask_depth":        round(cur_ask_depth, 2),
                        "ob_bid_delta":        ob_bid_delta,
                        "ob_ask_delta":        ob_ask_delta,
                        "volume_buy_ratio":    buy_ratio,
                        "p_market":            poly_up_mid,
                        "poly_spread":         poly_spread,
                        "poly_fill_up":        poly_fill_up,
                        "poly_fill_down":      poly_fill_down,
                        "poly_slip_up":        poly_slip_up,
                        "poly_deviation":      poly_deviation,
                        "price_bucket":        price_bucket,
                        "interact_momentum_x_vol":      interact_momentum_x_vol,
                        "interact_ob_x_spread":         interact_ob_x_spread,
                        "interact_liq_x_price_pos":     interact_liq_x_price_pos,
                        "interact_momentum_x_progress": interact_momentum_x_progress,
                        "liq_long_usd":        round(liq_long, 2),
                        "liq_short_usd":       round(liq_short, 2),
                        "liq_dominant_ratio":  round(max(liq_long, liq_short) / max(liq_total, 1.0), 4),
                        "regime":              _regime["composite"],
                        "session":             _regime["session"],
                        "activity":            _regime["activity"],
                        "day_type":            _regime["day_type"],
                        "momentum_score":      _regime["momentum_score"],
                        "volatility_pct":      _regime["volatility_pct"],
                        "flow_score":          _regime["flow_score"],
                        "funding_zscore":      _regime["funding_zscore"],
                        "funding_divergence":  round(float(_funding_cache.get("okx", 0.0)) - float(_funding_cache.get("binance", 0.0)), 8),
                        "prev_market_error":   round(abs(float(_prev_market.get("prev_p_market_final", 0.5)) - float(_prev_market.get("prev_outcome", 0.5))), 4),
                        "hour_sin":            _regime["hour_sin"],
                        "hour_cos":            _regime["hour_cos"],
                        "dow_sin":             _regime["dow_sin"],
                        "dow_cos":             _regime["dow_cos"],
                        "outcome_binary":      None,
                        "edge_realized":       None,
                        "resolved_outcome":    None,
                        "secs_to_resolution":  secs_to_res,
                        # Market-level rolling aggregates (mirror of market model features)
                        "p_market_std":           p_market_std,
                        "p_market_open":          round(p_market_open, 4) if p_market_open else None,
                        "avg_ob_imbalance_abs":   avg_ob_imbalance_abs,
                        "avg_funding_zscore_abs": avg_funding_zscore_abs,
                        "avg_momentum_abs":       avg_momentum_abs,
                        "btc_range_pct":          btc_range_pct,
                        # Spot vs futures OB divergence
                        "spot_ob_imbalance":         spot_div["spot_ob_imbalance"],
                        "ob_divergence":             spot_div["ob_divergence"],
                        "spot_bid_depth":            spot_div["spot_bid_depth"],
                        "spot_ask_depth":            spot_div["spot_ask_depth"],
                        # Polymarket order flow
                        "poly_flow_imb":             float(_poly_cache.get("poly_flow_imb", 0.0)),
                        "poly_depth_ratio":          float(_poly_cache.get("poly_depth_ratio", 0.5)),
                        "poly_trade_imb":            poly_flow["poly_trade_imb"],
                        "poly_up_buys":              poly_flow["poly_up_buys"],
                        "poly_down_buys":            poly_flow["poly_down_buys"],
                        "poly_trade_count":          poly_flow["poly_trade_count"],
                        "poly_large_pct":            poly_flow["poly_large_pct"],
                        # Intra-market memory (backward-looking summary)
                        "snapshots_seen":            float(cur["snapshots_seen"]),
                        "max_score":                 round(cur["max_score"], 4),
                        "cvd_trend":                 cvd_trend,
                        "pm_velocity":               pm_velocity,
                        # Velocity features
                        "delta_funding":             delta_funding,
                        "delta_basis":               delta_basis,
                        "delta_trade_imb":           delta_trade_imb,
                        "xex_spread":                xex_spread,
                        # Intra-market deltas
                        "delta_cvd":                 delta_cvd,
                        "delta_taker_buy":           delta_taker_buy,
                        "delta_momentum":            delta_momentum,
                        "delta_poly":                delta_poly,
                        "delta_score":               delta_score,
                        # Tick-level order flow (Binance aggTrades WebSocket)
                        "tick_cvd_30s":              tick_30["cvd"],
                        "tick_taker_buy_ratio_30s":  tick_30["taker_buy_ratio"],
                        "tick_large_buy_usd_30s":    tick_30["large_buy_usd"],
                        "tick_large_sell_usd_30s":   tick_30["large_sell_usd"],
                        "tick_intensity_30s":        tick_30["trade_intensity"],
                        "tick_vwap_disp_30s":        tick_30["vwap_displacement"],
                        "tick_cvd_60s":              tick_60["cvd"],
                        "tick_taker_buy_ratio_60s":  tick_60["taker_buy_ratio"],
                        "tick_intensity_60s":        tick_60["trade_intensity"],
                        # Sequence-aware features (LSTM-like)
                        "cvd_accel":               cvd_accel,
                        "momentum_accel":          momentum_accel,
                        "momentum_consistency_10": momentum_consistency_10,
                        "ob_flip_count_10":        ob_flip_count_10,
                        # Binance open interest + long/short ratio
                        "oi_value":            round(_oi_cache.get("open_interest", 0.0), 2),
                        "oi_change_5m":        round(_oi_cache.get("oi_change_5m", 0.0), 6),
                        "long_short_ratio":    _lsr_cache.get("long_short_ratio", 1.0),
                        "long_account_pct":    _lsr_cache.get("long_account_pct", 0.5),
                        "short_account_pct":   _lsr_cache.get("short_account_pct", 0.5),
                        # Deribit implied volatility (observation — not used for trading yet)
                        "iv_atm":   round(_iv_cache.get("atm_iv",   0.0), 2),
                        "iv_skew":  round(_iv_cache.get("skew_25d", 0.0), 2),
                        "iv_rank":  round(_iv_cache.get("iv_rank",  0.5), 4),
                        # Cache freshness — age of each data source at snapshot time.
                        **_cache_ages(),
                        # Snapshot trigger metadata (#3 architecture)
                        "snapshot_reason":    snap_reason,
                        "trigger_feature":    snap_feature,
                        "trigger_magnitude":  snap_magnitude,
                    })
                    # Update previous values for next cycle's change detection
                    cur["snap_prev_btc"]     = spot
                    cur["snap_prev_poly"]    = poly_up_mid
                    cur["snap_prev_ob"]      = ob_imb_now
                    cur["snap_prev_liq_imb"] = liq_imbalance
                    cur["snap_last_ts"]      = time.time()
            except Exception as e:
                log.warning(f"market_snapshot write failed: {e}")

            if time.time() - last_summary > 1800:
                log.info("=" * 60)
                log.info("ML QUANT ENGINE STATUS")
                if _ml_scores:
                    scores = list(_ml_scores.values())
                    log.info(f"  Active markets: {len(scores)}  "
                             f"score range: {min(scores):.3f}-{max(scores):.3f}  "
                             f"avg: {sum(scores)/len(scores):.3f}")
                # Pipeline health
                with _poly_trades_lock:
                    poly_buf_len = len(_poly_trades_buffer)
                with _binance_liq_lock:
                    liq_buf_len = len(_binance_liq_buffer)
                    liq_sides = {"BUY": 0, "SELL": 0}
                    for e in _binance_liq_buffer:
                        liq_sides[e.get("side", "?")] = liq_sides.get(e.get("side", "?"), 0) + 1
                poly_ws_ok = _poly_ws_instance and _poly_ws_instance.sock and _poly_ws_instance.sock.connected
                log.info(f"  Poly CLOB WS: {'CONNECTED' if poly_ws_ok else 'DEAD'}  "
                         f"buffer={poly_buf_len}  tokens={len(_poly_ws_token_ids)}")
                log.info(f"  Binance Liq buffer: {liq_buf_len}  "
                         f"BUY(short_liq)={liq_sides['BUY']}  SELL(long_liq)={liq_sides['SELL']}")
                log.info("=" * 60)
                last_summary = time.time()

            if time.time() - last_email > 3600:
                threading.Thread(target=_send_hourly_report,
                                 daemon=True).start()
                last_email = time.time()

            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    print("QUANT ENGINE v5.0 STARTING", flush=True)
    run()
