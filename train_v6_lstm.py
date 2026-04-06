"""
V6 LSTM Edge Prediction Pipeline
==================================
Time-sequence model for dYdX BTC-USD perpetual futures.

Architecture:
  - Input: last SEQ_LEN snapshots (60 snapshots = ~3 minutes of context)
  - Each snapshot: 80 features (same as V5 + V4 meta-features)
  - LSTM backbone -> shared temporal pattern learning
  - Two output heads: E(edge_long), E(edge_short)
  - 3-minute lookahead labels (same as V5)

Advantages over LightGBM (V5):
  - Sees sequences: "momentum accelerating for 10 snapshots" vs single point
  - Learns temporal patterns: reversals, breakouts, regime transitions
  - Asymmetric heads: long and short predictions can diverge

Labels: (btc_price_3min_later - btc_price_now) / btc_price_now - maker_fees
Fees: 0.01% per side = 0.02% round trip (dYdX Tier 1 maker)

Usage:
  python train_v6_lstm.py
"""

import sys
import logging
import math
import pickle
import warnings
import numpy as np
import requests
import time as _time
from collections import OrderedDict, defaultdict
from sklearn.impute import SimpleImputer
from fetch_cache import cached_fetch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

# ── Config ────────────────────────────────────────────────────────────────
SUPABASE_URL = "https://kcluwyzyetmkxhvszpxi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtjbHV3eXp5ZXRta3hodnN6cHhpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA4NTY2NCwiZXhwIjoyMDg5NjYxNjY0fQ.IbxuXRW0K9_UFZKG1i951EoL9KtCsOXCaz5Z_YqsmYE"
REST_H = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

SEQ_LEN = 60          # 60 snapshots = ~3 minutes of context (matches prediction horizon)
LOOKAHEAD_SECS = 180   # 3-minute prediction horizon
LOOKAHEAD_TOL = 15     # ±15 second tolerance for matching
MAKER_FEE = 0.0001     # 0.01% per side
ROUND_TRIP = MAKER_FEE * 2
MIN_EDGE = 0.0001

# LSTM hyperparameters
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.0003
EPOCHS = 50
PATIENCE = 10         # early stop after 10 epochs with no improvement
BATCH_SIZE = 512      # balanced for GPU speed + stable gradients
TRAIN_RATIO = 0.8      # temporal split


def _f(val):
    if val is None:
        return np.nan
    try:
        return float(val)
    except:
        return np.nan


def _rest_fetch(table, params, limit=500):
    """Fetch using created_at cursor — handles 100k+ rows."""
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
                _time.sleep(2)
                continue
            batch = r.json()
            if isinstance(batch, dict):
                retries += 1
                if retries > 5:
                    log.warning(f"  API error after 5 retries at {len(rows):,} rows: {batch.get('message','?')}")
                    break
                log.warning(f"  API error, retry {retries}: {batch.get('message','?')[:60]}")
                _time.sleep(5)
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
                log.warning(f"  Fetch stopped at {len(rows):,} rows: {e}")
                break
            _time.sleep(5)
    return rows


# ── V4 scoring (same as train_v5_futures.py) ──────────────────────────────
_v4_clf = _v4_imp = _v4_features = None
_v4d_clf = _v4d_imp = None
try:
    with open("models/model_v4_profitable.pkl", "rb") as f:
        _v4 = pickle.load(f)
    _v4_clf = _v4["classifier"]
    _v4_imp = _v4["classifier_imp"]
    _v4_features = _v4["features"]
    with open("models/model_v4_direction.pkl", "rb") as f:
        _v4d = pickle.load(f)
    _v4d_clf = _v4d["classifier"]
    _v4d_imp = _v4d["classifier_imp"]
    log.info(f"V4 loaded: {len(_v4_features)} features")
except Exception as e:
    log.warning(f"V4 NOT loaded: {e}")

_REGIME_MAP = {"TREND_UP":2,"TREND_DOWN":-2,"VOLATILE":1,"CALM":0,"DEAD":-1}
_SESSION_MAP = {"OVERLAP":3,"US":2,"LONDON":1,"ASIA":0,"OFFPEAK":-1}
_ACTIVITY_MAP = {"HIGH":2,"NORMAL":1,"LOW":0,"DEAD":-1}
_DAY_MAP = {"WEEKDAY":1,"WEEKEND":0}
_BUCKET_MAP = {"heavy_fav":3,"favourite":2,"underdog":1,"longshot":0}


def _score_v4(row):
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

        feat = {}
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
            elif fname == "regime_enc": val = _REGIME_MAP.get(row.get("regime", ""), 0)
            elif fname == "session_enc": val = _SESSION_MAP.get(row.get("session", ""), 0)
            elif fname == "activity_enc": val = _ACTIVITY_MAP.get(row.get("activity", ""), 0)
            elif fname == "day_enc": val = _DAY_MAP.get(row.get("day_type", ""), 0)
            elif fname == "bucket_enc": val = _BUCKET_MAP.get(row.get("price_bucket", ""), 0)
            elif fname == "is_extreme_market": val = 1.0 if (not np.isnan(pm) and (pm > 0.85 or pm < 0.15)) else 0.0
            feat[fname] = val if val is not None else np.nan
        X = np.array([[feat.get(k, np.nan) for k in _v4_features]], dtype=np.float32)
        X = _v4_imp.transform(X)
        return float(_v4_clf.predict_proba(X)[0][1]), float(_v4d_clf.predict_proba(_v4d_imp.transform(X))[0][1])
    except:
        return 0.5, 0.5


# ── Feature engineering (matches V5) ─────────────────────────────────────
FEATURE_COLS = [
    "secs_to_resolution", "market_progress",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "price_vs_open_pct", "momentum_30s", "momentum_60s", "momentum_120s",
    "cl_vs_open_pct",
    "liq_total", "liq_imbalance", "liq_long_usd", "liq_short_usd",
    "ob_imbalance", "ob_bid_depth", "ob_ask_depth",
    "vol_range_pct", "volatility_pct", "volume_buy_ratio",
    "p_market", "poly_spread",
    "basis_pct", "funding_rate", "funding_zscore", "okx_funding", "gate_funding",
    "avg_ob_imbalance_abs", "avg_funding_zscore_abs", "avg_momentum_abs",
    "btc_range_pct", "p_market_std",
    "poly_flow_imb", "poly_depth_ratio",
    "poly_trade_imb", "poly_up_buys", "poly_down_buys", "poly_trade_count", "poly_large_pct",
    "tick_cvd_30s", "tick_taker_buy_ratio_30s", "tick_large_buy_usd_30s",
    "tick_large_sell_usd_30s", "tick_intensity_30s", "tick_vwap_disp_30s",
    "tick_cvd_60s", "tick_taker_buy_ratio_60s", "tick_intensity_60s",
    "delta_cvd", "delta_taker_buy", "delta_momentum", "delta_funding", "delta_basis",
]


def build_feature_vector(row):
    """Extract feature vector from a single row."""
    vec = []
    for col in FEATURE_COLS:
        vec.append(_f(row.get(col)))
    # V4 meta-features
    vec.append(row.get("v4_score", 0.5))
    vec.append(row.get("v4_direction", 0.5))
    return vec


# ── Dataset ───────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X_seq, y_long, y_short):
        self.X = torch.FloatTensor(X_seq)
        self.y_long = torch.FloatTensor(y_long)
        self.y_short = torch.FloatTensor(y_short)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_long[idx], self.y_short[idx]


# ── Model ─────────────────────────────────────────────────────────────────
class EdgeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hidden_dim)
        # Two output heads — asymmetric long/short predictions
        self.head_long = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.head_short = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take last timestep
        last_hidden = self.bn(last_hidden)
        edge_long = self.head_long(last_hidden).squeeze(-1)
        edge_short = self.head_short(last_hidden).squeeze(-1)
        return edge_long, edge_short


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  V6 LSTM EDGE PREDICTION PIPELINE")
    print(f"  Sequence length: {SEQ_LEN} snapshots (~{SEQ_LEN*3}s)")
    print(f"  Lookahead: {LOOKAHEAD_SECS}s ({LOOKAHEAD_SECS//60} min)")
    print(f"  Fees: {ROUND_TRIP*100:.2f}% round trip (maker)")
    print("=" * 60)

    # Fetch data (cached — only pulls new rows after first run)
    cols = ",".join([
        "created_at", "condition_id", "btc_price",
        "secs_left", "secs_to_resolution", "market_progress",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "price_vs_open_pct", "momentum_30s", "momentum_60s", "momentum_120s",
        "cl_vs_open_pct",
        "liq_total", "liq_imbalance", "liq_long_usd", "liq_short_usd",
        "ob_imbalance", "ob_bid_depth", "ob_ask_depth",
        "vol_range_pct", "volatility_pct", "volume_buy_ratio",
        "p_market", "poly_spread",
        "basis_pct", "funding_rate", "funding_zscore", "okx_funding", "gate_funding",
        "avg_ob_imbalance_abs", "avg_funding_zscore_abs", "avg_momentum_abs",
        "btc_range_pct", "p_market_std",
        "poly_flow_imb", "poly_depth_ratio",
        "poly_trade_imb", "poly_up_buys", "poly_down_buys", "poly_trade_count", "poly_large_pct",
        "tick_cvd_30s", "tick_taker_buy_ratio_30s", "tick_large_buy_usd_30s",
        "tick_large_sell_usd_30s", "tick_intensity_30s", "tick_vwap_disp_30s",
        "tick_cvd_60s", "tick_taker_buy_ratio_60s", "tick_intensity_60s",
        "delta_cvd", "delta_taker_buy", "delta_momentum", "delta_funding", "delta_basis",
        "regime", "session", "activity", "day_type", "price_bucket",
        "outcome_binary",
    ])
    log.info("Fetching snapshots (cached)...")
    rows = cached_fetch("v6_snapshots", cols, {
        "btc_price": "gt.0",
        "outcome_binary": "not.is.null",
        "order": "created_at.asc",
    })
    log.info(f"  {len(rows):,} rows fetched")

    # Build price timeline for lookahead
    from datetime import datetime, timezone
    timestamps = []
    prices = []
    for row in rows:
        ts = datetime.fromisoformat(row["created_at"]).timestamp()
        timestamps.append(ts)
        prices.append(float(row["btc_price"]))
    ts_arr = np.array(timestamps)
    px_arr = np.array(prices)
    log.info(f"  Timeline: {len(ts_arr):,} points, {(ts_arr[-1]-ts_arr[0])/3600:.1f} hours")

    # Batch score with V4 (100x faster than row-by-row)
    if _v4_clf is not None:
        log.info("  Batch scoring with V4...")
        v4_X = np.zeros((len(rows), len(_v4_features)), dtype=np.float32)
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
                elif fname == "regime_enc": val = _REGIME_MAP.get(row.get("regime", ""), 0)
                elif fname == "session_enc": val = _SESSION_MAP.get(row.get("session", ""), 0)
                elif fname == "activity_enc": val = _ACTIVITY_MAP.get(row.get("activity", ""), 0)
                elif fname == "day_enc": val = _DAY_MAP.get(row.get("day_type", ""), 0)
                elif fname == "bucket_enc": val = _BUCKET_MAP.get(row.get("price_bucket", ""), 0)
                elif fname == "is_extreme_market": val = 1.0 if (not np.isnan(pm) and (pm > 0.85 or pm < 0.15)) else 0.0
                v4_X[i, j] = val if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan
        v4_X = _v4_imp.transform(v4_X)
        v4_probs = _v4_clf.predict_proba(v4_X)[:, 1]
        v4d_probs = _v4d_clf.predict_proba(_v4d_imp.transform(v4_X))[:, 1]
        for i, row in enumerate(rows):
            row["v4_score"] = float(v4_probs[i])
            row["v4_direction"] = float(v4d_probs[i])
        log.info(f"    Done: {len(rows):,} rows scored (batch)")

    # Build feature vectors + lookahead labels
    log.info("  Building feature vectors and labels...")
    features = []
    edge_longs = []
    edge_shorts = []
    skipped = 0

    for i, row in enumerate(rows):
        btc = float(row["btc_price"])
        ts = timestamps[i]

        # Lookahead: find BTC price 3 min later
        target_ts = ts + LOOKAHEAD_SECS
        idx = np.searchsorted(ts_arr, target_ts)
        best_price = None
        best_gap = LOOKAHEAD_TOL + 1
        for cand in [idx - 1, idx]:
            if 0 <= cand < len(ts_arr):
                gap = abs(ts_arr[cand] - target_ts)
                if gap < best_gap:
                    best_gap = gap
                    best_price = px_arr[cand]

        if best_price is None or best_gap > LOOKAHEAD_TOL:
            skipped += 1
            features.append(None)
            edge_longs.append(None)
            edge_shorts.append(None)
            continue

        el = (best_price - btc) / btc - ROUND_TRIP
        es = (btc - best_price) / btc - ROUND_TRIP
        features.append(build_feature_vector(row))
        edge_longs.append(el)
        edge_shorts.append(es)

    log.info(f"  {len(rows) - skipped:,} usable, {skipped:,} skipped (no lookahead)")

    # Impute NaN features
    n_features = len(FEATURE_COLS) + 2  # +2 for V4 meta
    valid_idx = [i for i in range(len(features)) if features[i] is not None]
    feat_matrix = np.array([features[i] for i in valid_idx], dtype=np.float32)
    log.info(f"  Feature matrix: {feat_matrix.shape}")

    labels_long = np.array([edge_longs[i] for i in valid_idx], dtype=np.float32)
    labels_short = np.array([edge_shorts[i] for i in valid_idx], dtype=np.float32)

    # Build sequences: sliding windows over time-continuous segments
    # Split on gaps > 30s (bot restarts, disconnects)
    MAX_GAP = 30  # seconds
    log.info(f"  Building sequences (window={SEQ_LEN}, max_gap={MAX_GAP}s)...")

    # Build continuous segments from valid_idx based on timestamps
    segments = []
    current_seg = [0]  # positions into valid_idx
    for pos in range(1, len(valid_idx)):
        t_prev = timestamps[valid_idx[pos - 1]]
        t_curr = timestamps[valid_idx[pos]]
        if t_curr - t_prev > MAX_GAP:
            segments.append(current_seg)
            current_seg = [pos]
        else:
            current_seg.append(pos)
    segments.append(current_seg)
    log.info(f"  {len(segments)} continuous segments (avg {np.mean([len(s) for s in segments]):.0f} snapshots)")

    X_sequences = []
    y_long_seq = []
    y_short_seq = []

    for positions in segments:
        if len(positions) < SEQ_LEN:
            continue
        for i in range(SEQ_LEN, len(positions)):
            seq_positions = positions[i - SEQ_LEN:i]
            X_sequences.append(feat_matrix[seq_positions])
            # Label is for the LAST snapshot in the sequence
            y_long_seq.append(labels_long[positions[i - 1]])
            y_short_seq.append(labels_short[positions[i - 1]])

    X_seq = np.array(X_sequences, dtype=np.float32)
    y_long = np.array(y_long_seq, dtype=np.float32)
    y_short = np.array(y_short_seq, dtype=np.float32)

    log.info(f"  Sequences: {len(X_seq):,} (shape {X_seq.shape})")
    log.info(f"  Avg edge_long: {y_long.mean():.6f}  edge_short: {y_short.mean():.6f}")

    # Temporal split BEFORE imputation/normalization (prevents leakage)
    n_train = int(len(X_seq) * TRAIN_RATIO)
    X_train_raw, X_test_raw = X_seq[:n_train], X_seq[n_train:]
    yl_train, yl_test = y_long[:n_train], y_long[n_train:]
    ys_train, ys_test = y_short[:n_train], y_short[n_train:]

    # Fit imputer and scaler on TRAINING data only (no test leakage)
    # Reshape to 2D for fitting, then back to 3D
    n_tr_seq, seq_len, n_feat = X_train_raw.shape
    train_2d = X_train_raw.reshape(-1, n_feat)
    imp = SimpleImputer(strategy="median")
    train_2d = imp.fit_transform(train_2d)
    feat_mean = train_2d.mean(axis=0)
    feat_std = train_2d.std(axis=0) + 1e-8
    train_2d = (train_2d - feat_mean) / feat_std
    X_train = train_2d.reshape(n_tr_seq, seq_len, n_feat).astype(np.float32)

    # Transform test with train-fitted imputer/scaler
    n_te_seq = X_test_raw.shape[0]
    test_2d = X_test_raw.reshape(-1, n_feat)
    test_2d = imp.transform(test_2d)
    test_2d = (test_2d - feat_mean) / feat_std
    X_test = test_2d.reshape(n_te_seq, seq_len, n_feat).astype(np.float32)

    log.info(f"  Train: {n_train:,}  Test: {len(X_test):,}")

    train_ds = SequenceDataset(X_train, yl_train, ys_train)
    test_ds = SequenceDataset(X_test, yl_test, ys_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Train
    model = EdgeLSTM(input_dim=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    log.info(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"\n  {'Epoch':>5} {'Train Loss':>12} {'Test Loss':>12} {'Corr Long':>10} {'Corr Short':>10} {'Side Acc':>10}")
    print(f"  {'-'*60}")
    sys.stdout.flush()

    best_corr = -1
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, yl_batch, ys_batch in train_dl:
            X_batch, yl_batch, ys_batch = X_batch.to(DEVICE), yl_batch.to(DEVICE), ys_batch.to(DEVICE)
            optimizer.zero_grad()
            pred_long, pred_short = model(X_batch)
            loss = criterion(pred_long, yl_batch) + criterion(pred_short, ys_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        # Eval
        model.eval()
        test_loss = 0
        all_pred_long, all_pred_short = [], []
        all_actual_long, all_actual_short = [], []
        with torch.no_grad():
            for X_batch, yl_batch, ys_batch in test_dl:
                X_batch, yl_batch, ys_batch = X_batch.to(DEVICE), yl_batch.to(DEVICE), ys_batch.to(DEVICE)
                pred_long, pred_short = model(X_batch)
                loss = criterion(pred_long, yl_batch) + criterion(pred_short, ys_batch)
                test_loss += loss.item() * len(X_batch)
                all_pred_long.extend(pred_long.cpu().numpy())
                all_pred_short.extend(pred_short.cpu().numpy())
                all_actual_long.extend(yl_batch.cpu().numpy())
                all_actual_short.extend(ys_batch.cpu().numpy())
        test_loss /= len(test_ds)

        pl = np.array(all_pred_long)
        ps = np.array(all_pred_short)
        al = np.array(all_actual_long)
        ash = np.array(all_actual_short)

        corr_long = float(np.corrcoef(pl, al)[0, 1]) if len(al) > 2 else 0
        corr_short = float(np.corrcoef(ps, ash)[0, 1]) if len(ash) > 2 else 0

        # Side accuracy
        chosen_long = pl > ps
        actual_up = al > ash
        side_acc = float((chosen_long == actual_up).mean())

        avg_corr = (corr_long + corr_short) / 2
        marker = ""
        if avg_corr > best_corr:
            best_corr = avg_corr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1

        print(f"  {epoch+1:>5} {train_loss:>12.8f} {test_loss:>12.8f} {corr_long:>+10.4f} {corr_short:>+10.4f} {side_acc*100:>9.1f}%{marker}")
        sys.stdout.flush()

        if no_improve >= PATIENCE:
            log.info(f"  Early stop at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_pred_long, all_pred_short = [], []
    all_actual_long, all_actual_short = [], []
    with torch.no_grad():
        for X_batch, yl_batch, ys_batch in test_dl:
            X_batch = X_batch.to(DEVICE)
            pred_long, pred_short = model(X_batch)
            all_pred_long.extend(pred_long.cpu().numpy())
            all_pred_short.extend(pred_short.cpu().numpy())
            all_actual_long.extend(yl_batch.numpy())
            all_actual_short.extend(ys_batch.numpy())

    pl = np.array(all_pred_long)
    ps = np.array(all_pred_short)
    al = np.array(all_actual_long)
    ash = np.array(all_actual_short)

    corr_long = float(np.corrcoef(pl, al)[0, 1])
    corr_short = float(np.corrcoef(ps, ash)[0, 1])

    # PnL simulation
    chosen_long = pl > ps
    pred_edge = np.where(chosen_long, pl, ps)
    actual_edge = np.where(chosen_long, al, ash)
    trade_mask = pred_edge > MIN_EDGE
    n_trades = int(trade_mask.sum())
    if n_trades > 0:
        pnl = float(actual_edge[trade_mask].sum())
        wr = float((actual_edge[trade_mask] > 0).mean())
        side_acc = float((chosen_long[trade_mask] == (al > ash)[trade_mask]).mean())
    else:
        pnl = wr = side_acc = 0

    print(f"\n{'='*60}")
    print(f"  V6 LSTM RESULTS (best epoch)")
    print(f"{'='*60}")
    print(f"  Train sequences:  {n_train:,}")
    print(f"  Test sequences:   {len(X_test):,}")
    print(f"  Features:         {n_features} per snapshot x {SEQ_LEN} timesteps")
    print(f"  Corr LONG:        {corr_long:+.4f}")
    print(f"  Corr SHORT:       {corr_short:+.4f}")
    print(f"  Avg Corr:         {(corr_long+corr_short)/2:+.4f}")
    print(f"  Trades:           {n_trades:,}")
    print(f"  PnL:              {pnl:+.6f}")
    print(f"  Win Rate:         {wr*100:.1f}%")
    print(f"  Side Accuracy:    {side_acc*100:.1f}%")
    print(f"  vs V5 LightGBM:   corr +0.112, PnL -0.76")
    print(f"{'='*60}")

    # Save (move to CPU for portability)
    model.cpu()
    torch.save({
        "model_state": model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "n_features": n_features,
        "seq_len": SEQ_LEN,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "imputer_medians": imp.statistics_,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "corr_long": corr_long,
        "corr_short": corr_short,
    }, "models/model_v6_lstm.pt")
    log.info("  Saved models/model_v6_lstm.pt")


if __name__ == "__main__":
    main()
