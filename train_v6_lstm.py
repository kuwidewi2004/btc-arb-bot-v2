"""
V6 LSTM Edge Prediction Pipeline
==================================
Time-sequence model for dYdX BTC-USD perpetual futures.

Architecture:
  - Input: 60 consecutive snapshots (~3 min of context at ~3s intervals)
  - 42 pure BTC microstructure features (no Polymarket)
  - 3-layer LSTM (128 hidden) with attention pooling
  - 6 output heads: edge_long, edge_short, MFE, MAE, time_in_profit, path_efficiency
  - Multi-component loss: Huber + edge weighting + ranking

Labels: (btc_price_3min_later - btc_price_now) / btc_price_now - 0.02% maker fees
Evaluation: Walk-forward (5 folds), 3-way split (train/val/test), Spearman correlation

Usage:
  python train_v6_lstm.py           # full eval (5 folds, ~50 min)
  python train_v6_lstm.py --quick   # quick eval (3 folds, ~15 min)
  python train_v6_lstm.py --fast    # train + save only, no eval (~10 min)
  python train_v6_lstm.py --pnl-loss  # experimental PnL-aware loss
"""

import os
import sys
import logging
import math
import pickle
import random
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

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

# ── Config ────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kcluwyzyetmkxhvszpxi.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", os.environ.get("SUPABASE_SERVICE_KEY", ""))
if not SUPABASE_KEY:
    # Fallback: read from fetch_cache (temporary until full env var migration)
    try:
        from fetch_cache import SUPABASE_KEY as _fk
        SUPABASE_KEY = _fk
    except ImportError:
        raise RuntimeError("SUPABASE_KEY env var required")
REST_H = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

SEQ_LEN = 60          # 60 snapshots = ~3 minutes of context (matches prediction horizon)
LOOKAHEAD_SECS = 180   # 3-minute prediction horizon
LOOKAHEAD_TOL = 15     # ±15 second tolerance for matching
MAKER_FEE = 0.0001     # 0.01% per side
ROUND_TRIP = MAKER_FEE * 2
MIN_EDGE = 0.0002      # higher threshold reduces overtrading on noise
TRAIN_STRIDE = 1       # no stride (need more data before striding helps)
PURGE_GAP = 30         # ~90s gap between train/val/test to prevent sequence overlap leakage


def safe_corr(a, b):
    """Spearman rank correlation — robust to outliers, measures ranking quality."""
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ic, _ = spearmanr(a, b)
    return float(ic) if np.isfinite(ic) else 0.0


# LSTM hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
LR = 0.001
EPOCHS = 80
PNL_LOSS = "--pnl-loss" in sys.argv  # experiment: add PnL-aware loss component
QUICK_EVAL = "--quick" in sys.argv   # faster eval: 3 folds, lower patience
PATIENCE = 6 if QUICK_EVAL else 12
BATCH_SIZE = 512      # balanced for GPU speed + stable gradients
WF_FOLDS = 3 if QUICK_EVAL else 5


def _f(val):
    if val is None:
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
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


# ── Feature engineering ───────────────────────────────────────────────────
FEATURE_COLS = [
    # Time features
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # BTC price features
    "price_vs_open_pct", "momentum_30s", "momentum_60s", "momentum_120s",
    "cl_vs_open_pct",
    # Liquidation
    "liq_total", "liq_imbalance", "liq_long_usd", "liq_short_usd",
    # Order book
    "ob_imbalance", "ob_bid_depth", "ob_ask_depth",
    # Volatility
    "vol_range_pct", "volatility_pct", "volume_buy_ratio",
    # Derivatives
    "basis_pct", "funding_rate", "funding_zscore", "okx_funding", "gate_funding",
    # Rolling aggregates
    "avg_ob_imbalance_abs", "avg_funding_zscore_abs", "avg_momentum_abs",
    "btc_range_pct",
    # Tick data
    "tick_cvd_30s", "tick_taker_buy_ratio_30s", "tick_large_buy_usd_30s",
    "tick_large_sell_usd_30s", "tick_intensity_30s", "tick_vwap_disp_30s",
    "tick_cvd_60s", "tick_taker_buy_ratio_60s", "tick_intensity_60s",
    # Velocity
    "delta_cvd", "delta_taker_buy", "delta_momentum", "delta_funding", "delta_basis",
]


def build_feature_vector(row):
    """Extract feature vector from a single row."""
    vec = []
    for col in FEATURE_COLS:
        vec.append(_f(row.get(col)))
    return vec


# ── Dataset ───────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X_seq, y_long, y_short, y_mfe=None, y_mae=None, y_tip=None, y_peff=None):
        self.X = torch.FloatTensor(X_seq)
        self.y_long = torch.FloatTensor(y_long)
        self.y_short = torch.FloatTensor(y_short)
        self.y_mfe = torch.FloatTensor(y_mfe) if y_mfe is not None else torch.zeros(len(X_seq))
        self.y_mae = torch.FloatTensor(y_mae) if y_mae is not None else torch.zeros(len(X_seq))
        self.y_tip = torch.FloatTensor(y_tip) if y_tip is not None else torch.zeros(len(X_seq))
        self.y_peff = torch.FloatTensor(y_peff) if y_peff is not None else torch.zeros(len(X_seq))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y_long[idx], self.y_short[idx],
                self.y_mfe[idx], self.y_mae[idx], self.y_tip[idx], self.y_peff[idx])


# ── Model ─────────────────────────────────────────────────────────────────
class EdgeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Attention pooling: learns which timesteps matter + last hidden state
        self.attn_w = nn.Linear(hidden_dim, 1)
        pool_dim = hidden_dim * 2  # attention-weighted sum + last hidden
        self.ln = nn.LayerNorm(pool_dim)
        # Edge prediction heads
        self.head_long = nn.Sequential(
            nn.Linear(pool_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.head_short = nn.Sequential(
            nn.Linear(pool_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        # Path quality heads
        self.head_mfe = nn.Sequential(nn.Linear(pool_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.head_mae = nn.Sequential(nn.Linear(pool_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.head_tip = nn.Sequential(nn.Linear(pool_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())  # 0-1
        self.head_peff = nn.Sequential(nn.Linear(pool_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()) # 0-1

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Attention pooling
        attn_scores = self.attn_w(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        last = lstm_out[:, -1, :]
        h = torch.cat([attn_pool, last], dim=1)
        h = self.ln(h)
        edge_long = self.head_long(h).squeeze(-1)
        edge_short = self.head_short(h).squeeze(-1)
        mfe = self.head_mfe(h).squeeze(-1)
        mae = self.head_mae(h).squeeze(-1)
        tip = self.head_tip(h).squeeze(-1)
        peff = self.head_peff(h).squeeze(-1)
        return edge_long, edge_short, mfe, mae, tip, peff


# ── Main ──────────────────────────────────────────────────────────────────
FAST_MODE = "--fast" in sys.argv  # skip walk-forward, just train final model

def main():
    print("=" * 60)
    print(f"  V6 LSTM EDGE PREDICTION PIPELINE {'(FAST MODE)' if FAST_MODE else ''}")
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

    # Build feature vectors + lookahead labels + MFE/MAE
    log.info("  Building feature vectors and labels (with MFE/MAE)...")
    features = []
    edge_longs = []
    edge_shorts = []
    mfe_longs = []   # max favorable excursion for long
    mae_longs = []   # max adverse excursion for long
    tip_list = []    # time in profit (fraction of window in green)
    peff_list = []   # path efficiency (net move / total wiggle)
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
            mfe_longs.append(None)
            mae_longs.append(None)
            tip_list.append(None)
            peff_list.append(None)
            continue

        raw_return = (best_price - btc) / btc

        # Path analysis: scan all prices between now and target
        start_idx = np.searchsorted(ts_arr, ts)
        end_idx = np.searchsorted(ts_arr, target_ts)
        if end_idx > start_idx and end_idx <= len(px_arr):
            path_prices = px_arr[start_idx:end_idx]
            path_returns = (path_prices - btc) / btc
            mfe_long = float(path_returns.max())   # best exit for long
            mae_long = float(path_returns.min())   # best exit for short (flipped)
            tip = float((path_returns > 0).mean())
            total_wiggle = float(np.abs(np.diff(path_returns)).sum())
            peff = abs(raw_return) / (total_wiggle + 1e-10)
            peff = min(peff, 1.0)
        else:
            mfe_long = raw_return
            mae_long = raw_return
            tip = 0.5
            peff = 0.5

        # Labels: endpoint return minus fees (MFE labels were biased positive — see kill tests)
        el = raw_return - ROUND_TRIP
        es = -raw_return - ROUND_TRIP

        features.append(build_feature_vector(row))
        edge_longs.append(el)
        edge_shorts.append(es)
        mfe_longs.append(mfe_long)
        mae_longs.append(mae_long)
        tip_list.append(tip)
        peff_list.append(peff)

    log.info(f"  {len(rows) - skipped:,} usable, {skipped:,} skipped (no lookahead)")

    # Impute NaN features
    n_features = len(FEATURE_COLS)
    valid_idx = [i for i in range(len(features)) if features[i] is not None]
    feat_matrix = np.array([features[i] for i in valid_idx], dtype=np.float32)
    log.info(f"  Feature matrix: {feat_matrix.shape}")

    labels_long = np.array([edge_longs[i] for i in valid_idx], dtype=np.float32)
    labels_short = np.array([edge_shorts[i] for i in valid_idx], dtype=np.float32)
    labels_mfe = np.array([mfe_longs[i] for i in valid_idx], dtype=np.float32)
    labels_mae = np.array([mae_longs[i] for i in valid_idx], dtype=np.float32)
    labels_tip = np.array([tip_list[i] for i in valid_idx], dtype=np.float32)
    labels_peff = np.array([peff_list[i] for i in valid_idx], dtype=np.float32)
    log.info(f"  MFE: mean={labels_mfe.mean():.6f}  MAE: mean={labels_mae.mean():.6f}")
    log.info(f"  Time-in-profit: mean={labels_tip.mean():.3f}  Path-eff: mean={labels_peff.mean():.3f}")

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
    y_mfe_seq = []
    y_mae_seq = []
    y_tip_seq = []
    y_peff_seq = []

    for positions in segments:
        if len(positions) < SEQ_LEN:
            continue
        for i in range(SEQ_LEN, len(positions)):
            seq_positions = positions[i - SEQ_LEN:i]
            # Reject sequences with irregular intra-sequence timing (>15s gap)
            seq_ts = [timestamps[valid_idx[p]] for p in seq_positions]
            max_intra_gap = max(seq_ts[j+1] - seq_ts[j] for j in range(len(seq_ts)-1))
            if max_intra_gap > 15:
                continue
            X_sequences.append(feat_matrix[seq_positions])
            last_pos = positions[i - 1]
            y_long_seq.append(labels_long[last_pos])
            y_short_seq.append(labels_short[last_pos])
            y_mfe_seq.append(labels_mfe[last_pos])
            y_mae_seq.append(labels_mae[last_pos])
            y_tip_seq.append(labels_tip[last_pos])
            y_peff_seq.append(labels_peff[last_pos])

    X_seq = np.array(X_sequences, dtype=np.float32)
    y_long = np.array(y_long_seq, dtype=np.float32)
    y_short = np.array(y_short_seq, dtype=np.float32)
    y_mfe = np.array(y_mfe_seq, dtype=np.float32)
    y_mae = np.array(y_mae_seq, dtype=np.float32)
    y_tip = np.array(y_tip_seq, dtype=np.float32)
    y_peff = np.array(y_peff_seq, dtype=np.float32)

    log.info(f"  Sequences: {len(X_seq):,} (shape {X_seq.shape})")
    log.info(f"  Avg edge_long: {y_long.mean():.6f}  edge_short: {y_short.mean():.6f}")
    log.info(f"  Avg MFE: {y_mfe.mean():.6f}  MAE: {y_mae.mean():.6f}")

    # Filter noisy labels — only train on meaningful edges (|edge| > threshold)
    NOISE_FLOOR = 0.0003
    meaningful = (np.abs(y_long) > NOISE_FLOOR) | (np.abs(y_short) > NOISE_FLOOR)
    n_before = len(X_seq)
    X_seq = X_seq[meaningful]
    y_long = y_long[meaningful]
    y_short = y_short[meaningful]
    y_mfe = y_mfe[meaningful]
    y_mae = y_mae[meaningful]
    y_tip = y_tip[meaningful]
    y_peff = y_peff[meaningful]
    log.info(f"  Filtered noisy labels: {n_before:,} → {len(X_seq):,} ({meaningful.mean()*100:.1f}% kept)")

    # ── Walk-forward evaluation ──────────────────────────────────────
    n_total = len(X_seq)
    avg_corr_l = avg_corr_s = 0.0

    if FAST_MODE:
        log.info("  FAST MODE — skipping walk-forward, training final model only")
        all_fold_results = []
    else:
        fold_size = n_total // (WF_FOLDS + 1)  # +1 so first fold has training data
        all_fold_results = []
        log.info(f"  Walk-forward: {WF_FOLDS} folds, ~{fold_size:,} sequences each")

    for fold_i in range(0 if FAST_MODE else WF_FOLDS):
        train_end = fold_size * (fold_i + 1)
        test_end = min(train_end + fold_size, n_total)
        if test_end <= train_end:
            continue

        # 3-way split with purge gap: train / [gap] / val (calibration) / [gap] / test (honest)
        val_start = train_end + PURGE_GAP
        val_end = val_start + fold_size // 2
        val_end = min(val_end, test_end - PURGE_GAP)
        test_start = val_end + PURGE_GAP
        if test_start >= test_end or val_start >= val_end:
            continue  # not enough data for this fold

        # Slice views (no copy) to avoid OOM
        X_tr_raw = X_seq[:train_end:TRAIN_STRIDE]
        yl_tr = y_long[:train_end:TRAIN_STRIDE]
        ys_tr = y_short[:train_end:TRAIN_STRIDE]
        mfe_tr = y_mfe[:train_end:TRAIN_STRIDE]
        mae_tr = y_mae[:train_end:TRAIN_STRIDE]
        tip_tr = y_tip[:train_end:TRAIN_STRIDE]
        peff_tr = y_peff[:train_end:TRAIN_STRIDE]

        X_val_raw = X_seq[val_start:val_end]
        X_te_raw = X_seq[test_start:test_end]
        yl_val = y_long[val_start:val_end]
        ys_val = y_short[val_start:val_end]
        mfe_val = y_mfe[val_start:val_end]
        mae_val = y_mae[val_start:val_end]
        tip_val = y_tip[val_start:val_end]
        peff_val = y_peff[val_start:val_end]
        # Raw labels for PnL — test set only (untouched by threshold fitting)
        yl_te_raw = y_long[test_start:test_end]
        ys_te_raw = y_short[test_start:test_end]
        # Val raw labels for threshold fitting
        yl_val_raw = y_long[val_start:val_end]
        ys_val_raw = y_short[val_start:val_end]

        # Fit imputer/scaler on train sample only
        n_tr_seq, sl, n_feat = X_tr_raw.shape
        CHUNK = 20000  # smaller chunks to avoid OOM
        sample_n = min(CHUNK, n_tr_seq)
        sample_2d = X_tr_raw[:sample_n].copy().reshape(-1, n_feat)
        imp = SimpleImputer(strategy="median")
        imp.fit(sample_2d)
        sample_imp = imp.transform(sample_2d)
        feat_mean = sample_imp.mean(axis=0)
        feat_std = sample_imp.std(axis=0) + 1e-8
        del sample_2d, sample_imp

        # Normalize in chunks — process into list of small arrays to avoid one big allocation
        def _normalize_to_list(X_raw):
            chunks = []
            for c in range(0, len(X_raw), CHUNK):
                ce = min(c + CHUNK, len(X_raw))
                chunk = X_raw[c:ce].copy().reshape(-1, n_feat)
                chunk = imp.transform(chunk)
                chunk = (chunk - feat_mean) / feat_std
                chunks.append(chunk.reshape(ce - c, sl, n_feat).astype(np.float32))
            return np.concatenate(chunks, axis=0) if chunks else np.empty((0, sl, n_feat), dtype=np.float32)

        X_train = _normalize_to_list(X_tr_raw)
        X_val = _normalize_to_list(X_val_raw)
        X_test = _normalize_to_list(X_te_raw)

        # Val set for training loss eval + early stopping
        train_ds = SequenceDataset(X_train, yl_tr, ys_tr, mfe_tr, mae_tr, tip_tr, peff_tr)
        val_ds = SequenceDataset(X_val, yl_val, ys_val, mfe_val, mae_val, tip_val, peff_val)
        _dl_kwargs = {"num_workers": 2, "pin_memory": True} if DEVICE.type == "cuda" else {}
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **_dl_kwargs)
        test_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **_dl_kwargs)  # early stop on val

        # Fresh model per fold
        model = EdgeLSTM(input_dim=n_features).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
        criterion = nn.HuberLoss(delta=0.001)  # robust to outliers

        if fold_i == 0:
            log.info(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

        print(f"\n  ── Fold {fold_i+1}/{WF_FOLDS}  train={len(X_tr_raw):,}(stride={TRAIN_STRIDE})  val={len(X_val_raw):,}  test={len(X_te_raw):,}  purge={PURGE_GAP} ──")
        print(f"  {'Epoch':>5} {'TrLoss':>10} {'TeLoss':>10} {'CorrL':>8} {'CorrS':>8} {'SideAcc':>8} {'LR':>10}")
        sys.stdout.flush()

        best_score = -1
        best_state = None
        no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for X_b, yl_b, ys_b, mfe_b, mae_b, tip_b, peff_b in train_dl:
                X_b = X_b.to(DEVICE)
                yl_b, ys_b = yl_b.to(DEVICE), ys_b.to(DEVICE)
                mfe_b, mae_b = mfe_b.to(DEVICE), mae_b.to(DEVICE)
                tip_b, peff_b = tip_b.to(DEVICE), peff_b.to(DEVICE)
                optimizer.zero_grad()
                pl, ps, p_mfe, p_mae, p_tip, p_peff = model(X_b)

                # Base regression loss (asymmetric Huber)
                base_loss = 0.55 * criterion(pl, yl_b) + 0.45 * criterion(ps, ys_b)

                # Path quality auxiliary loss (MFE + MAE + time-in-profit + path efficiency)
                path_loss = (criterion(p_mfe, mfe_b) + criterion(p_mae, mae_b)
                             + nn.MSELoss()(p_tip, tip_b) + nn.MSELoss()(p_peff, peff_b))

                # Edge weighting: emphasize large moves (capped at 5x)
                weights = (torch.abs(yl_b) + torch.abs(ys_b)).clamp(max=0.005)
                weights = weights / (weights.mean() + 1e-8)
                weighted_loss = (weights * (0.55 * (pl - yl_b)**2 + 0.45 * (ps - ys_b)**2)).mean()

                # Ranking loss: "is long better than short?" direction
                rank_target = torch.sign(yl_b - ys_b)
                rank_pred = pl - ps
                rank_loss = torch.relu(0.001 - rank_pred * rank_target).mean()

                loss = 0.4 * base_loss + 0.2 * path_loss + 0.2 * weighted_loss + 0.2 * rank_loss

                # PnL-aware loss (experiment)
                if PNL_LOSS:
                    pred_edge = torch.maximum(pl, ps)
                    direction = (pl > ps).float() * 2 - 1
                    pnl_est = (direction * pred_edge).mean()
                    loss = 0.7 * loss + 0.3 * (-pnl_est)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_b)
            train_loss /= len(train_ds)

            model.eval()
            test_loss = 0
            all_pl, all_ps, all_al, all_as = [], [], [], []
            with torch.no_grad():
                for X_b, yl_b, ys_b, mfe_b, mae_b, tip_b, peff_b in test_dl:
                    X_b, yl_b, ys_b = X_b.to(DEVICE), yl_b.to(DEVICE), ys_b.to(DEVICE)
                    pl, ps, p_mfe, p_mae, p_tip, p_peff = model(X_b)
                    loss = 0.55 * criterion(pl, yl_b) + 0.45 * criterion(ps, ys_b)
                    test_loss += loss.item() * len(X_b)
                    all_pl.extend(pl.cpu().numpy())
                    all_ps.extend(ps.cpu().numpy())
                    all_al.extend(yl_b.cpu().numpy())
                    all_as.extend(ys_b.cpu().numpy())
            test_loss /= len(val_ds)

            scheduler.step(test_loss)

            pl_arr = np.array(all_pl)
            ps_arr = np.array(all_ps)
            al_arr = np.array(all_al)
            as_arr = np.array(all_as)

            corr_l = safe_corr(pl_arr, al_arr)
            corr_s = safe_corr(ps_arr, as_arr)
            chosen_up = pl_arr > ps_arr
            actual_up = al_arr > as_arr
            side_acc = float((chosen_up == actual_up).mean())

            # Blended score: correlation + economic performance
            avg_corr = (corr_l + corr_s) / 2
            pred_edge_ep = np.where(chosen_up, pl_arr, ps_arr)
            actual_edge_ep = np.where(chosen_up, al_arr, as_arr)
            top_mask = pred_edge_ep > np.percentile(pred_edge_ep, 80) if len(pred_edge_ep) > 10 else pred_edge_ep > 0
            pnl_per_trade = float(actual_edge_ep[top_mask].mean()) if top_mask.sum() > 0 else 0
            # Score: corr matters + PnL matters (scaled to similar magnitude)
            score = avg_corr + 300 * pnl_per_trade  # 300x scale factor ~= corr units

            marker = ""
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                marker = " *"
            else:
                no_improve += 1

            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  {epoch+1:>5} {train_loss:>10.8f} {test_loss:>10.8f} {corr_l:>+8.4f} {corr_s:>+8.4f} {side_acc*100:>7.1f}% {cur_lr:>10.6f}{marker}")
            sys.stdout.flush()

            if no_improve >= PATIENCE:
                log.info(f"    Early stop at epoch {epoch+1}")
                break

        # Evaluate fold: predict on VAL (for calibration), then TEST (honest metrics)
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        def _predict(X_data):
            pls, pss, pmfes, pmaes, ptips, ppeffs = [], [], [], [], [], []
            zeros = np.zeros(len(X_data))
            ds = SequenceDataset(X_data, zeros, zeros, zeros, zeros, zeros, zeros)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, **_dl_kwargs)
            with torch.no_grad():
                for X_b, _, _, _, _, _, _ in dl:
                    X_b = X_b.to(DEVICE)
                    pl, ps, p_mfe, p_mae, p_tip, p_peff = model(X_b)
                    pls.extend(pl.cpu().numpy())
                    pss.extend(ps.cpu().numpy())
                    pmfes.extend(p_mfe.cpu().numpy())
                    pmaes.extend(p_mae.cpu().numpy())
                    ptips.extend(p_tip.cpu().numpy())
                    ppeffs.extend(p_peff.cpu().numpy())
            return (np.array(pls), np.array(pss), np.array(pmfes),
                    np.array(pmaes), np.array(ptips), np.array(ppeffs))

        # Step 1: Predict on VAL → fit threshold + isotonic calibration
        val_pl, val_ps, val_mfe, val_mae, val_tip_pred, val_peff_pred = _predict(X_val)
        val_chosen_long = val_pl > val_ps
        val_pred_edge = np.where(val_chosen_long, val_pl, val_ps)
        val_actual_edge = np.where(val_chosen_long, yl_val_raw, ys_val_raw)

        # Fit threshold on val
        best_pnl_thresh = MIN_EDGE
        best_pnl_val = -999
        for pct in [70, 75, 80, 85, 90, 95]:
            t = np.percentile(val_pred_edge, pct) if len(val_pred_edge) > 0 else MIN_EDGE
            t = max(t, MIN_EDGE)
            mask = val_pred_edge > t
            n = int(mask.sum())
            if n > 10:
                p = float(val_actual_edge[mask].sum())
                if p > best_pnl_val:
                    best_pnl_val = p
                    best_pnl_thresh = t

        # Fit isotonic calibration on val: predicted edge → actual edge
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        if len(val_pred_edge) > 50:
            iso_reg.fit(val_pred_edge, val_actual_edge)

        # Step 2: Predict on TEST (untouched) → apply val-fitted threshold + calibration
        te_pl, te_ps, te_mfe, te_mae, te_tip, te_peff = _predict(X_test)
        chosen_long = te_pl > te_ps
        pred_edge_raw = np.where(chosen_long, te_pl, te_ps)
        # Calibrate predictions using val-fitted isotonic regression
        if len(val_pred_edge) > 50:
            pred_edge = iso_reg.predict(pred_edge_raw)
        else:
            pred_edge = pred_edge_raw
        actual_edge = np.where(chosen_long, yl_te_raw, ys_te_raw)

        corr_l = safe_corr(te_pl, yl_te_raw)
        corr_s = safe_corr(te_ps, ys_te_raw)

        # Apply val-fitted threshold + path quality filter
        # Quality score: MFE upside, limited MAE pain, high time-in-profit, high path efficiency
        quality = te_mfe - np.abs(te_mae) + te_tip * 0.001 + te_peff * 0.001
        trade_mask = (pred_edge_raw > best_pnl_thresh) & (quality > np.median(quality))
        n_trades = int(trade_mask.sum())

        if n_trades > 0:
            pnl = float(actual_edge[trade_mask].sum())
            wr = float((actual_edge[trade_mask] > 0).mean())
            s_acc = float((chosen_long[trade_mask] == (yl_te_raw > ys_te_raw)[trade_mask]).mean())
        else:
            pnl = wr = s_acc = 0

        n_val = len(X_val)
        n_test = len(X_test)
        print(f"    Fold {fold_i+1} (val={n_val},test={n_test}): corr_l={corr_l:+.4f} corr_s={corr_s:+.4f} trades={n_trades} PnL={pnl:+.4f} WR={wr*100:.1f}% thresh={best_pnl_thresh:.5f}")
        all_fold_results.append({
            "corr_long": corr_l, "corr_short": corr_s,
            "n_trades": n_trades, "pnl": pnl, "wr": wr, "side_acc": s_acc,
            # Save predictions for ensemble eval
            "test_indices": (test_start, test_end),
            "pred_long": te_pl, "pred_short": te_ps,
            "actual_long": yl_te_raw, "actual_short": ys_te_raw,
            "pred_mfe": te_mfe, "pred_peff": te_peff,
        })

    # ── Summary across folds ─────────────────────────────────────────
    if all_fold_results:
        avg_corr_l = np.mean([r["corr_long"] for r in all_fold_results])
        avg_corr_s = np.mean([r["corr_short"] for r in all_fold_results])
        total_pnl = sum(r["pnl"] for r in all_fold_results)
        total_trades = sum(r["n_trades"] for r in all_fold_results)
        avg_wr = np.mean([r["wr"] for r in all_fold_results if r["n_trades"] > 0]) if any(r["n_trades"] > 0 for r in all_fold_results) else 0
        avg_side = np.mean([r["side_acc"] for r in all_fold_results if r["n_trades"] > 0]) if any(r["n_trades"] > 0 for r in all_fold_results) else 0

        print(f"\n{'='*60}")
        print(f"  V6 LSTM RESULTS ({WF_FOLDS}-fold walk-forward)")
        print(f"{'='*60}")
        print(f"  Total sequences:  {n_total:,}")
        print(f"  Features:         {n_features} per snapshot x {SEQ_LEN} timesteps")
        print(f"  Labels:           raw edge (after fees)")
        print(f"  Corr LONG:        {avg_corr_l:+.4f}")
        print(f"  Corr SHORT:       {avg_corr_s:+.4f}")
        print(f"  Avg Corr:         {(avg_corr_l+avg_corr_s)/2:+.4f}")
        print(f"  Trades:           {total_trades:,}")
        print(f"  PnL:              {total_pnl:+.6f}")
        print(f"  Win Rate:         {avg_wr*100:.1f}%")
        print(f"  Side Accuracy:    {avg_side*100:.1f}%")
        print(f"{'='*60}")

    # Save fold predictions for ensemble eval
    if all_fold_results and not FAST_MODE:
        os.makedirs("cache", exist_ok=True)
        np.savez("cache/v6_fold_predictions.npz",
                 **{f"fold_{i}_{k}": np.array(r[k]) if isinstance(r[k], np.ndarray) else np.array(r[k])
                    for i, r in enumerate(all_fold_results)
                    for k in ["pred_long", "pred_short", "actual_long", "actual_short", "pred_mfe", "pred_peff", "test_indices"]})
        log.info("  Saved fold predictions to cache/v6_fold_predictions.npz")

    # ── Train final model on all data for deployment ─────────────────
    log.info("  Training final model on all data...")
    n_all = len(X_seq)

    # Compute imputer/scaler stats from a sample (avoids OOM on full reshape)
    sample_size = min(100000, n_all)
    sample_2d = X_seq[:sample_size].reshape(-1, n_features)
    imp = SimpleImputer(strategy="median")
    imp.fit(sample_2d)
    sample_imp = imp.transform(sample_2d)
    feat_mean = sample_imp.mean(axis=0)
    feat_std = sample_imp.std(axis=0) + 1e-8
    del sample_2d, sample_imp

    # Normalize in chunks to avoid OOM
    CHUNK = 50000
    X_all = np.empty_like(X_seq)
    for c_start in range(0, n_all, CHUNK):
        c_end = min(c_start + CHUNK, n_all)
        chunk = X_seq[c_start:c_end].reshape(-1, n_features)
        chunk = imp.transform(chunk)
        chunk = (chunk - feat_mean) / feat_std
        X_all[c_start:c_end] = chunk.reshape(c_end - c_start, SEQ_LEN, n_features)
    X_all = X_all.astype(np.float32)

    final_ds = SequenceDataset(X_all, y_long, y_short, y_mfe, y_mae, y_tip, y_peff)
    _dl_kwargs = {"num_workers": 2, "pin_memory": True} if DEVICE.type == "cuda" else {}
    final_dl = DataLoader(final_ds, batch_size=BATCH_SIZE, shuffle=True, **_dl_kwargs)

    model = EdgeLSTM(input_dim=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    criterion = nn.HuberLoss(delta=0.001)  # same as fold training

    # Train for fewer epochs on full data (no early stop needed)
    FINAL_EPOCHS = 8  # enough to converge on full data
    for epoch in range(FINAL_EPOCHS):
        model.train()
        epoch_loss = 0
        for X_b, yl_b, ys_b, mfe_b, mae_b, tip_b, peff_b in final_dl:
            X_b = X_b.to(DEVICE)
            yl_b, ys_b = yl_b.to(DEVICE), ys_b.to(DEVICE)
            mfe_b, mae_b = mfe_b.to(DEVICE), mae_b.to(DEVICE)
            tip_b, peff_b = tip_b.to(DEVICE), peff_b.to(DEVICE)
            optimizer.zero_grad()
            pl, ps, p_mfe, p_mae, p_tip, p_peff = model(X_b)
            base_loss = 0.55 * criterion(pl, yl_b) + 0.45 * criterion(ps, ys_b)
            path_loss = (criterion(p_mfe, mfe_b) + criterion(p_mae, mae_b)
                         + nn.MSELoss()(p_tip, tip_b) + nn.MSELoss()(p_peff, peff_b))
            weights = (torch.abs(yl_b) + torch.abs(ys_b)).clamp(max=0.005)
            weights = weights / (weights.mean() + 1e-8)
            weighted_loss = (weights * (0.55 * (pl - yl_b)**2 + 0.45 * (ps - ys_b)**2)).mean()
            rank_target = torch.sign(yl_b - ys_b)
            rank_loss = torch.relu(0.001 - (pl - ps) * rank_target).mean()
            loss = 0.4 * base_loss + 0.2 * path_loss + 0.2 * weighted_loss + 0.2 * rank_loss
            if PNL_LOSS:
                pred_edge = torch.maximum(pl, ps)
                direction = (pl > ps).float() * 2 - 1
                loss = 0.7 * loss + 0.3 * (-(direction * pred_edge).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= len(final_ds)
        scheduler.step(epoch_loss)
    log.info(f"    Final model trained ({FINAL_EPOCHS} epochs, loss={epoch_loss:.8f})")

    # Save (move to CPU for portability)
    os.makedirs("models", exist_ok=True)
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
        "label_type": "raw_edge",
        "corr_long": avg_corr_l,
        "corr_short": avg_corr_s,
    }, "models/model_v6_lstm.pt")
    log.info("  Saved models/model_v6_lstm.pt")


if __name__ == "__main__":
    main()
