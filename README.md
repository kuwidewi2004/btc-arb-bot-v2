# BTC Arbitrage Bot v2

ML-driven BTC trading system for dYdX v4 perpetual futures, with Polymarket signal integration.

## Architecture

```
Data Collection (Railway)
  quant_engine.py  -- 9 real-time feeds -> Supabase snapshots every ~3s
  resolver.py      -- resolves market outcomes for training labels

Training (Local, GPU)
  train_model_v4_rest.py  -- V4: Polymarket P(profitable) classifier (LightGBM)
  train_v5_futures.py     -- V5: BTC 3-min edge regressor (LightGBM, 60 features)
  train_v6_lstm.py        -- V6: BTC 3-min edge sequence model (LSTM, 42 features)
  eval_ensemble.py        -- V5+V6 ensemble evaluation

Execution
  execution.py     -- dYdX v4 order execution
  start.py         -- Railway entry point
```

## Data Pipeline

- **9 real-time feeds**: Coinbase WS, Chainlink WS, Binance (liquidations, trades, depth), Polymarket CLOB WS, OKX WS, OKX REST (LSR), Deribit REST (IV)
- **~30k snapshots/day** stored in Supabase
- **Cached locally** via `fetch_cache.py` with GitHub Release archiving

## Models

### V4 -- Polymarket Classifier
- Predicts P(profitable) for Polymarket binary markets
- 75 features, walk-forward validation
- Used as meta-feature in V5 (fold-specific to avoid leakage)

### V5 -- Futures Edge Regressor
- Dual LightGBM regressors: E(edge_long), E(edge_short)
- 60 BTC microstructure features (no Polymarket)
- 3-minute lookahead, 0.02% maker fees
- Walk-forward by condition_id, fold-specific V4 scoring
- Extended eval: regime/session breakdown, confidence calibration

### V6 -- LSTM Sequence Model
- 3-layer LSTM (128 hidden) with attention pooling
- 42 pure BTC features, 60-snapshot sequences (~3 min)
- 6 output heads: edge_long, edge_short, MFE, MAE, time_in_profit, path_efficiency
- Multi-component loss: Huber + edge weighting + ranking
- 3-way split (train/val/test) with isotonic calibration
- GPU training (CUDA), Spearman rank correlation

## Training

```bash
# V4 (Polymarket)
python train_model_v4_rest.py

# V5 (Futures, ~30 min)
python train_v5_futures.py

# V6 (LSTM)
python train_v6_lstm.py           # full eval, 5 folds (~50 min)
python train_v6_lstm.py --quick   # quick eval, 3 folds (~15 min)
python train_v6_lstm.py --fast    # train + save only (~10 min)

# Ensemble
python eval_ensemble.py           # combines V5+V6 predictions
```

## Deployment

Runs on Railway with ONNX models (no LightGBM dependency in production).

```bash
# Set environment variables
SUPABASE_URL=...
SUPABASE_KEY=...

# Start
python start.py
```

## Key Findings

- Polymarket features don't predict BTC price direction (removed from V5/V6)
- V4 meta-feature leaks information when not fold-specific (fixed)
- MFE-based labels are biased positive by construction (reverted to endpoint)
- Model has real signal (Spearman +0.04-0.10) but not yet profitable after fees
- Edge exists primarily in VOLATILE/TREND regimes (~20% of data)
- 80% CALM regime data dilutes signal -- regime filter recommended for production
