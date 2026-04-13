"""
V5 + V6 Honest Ensemble Evaluation
====================================
Loads out-of-sample fold predictions saved by V5 and V6 walk-forward runs.
No retraining — just combines existing honest predictions.

Usage:
  1. Run train_v5_futures.py (saves cache/v5_fold_predictions.npz)
  2. Run train_v6_lstm.py    (saves cache/v6_fold_predictions.npz)
  3. Run eval_ensemble.py    (combines + evaluates)
"""

import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MIN_EDGE = 0.0002

def safe_corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else 0.0

# ── Load V5 predictions ──────────────────────────────────────────────────
log.info("Loading V5 fold predictions...")
try:
    v5_data = np.load("cache/v5_fold_predictions.npz")
except FileNotFoundError:
    log.error("cache/v5_fold_predictions.npz not found. Run train_v5_futures.py first.")
    sys.exit(1)

# Collect all V5 test predictions keyed by row index
v5_pred_up_map = {}    # row_idx → pred_up
v5_pred_down_map = {}  # row_idx → pred_down
v5_actual_long_map = {}
v5_actual_short_map = {}

n_v5_folds = 0
while f"fold_{n_v5_folds}_test_row_indices" in v5_data:
    n_v5_folds += 1

log.info(f"  V5: {n_v5_folds} folds")

for fi in range(n_v5_folds):
    row_idx = v5_data[f"fold_{fi}_test_row_indices"]
    pred_up = v5_data[f"fold_{fi}_pred_up"]
    pred_dn = v5_data[f"fold_{fi}_pred_down"]
    act_long = v5_data[f"fold_{fi}_actual_long"]
    act_short = v5_data[f"fold_{fi}_actual_short"]

    for j in range(len(row_idx)):
        ri = int(row_idx[j])
        v5_pred_up_map[ri] = pred_up[j]
        v5_pred_down_map[ri] = pred_dn[j]
        v5_actual_long_map[ri] = act_long[j]
        v5_actual_short_map[ri] = act_short[j]

log.info(f"  V5 predictions: {len(v5_pred_up_map):,} unique rows")

# ── Load V6 predictions ──────────────────────────────────────────────────
log.info("Loading V6 fold predictions...")
try:
    v6_data = np.load("cache/v6_fold_predictions.npz")
except FileNotFoundError:
    log.error("cache/v6_fold_predictions.npz not found. Run train_v6_lstm.py first.")
    sys.exit(1)

# V6 predictions are indexed by sequence position, not row index
# We need to align them — for now collect by fold
n_v6_folds = 0
while f"fold_{n_v6_folds}_pred_long" in v6_data:
    n_v6_folds += 1

log.info(f"  V6: {n_v6_folds} folds")

# Collect V6 predictions sequentially (test sets are contiguous temporal slices)
v6_all_pred_long = []
v6_all_pred_short = []
v6_all_actual_long = []
v6_all_actual_short = []
v6_all_pred_mfe = []
v6_all_pred_peff = []

for fi in range(n_v6_folds):
    v6_all_pred_long.extend(v6_data[f"fold_{fi}_pred_long"])
    v6_all_pred_short.extend(v6_data[f"fold_{fi}_pred_short"])
    v6_all_actual_long.extend(v6_data[f"fold_{fi}_actual_long"])
    v6_all_actual_short.extend(v6_data[f"fold_{fi}_actual_short"])
    if f"fold_{fi}_pred_mfe" in v6_data:
        v6_all_pred_mfe.extend(v6_data[f"fold_{fi}_pred_mfe"])
    if f"fold_{fi}_pred_peff" in v6_data:
        v6_all_pred_peff.extend(v6_data[f"fold_{fi}_pred_peff"])

v6_pl = np.array(v6_all_pred_long)
v6_ps = np.array(v6_all_pred_short)
v6_al = np.array(v6_all_actual_long)
v6_as = np.array(v6_all_actual_short)
v6_mfe = np.array(v6_all_pred_mfe) if v6_all_pred_mfe else np.zeros(len(v6_pl))
v6_peff = np.array(v6_all_pred_peff) if v6_all_pred_peff else np.zeros(len(v6_pl))

log.info(f"  V6 predictions: {len(v6_pl):,}")

# ── V5-only evaluation ───────────────────────────────────────────────────
v5_rows = sorted(v5_pred_up_map.keys())
v5_pu = np.array([v5_pred_up_map[r] for r in v5_rows])
v5_pd = np.array([v5_pred_down_map[r] for r in v5_rows])
v5_al = np.array([v5_actual_long_map[r] for r in v5_rows])
v5_as = np.array([v5_actual_short_map[r] for r in v5_rows])
v5_chosen = v5_pu > v5_pd
v5_edge = np.where(v5_chosen, v5_pu, v5_pd)
v5_actual = np.where(v5_chosen, v5_al, v5_as)
v5_mask = v5_edge > MIN_EDGE

# ── V6-only evaluation ───────────────────────────────────────────────────
v6_chosen = v6_pl > v6_ps
v6_edge = np.where(v6_chosen, v6_pl, v6_ps)
v6_actual = np.where(v6_chosen, v6_al, v6_as)
v6_mask = v6_edge > MIN_EDGE

print("\n" + "=" * 60)
print("  ENSEMBLE EVALUATION (from saved walk-forward predictions)")
print("=" * 60)

for name, mask, actual, edge, chosen in [
    ("V5 alone", v5_mask, v5_actual, v5_edge, v5_chosen),
    ("V6 alone", v6_mask, v6_actual, v6_edge, v6_chosen),
]:
    n = int(mask.sum())
    if n > 0:
        pnl = float(actual[mask].sum())
        wr = float((actual[mask] > 0).mean())
        corr = safe_corr(edge, actual)
    else:
        pnl = wr = corr = 0
    print(f"\n  {name}: trades={n:,}  PnL={pnl:+.4f}  WR={wr*100:.1f}%  corr={corr:+.4f}")

# ── Ensemble: align V5 and V6 by position ─────────────────────────────────
# V6 predictions are sequential (from walk-forward test sets)
# V5 predictions are by row index
# We can't directly align them without shared indices
# Instead, evaluate each independently and report combined decision rules

print(f"\n  NOTE: V5 has {len(v5_rows):,} predictions, V6 has {len(v6_pl):,}")
print(f"  These come from different fold structures (V5={n_v5_folds} folds, V6={n_v6_folds} folds)")
print(f"  Direct combination requires same data splits — showing independent results")

# ── V6 with quality filtering (what ensemble would use) ──────────────────
print(f"\n  V6 with quality filtering:")
for name, mask_fn in [
    ("V6 edge > threshold", lambda: v6_edge > MIN_EDGE),
    ("V6 top 20%", lambda: v6_edge > np.percentile(v6_edge, 80)),
    ("V6 top 10%", lambda: v6_edge > np.percentile(v6_edge, 90)),
    ("V6 edge + MFE > 0", lambda: (v6_edge > MIN_EDGE) & (v6_mfe > 0)),
    ("V6 edge + quality", lambda: (v6_edge > MIN_EDGE) & ((v6_mfe - np.abs(v6_mfe)) > np.median(v6_mfe - np.abs(v6_mfe)))),
]:
    mask = mask_fn()
    n = int(mask.sum())
    if n > 0:
        pnl = float(v6_actual[mask].sum())
        wr = float((v6_actual[mask] > 0).mean())
    else:
        pnl = wr = 0
    print(f"    {name:<30} trades={n:>6,}  PnL={pnl:>+8.4f}  WR={wr*100:>5.1f}%")

# ── V5 with thresholds ───────────────────────────────────────────────────
print(f"\n  V5 with quality filtering:")
for name, mask_fn in [
    ("V5 edge > threshold", lambda: v5_edge > MIN_EDGE),
    ("V5 top 20%", lambda: v5_edge > np.percentile(v5_edge, 80)),
    ("V5 top 10%", lambda: v5_edge > np.percentile(v5_edge, 90)),
]:
    mask = mask_fn()
    n = int(mask.sum())
    if n > 0:
        pnl = float(v5_actual[mask].sum())
        wr = float((v5_actual[mask] > 0).mean())
    else:
        pnl = wr = 0
    print(f"    {name:<30} trades={n:>6,}  PnL={pnl:>+8.4f}  WR={wr*100:>5.1f}%")

print(f"\n{'='*60}")
print(f"  To enable direct V5+V6 ensemble combination,")
print(f"  both models need to use the same temporal walk-forward splits.")
print(f"  This is a next-session task.")
print(f"{'='*60}")
