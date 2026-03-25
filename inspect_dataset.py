"""
ML Dataset Inspector
====================
Pulls all resolved market_snapshots from Supabase, applies cleaning rules,
and prints a structured report covering:

  1. Row / market counts and data completeness
  2. Null rates per feature
  3. Outcome balance and p_market distribution
  4. Feature correlation with outcome_binary
  5. p_market Brier score — the baseline your model must beat
  6. Leakage zone summary
  7. Per-phase signal quality

Run on Railway (or locally) with env vars set:
  SUPABASE_URL=https://xxx.supabase.co
  SUPABASE_KEY=your_service_role_key

Install deps if needed:
  pip install requests numpy
"""

import os
import sys
import math
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ CONFIG ---

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
_SB_PAGE_SIZE = 1000

# Columns we care about for the ML feature set
FEATURE_COLS = [
    "secs_left",
    "secs_to_resolution",
    "market_progress",
    "price_vs_open_pct",
    "price_vs_open_score",
    "momentum_10s",
    "momentum_30s",
    "momentum_60s",
    "momentum_120s",
    "cl_divergence",
    "cl_age",
    "cl_vs_open_pct",
    "liq_imbalance",
    "liq_delta",
    "liq_accel",
    "liq_total",
    "ob_imbalance",
    "ob_spread_pct",
    "ob_bid_delta",
    "ob_ask_delta",
    "vol_range_pct",
    "volume_buy_ratio",
    "p_market",
    "poly_spread",
    "poly_slip_up",
    "poly_deviation",
    "basis_pct",
    "funding_rate",
    "funding_zscore",
    "momentum_score",
    "volatility_pct",
    "flow_score",
    "liquidity_score",
    "interact_momentum_x_vol",
    "interact_ob_x_spread",
    "interact_liq_x_price_pos",
    "interact_momentum_x_progress",
]

LABEL_COLS = [
    "outcome_binary",
    "resolved_outcome",
    "edge_realized",
    "price_vs_resolution_pct",
    "btc_resolution_price",
]

META_COLS = [
    "condition_id",
    "created_at",
    "market_end_time",
    "secs_left",
    "phase_early",
    "phase_mid",
    "phase_late",
    "phase_final",
    "price_bucket",
    "regime",
    "session",
    "activity",
    "day_type",
]

# Leakage threshold — rows with less than this many seconds to resolution
# are too late-market to train on safely
LEAKAGE_SECS = 60

# Corrupted momentum threshold — rows where momentum_120s is exactly 0.0
# but secs_left > 120 are likely pre-fix startup artefacts
CORRUPT_MOMENTUM_SECS = 120


# ---------------------------------------------------------------- SUPABASE ---

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET", "POST", "PATCH"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

_session = _make_session()


def _sb_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


def _sb_fetch_all(table: str, params: dict) -> list:
    """Paginated Supabase fetch — identical pattern to resolver.py."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY must be set")
        sys.exit(1)

    rows   = []
    offset = 0
    while True:
        range_header = f"{offset}-{offset + _SB_PAGE_SIZE - 1}"
        try:
            resp = _session.get(
                f"{SUPABASE_URL}/rest/v1/{table}",
                headers={**_sb_headers(), "Range": range_header},
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            rows.extend(page)
            log.info(f"  fetched {len(rows)} rows so far...")

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
            log.warning(f"Fetch failed (offset={offset}): {e}")
            break

    return rows


# ------------------------------------------------------------------ HELPERS ---

def _f(v) -> float:
    """Safe float conversion — returns None on null/invalid."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(vals) -> float:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _std(vals) -> float:
    clean = [v for v in vals if v is not None]
    if len(clean) < 2:
        return None
    m = sum(clean) / len(clean)
    return math.sqrt(sum((x - m) ** 2 for x in clean) / len(clean))


def _percentile(vals, p) -> float:
    clean = sorted(v for v in vals if v is not None)
    if not clean:
        return None
    idx = (len(clean) - 1) * p / 100
    lo  = int(idx)
    hi  = lo + 1
    if hi >= len(clean):
        return clean[lo]
    return clean[lo] + (idx - lo) * (clean[hi] - clean[lo])


def _corr(xs, ys) -> float:
    """Pearson correlation between two lists, ignoring rows where either is None."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 10:
        return None
    n  = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num  = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    dx   = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    dy   = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _brier(probs, outcomes) -> float:
    """Brier score = mean((prob - outcome)^2). Lower is better. 0.25 = random."""
    pairs = [(p, o) for p, o in zip(probs, outcomes)
             if p is not None and o is not None]
    if not pairs:
        return None
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def _div(a, b) -> str:
    if b == 0:
        return "n/a"
    return f"{a / b * 100:.1f}%"


def _fmt(v, decimals=4) -> str:
    if v is None:
        return "null"
    return f"{v:.{decimals}f}"


def _bar(frac, width=30) -> str:
    """Simple ASCII bar chart for a fraction 0–1."""
    if frac is None:
        return " " * width
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _sep(char="─", width=72):
    print(char * width)


# ------------------------------------------------------------------- MAIN ---

def main():
    _sep("═")
    print("  ML DATASET INSPECTOR")
    print(f"  Supabase: {SUPABASE_URL[:40]}..." if SUPABASE_URL else "  !! NO SUPABASE_URL SET !!")
    _sep("═")

    # ── 1. Pull data ──────────────────────────────────────────────────────────
    print("\n[1/7] Pulling resolved market_snapshots from Supabase...")

    all_cols = ",".join(["id", "condition_id", "created_at", "market_end_time",
                         "resolved_outcome", "outcome_binary", "edge_realized",
                         "price_vs_resolution_pct", "btc_resolution_price",
                         "secs_left", "secs_to_resolution", "market_progress",
                         "phase_early", "phase_mid", "phase_late", "phase_final",
                         "price_bucket", "regime", "session", "activity", "day_type",
                         "p_market", "poly_spread", "poly_slip_up", "poly_deviation",
                         "price_vs_open_pct", "price_vs_open_score",
                         "momentum_10s", "momentum_30s", "momentum_60s", "momentum_120s",
                         "momentum_score", "volatility_pct", "flow_score",
                         "liquidity_score", "funding_zscore",
                         "cl_divergence", "cl_age", "cl_vs_open_pct",
                         "liq_imbalance", "liq_delta", "liq_accel", "liq_total",
                         "ob_imbalance", "ob_spread_pct", "ob_bid_delta", "ob_ask_delta",
                         "vol_range_pct", "volume_buy_ratio",
                         "basis_pct", "funding_rate",
                         "interact_momentum_x_vol", "interact_ob_x_spread",
                         "interact_liq_x_price_pos", "interact_momentum_x_progress"])

    rows = _sb_fetch_all("market_snapshots", {
        "resolved_outcome": "not.is.null",
        "select":           all_cols,
        "order":            "created_at.asc",
    })

    if not rows:
        log.error("No resolved rows returned — check credentials and schema")
        sys.exit(1)

    print(f"  → {len(rows)} rows fetched")

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    print("\n[2/7] Applying cleaning rules...")

    corrupted_momentum = 0
    secs_to_res_reconstructed = 0

    for row in rows:
        # Null out corrupted momentum_120s (pre-fix-5 startup artefacts)
        m120 = _f(row.get("momentum_120s"))
        sl   = _f(row.get("secs_left")) or 0
        if m120 == 0.0 and sl > CORRUPT_MOMENTUM_SECS:
            row["momentum_120s"] = None
            corrupted_momentum += 1

        # Reconstruct secs_to_resolution from secs_left if missing (pre-schema-fix rows)
        if row.get("secs_to_resolution") is None and row.get("secs_left") is not None:
            row["secs_to_resolution"] = row["secs_left"]
            secs_to_res_reconstructed += 1

    print(f"  → Nulled {corrupted_momentum} corrupted momentum_120s values")
    print(f"  → Reconstructed secs_to_resolution from secs_left for "
          f"{secs_to_res_reconstructed} rows")

    # Tag leakage rows
    leakage_rows  = []
    training_rows = []
    for row in rows:
        str_ = _f(row.get("secs_to_resolution")) or _f(row.get("secs_left")) or 999
        if str_ < LEAKAGE_SECS:
            leakage_rows.append(row)
        else:
            training_rows.append(row)

    # ── 3. Row / market counts ────────────────────────────────────────────────
    print("\n[3/7] Row and market counts")
    _sep()

    all_markets     = set(r["condition_id"] for r in rows)
    train_markets   = set(r["condition_id"] for r in training_rows)
    leakage_markets = set(r["condition_id"] for r in leakage_rows)

    print(f"  Total resolved rows       : {len(rows):>7,}")
    print(f"  Training rows (≥{LEAKAGE_SECS}s left)  : {len(training_rows):>7,}  "
          f"({_div(len(training_rows), len(rows))})")
    print(f"  Leakage rows  (<{LEAKAGE_SECS}s left)  : {len(leakage_rows):>7,}  "
          f"({_div(len(leakage_rows), len(rows))})")
    print(f"  Unique markets (total)    : {len(all_markets):>7,}")
    print(f"  Markets in training set   : {len(train_markets):>7,}")
    print(f"  Avg rows per market       : {len(rows) / max(len(all_markets), 1):>7.1f}")
    print(f"  Avg training rows/market  : "
          f"{len(training_rows) / max(len(train_markets), 1):.1f}")

    # Date range
    dates = sorted(r["created_at"] for r in rows if r.get("created_at"))
    if dates:
        print(f"  Date range                : {dates[0][:19]}  →  {dates[-1][:19]}")

    # ── 4. Outcome balance ────────────────────────────────────────────────────
    print("\n[4/7] Outcome balance and p_market distribution")
    _sep()

    outcomes = [r.get("resolved_outcome") for r in training_rows]
    up_rows   = [r for r in training_rows if r.get("resolved_outcome") == "UP"]
    down_rows = [r for r in training_rows if r.get("resolved_outcome") == "DOWN"]

    print(f"  UP   markets : {len(set(r['condition_id'] for r in up_rows)):>4}  "
          f"rows: {len(up_rows):>6,}  ({_div(len(up_rows), len(training_rows))})")
    print(f"  DOWN markets : {len(set(r['condition_id'] for r in down_rows)):>4}  "
          f"rows: {len(down_rows):>6,}  ({_div(len(down_rows), len(training_rows))})")

    # p_market distribution
    pm_all  = [_f(r.get("p_market")) for r in training_rows]
    pm_up   = [_f(r.get("p_market")) for r in up_rows]
    pm_down = [_f(r.get("p_market")) for r in down_rows]

    print()
    print(f"  p_market (all training rows):")
    print(f"    mean={_fmt(_mean(pm_all), 4)}  "
          f"std={_fmt(_std(pm_all), 4)}  "
          f"p10={_fmt(_percentile(pm_all, 10), 4)}  "
          f"p50={_fmt(_percentile(pm_all, 50), 4)}  "
          f"p90={_fmt(_percentile(pm_all, 90), 4)}")
    print(f"  p_market when UP   resolved: mean={_fmt(_mean(pm_up), 4)}")
    print(f"  p_market when DOWN resolved: mean={_fmt(_mean(pm_down), 4)}")

    # p_market bucket distribution
    buckets = {"<0.40": 0, "0.40-0.45": 0, "0.45-0.50": 0,
               "0.50-0.55": 0, "0.55-0.60": 0, ">0.60": 0}
    for p in pm_all:
        if p is None:
            continue
        if p < 0.40:   buckets["<0.40"] += 1
        elif p < 0.45: buckets["0.40-0.45"] += 1
        elif p < 0.50: buckets["0.45-0.50"] += 1
        elif p < 0.55: buckets["0.50-0.55"] += 1
        elif p < 0.60: buckets["0.55-0.60"] += 1
        else:          buckets[">0.60"] += 1

    print()
    print("  p_market bucket distribution:")
    total_pm = sum(buckets.values())
    for label, count in buckets.items():
        frac = count / total_pm if total_pm else 0
        print(f"    {label:12s} {_bar(frac, 25)} {count:5,}  ({frac*100:.1f}%)")

    # ── 5. Brier score baseline ───────────────────────────────────────────────
    print("\n[5/7] Brier score — p_market baseline")
    _sep()

    ob_train = [_f(r.get("outcome_binary")) for r in training_rows]
    pm_train = [_f(r.get("p_market"))       for r in training_rows]

    brier_market = _brier(pm_train, ob_train)
    brier_random = 0.25
    brier_perfect = 0.0

    print(f"  Random baseline (0.5 always)  : 0.2500")
    print(f"  p_market Brier score          : {_fmt(brier_market, 4)}  ← your model must beat this")
    print(f"  Perfect score                 : 0.0000")
    print()

    # p_market accuracy (directional) as sanity check
    correct = sum(1 for p, o in zip(pm_train, ob_train)
                  if p is not None and o is not None
                  and ((p >= 0.5 and o == 1.0) or (p < 0.5 and o == 0.0)))
    total_valid = sum(1 for p, o in zip(pm_train, ob_train)
                      if p is not None and o is not None)
    print(f"  p_market directional accuracy : {_div(correct, total_valid)}  "
          f"({correct:,} / {total_valid:,})")
    print()
    print(f"  Interpretation:")
    if brier_market is not None:
        improvement_needed = brier_market - 0.23  # ~0.23 is a decent model
        print(f"    p_market Brier {brier_market:.4f} means the crowd is already capturing")
        print(f"    most of the signal. Your model needs Brier < {brier_market:.4f} to add value.")
        print(f"    A strong model on this type of data typically reaches ~0.22–0.23.")

    # ── 6. Null rates per feature ─────────────────────────────────────────────
    print("\n[6/7] Feature null rates (training rows only)")
    _sep()

    print(f"  {'Feature':<35} {'Non-null':>8}  {'Null%':>6}  Bar")
    print(f"  {'─'*35} {'─'*8}  {'─'*6}  {'─'*25}")

    n = len(training_rows)
    null_summary = {}
    for col in FEATURE_COLS:
        non_null = sum(1 for r in training_rows if r.get(col) is not None)
        null_pct = (n - non_null) / n if n else 0
        null_summary[col] = null_pct
        bar = _bar(1 - null_pct, 25)
        flag = " ⚠" if null_pct > 0.10 else ""
        print(f"  {col:<35} {non_null:>8,}  {null_pct*100:>5.1f}%  {bar}{flag}")

    high_null = [(c, p) for c, p in null_summary.items() if p > 0.10]
    if high_null:
        print()
        print(f"  ⚠  Features with >10% nulls — handle before training:")
        for col, pct in sorted(high_null, key=lambda x: -x[1]):
            print(f"     {col:<35} {pct*100:.1f}% null")

    # ── 7. Feature correlations with outcome_binary ───────────────────────────
    print("\n[7/7] Feature correlation with outcome_binary")
    _sep()
    print("  (Pearson r — magnitude matters, sign tells direction)")
    print(f"  {'Feature':<35} {'r':>7}  {'|r| bar'}")
    print(f"  {'─'*35} {'─'*7}  {'─'*30}")

    ob_vals = [_f(r.get("outcome_binary")) for r in training_rows]

    correlations = {}
    for col in FEATURE_COLS:
        feat_vals = [_f(r.get(col)) for r in training_rows]
        r = _corr(feat_vals, ob_vals)
        if r is not None:
            correlations[col] = r

    # Sort by absolute correlation descending
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for col, r in sorted_corr:
        bar = _bar(abs(r), 30)
        sign = "+" if r >= 0 else "-"
        flag = " ◀ strong" if abs(r) > 0.15 else (" ◀ moderate" if abs(r) > 0.07 else "")
        print(f"  {col:<35} {sign}{abs(r):.4f}  {bar}{flag}")

    # ── 8. Phase breakdown ────────────────────────────────────────────────────
    print("\n[+] Bonus: signal quality by market phase")
    _sep()

    phase_map = {
        "phase_early": "Early  (>200s left)",
        "phase_mid":   "Mid    (100-200s)  ",
        "phase_late":  "Late   (30-100s)   ",
        "phase_final": "Final  (<30s)      ",
    }

    print(f"  {'Phase':<22} {'Rows':>7}  {'p_mkt Brier':>12}  {'p_mkt Acc':>10}  "
          f"{'Avg |mom30|':>12}")
    print(f"  {'─'*22} {'─'*7}  {'─'*12}  {'─'*10}  {'─'*12}")

    for phase_col, phase_label in phase_map.items():
        phase_rows = [r for r in training_rows if r.get(phase_col) == 1]
        if not phase_rows:
            continue

        ph_ob = [_f(r.get("outcome_binary")) for r in phase_rows]
        ph_pm = [_f(r.get("p_market"))       for r in phase_rows]
        ph_m30= [abs(_f(r.get("momentum_30s"))) for r in phase_rows
                 if _f(r.get("momentum_30s")) is not None]

        brier = _brier(ph_pm, ph_ob)
        acc   = sum(1 for p, o in zip(ph_pm, ph_ob)
                    if p is not None and o is not None
                    and ((p >= 0.5 and o == 1.0) or (p < 0.5 and o == 0.0)))
        acc_n = sum(1 for p, o in zip(ph_pm, ph_ob) if p is not None and o is not None)
        avg_mom = _mean(ph_m30)

        print(f"  {phase_label} {len(phase_rows):>7,}  "
              f"{_fmt(brier, 4):>12}  "
              f"{_div(acc, acc_n):>10}  "
              f"{_fmt(avg_mom, 5):>12}")

    # ── 9. Leakage zone summary ───────────────────────────────────────────────
    print("\n[+] Leakage zone summary (rows with <60s to resolution)")
    _sep()

    lk_ob = [_f(r.get("outcome_binary")) for r in leakage_rows]
    lk_pm = [_f(r.get("p_market"))       for r in leakage_rows]
    lk_brier = _brier(lk_pm, lk_ob)
    lk_acc = sum(1 for p, o in zip(lk_pm, lk_ob)
                 if p is not None and o is not None
                 and ((p >= 0.5 and o == 1.0) or (p < 0.5 and o == 0.0)))
    lk_acc_n = sum(1 for p, o in zip(lk_pm, lk_ob)
                   if p is not None and o is not None)

    print(f"  Leakage rows              : {len(leakage_rows):,}")
    print(f"  p_market Brier (leakage)  : {_fmt(lk_brier, 4)}  "
          f"(vs {_fmt(brier_market, 4)} in training set)")
    print(f"  p_market accuracy         : {_div(lk_acc, lk_acc_n)}")
    print()
    print(f"  These rows are EXCLUDED from training.")
    print(f"  Use them only to measure how well your model predicts at resolution time.")

    # ── 10. Train/test split preview ─────────────────────────────────────────
    print("\n[+] Recommended train/test split (temporal, by market)")
    _sep()

    sorted_markets = sorted(
        train_markets,
        key=lambda cid: min(
            r["created_at"] for r in training_rows
            if r["condition_id"] == cid and r.get("created_at")
        )
    )
    split_idx   = int(len(sorted_markets) * 0.80)
    train_set   = set(sorted_markets[:split_idx])
    test_set    = set(sorted_markets[split_idx:])

    train_data  = [r for r in training_rows if r["condition_id"] in train_set]
    test_data   = [r for r in training_rows if r["condition_id"] in test_set]

    print(f"  Train markets : {len(train_set):>4}  ({len(train_data):,} rows)  "
          f"earliest {sorted_markets[0][:12] if sorted_markets else '?'}...")
    print(f"  Test  markets : {len(test_set):>4}  ({len(test_data):,} rows)  "
          f"latest   {sorted_markets[-1][:12] if sorted_markets else '?'}...")
    print()
    print(f"  Split rule: all rows from a given condition_id go entirely")
    print(f"  into train OR test — never split across both.")
    print(f"  Temporal order preserved — test set is strictly more recent.")

    # ── Done ──────────────────────────────────────────────────────────────────
    _sep("═")
    print("  INSPECTION COMPLETE")
    print()
    print("  Next steps:")
    print("  1. Note any features with >10% nulls above — decide impute or drop")
    print("  2. Note the top correlated features — these are your first model inputs")
    print("  3. Run the ALTER TABLE + UPDATE SQL if secs_to_resolution is still null")
    print("  4. Build train/test split using the market-level temporal split above")
    print("  5. Train logistic regression first — verify Brier < p_market baseline")
    _sep("═")


if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: set SUPABASE_URL and SUPABASE_KEY environment variables")
        sys.exit(1)
    main()
