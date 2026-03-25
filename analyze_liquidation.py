"""
Liquidation Cascade — Deep Dive Analysis
=========================================
Pulls all Liquidation Cascade trades + signal_log evaluations and
analyzes exactly when the strategy wins vs loses.

Answers:
  - What signal conditions lead to wins vs losses?
  - Which regimes / sessions have edge?
  - What liq_total / liq_imbalance thresholds actually predict outcome?
  - How does win rate vary by secs_left, vol_range, funding regime?

Usage:
  $env:DB_URL="postgresql://..."
  python analyze_liquidation.py
"""

import os
import sys
import math
import logging
from collections import defaultdict

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

DB_URL = os.environ.get("DB_URL", "")


# ─────────────────────────────────────────────────────────────── FETCH ────────

def connect():
    if not DB_URL:
        log.error("Set DB_URL")
        sys.exit(1)
    try:
        return psycopg2.connect(DB_URL, connect_timeout=15)
    except Exception as e:
        log.error(f"Connection failed: {e}")
        sys.exit(1)


def fetch_liq_trades(conn) -> list:
    """All resolved Liquidation Cascade trades with signal_data."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            t.id,
            t.condition_id,
            t.side,
            t.price,
            t.size,
            t.actual_win,
            t.resolved_outcome,
            t.pnl,
            t.signal_data,
            t.regime,
            t.session,
            t.activity,
            t.day_type,
            t.created_at
        FROM trades t
        WHERE t.strategy        = 'Liquidation Cascade'
          AND t.action          = 'OPEN'
          AND t.resolved_outcome IS NOT NULL
          AND t.actual_win       IS NOT NULL
        ORDER BY t.created_at ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    log.info(f"Fetched {len(rows)} Liquidation Cascade trades")
    return rows


def fetch_liq_snapshots(conn) -> list:
    """
    Snapshot rows joined to nearest signal_log entry for Liquidation Cascade.
    Only for markets where a trade was placed.
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            ms.condition_id,
            ms.secs_left,
            ms.outcome_binary,
            ms.resolved_outcome,
            ms.liq_total,
            ms.liq_long,
            ms.liq_short,
            ms.liq_imbalance,
            ms.liq_delta,
            ms.liq_accel,
            ms.vol_range_pct,
            ms.momentum_30s,
            ms.momentum_60s,
            ms.ob_imbalance,
            ms.p_market,
            ms.funding_rate,
            ms.funding_zscore,
            ms.regime,
            ms.session,
            ms.activity,
            ms.phase_early,
            ms.phase_mid,
            ms.phase_late,
            ms.phase_final
        FROM market_snapshots ms
        WHERE ms.resolved_outcome IS NOT NULL
          AND ms.outcome_binary   IS NOT NULL
          AND ms.liq_total        IS NOT NULL
        ORDER BY ms.created_at ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    log.info(f"Fetched {len(rows):,} snapshot rows with liq data")
    return rows


# ─────────────────────────────────────────────────────────── HELPERS ──────────

def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _sep(char="─", w=68):
    print(char * w)


def _pct(n, d):
    return f"{n/d*100:.1f}%" if d > 0 else "n/a"


def _mean(vals):
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _fmt(v, d=2):
    return f"{v:.{d}f}" if v is not None else "null"


def bucket_analysis(rows, bucket_fn, label_fn, y_fn, title):
    """
    Generic bucketed win rate analysis.
    bucket_fn(row) -> bucket key
    label_fn(key)  -> display string
    y_fn(row)      -> 1/0 outcome
    """
    buckets = defaultdict(lambda: {"wins": 0, "total": 0, "vals": []})
    for row in rows:
        k = bucket_fn(row)
        if k is None:
            continue
        y = y_fn(row)
        if y is None:
            continue
        buckets[k]["wins"]  += y
        buckets[k]["total"] += 1

    print(f"\n  {title}")
    _sep()
    print(f"  {'Bucket':<30} {'N':>6}  {'Wins':>5}  {'WR':>7}  Bar")
    print(f"  {'─'*30} {'─'*6}  {'─'*5}  {'─'*7}  {'─'*20}")
    for k in sorted(buckets.keys()):
        b   = buckets[k]
        n   = b["total"]
        w   = b["wins"]
        wr  = w / n if n > 0 else 0
        bar = "#" * int(wr * 20)
        lbl = label_fn(k)
        print(f"  {lbl:<30} {n:>6,}  {w:>5}  {_pct(w,n):>7}  {bar}")


# ─────────────────────────────────────────────────────────────── MAIN ─────────

def main():
    conn   = connect()
    trades = fetch_liq_trades(conn)
    snaps  = fetch_liq_snapshots(conn)
    conn.close()

    if not trades:
        print("No Liquidation Cascade trades found.")
        return

    import json

    # Parse signal_data JSON
    for t in trades:
        sd = t.get("signal_data") or {}
        if isinstance(sd, str):
            try:
                sd = json.loads(sd)
            except Exception:
                sd = {}
        t["sig"] = sd

    wins   = [t for t in trades if t["actual_win"]]
    losses = [t for t in trades if not t["actual_win"]]
    n      = len(trades)
    w      = len(wins)

    print()
    print("=" * 68)
    print("  LIQUIDATION CASCADE — DEEP DIVE ANALYSIS")
    print("=" * 68)
    print(f"  Total trades : {n}")
    print(f"  Wins         : {w}  ({_pct(w, n)})")
    print(f"  Losses       : {len(losses)}  ({_pct(len(losses), n)})")
    print(f"  Total PnL    : ${sum(_f(t['pnl']) or 0 for t in trades):+.2f}")

    # ── 1. Signal conditions: wins vs losses ──────────────────────────────────
    print()
    print("  [1] SIGNAL CONDITIONS — wins vs losses")
    _sep()

    def _sig_stats(group, label):
        ll    = [_f(t["sig"].get("long_liqs"))  for t in group]
        sl    = [_f(t["sig"].get("short_liqs")) for t in group]
        vr    = [_f(t["sig"].get("vol_range"))  for t in group]
        inten = [_f(t["sig"].get("intensity"))  for t in group]
        total = [l + s for l, s in zip(ll, sl)
                 if l is not None and s is not None]
        imbal = [(l - s) / (l + s) for l, s in zip(ll, sl)
                 if l is not None and s is not None and (l + s) > 0]

        print(f"\n  {label} (n={len(group)})")
        print(f"    long_liqs  : mean=${_fmt(_mean(ll), 0)}  "
              f"  short_liqs: mean=${_fmt(_mean(sl), 0)}")
        print(f"    total_liqs : mean=${_fmt(_mean(total), 0)}")
        print(f"    imbalance  : mean={_fmt(_mean(imbal), 3)}")
        print(f"    vol_range  : mean={_fmt(_mean(vr), 4)}")
        print(f"    intensity  : mean={_fmt(_mean(inten), 3)}")

    _sig_stats(wins,   "WINS")
    _sig_stats(losses, "LOSSES")

    # ── 2. Win rate by side ────────────────────────────────────────────────────
    print()
    print("  [2] WIN RATE BY SIDE (UP = bet longs liquidated / short squeeze)")
    _sep()
    for side in ("UP", "DOWN"):
        side_trades = [t for t in trades if t["side"] == side]
        side_wins   = [t for t in side_trades if t["actual_win"]]
        print(f"  {side:<6} : {len(side_wins)}/{len(side_trades)}  "
              f"({_pct(len(side_wins), len(side_trades))})")

    # ── 3. Win rate by session ────────────────────────────────────────────────
    bucket_analysis(
        trades,
        bucket_fn = lambda r: r.get("session", "UNKNOWN"),
        label_fn  = lambda k: k,
        y_fn      = lambda r: int(r["actual_win"]),
        title     = "[3] WIN RATE BY SESSION",
    )

    # ── 4. Win rate by activity ───────────────────────────────────────────────
    bucket_analysis(
        trades,
        bucket_fn = lambda r: r.get("activity", "UNKNOWN"),
        label_fn  = lambda k: k,
        y_fn      = lambda r: int(r["actual_win"]),
        title     = "[4] WIN RATE BY ACTIVITY LEVEL",
    )

    # ── 5. Win rate by regime ─────────────────────────────────────────────────
    bucket_analysis(
        trades,
        bucket_fn = lambda r: r.get("regime", "UNKNOWN"),
        label_fn  = lambda k: k,
        y_fn      = lambda r: int(r["actual_win"]),
        title     = "[5] WIN RATE BY REGIME",
    )

    # ── 6. Win rate by liq_total bucket ──────────────────────────────────────
    def liq_bucket(row):
        ll = _f(row["sig"].get("long_liqs"))
        sl = _f(row["sig"].get("short_liqs"))
        if ll is None or sl is None:
            return None
        t = ll + sl
        if t < 50_000:   return "< $50k"
        if t < 150_000:  return "$50k–$150k"
        if t < 300_000:  return "$150k–$300k"
        if t < 500_000:  return "$300k–$500k"
        return "> $500k"

    bucket_analysis(
        trades,
        bucket_fn = liq_bucket,
        label_fn  = lambda k: k,
        y_fn      = lambda r: int(r["actual_win"]),
        title     = "[6] WIN RATE BY TOTAL LIQUIDATION SIZE",
    )

    # ── 7. Win rate by imbalance strength ─────────────────────────────────────
    def imbal_bucket(row):
        ll = _f(row["sig"].get("long_liqs"))
        sl = _f(row["sig"].get("short_liqs"))
        if ll is None or sl is None:
            return None
        t = ll + sl
        if t == 0:
            return None
        imb = max(ll, sl) / t
        if imb < 0.70:  return "0.65–0.70 (weak)"
        if imb < 0.80:  return "0.70–0.80"
        if imb < 0.90:  return "0.80–0.90"
        return "0.90–1.00 (strong)"

    bucket_analysis(
        trades,
        bucket_fn = imbal_bucket,
        label_fn  = lambda k: k,
        y_fn      = lambda r: int(r["actual_win"]),
        title     = "[7] WIN RATE BY IMBALANCE STRENGTH (dominant/total)",
    )

    # ── 8. Win rate by vol_range ──────────────────────────────────────────────
    def vol_bucket(row):
        vr = _f(row["sig"].get("vol_range"))
        if vr is None:
            return None
        if vr < 0.05:   return "< 0.05% (dead)"
        if vr < 0.10:   return "0.05–0.10%"
        if vr < 0.15:   return "0.10–0.15%"
        if vr < 0.25:   return "0.15–0.25%"
        return "> 0.25% (volatile)"

    bucket_analysis(
        trades,
        bucket_fn = vol_bucket,
        label_fn  = lambda k: k,
        y_fn      = lambda r: int(r["actual_win"]),
        title     = "[8] WIN RATE BY VOLATILITY (vol_range_pct)",
    )

    # ── 9. Win rate by intensity ──────────────────────────────────────────────
    def intensity_bucket(row):
        i = _f(row["sig"].get("intensity"))
        if i is None:
            return None
        if i < 0.10:  return "0.00–0.10 (very weak)"
        if i < 0.20:  return "0.10–0.20"
        if i < 0.30:  return "0.20–0.30"
        if i < 0.50:  return "0.30–0.50"
        return "0.50+ (strong)"

    bucket_analysis(
        trades,
        bucket_fn = intensity_bucket,
        label_fn  = lambda k: k,
        y_fn      = lambda r: int(r["actual_win"]),
        title     = "[9] WIN RATE BY KELLY INTENSITY",
    )

    # ── 10. Snapshot-level analysis: does liq signal predict outcome? ─────────
    print()
    print("  [10] SNAPSHOT ANALYSIS — liq_total vs outcome across all markets")
    _sep()
    print("  (all resolved snapshot rows, not just traded markets)")

    def snap_liq_bucket(row):
        lt = _f(row.get("liq_total"))
        if lt is None:
            return None
        if lt == 0:        return "0 (no activity)"
        if lt < 10_000:    return "< $10k"
        if lt < 50_000:    return "$10k–$50k"
        if lt < 150_000:   return "$50k–$150k"
        if lt < 300_000:   return "$150k–$300k"
        return "> $300k"

    bucket_analysis(
        snaps,
        bucket_fn = snap_liq_bucket,
        label_fn  = lambda k: k,
        y_fn      = lambda r: int(_f(r.get("outcome_binary")) or 0),
        title     = "liq_total bucket vs UP outcome rate",
    )

    # ── 11. liq_imbalance direction vs outcome ────────────────────────────────
    print()
    print("  [11] LIQ_IMBALANCE DIRECTION vs OUTCOME")
    _sep()
    print("  liq_imbalance > 0 = more long liqs (bearish signal = bet DOWN)")
    print("  liq_imbalance < 0 = more short liqs (bullish signal = bet UP)")
    print()

    up_when_pos   = sum(1 for r in snaps
                        if _f(r.get("liq_imbalance")) and
                        _f(r.get("liq_imbalance")) > 0.1 and
                        _f(r.get("outcome_binary")) == 1.0 and
                        _f(r.get("liq_total")) and
                        _f(r.get("liq_total")) > 10_000)
    tot_when_pos  = sum(1 for r in snaps
                        if _f(r.get("liq_imbalance")) and
                        _f(r.get("liq_imbalance")) > 0.1 and
                        _f(r.get("liq_total")) and
                        _f(r.get("liq_total")) > 10_000)

    up_when_neg   = sum(1 for r in snaps
                        if _f(r.get("liq_imbalance")) and
                        _f(r.get("liq_imbalance")) < -0.1 and
                        _f(r.get("outcome_binary")) == 1.0 and
                        _f(r.get("liq_total")) and
                        _f(r.get("liq_total")) > 10_000)
    tot_when_neg  = sum(1 for r in snaps
                        if _f(r.get("liq_imbalance")) and
                        _f(r.get("liq_imbalance")) < -0.1 and
                        _f(r.get("liq_total")) and
                        _f(r.get("liq_total")) > 10_000)

    print(f"  Long-heavy liqs (imbal > 0.1, total > $10k):")
    print(f"    UP rate: {_pct(up_when_pos, tot_when_pos)}  (n={tot_when_pos})")
    print(f"    Strategy bets DOWN in this case")
    print()
    print(f"  Short-heavy liqs (imbal < -0.1, total > $10k):")
    print(f"    UP rate: {_pct(up_when_neg, tot_when_neg)}  (n={tot_when_neg})")
    print(f"    Strategy bets UP in this case")

    # ── 12. Regime filter recommendation ─────────────────────────────────────
    print()
    print("  [12] REGIME FILTER RECOMMENDATION")
    _sep()

    regime_wins = defaultdict(lambda: {"w": 0, "n": 0})
    for t in trades:
        r = t.get("regime", "UNKNOWN")
        regime_wins[r]["n"] += 1
        if t["actual_win"]:
            regime_wins[r]["w"] += 1

    session_wins = defaultdict(lambda: {"w": 0, "n": 0})
    for t in trades:
        s = t.get("session", "UNKNOWN")
        session_wins[s]["n"] += 1
        if t["actual_win"]:
            session_wins[s]["w"] += 1

    print()
    print("  Regimes where WR > 65% (keep trading):")
    for regime, v in sorted(regime_wins.items(), key=lambda x: x[0] or ''):
        if not regime:
            continue
        wr = v["w"] / v["n"] if v["n"] > 0 else 0
        if wr >= 0.65 and v["n"] >= 5:
            print(f"    {regime:<15} {v['w']}/{v['n']}  WR={wr*100:.1f}%  [KEEP]")

    print()
    print("  Regimes where WR < 50% (consider skipping):")
    for regime, v in sorted(regime_wins.items(), key=lambda x: x[0] or ''):
        if not regime:
            continue
        wr = v["w"] / v["n"] if v["n"] > 0 else 0
        if wr < 0.50 and v["n"] >= 5:
            print(f"    {regime:<15} {v['w']}/{v['n']}  WR={wr*100:.1f}%  [SKIP?]")

    print()
    print("  Sessions where WR > 65% (keep trading):")
    for session, v in sorted(session_wins.items(), key=lambda x: x[0] or ''):
        if not session:
            continue
        wr = v["w"] / v["n"] if v["n"] > 0 else 0
        if wr >= 0.65 and v["n"] >= 5:
            print(f"    {session:<15} {v['w']}/{v['n']}  WR={wr*100:.1f}%  [KEEP]")

    print()
    print("  Sessions where WR < 50% (consider skipping):")
    for session, v in sorted(session_wins.items(), key=lambda x: x[0] or ''):
        if not session:
            continue
        wr = v["w"] / v["n"] if v["n"] > 0 else 0
        if wr < 0.50 and v["n"] >= 5:
            print(f"    {session:<15} {v['w']}/{v['n']}  WR={wr*100:.1f}%  [SKIP?]")

    # ── 13. Source breakdown ──────────────────────────────────────────────────
    print()
    print("  [13] DATA SOURCE (OKX vs Binance)")
    _sep()
    sources = defaultdict(lambda: {"w": 0, "n": 0})
    for t in trades:
        src = t["sig"].get("source", "unknown")
        sources[src]["n"] += 1
        if t["actual_win"]:
            sources[src]["w"] += 1
    for src, v in sorted(sources.items(), key=lambda x: x[0] or ''):
        wr = v["w"] / v["n"] if v["n"] > 0 else 0
        print(f"  {src:<20} {v['w']}/{v['n']}  WR={wr*100:.1f}%")

    # ── 14. PnL breakdown ─────────────────────────────────────────────────────
    print()
    print("  [14] PnL BREAKDOWN")
    _sep()
    total_pnl = sum(_f(t["pnl"]) or 0 for t in trades)
    win_pnl   = sum(_f(t["pnl"]) or 0 for t in wins)
    loss_pnl  = sum(_f(t["pnl"]) or 0 for t in losses)
    avg_win   = _mean([_f(t["pnl"]) for t in wins])
    avg_loss  = _mean([_f(t["pnl"]) for t in losses])

    print(f"  Total PnL    : ${total_pnl:+.2f}")
    print(f"  Win PnL      : ${win_pnl:+.2f}")
    print(f"  Loss PnL     : ${loss_pnl:+.2f}")
    print(f"  Avg win      : ${_fmt(avg_win, 2)}")
    print(f"  Avg loss     : ${_fmt(avg_loss, 2)}")
    if avg_win and avg_loss and avg_loss != 0:
        print(f"  Win/loss ratio: {abs(avg_win / avg_loss):.2f}x")

    print()
    print("=" * 68)
    print("  ANALYSIS COMPLETE")
    print("=" * 68)
    print()
    print("  Key questions to answer from this output:")
    print("  1. Does imbalance direction actually predict outcome? (section 11)")
    print("  2. Which regimes/sessions have real edge? (sections 3-5, 12)")
    print("  3. Is there a minimum liq_total threshold that filters noise? (6)")
    print("  4. Does higher intensity/imbalance actually mean higher WR? (7,9)")
    print("  5. What does 'good' vs 'bad' liq signal look like? (1)")


if __name__ == "__main__":
    if not DB_URL:
        print("ERROR: set DB_URL environment variable")
        sys.exit(1)
    main()
