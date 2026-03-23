"""
Resolution Tracker v3
======================
Runs alongside multi_strategy.py on Railway.

Every 60 seconds it:
  1. Fetches all OPEN trades from Supabase where:
       - resolved_outcome IS NULL  (not yet resolved)
       - market_end_time IS NOT NULL (bot wrote it)
       - market_end_time is between 1 minute and 15 minutes ago
         (market has had time to settle, but we haven't given up yet)
  2. Checks Polymarket CLOB API for the actual market outcome
  3. If resolved: patches the exact row (by trade_id) with:
       - resolved_outcome (UP / DOWN / VOID)
       - actual_win (True / False / None for VOID)
       - polymarket_final_price
       - resolved_at timestamp
       - pnl / flat_pnl / edge based on real outcome
  4. If not yet resolved and market_end_time < 15 min ago: skips — will retry
  5. If market_end_time > 15 min ago and still unresolved: marks VOID + gave_up_at

Patching is done by trade_id (UUID), not by condition_id, so each row is
updated independently with no risk of cross-contamination between strategies.

Summary queries only OPEN rows where resolved_outcome IS NOT NULL,
so all win-rate stats are 100% ground truth from Polymarket.

Supabase schema additions required (run once):
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS trade_id         uuid;
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS market_end_time  timestamptz;
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS resolved_outcome text;
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS actual_win       boolean;
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS polymarket_final_price float;
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS resolved_at      timestamptz;
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS flat_pnl         float;
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS edge             float;
    ALTER TABLE trades ADD COLUMN IF NOT EXISTS gave_up_at       timestamptz;

Usage on Railway: add to Procfile as second worker
Or run locally: python resolver.py
"""

import os
import time
import logging
import json
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

CLOB_API       = "https://clob.polymarket.com"
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")
CHECK_INTERVAL = 60    # seconds between resolver loops
TAKER_FEE      = 0.0025
MIN_AGE_SECS   = 60    # don't check a market until at least 1 min after end_time
MAX_AGE_SECS   = 900   # give up after 15 minutes, mark VOID


# ---------------------------------------------------------------- SUPABASE ---

def sb_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


def fetch_pending_trades() -> list:
    """
    Fetch OPEN trades that:
      - have no resolved_outcome yet
      - have a market_end_time we can reason about
      - market ended at least MIN_AGE_SECS ago (give Polymarket time to settle)

    We fetch all unresolved trades and let the caller decide whether to skip
    (too young) or give up (too old), so the logic is in one place.
    """
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={**sb_headers(), "Range": "0-999"},
            params={
                "action":           "eq.OPEN",
                "resolved_outcome": "is.null",
                "market_end_time":  "not.is.null",
                "select":           "id,trade_id,strategy,side,price,size,"
                                    "condition_id,question,market_end_time,created_at",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch pending trades: {e}")
        return []


def patch_trade(trade_id: str, outcome_data: dict) -> bool:
    """
    Patch a single trade row by trade_id (UUID).
    This is the only write path for outcome fields — multi_strategy.py never writes these.
    """
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={**sb_headers(), "Prefer": "return=minimal"},
            params={"trade_id": f"eq.{trade_id}"},
            json=outcome_data,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"Failed to patch trade {trade_id}: {e}")
        return False


def resolve_snapshots(outcome: str, condition_id: str):
    """Update all snapshot rows for this market with the resolved outcome."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/signal_snapshots",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            params={"condition_id": f"eq.{condition_id}", "resolved_outcome": "is.null"},
            json={"resolved_outcome": outcome},
            timeout=10,
        )
        if resp.status_code in (200, 204):
            log.info(f"Updated snapshots for {condition_id[:12]}... → {outcome}")
    except Exception as e:
        log.warning(f"Snapshot resolution failed: {e}")


def fetch_strategy_summary() -> list:
    """
    Fetch resolved trades for win-rate summary.
    Queries OPEN rows where resolved_outcome IS NOT NULL —
    these are the only rows with real Polymarket outcome data.
    """
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={**sb_headers(), "Range": "0-9999"},
            params={
                "action":           "eq.OPEN",
                "resolved_outcome": "not.is.null",
                "select":           "strategy,actual_win,pnl,flat_pnl,resolved_outcome",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch summary: {e}")
        return []


# --------------------------------------------------------- POLYMARKET --------

def fetch_market_outcome(condition_id: str) -> dict:
    """
    Fetch the actual resolved outcome from Polymarket CLOB API.
    Uses https://clob.polymarket.com/markets/{condition_id} which correctly
    filters by condition ID — the Gamma API conditionId filter is broken.

    Returns:
      {"resolved": True,  "outcome": "UP"|"DOWN", "up_price": float, "down_price": float}
      {"resolved": False}      — market not settled yet, try again later
      {"resolved": "ZERO_PRICES"} — closed but no winner set yet
    """
    try:
        resp = requests.get(
            f"{CLOB_API}/markets/{condition_id}",
            timeout=10,
        )
        resp.raise_for_status()
        m = resp.json()

        if not m:
            return {"resolved": False}

        closed = m.get("closed", False)
        log.debug(f"CLOB closed={closed} tokens={m.get('tokens')}")

        if not closed:
            return {"resolved": False}

        tokens     = m.get("tokens", [])
        up_token   = next((t for t in tokens if t.get("outcome", "").lower() == "up"),   None)
        down_token = next((t for t in tokens if t.get("outcome", "").lower() == "down"), None)

        if not up_token or not down_token:
            log.warning(f"Unexpected token structure for {condition_id}: {tokens}")
            return {"resolved": "ZERO_PRICES"}

        if up_token.get("winner"):
            return {"resolved": True, "outcome": "UP",   "up_price": 1.0, "down_price": 0.0}
        elif down_token.get("winner"):
            return {"resolved": True, "outcome": "DOWN", "up_price": 0.0, "down_price": 1.0}
        else:
            # Closed but neither token marked as winner yet
            return {"resolved": "ZERO_PRICES"}

    except Exception as e:
        log.warning(f"Outcome fetch failed for {condition_id}: {e}")
        return {"resolved": False}


# --------------------------------------------------------- MAIN LOOP ---------

def resolve_pending_trades():
    """
    Check all pending trades and write real outcomes where available.
    Skips trades whose markets ended too recently (< MIN_AGE_SECS).
    Gives up on trades whose markets ended too long ago (> MAX_AGE_SECS).
    """
    trades = fetch_pending_trades()

    if not trades:
        log.info("No pending unresolved trades.")
        return 0

    now = datetime.now(timezone.utc)
    log.info(f"Checking {len(trades)} unresolved trade(s)...")

    resolved_count = 0
    skipped_young  = 0
    gave_up_count  = 0

    # Cache outcomes per condition_id to avoid hammering the API
    # when multiple strategies traded the same market.
    outcome_cache: dict = {}

    for trade in trades:
        trade_id     = trade.get("trade_id")
        condition_id = trade.get("condition_id", "")

        if not trade_id:
            log.warning(f"Trade row id={trade.get('id')} has no trade_id UUID — skipping. "
                        f"Run the backfill migration to add trade_id to old rows.")
            continue

        if not condition_id:
            log.warning(f"Trade {trade_id} has no condition_id — skipping.")
            continue

        # Parse market_end_time
        end_time_str = trade.get("market_end_time")
        if not end_time_str:
            log.warning(f"Trade {trade_id} has no market_end_time — skipping.")
            continue

        try:
            market_end = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
        except ValueError:
            log.warning(f"Trade {trade_id} has unparseable market_end_time: {end_time_str}")
            continue

        age_secs = (now - market_end).total_seconds()

        # Too young — Polymarket likely hasn't settled yet, come back next cycle.
        if age_secs < MIN_AGE_SECS:
            skipped_young += 1
            log.debug(f"Trade {trade_id[:8]}... market only {age_secs:.0f}s old — waiting")
            continue

        # Too old — we've been trying long enough. Mark VOID and move on.
        if age_secs > MAX_AGE_SECS:
            log.warning(f"Trade {trade_id[:8]}... market {age_secs:.0f}s old with no resolution "
                        f"— marking VOID (gave_up)")
            patch_trade(trade_id, {
                "resolved_outcome": "VOID",
                "actual_win":       None,
                "resolved_at":      now.isoformat(),
                "gave_up_at":       now.isoformat(),
                "pnl":              -(float(trade.get("size", 0)) + float(trade.get("fee", 0))),
                "flat_pnl":         -(float(trade.get("size", 0)) * (1 + TAKER_FEE * 2)),
                "edge":             -1.0,
            })
            gave_up_count += 1
            time.sleep(0.1)
            continue

        # Check outcome (cached per condition_id to save API calls)
        if condition_id not in outcome_cache:
            outcome_cache[condition_id] = fetch_market_outcome(condition_id)
            time.sleep(0.2)  # light rate limiting

        result = outcome_cache[condition_id]

        # Not resolved yet — try again next cycle.
        if result["resolved"] is False:
            log.info(f"Trade {trade_id[:8]}... ({condition_id[:12]}...) not resolved yet "
                     f"(age={age_secs:.0f}s) — will retry")
            continue

        # Zero prices — closed but winner not set yet.
        # Wait until MAX_AGE_SECS then give up.
        if result["resolved"] == "ZERO_PRICES":
            if age_secs > MAX_AGE_SECS:
                log.warning(f"Trade {trade_id[:8]}... zero prices for {age_secs:.0f}s — marking VOID")
                patch_trade(trade_id, {
                    "resolved_outcome": "VOID",
                    "actual_win":       None,
                    "resolved_at":      now.isoformat(),
                    "gave_up_at":       now.isoformat(),
                    "pnl":              -(float(trade.get("size", 0)) + float(trade.get("fee", 0))),
                    "flat_pnl":         -(float(trade.get("size", 0)) * (1 + TAKER_FEE * 2)),
                    "edge":             -1.0,
                })
                gave_up_count += 1
            else:
                log.info(f"Trade {trade_id[:8]}... closed, winner not set yet (age={age_secs:.0f}s) — waiting")
            continue

        # We have a real outcome — write it.
        outcome    = result["outcome"]   # "UP" or "DOWN"
        up_price   = result["up_price"]
        down_price = result["down_price"]

        side       = trade.get("side", "")
        won        = (side == outcome)
        size       = float(trade.get("size", 0))
        entry_px   = float(trade.get("price", 0.5))
        fee        = size * TAKER_FEE * 2  # round-trip fee

        if won:
            actual_pnl = size * (1 / entry_px - 1) - fee
            flat_pnl   = (size * (1 / entry_px - 1)) - (size * TAKER_FEE * 2)
            edge       = round(1 / entry_px - 1, 4)
        else:
            actual_pnl = -size - fee
            flat_pnl   = -size - (size * TAKER_FEE * 2)
            edge       = -1.0

        final_price = up_price if side == "UP" else down_price

        outcome_data = {
            "resolved_outcome":       outcome,
            "actual_win":             won,
            "polymarket_final_price": final_price,
            "resolved_at":            now.isoformat(),
            "pnl":                    round(actual_pnl, 4),
            "flat_pnl":               round(flat_pnl, 4),
            "edge":                   edge,
        }

        if patch_trade(trade_id, outcome_data):
            status = "WIN " if won else "LOSS"
            log.info(f"Resolved: {status} | trade_id={trade_id[:8]}... | "
                     f"{side} vs {outcome} | PnL={actual_pnl:+.3f} | "
                     f"age={age_secs:.0f}s | {trade.get('strategy','')} | "
                     f"{trade.get('question','')[:50]}")
            resolve_snapshots(outcome, condition_id)
            resolved_count += 1

    if skipped_young:
        log.info(f"Skipped {skipped_young} trade(s) — markets too recent, will check next cycle")
    if gave_up_count:
        log.info(f"Gave up on {gave_up_count} trade(s) — marked VOID after >{MAX_AGE_SECS}s")

    return resolved_count


def print_summary():
    """Print win-rate summary per strategy using real Polymarket outcomes."""
    trades = fetch_strategy_summary()
    if not trades:
        log.info("No resolved trades to summarise yet.")
        return

    stats = defaultdict(lambda: {"wins": 0, "losses": 0, "voids": 0, "pnl": 0.0, "flat_pnl": 0.0})

    for t in trades:
        s   = t.get("strategy", "unknown")
        won = t.get("actual_win")
        out = t.get("resolved_outcome", "")

        if out == "VOID":
            stats[s]["voids"] += 1
        elif won is True:
            stats[s]["wins"] += 1
        elif won is False:
            stats[s]["losses"] += 1

        stats[s]["pnl"]      += float(t.get("pnl") or 0)
        stats[s]["flat_pnl"] += float(t.get("flat_pnl") or 0)

    log.info("=" * 65)
    log.info("STRATEGY PERFORMANCE — ground truth from Polymarket outcomes")
    log.info("=" * 65)
    for strategy, s in sorted(stats.items()):
        decided = s["wins"] + s["losses"]
        wr      = s["wins"] / decided * 100 if decided else 0
        log.info(
            f"{strategy:<25} | WR={wr:.0f}% ({s['wins']}W/{s['losses']}L) | "
            f"PnL=${s['pnl']:+.2f} | flat=${s['flat_pnl']:+.2f} | "
            f"void={s['voids']}"
        )
    log.info("=" * 65)


def run():
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY environment variables required.")
        return

    log.info("Resolution Tracker v3 — using CLOB API for outcomes")
    log.info(f"Patience window: {MIN_AGE_SECS}s – {MAX_AGE_SECS}s after market end")
    log.info(f"Patching by trade_id (UUID) — one row per trade, resolver owns all outcomes")
    log.info(f"Checking every {CHECK_INTERVAL}s")

    last_summary = 0

    while True:
        try:
            count = resolve_pending_trades()
            if count > 0:
                log.info(f"Resolved {count} new trade(s) this cycle")

            if time.time() - last_summary > 1800:
                print_summary()
                last_summary = time.time()

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log.info("Resolver stopped.")
            print_summary()
            break
        except Exception as e:
            log.error(f"Error in resolver loop: {e}", exc_info=True)
            time.sleep(30)


if __name__ == "__main__":
    run()
