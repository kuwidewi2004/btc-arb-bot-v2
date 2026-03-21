"""
Resolution Tracker v2
======================
Runs alongside multi_strategy.py on Railway.
Every 60 seconds it:
  1. Fetches all OPEN trades from Supabase that haven't been resolved
  2. Checks Polymarket for the actual market outcome
  3. Updates the trade row in Supabase with:
     - resolved_outcome (UP/DOWN)
     - actual_win (True/False)
     - polymarket_final_price
     - resolved_at timestamp
     - corrected pnl based on actual outcome

This gives you ground truth win rates per strategy.

Usage on Railway: add to Procfile as second worker
Or run locally: python resolver.py
"""

import os
import time
import logging
import json
import requests
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

GAMMA_API       = "https://gamma-api.polymarket.com"
SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
CHECK_INTERVAL  = 60
TAKER_FEE       = 0.0025


# ---------------------------------------------------------------- SUPABASE ---

def sb_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


def fetch_open_trades() -> list:
    """Fetch all OPEN trades from Supabase that haven't been resolved yet."""
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades?action=eq.OPEN&resolved_outcome=is.null&select=id,strategy,side,price,size,condition_id,question,created_at",
            headers={**sb_headers(), "Range": "0-999"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch open trades: {e}")
        return []


def update_trade_outcome(trade_id: int, outcome_data: dict):
    """Update a trade row in Supabase with resolution data."""
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={**sb_headers(), "Prefer": "return=minimal"},
            params={"id": f"eq.{trade_id}"},
            json=outcome_data,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"Failed to update trade {trade_id}: {e}")
        return False


def fetch_strategy_summary() -> list:
    """Fetch win rate summary per strategy from Supabase."""
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={**sb_headers(), "Range": "0-9999"},
            params={
                "action":  "eq.CLOSE",
                "select":  "strategy,actual_win,pnl",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch summary: {e}")
        return []


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


# --------------------------------------------------------- POLYMARKET --------

def fetch_market_outcome(condition_id: str) -> dict:
    """
    Fetch the actual resolved outcome from Polymarket.
    Returns dict with resolved, outcome, up_price, down_price.
    """
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={"conditionId": condition_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {"resolved": False}

        m      = data[0] if isinstance(data, list) else data
        closed = m.get("closed", False)
        if not closed:
            return {"resolved": False}

        outcome_prices = json.loads(m.get("outcomePrices", "[0,0]"))
        up_price   = float(outcome_prices[0])
        down_price = float(outcome_prices[1])

        if up_price > 0.9:
            outcome = "UP"
        elif down_price > 0.9:
            outcome = "DOWN"
        else:
            return {"resolved": False}

        return {
            "resolved":   True,
            "outcome":    outcome,
            "up_price":   up_price,
            "down_price": down_price,
        }
    except Exception as e:
        log.warning(f"Outcome fetch failed for {condition_id}: {e}")
        return {"resolved": False}


# --------------------------------------------------------- MAIN LOOP ---------

def resolve_open_trades():
    """Check all open trades and update with outcomes where available."""
    open_trades = fetch_open_trades()

    if not open_trades:
        log.info("No unresolved open trades found.")
        return 0

    log.info(f"Checking {len(open_trades)} unresolved trades...")
    resolved_count = 0

    for trade in open_trades:
        condition_id = trade.get("condition_id", "")
        if not condition_id:
            continue

        result = fetch_market_outcome(condition_id)
        log.info(f"Outcome for {condition_id[:16]}: {result}")
        if not result["resolved"]:
            continue

        outcome  = result["outcome"]
        side     = trade.get("side", "")
        won      = (side == outcome)
        size     = float(trade.get("size", 0))
        entry_px = float(trade.get("price", 0.5))
        fee      = size * TAKER_FEE * 2

        # Calculate actual PnL based on real outcome
        if won:
            actual_pnl = size * (1 / entry_px - 1) - fee
        else:
            actual_pnl = -size - fee

        final_price = result["up_price"] if side == "UP" else result["down_price"]

        flat_pnl = (10 * (1 / entry_px - 1) - 10 * TAKER_FEE * 2) if won                    else (-10 - 10 * TAKER_FEE * 2)
        edge     = round((1 / entry_px - 1) if won else -1.0, 4)
        outcome_data = {
            "resolved_outcome":       outcome,
            "actual_win":             won,
            "polymarket_final_price": final_price,
            "resolved_at":            datetime.now(timezone.utc).isoformat(),
            "pnl":                    round(actual_pnl, 4),
            "flat_pnl":               round(flat_pnl, 4),
            "edge":                   edge,
        }

        if update_trade_outcome(trade["id"], outcome_data):
            status = "WIN " if won else "LOSS"
            log.info(f"Resolved: {status} | {side} vs {outcome} | "
                     f"PnL={actual_pnl:+.3f} | {trade.get('strategy','')} | "
                     f"{trade.get('question','')[:50]}")
            resolve_snapshots(outcome, condition_id)
            resolved_count += 1

        time.sleep(0.2)  # avoid hammering APIs

    return resolved_count


def print_summary():
    """Print win rate summary per strategy."""
    trades = fetch_strategy_summary()
    if not trades:
        return

    from collections import defaultdict
    stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})

    for t in trades:
        s = t.get("strategy", "unknown")
        if t.get("actual_win"):
            stats[s]["wins"] += 1
        else:
            stats[s]["losses"] += 1
        stats[s]["pnl"] += float(t.get("pnl") or 0)

    log.info("=" * 55)
    log.info("STRATEGY PERFORMANCE (actual Polymarket outcomes)")
    log.info("=" * 55)
    for strategy, s in sorted(stats.items()):
        total = s["wins"] + s["losses"]
        wr    = s["wins"] / total * 100 if total else 0
        log.info(f"{strategy:<25} | WR={wr:.0f}% | "
                 f"PnL=${s['pnl']:+.2f} | {total} trades")
    log.info("=" * 55)


def run():
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY environment variables required.")
        return

    log.info("Resolution Tracker v2 — writing outcomes to Supabase")
    log.info(f"Checking every {CHECK_INTERVAL}s")

    last_summary = 0

    while True:
        try:
            count = resolve_open_trades()
            if count > 0:
                log.info(f"Resolved {count} new trade(s)")

            if time.time() - last_summary > 1800:
                print_summary()
                last_summary = time.time()

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log.info("Resolver stopped.")
            print_summary()
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            time.sleep(30)


if __name__ == "__main__":
    run()
