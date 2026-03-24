"""
Resolution Tracker v3.2
========================
Runs alongside multi_strategy.py on Railway.

Every 60 seconds it:
  1. Fetches all OPEN trades from Supabase where:
       - resolved_outcome IS NULL  (not yet resolved)
       - market_end_time IS NOT NULL (bot wrote it)
  2. Checks Polymarket CLOB API for the actual market outcome.
     Uses https://clob.polymarket.com/markets/{condition_id} and reads
     winner=true from the tokens array.
  3. If resolved: patches the exact row by integer id (primary key) with:
       - resolved_outcome (UP / DOWN / VOID)
       - actual_win (True / False / None for VOID)
       - polymarket_final_price
       - resolved_at timestamp
       - pnl / flat_pnl / edge based on real outcome
  4. If not yet resolved: skips — will retry next cycle
  5. If market_end_time > 1 hour ago and still unresolved: marks VOID + gave_up_at

INDEPENDENT RESOLUTION (v3.2):
  - signal_snapshots and signal_log are resolved independently of trades.
  - Any condition_id with unresolved rows in either table is checked against
    the CLOB API and patched, even if no trade was placed on that market.
  - Outcome cache is shared within each cycle to avoid duplicate CLOB calls.

NOTE: Patching is done by integer id (primary key) not trade_id UUID,
because Supabase REST API uuid column filtering is unreliable.
trade_id is still stored on every row for reference and traceability.

Summary queries only OPEN rows where resolved_outcome IS NOT NULL,
so all win-rate stats are 100% ground truth from Polymarket.
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone
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
CHECK_INTERVAL = 60
TAKER_FEE      = 0.0025
MIN_AGE_SECS   = 60
MAX_AGE_SECS   = 3600


# ---------------------------------------------------------------- SUPABASE ---

def sb_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


def fetch_pending_trades() -> list:
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={**sb_headers(), "Range": "0-999"},
            params={
                "action":           "eq.OPEN",
                "resolved_outcome": "is.null",
                "market_end_time":  "not.is.null",
                "select":           "id,trade_id,strategy,side,price,size,fee,"
                                    "condition_id,question,market_end_time,created_at",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"Failed to fetch pending trades: {e}")
        return []


def fetch_unresolved_condition_ids() -> set:
    """
    Collect all distinct condition_ids that have unresolved rows in
    signal_log or market_snapshots — regardless of whether a trade exists.
    """
    condition_ids = set()
    for table in ("signal_log", "market_snapshots"):
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/{table}",
                headers={**sb_headers(), "Range": "0-999"},
                params={
                    "resolved_outcome": "is.null",
                    "select":           "condition_id",
                },
                timeout=10,
            )
            resp.raise_for_status()
            for row in resp.json():
                cid = row.get("condition_id")
                if cid:
                    condition_ids.add(cid)
        except Exception as e:
            log.warning(f"Failed to fetch unresolved condition_ids from {table}: {e}")
    return condition_ids


def patch_trade(row_id: int, trade_id: str, outcome_data: dict) -> bool:
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={**sb_headers(), "Prefer": "return=representation"},
            params={"id": f"eq.{row_id}"},
            json=outcome_data,
            timeout=10,
        )
        resp.raise_for_status()
        updated = resp.json()
        if not updated:
            log.warning(f"patch_trade: no rows matched id={row_id} (trade_id={trade_id[:8]}...)")
            return False
        return True
    except Exception as e:
        log.warning(f"Failed to patch trade id={row_id}: {e}")
        return False


def resolve_signal_logs(outcome: str, condition_id: str):
    """
    Patch all signal_log rows for this condition_id with the resolved outcome.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/signal_log",
            headers={**sb_headers(), "Prefer": "return=representation"},
            params={"condition_id": f"eq.{condition_id}", "resolved_outcome": "is.null"},
            json={"resolved_outcome": outcome},
            timeout=10,
        )
        updated = resp.json() if resp.content else []
        if updated:
            log.info(f"Updated {len(updated)} signal_log row(s) for {condition_id[:12]}... → {outcome}")
    except Exception as e:
        log.warning(f"Signal log resolution failed: {e}")


def resolve_market_snapshots(outcome: str, condition_id: str):
    """
    Patch all market_snapshots rows for this condition_id with the resolved outcome.
    This is the primary ML training label — without it the snapshot rows are unlabelled.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/market_snapshots",
            headers={**sb_headers(), "Prefer": "return=representation"},
            params={"condition_id": f"eq.{condition_id}", "resolved_outcome": "is.null"},
            json={"resolved_outcome": outcome},
            timeout=10,
        )
        updated = resp.json() if resp.content else []
        if updated:
            log.info(f"Updated {len(updated)} market_snapshot(s) for {condition_id[:12]}... → {outcome}")
    except Exception as e:
        log.warning(f"Market snapshot resolution failed: {e}")


def fetch_strategy_summary() -> list:
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
    Fetch resolved outcome from Polymarket CLOB API.
    Checks tokens[].winner — when resolved, one token has winner=true.
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

        tokens     = m.get("tokens", [])
        up_token   = next((t for t in tokens if t.get("outcome", "").lower() == "up"),   None)
        down_token = next((t for t in tokens if t.get("outcome", "").lower() == "down"), None)

        if not up_token or not down_token:
            log.warning(f"Unexpected token structure for {condition_id[:12]}...: {tokens}")
            return {"resolved": False}

        log.debug(f"CLOB up=winner:{up_token.get('winner')} "
                  f"down=winner:{down_token.get('winner')} "
                  f"for {condition_id[:12]}...")

        if up_token.get("winner"):
            return {"resolved": True, "outcome": "UP",   "up_price": 1.0, "down_price": 0.0}
        elif down_token.get("winner"):
            return {"resolved": True, "outcome": "DOWN", "up_price": 0.0, "down_price": 1.0}
        else:
            return {"resolved": "ZERO_PRICES"}

    except Exception as e:
        log.warning(f"Outcome fetch failed for {condition_id}: {e}")
        return {"resolved": False}


# --------------------------------------------------------- MAIN LOOP ---------

def resolve_pending_trades(outcome_cache: dict) -> int:
    trades = fetch_pending_trades()

    if not trades:
        log.info("No pending unresolved trades.")
        return 0

    now = datetime.now(timezone.utc)
    log.info(f"Checking {len(trades)} unresolved trade(s)...")

    resolved_count = 0
    skipped_young  = 0
    gave_up_count  = 0
    waiting_count  = 0

    for trade in trades:
        row_id       = trade.get("id")
        trade_id     = trade.get("trade_id", "unknown")
        condition_id = trade.get("condition_id", "")

        if not row_id:
            log.warning("Trade has no integer id — skipping.")
            continue

        if not condition_id:
            log.warning(f"Trade id={row_id} has no condition_id — skipping.")
            continue

        end_time_str = trade.get("market_end_time")
        if not end_time_str:
            log.warning(f"Trade id={row_id} has no market_end_time — skipping.")
            continue

        try:
            market_end = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
        except ValueError:
            log.warning(f"Trade id={row_id} unparseable market_end_time: {end_time_str}")
            continue

        age_secs = (now - market_end).total_seconds()

        if age_secs < MIN_AGE_SECS:
            skipped_young += 1
            continue

        if age_secs > MAX_AGE_SECS:
            log.warning(f"Trade id={row_id} ({trade_id[:8]}...) market {age_secs:.0f}s old — marking VOID")
            void_fee = float(trade.get("size", 0)) * TAKER_FEE
            patch_trade(row_id, trade_id, {
                "resolved_outcome": "VOID",
                "actual_win":       None,
                "resolved_at":      now.isoformat(),
                "gave_up_at":       now.isoformat(),
                "pnl":              round(-void_fee, 4),
                "flat_pnl":         round(-void_fee, 4),
                "edge":             0.0,
            })
            gave_up_count += 1
            time.sleep(0.1)
            continue

        if condition_id not in outcome_cache:
            outcome_cache[condition_id] = fetch_market_outcome(condition_id)
            time.sleep(0.3)

        result = outcome_cache[condition_id]

        if result["resolved"] is False:
            waiting_count += 1
            log.debug(f"Trade id={row_id} not resolved yet (age={age_secs:.0f}s)")
            continue

        if result["resolved"] == "ZERO_PRICES":
            if age_secs > MAX_AGE_SECS:
                log.warning(f"Trade id={row_id} no winner after {age_secs:.0f}s — marking VOID")
                void_fee = float(trade.get("size", 0)) * TAKER_FEE
                patch_trade(row_id, trade_id, {
                    "resolved_outcome": "VOID",
                    "actual_win":       None,
                    "resolved_at":      now.isoformat(),
                    "gave_up_at":       now.isoformat(),
                    "pnl":              round(-void_fee, 4),
                    "flat_pnl":         round(-void_fee, 4),
                    "edge":             0.0,
                })
                gave_up_count += 1
            else:
                waiting_count += 1
                log.info(f"Trade id={row_id} ({trade_id[:8]}...) closed, no winner yet "
                         f"(age={age_secs:.0f}s) — waiting")
            continue

        outcome    = result["outcome"]
        up_price   = result["up_price"]
        down_price = result["down_price"]
        side       = trade.get("side", "")
        won        = (side == outcome)
        size       = float(trade.get("size", 0))
        entry_px   = float(trade.get("price", 0.5))
        fee        = size * TAKER_FEE * 2

        edge = round(1 / entry_px - 1, 4)

        if won:
            actual_pnl = size * (1 / entry_px - 1) - fee
            flat_pnl   = (size * (1 / entry_px - 1)) - (size * TAKER_FEE * 2)
        else:
            actual_pnl = -size - fee
            flat_pnl   = -size - (size * TAKER_FEE * 2)

        final_price  = up_price if side == "UP" else down_price
        outcome_data = {
            "resolved_outcome":       outcome,
            "actual_win":             won,
            "polymarket_final_price": final_price,
            "resolved_at":            now.isoformat(),
            "pnl":                    round(actual_pnl, 4),
            "flat_pnl":               round(flat_pnl, 4),
            "edge":                   edge,
        }

        if patch_trade(row_id, trade_id, outcome_data):
            status = "WIN " if won else "LOSS"
            log.info(f"Resolved: {status} | id={row_id} trade_id={trade_id[:8]}... | "
                     f"{side} vs {outcome} | PnL={actual_pnl:+.3f} | "
                     f"age={age_secs:.0f}s | {trade.get('strategy','')} | "
                     f"{trade.get('question','')[:50]}")
            resolve_signal_logs(outcome, condition_id)
            resolve_market_snapshots(outcome, condition_id)
            resolved_count += 1

    if skipped_young:
        log.info(f"Skipped {skipped_young} trade(s) — markets too recent")
    if waiting_count:
        log.info(f"Waiting on {waiting_count} trade(s) — Polymarket not resolved yet")
    if gave_up_count:
        log.info(f"Gave up on {gave_up_count} trade(s) — marked VOID after >{MAX_AGE_SECS}s")

    return resolved_count


def resolve_independent_signals(outcome_cache: dict):
    """
    Independently resolve signal_log for any condition_id that has unresolved
    rows, even if no trade was placed on that market.
    Shares outcome_cache with resolve_pending_trades to avoid duplicate CLOB calls.
    """
    condition_ids = fetch_unresolved_condition_ids()

    if not condition_ids:
        return

    log.info(f"Checking {len(condition_ids)} condition_id(s) for independent signal resolution...")

    for condition_id in condition_ids:
        if condition_id not in outcome_cache:
            outcome_cache[condition_id] = fetch_market_outcome(condition_id)
            time.sleep(0.3)

        result = outcome_cache[condition_id]

        if result.get("resolved") is True:
            outcome = result["outcome"]
            resolve_signal_logs(outcome, condition_id)
            resolve_market_snapshots(outcome, condition_id)
        elif result.get("resolved") == "ZERO_PRICES":
            log.debug(f"condition_id={condition_id[:12]}... closed but no winner yet — will retry")
        else:
            log.debug(f"condition_id={condition_id[:12]}... not resolved yet — will retry")


def print_summary():
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

    log.info("Resolution Tracker v3.2 — CLOB API, patching by integer id")
    log.info("Resolves: trades + signal_snapshots + signal_log (independent of trades)")
    log.info(f"Patience window: {MIN_AGE_SECS}s – {MAX_AGE_SECS}s after market end")
    log.info(f"Checking every {CHECK_INTERVAL}s")

    last_summary = 0

    while True:
        try:
            # Shared cache so we don't double-hit the CLOB API
            # for the same condition_id within one cycle
            outcome_cache: dict = {}

            count = resolve_pending_trades(outcome_cache)
            if count > 0:
                log.info(f"Resolved {count} new trade(s) this cycle")

            # Independently resolve snapshots/signal_log even with no trades
            resolve_independent_signals(outcome_cache)

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
