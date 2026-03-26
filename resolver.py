"""
Resolution Tracker v3.5
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

INDEPENDENT RESOLUTION (v3.2+):
  - signal_snapshots and signal_log are resolved independently of trades.
  - Any condition_id with unresolved rows in either table is checked against
    the CLOB API and patched, even if no trade was placed on that market.
  - Outcome cache is shared within each cycle to avoid duplicate CLOB calls.

NOTE: Patching is done by integer id (primary key) not trade_id UUID,
because Supabase REST API uuid column filtering is unreliable.
trade_id is still stored on every row for reference and traceability.

Summary queries only OPEN rows where resolved_outcome IS NOT NULL,
so all win-rate stats are 100% ground truth from Polymarket.

Fixes applied (v3.5):
  - fetch_btc_price() added — Coinbase primary, Kraken fallback.
    Called once per resolved market to capture BTC price at resolution.
  - resolve_market_snapshots() now patches three additional fields:
      btc_resolution_price   — BTC spot price at the moment of resolution
                               (same value for all rows in a market)
      price_vs_resolution_pct — per-row: how much BTC moved between the
                               snapshot and resolution. Critical ML feature
                               for understanding reversal patterns.
                               = (btc_resolution - btc_at_snapshot) / btc_at_snapshot * 100
      secs_to_resolution     — per-row: how many seconds remained between
                               the snapshot and market resolution.
                               = (market_end_time - snapshot created_at)
  - resolve_pending_trades() now fetches BTC price at resolution and passes
    it to resolve_market_snapshots() and resolve_signal_logs().
  - resolve_independent_signals() similarly passes btc_resolution_price.
  - Required new Supabase columns (add before deploying):
      ALTER TABLE market_snapshots
        ADD COLUMN IF NOT EXISTS btc_resolution_price   numeric,
        ADD COLUMN IF NOT EXISTS price_vs_resolution_pct numeric,
        ADD COLUMN IF NOT EXISTS secs_to_resolution     numeric;
"""

import os
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

# Supabase page size for paginated fetches
_SB_PAGE_SIZE  = 1000


# --------------------------------------------------------- HTTP SESSION ------

def _make_session() -> requests.Session:
    """Return a Session with automatic retry + exponential backoff."""
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


# --------------------------------------------------------- BTC PRICE ----------

COINBASE_API = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
KRAKEN_API   = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"

def fetch_btc_price() -> float:
    """
    Fetch current BTC spot price at the moment of market resolution.
    Coinbase primary, Kraken fallback — same sources as multi_strategy.py.
    Called once per resolved market so the resolution price is captured
    as close to the actual resolution moment as possible.
    Returns 0.0 on failure — callers must handle gracefully.
    """
    try:
        r = _session.get(COINBASE_API, timeout=5)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception:
        pass
    try:
        r = _session.get(KRAKEN_API, timeout=5)
        r.raise_for_status()
        res = r.json()["result"]
        return float(res[list(res.keys())[0]]["c"][0])
    except Exception:
        log.warning("fetch_btc_price: both Coinbase and Kraken failed")
        return 0.0


# ---------------------------------------------------------------- SUPABASE ---

def sb_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


def _sb_fetch_all(table: str, params: dict) -> list:
    """
    FIX (#8): Paginate through Supabase results using Content-Range header.
    Supabase returns 'Content-Range: 0-999/2345' — we iterate until we
    have fetched all rows rather than silently truncating at page size.
    """
    rows   = []
    offset = 0

    while True:
        range_header = f"{offset}-{offset + _SB_PAGE_SIZE - 1}"
        try:
            resp = _session.get(
                f"{SUPABASE_URL}/rest/v1/{table}",
                headers={**sb_headers(), "Range": range_header},
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            rows.extend(page)

            # Parse Content-Range: "0-999/2345"
            content_range = resp.headers.get("Content-Range", "")
            if "/" in content_range:
                try:
                    total = int(content_range.split("/")[1])
                    if offset + _SB_PAGE_SIZE >= total:
                        break   # fetched everything
                except ValueError:
                    break
            else:
                # No total available — if page was full, try next page
                if len(page) < _SB_PAGE_SIZE:
                    break

            offset += _SB_PAGE_SIZE

        except Exception as e:
            log.warning(f"Supabase paginated fetch failed ({table}, offset={offset}): {e}")
            break

    return rows


def fetch_pending_trades() -> list:
    return _sb_fetch_all("trades", {
        "action":           "eq.OPEN",
        "resolved_outcome": "is.null",
        "market_end_time":  "not.is.null",
        "select":           "id,trade_id,strategy,side,price,size,fee,"
                            "condition_id,question,market_end_time,created_at",
    })


def fetch_unresolved_condition_ids() -> set:
    """
    Collect all distinct condition_ids that have unresolved rows in
    signal_log or market_snapshots — regardless of whether a trade exists.
    """
    condition_ids = set()
    for table in ("signal_log", "market_snapshots"):
        try:
            rows = _sb_fetch_all(table, {
                "resolved_outcome": "is.null",
                "select":           "condition_id",
            })
            for row in rows:
                cid = row.get("condition_id")
                if cid:
                    condition_ids.add(cid)
        except Exception as e:
            log.warning(f"Failed to fetch unresolved condition_ids from {table}: {e}")
    return condition_ids


def patch_trade(row_id: int, trade_id: str, outcome_data: dict) -> bool:
    try:
        resp = _session.patch(
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
    Patch all signal_log rows for this condition_id with resolved outcome data.

    Writes per-row:
      resolved_outcome  — "UP" or "DOWN"
      outcome_binary    — 1.0 if UP won, 0.0 if DOWN won
      actual_win        — True if the crowd's chosen side won
                          (crowd side = UP if p_market >= 0.5, else DOWN)

    These three columns make signal_log self-contained for ML analysis —
    no join to trades or market_snapshots needed to compute win rates.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        outcome_binary = 1.0 if outcome == "UP" else 0.0

        # Step 1: fetch unresolved signal_log rows for this market
        # so we can compute actual_win per row based on p_market
        fetch_resp = _session.get(
            f"{SUPABASE_URL}/rest/v1/signal_log",
            headers=sb_headers(),
            params={
                "condition_id":    f"eq.{condition_id}",
                "resolved_outcome": "is.null",
                "select":          "id,p_market,poly_fill_up,poly_fill_down",
            },
            timeout=10,
        )
        rows = fetch_resp.json() if fetch_resp.content else []
        if not rows:
            return

        # Step 2: batch update all rows with resolved fields
        # actual_win: crowd's side = UP if p_market >= 0.5, DOWN otherwise
        # For each row, compute whether that side won
        patched = 0
        for row in rows:
            pm = row.get("p_market")
            if pm is not None:
                crowd_side  = "UP" if float(pm) >= 0.5 else "DOWN"
                actual_win  = (crowd_side == outcome)
            else:
                actual_win = None

            patch_payload = {
                "resolved_outcome": outcome,
                "outcome_binary":   outcome_binary,
                "actual_win":       actual_win,
            }

            _session.patch(
                f"{SUPABASE_URL}/rest/v1/signal_log",
                headers=sb_headers(),
                params={"id": f"eq.{row['id']}"},
                json=patch_payload,
                timeout=10,
            )
            patched += 1

        log.info(f"Resolved {patched} signal_log row(s) for "
                 f"{condition_id[:12]}... → {outcome} "
                 f"(outcome_binary={outcome_binary})")

    except Exception as e:
        log.warning(f"Signal log resolution failed: {e}")


def resolve_market_snapshots(outcome: str, condition_id: str,
                             btc_resolution_price: float = 0.0,
                             market_end_time: str = ""):
    """
    Patch all market_snapshots rows for this condition_id with the resolved outcome.
    This is the primary ML training table — without it the snapshot rows are unlabelled.

    Patches six fields per row:
      resolved_outcome        — "UP" or "DOWN" (classification label)
      outcome_binary          — 1.0 if UP won, 0.0 if DOWN won
      edge_realized           — outcome_binary - p_market (per-row profitability label)
      btc_resolution_price    — BTC spot price at market resolution (same for all rows)
      price_vs_resolution_pct — per-row: % BTC moved from snapshot to resolution
                                = (btc_resolution - btc_at_snapshot) / btc_at_snapshot * 100
                                Positive = BTC rose between snapshot and resolution.
                                Critical for learning reversal patterns — a snapshot
                                where BTC was rising but price_vs_resolution_pct is
                                negative means BTC reversed before resolution.
      secs_to_resolution      — per-row: seconds between snapshot and market end.
                                = market_end_time - snapshot created_at
                                Gives the model a continuous time-to-resolution signal
                                complementary to the phase flags.

    All per-row fields are computed individually to reflect conditions at each
    specific snapshot moment rather than a single market-level value.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        outcome_binary = 1.0 if outcome == "UP" else 0.0

        # Parse market end time for secs_to_resolution computation
        market_end_dt = None
        if market_end_time:
            try:
                market_end_dt = datetime.fromisoformat(
                    market_end_time.replace("Z", "+00:00"))
            except Exception:
                pass

        # Fetch all unresolved snapshot rows — need btc_price and created_at
        # in addition to p_market for the new per-row computations
        rows = _sb_fetch_all("market_snapshots", {
            "condition_id":     f"eq.{condition_id}",
            "resolved_outcome": "is.null",
            "select":           "id,p_market,btc_price,created_at",
        })

        if not rows:
            return

        updated_count = 0
        for row in rows:
            row_id          = row.get("id")
            p_market        = row.get("p_market")
            btc_at_snapshot = row.get("btc_price")
            created_at_str  = row.get("created_at")

            if not row_id:
                continue

            # edge_realized: how much edge did this snapshot have?
            # Positive = market underpriced the winning side at this moment
            edge_realized = round(outcome_binary - p_market, 6) \
                            if p_market is not None else None

            # price_vs_resolution_pct: did BTC move toward or away from
            # where it was at snapshot time? Captures reversal dynamics.
            price_vs_res_pct = None
            if btc_resolution_price and btc_at_snapshot:
                try:
                    price_vs_res_pct = round(
                        (btc_resolution_price - float(btc_at_snapshot))
                        / float(btc_at_snapshot) * 100, 6)
                except Exception:
                    pass

            # secs_to_resolution: continuous time remaining at snapshot
            secs_to_res = None
            if market_end_dt and created_at_str:
                try:
                    snap_dt = datetime.fromisoformat(
                        created_at_str.replace("Z", "+00:00"))
                    secs_to_res = round(
                        (market_end_dt - snap_dt).total_seconds(), 1)
                except Exception:
                    pass

            patch_payload = {
                "resolved_outcome":        outcome,
                "outcome_binary":          outcome_binary,
                "edge_realized":           edge_realized,
                "btc_resolution_price":    btc_resolution_price or None,
                "price_vs_resolution_pct": price_vs_res_pct,
                "secs_to_resolution":      secs_to_res,
            }

            try:
                resp = _session.patch(
                    f"{SUPABASE_URL}/rest/v1/market_snapshots",
                    headers={**sb_headers(), "Prefer": "return=minimal"},
                    params={"id": f"eq.{row_id}"},
                    json=patch_payload,
                    timeout=10,
                )
                resp.raise_for_status()
                updated_count += 1
            except Exception as e:
                log.warning(f"market_snapshot patch failed for id={row_id}: {e}")

        if updated_count:
            log.info(f"Patched {updated_count} snapshot(s) for "
                     f"{condition_id[:12]}... → {outcome} "
                     f"btc_res=${btc_resolution_price:.2f}"
                     if btc_resolution_price else
                     f"Patched {updated_count} snapshot(s) for "
                     f"{condition_id[:12]}... → {outcome} (no BTC price)")
    except Exception as e:
        log.warning(f"Market snapshot resolution failed: {e}")


def fetch_strategy_summary() -> list:
    return _sb_fetch_all("trades", {
        "action":           "eq.OPEN",
        "resolved_outcome": "not.is.null",
        "select":           "strategy,actual_win,pnl,flat_pnl,resolved_outcome",
    })


# --------------------------------------------------------- POLYMARKET --------

def fetch_market_outcome(condition_id: str) -> dict:
    """
    Fetch resolved outcome from Polymarket CLOB API.
    Checks tokens[].winner — when resolved, one token has winner=true.
    """
    try:
        resp = _session.get(
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

def _build_void_payload(size: float, now: datetime) -> dict:
    void_fee = size * TAKER_FEE
    return {
        "resolved_outcome": "VOID",
        "actual_win":       None,
        "resolved_at":      now.isoformat(),
        "gave_up_at":       now.isoformat(),
        "pnl":              round(-void_fee, 4),
        "flat_pnl":         round(-void_fee, 4),
        "edge":             0.0,
    }


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

        # --- Fetch outcome before age-based decisions so we don't double-VOID ---
        # FIX (#6): Always fetch the outcome first, then decide what to do.
        # This prevents the ZERO_PRICES branch from double-VOIDing a trade that
        # the early age check already handled.
        if condition_id not in outcome_cache:
            outcome_cache[condition_id] = fetch_market_outcome(condition_id)
            time.sleep(0.3)

        result = outcome_cache[condition_id]

        # If we have a clean resolution, handle it immediately regardless of age
        if result.get("resolved") is True:
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

                # Fetch BTC price at resolution moment — used to compute
                # price_vs_resolution_pct per snapshot row.
                # Cached in outcome_cache so we don't refetch for same condition_id.
                btc_res_key = f"btc_{condition_id}"
                if btc_res_key not in outcome_cache:
                    outcome_cache[btc_res_key] = fetch_btc_price()
                btc_res_price = outcome_cache[btc_res_key]

                resolve_signal_logs(outcome, condition_id)
                resolve_market_snapshots(
                    outcome, condition_id,
                    btc_resolution_price=btc_res_price,
                    market_end_time=end_time_str,
                )
                resolved_count += 1
            continue

        # Not resolved yet — check if we should give up
        if age_secs > MAX_AGE_SECS:
            log.warning(f"Trade id={row_id} ({trade_id[:8]}...) market {age_secs:.0f}s old — marking VOID")
            size = float(trade.get("size", 0))
            patch_trade(row_id, trade_id, _build_void_payload(size, now))
            gave_up_count += 1
            time.sleep(0.1)
            continue

        # Still within patience window — wait
        if result.get("resolved") == "ZERO_PRICES":
            log.info(f"Trade id={row_id} ({trade_id[:8]}...) closed, no winner yet "
                     f"(age={age_secs:.0f}s) — waiting")
        else:
            log.debug(f"Trade id={row_id} not resolved yet (age={age_secs:.0f}s)")
        waiting_count += 1

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

            # Fetch BTC price at resolution — cached per condition_id
            btc_res_key = f"btc_{condition_id}"
            if btc_res_key not in outcome_cache:
                outcome_cache[btc_res_key] = fetch_btc_price()
            btc_res_price = outcome_cache[btc_res_key]

            resolve_signal_logs(outcome, condition_id)
            resolve_market_snapshots(
                outcome, condition_id,
                btc_resolution_price=btc_res_price,
            )
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

    log.info("Resolution Tracker v3.5 — CLOB API, patching by integer id")
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
