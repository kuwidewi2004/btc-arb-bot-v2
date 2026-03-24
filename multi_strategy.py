"""
Multi-Strategy Mechanical Edge Simulator v3
=============================================
Runs 6 independent mechanical edge strategies simultaneously in dry run.

Strategies:
  1. Chainlink lag arb     — Coinbase (real-time) vs Chainlink (lagging oracle)
  2. Funding rate reversion — extreme funding predicts reversion (OKX data)
  3. Liquidation cascade   — large liquidations predict direction (OKX data)
  4. Basis arbitrage       — futures premium/discount mean reversion (OKX data)
  5. Odds mispricing       — Polymarket odds deviating from 50/50
  6. Volume clock          — aggressive order flow before resolution

Data sources (all work on Railway US servers):
  - Spot prices:    Coinbase + Kraken fallback
  - Funding/Basis:  OKX public API (no geo-blocking)
  - Volume:         OKX klines (no geo-blocking)
  - Liquidations:   OKX REST (poll) + Binance WebSocket (real-time)
                    — combined covers ~80-90% of BTC perp liq volume
  - Chainlink:      Polymarket RTDS WebSocket (crypto_prices_chainlink)
                    — exact same feed Polymarket uses for resolution

CHANGES (v3):
  - Each trade gets a UUID (trade_id) written to Supabase on OPEN
  - StrategyTrade dataclass carries trade_id for reference
  - supabase_insert on CLOSE removed — resolver.py owns all outcome writes
  - Local in-memory balance/wins/losses tracking unchanged (used for live log only)
  - market_end_time written to OPEN row so resolver can filter intelligently

CHANGES (v3.1):
  - Chainlink price now sourced from Polymarket RTDS WebSocket
    (wss://ws-subscriptions-clob.polymarket.com/ws/) using the
    crypto_prices_chainlink topic — this is the exact feed Polymarket
    uses to resolve BTC up/down markets, replacing the delayed
    data.chain.link proxy which showed stale display-only data
  - Background thread maintains WebSocket connection with auto-reconnect
  - fetch_chainlink_price() now reads from shared in-memory cache
"""

import os
import time
import logging
import json
import sys
import io
import uuid
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import requests
import numpy as np
import websocket

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multi_strategy.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ CONFIG ---

POLYMARKET_HOST  = "https://clob.polymarket.com"
GAMMA_API        = "https://gamma-api.polymarket.com"
OKX_BASE         = "https://www.okx.com"
BTC_SYMBOL       = "BTCUSDT"
ETH_SYMBOL       = "ETHUSDT"
OKX_BTC          = "BTC-USDT-SWAP"

TAKER_FEE        = 0.0025
STARTING_BALANCE = 100.0
MIN_BET          = 3.0
POLL_SEC         = 5

# Kelly criterion sizing config
KELLY_FRACTION   = 0.25   # fractional Kelly — 25% of full Kelly to reduce variance
MAX_BET_PCT      = 0.15   # hard cap: never more than 15% of portfolio per trade
# Assumed win probabilities by signal intensity (prior — replaced by real data over time)
KELLY_P_WEAK     = 0.54   # intensity < 0.3
KELLY_P_MED      = 0.57   # intensity 0.3–0.7
KELLY_P_STRONG   = 0.60   # intensity > 0.7

SKIP_HOURS_UTC   = {}


# Per-strategy win rates fetched from Supabase at startup and refreshed every 30min.
# Falls back to conservative priors when sample size is below MIN_KELLY_SAMPLES.
_win_rates: dict = {}   # {"Strategy Name": float}  e.g. {"Liquidation Cascade": 0.62}
MIN_KELLY_SAMPLES = 30  # minimum resolved trades before using real win rate


def fetch_win_rates() -> dict:
    """
    Query Supabase for per-strategy win rates from resolved trades.
    Only uses strategies with >= MIN_KELLY_SAMPLES resolved trades.
    Returns dict of {strategy_name: win_rate_float}.
    Logs which strategies have enough data and which are still on priors.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {}
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "application/json",
                "Range":         "0-9999",
            },
            params={
                "action":           "eq.OPEN",
                "resolved_outcome": "not.is.null",
                "select":           "strategy,actual_win",
            },
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        log.warning(f"fetch_win_rates: Supabase query failed — {e}")
        return {}

    from collections import defaultdict
    counts  = defaultdict(lambda: {"wins": 0, "total": 0})
    for row in rows:
        name = row.get("strategy", "")
        won  = row.get("actual_win")
        if not name or won is None:
            continue
        counts[name]["total"] += 1
        if won:
            counts[name]["wins"] += 1

    rates = {}
    for name, c in counts.items():
        if c["total"] >= MIN_KELLY_SAMPLES:
            wr = c["wins"] / c["total"]
            rates[name] = round(wr, 4)
            log.info(f"Kelly win rate [{name}]: {wr:.1%} "
                     f"({c['wins']}W/{c['total'] - c['wins']}L) — using real rate")
        else:
            log.info(f"Kelly win rate [{name}]: only {c['total']} samples "
                     f"(need {MIN_KELLY_SAMPLES}) — using prior")

    return rates


def kelly_size(intensity: float, entry_price: float, bankroll: float,
               strategy_name: str = "", win_prob: float = 0.0) -> float:
    """
    Fractional Kelly criterion bet sizing.

    Args:
        intensity:     Signal strength 0–1 (scales the Kelly allocation).
        entry_price:   Taker fill price (e.g. 0.45). Determines net odds b = 1/p - 1.
        bankroll:      Current available portfolio balance.
        strategy_name: Used to look up real win rate from _win_rates if available.
        win_prob:      Hard override win probability (0 = use _win_rates or prior).

    Win probability resolution order:
        1. win_prob argument if > 0 (hard override)
        2. _win_rates[strategy_name] if >= MIN_KELLY_SAMPLES resolved trades
        3. Conservative prior based on intensity tier (KELLY_P_WEAK/MED/STRONG)

    Kelly formula: f* = (p*b - q) / b
        p = estimated win probability
        q = 1 - p
        b = net odds = (1 / entry_price) - 1

    Returns fractional Kelly * bankroll, capped at MAX_BET_PCT of bankroll.
    Returns 0.0 if Kelly is zero or negative (no edge at this price).
    """
    if entry_price <= 0 or entry_price >= 1:
        return MIN_BET

    # Net odds on a Polymarket binary
    b = (1.0 / entry_price) - 1.0

    # Win probability — real rate > prior, hard override > both
    if win_prob > 0:
        p = win_prob
    elif strategy_name and strategy_name in _win_rates:
        p = _win_rates[strategy_name]
    elif intensity < 0.3:
        p = KELLY_P_WEAK
    elif intensity < 0.7:
        p = KELLY_P_MED
    else:
        p = KELLY_P_STRONG

    q = 1.0 - p

    # Full Kelly fraction
    kelly_f = (p * b - q) / b

    if kelly_f <= 0:
        # Negative Kelly = no edge at this price — skip
        return 0.0

    # Fractional Kelly
    frac_kelly = kelly_f * KELLY_FRACTION

    # Scale by signal intensity within the Kelly allocation
    frac_kelly *= intensity

    # Bet size in dollars
    size = frac_kelly * bankroll

    # Hard cap: never exceed MAX_BET_PCT of bankroll regardless of Kelly
    size = min(size, bankroll * MAX_BET_PCT)

    # Floor
    size = max(size, MIN_BET)

    return round(size, 4)


# ---------------------------------------------------------------- SUPABASE ---

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


def supabase_insert(record: dict):
    """Insert a trade record into Supabase. Never blocks on failure."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=record,
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Supabase insert failed: {e}")


def supabase_signal_log(entry: dict):
    """
    Log every signal evaluation that did NOT fire to Supabase signal_log table.

    Required Supabase table:
        CREATE TABLE IF NOT EXISTS signal_log (
            id               bigserial PRIMARY KEY,
            created_at       timestamptz DEFAULT now(),
            strategy         text,
            condition_id     text,
            secs_left        float,
            signal_value     float,
            threshold        float,
            fired            boolean,
            reason           text,
            market_question  text
        );
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/signal_log",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=entry,
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Supabase signal_log insert failed: {e}")


def _log_signal(strategy: str, market, secs_left: float,
                signal_value: float, threshold: float, reason: str):
    """Helper — builds and sends a signal log entry for a non-firing evaluation."""
    entry = {
        "strategy":        strategy,
        "condition_id":    market.condition_id if market else "",
        "market_question": market.question if market else "",
        "secs_left":       round(secs_left, 1),
        "signal_value":    round(signal_value, 6),
        "threshold":       round(threshold, 6),
        "fired":           False,
        "reason":          reason,
        # Regime labels — allows slicing win rates by market condition
        "regime":          _regime.get("composite", "UNKNOWN"),
        "session":         _regime.get("session", "UNKNOWN"),
        "activity":        _regime.get("activity", "UNKNOWN"),
        "day_type":        _regime.get("day_type", "UNKNOWN"),
    }
    log.debug(f"[SIGNAL] {strategy} | {reason} | value={signal_value:.6f} threshold={threshold:.6f} "
              f"| regime={entry['regime']} session={entry['session']}")
    supabase_signal_log(entry)


# --------------------------------------------------------------- DATACLASSES --

@dataclass
class Market:
    condition_id:  str
    up_token_id:   str
    down_token_id: str
    question:      str
    end_time:      datetime


@dataclass
class StrategyTrade:
    strategy:    str
    side:        str
    size:        float
    entry_price: float
    market:      Market
    trade_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_data: dict = field(default_factory=dict)
    entry_time:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------- SHARED PORTFOLIO ---
# One capital pool shared across all strategies.
# Each strategy draws from and returns to the same balance,
# so combined exposure is visible and capped in one place.

# Maximum fraction of portfolio that can be committed in one direction
# on a single 5-minute market across all strategies combined.
MAX_DIRECTIONAL_EXPOSURE = 0.40   # 40% of portfolio per direction per market


class SharedPortfolio:
    """
    Single capital pool for all strategies.
    Also tracks per-market directional exposure so correlated bets
    across strategies cannot silently stack beyond the cap.
    """
    def __init__(self, balance: float):
        self._lock      = threading.Lock()
        self.balance    = balance
        self.start_bal  = balance
        self._wins:   dict = {}
        self._losses: dict = {}
        # exposure[condition_id][direction] = total size committed
        self._exposure: dict = {}

    def available(self) -> float:
        with self._lock:
            return self.balance

    def directional_exposure(self, condition_id: str, direction: str) -> float:
        """Return total size already committed in this direction on this market."""
        with self._lock:
            return self._exposure.get(condition_id, {}).get(direction, 0.0)

    def exposure_cap_remaining(self, condition_id: str, direction: str) -> float:
        """How much more size is allowed in this direction before hitting the cap."""
        with self._lock:
            cap       = self.balance * MAX_DIRECTIONAL_EXPOSURE
            committed = self._exposure.get(condition_id, {}).get(direction, 0.0)
            return max(0.0, cap - committed)

    def debit(self, size: float, fee: float,
              condition_id: str = "", direction: str = ""):
        with self._lock:
            self.balance -= (size + fee)
            if condition_id and direction:
                if condition_id not in self._exposure:
                    self._exposure[condition_id] = {}
                prev = self._exposure[condition_id].get(direction, 0.0)
                self._exposure[condition_id][direction] = prev + size

    def credit(self, amount: float,
               condition_id: str = "", direction: str = "", size: float = 0.0):
        with self._lock:
            self.balance += amount
            if condition_id and direction and size > 0:
                prev = self._exposure.get(condition_id, {}).get(direction, 0.0)
                self._exposure.setdefault(condition_id, {})[direction] = max(0.0, prev - size)

    def clear_exposure(self, condition_id: str):
        """Called when a market expires — clears all exposure tracking for it."""
        with self._lock:
            self._exposure.pop(condition_id, None)

    def record_win(self, strategy: str):
        self._wins[strategy]   = self._wins.get(strategy, 0) + 1

    def record_loss(self, strategy: str):
        self._losses[strategy] = self._losses.get(strategy, 0) + 1

    def summary_lines(self) -> list:
        lines = [f"Portfolio balance: ${self.balance:.2f} | "
                 f"PnL={self.balance - self.start_bal:+.2f}"]
        for name in sorted(set(list(self._wins) + list(self._losses))):
            w = self._wins.get(name, 0)
            l = self._losses.get(name, 0)
            total = w + l
            wr    = (w / total * 100) if total else 0
            lines.append(f"  {name}: WR={wr:.0f}% ({w}W/{l}L) — ground truth in Supabase")
        return lines


# Singleton — instantiated once in run(), passed to all trackers
_portfolio: Optional["SharedPortfolio"] = None


# --------------------------------------------------------- STRATEGY TRACKER --

class StrategyTracker:
    def __init__(self, name: str, portfolio: "SharedPortfolio"):
        self.name      = name
        self.portfolio = portfolio
        self.trades    = []
        self.logfile   = f"strategy_{name.lower().replace(' ', '_')}.jsonl"

    # Convenience: delegate balance reads to the shared pool
    @property
    def balance(self) -> float:
        return self.portfolio.available()

    def open(self, side: str, price: float, size: float,
             signal_data: dict, market: Market) -> str:
        """
        Records an open position locally and writes ONE row to Supabase.
        Enforces per-market directional exposure cap before opening.
        Returns the trade_id so the caller can store it on StrategyTrade.
        """
        # Enforce directional exposure cap — cap size to whatever room remains
        cap_remaining = self.portfolio.exposure_cap_remaining(market.condition_id, side)
        if cap_remaining <= 0:
            log.info(f"[{self.name}] Exposure cap reached for {side} on "
                     f"{market.condition_id[:12]}... — skipping")
            return ""
        size = min(size, cap_remaining)
        if size < MIN_BET:
            log.info(f"[{self.name}] Remaining cap ${cap_remaining:.2f} below MIN_BET — skipping")
            return ""

        fee = size * TAKER_FEE
        self.portfolio.debit(size, fee, market.condition_id, side)
        trade_id = str(uuid.uuid4())

        entry = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "action":       "OPEN",
            "strategy":     self.name,
            "side":         side,
            "price":        price,
            "size":         size,
            "fee":          round(fee, 4),
            "balance":      round(self.balance, 2),
            "condition_id": market.condition_id,
            "question":     market.question,
            **signal_data,
        }
        self.trades.append(entry)
        with open(self.logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Write to Supabase — this is the ONLY row we write per trade.
        # Outcome fields (actual_win, resolved_outcome, pnl, etc.) will be
        # patched later by resolver.py using trade_id as the key.
        supabase_insert({
            "trade_id":        trade_id,
            "strategy":        self.name,
            "action":          "OPEN",
            "side":            side,
            "price":           price,
            "size":            size,
            "fee":             round(size * TAKER_FEE, 4),
            "condition_id":    market.condition_id,
            "question":        market.question,
            "market_end_time": market.end_time.isoformat(),
            "signal_data":     signal_data,
            "regime":          _regime.get("composite", "UNKNOWN"),
            "session":         _regime.get("session",   "UNKNOWN"),
            "activity":        _regime.get("activity",  "UNKNOWN"),
            "day_type":        _regime.get("day_type",  "UNKNOWN"),
            "kelly_fraction":  round(KELLY_FRACTION, 4),
            "max_bet_pct":     round(MAX_BET_PCT, 4),
        })
        log.info(f"[{self.name}] OPEN {side} | size={size:.2f} @ {price:.4f} | "
                 f"portfolio=${self.portfolio.available():.2f} | trade_id={trade_id[:8]}... | "
                 f"{json.dumps(signal_data)}")
        return trade_id

    def close(self, side: str, entry_price: float, exit_price: float,
              size: float, market: Market, reason: str = "RESOLVED"):
        """
        Updates shared portfolio balance and per-strategy win/loss counters only.
        Does NOT write to Supabase — resolver.py owns all outcome writes.
        """
        if side == "UP":
            pnl = (exit_price - entry_price) * size / entry_price
        else:
            pnl = (entry_price - exit_price) * size / entry_price
        # Round-trip fee: open taker + close taker — matches resolver.py accounting
        fee = size * TAKER_FEE * 2
        pnl -= fee
        self.portfolio.credit(size + pnl, market.condition_id, side, size)
        if pnl > 0:
            self.portfolio.record_win(self.name)
        else:
            self.portfolio.record_loss(self.name)

        w     = self.portfolio._wins.get(self.name, 0)
        l     = self.portfolio._losses.get(self.name, 0)
        total = w + l
        wr    = (w / total * 100) if total else 0

        entry = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "action":       "CLOSE_LOCAL",
            "strategy":     self.name,
            "reason":       reason,
            "side":         side,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "size":         size,
            "pnl":          round(pnl, 4),
            "fee":          round(fee, 4),   # round-trip: open + close taker
            "portfolio":    round(self.portfolio.available(), 2),
            "condition_id": market.condition_id,
        }
        self.trades.append(entry)
        with open(self.logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # No supabase_insert here — resolver writes the real outcome to the OPEN row.
        log.info(f"[{self.name}] CLOSE_LOCAL {side} ({reason}) | exit={exit_price:.2f} | "
                 f"PnL={pnl:+.3f} | portfolio=${self.portfolio.available():.2f} | "
                 f"WR={wr:.0f}% ({w}W/{l}L)")

    def void(self, pos: "StrategyTrade", market: Market):
        """
        Market was voided — refund the position size (minus fees already paid).
        Does not count as win or loss. Does not write to Supabase —
        resolver.py will mark the row VOID when it confirms the market outcome.
        """
        self.portfolio.credit(pos.size, market.condition_id, pos.side, pos.size)  # refund stake, fees are lost
        entry = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "action":       "VOID_LOCAL",
            "strategy":     self.name,
            "side":         pos.side,
            "entry_price":  pos.entry_price,
            "size":         pos.size,
            "pnl":          0.0,
            "portfolio":    round(self.portfolio.available(), 2),
            "condition_id": market.condition_id,
        }
        self.trades.append(entry)
        with open(self.logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")
        log.info(f"[{self.name}] VOID_LOCAL {pos.side} | size refunded={pos.size:.2f} | "
                 f"portfolio=${self.portfolio.available():.2f}")

    def summary(self) -> str:
        w     = self.portfolio._wins.get(self.name, 0)
        l     = self.portfolio._losses.get(self.name, 0)
        total = w + l
        wr    = (w / total * 100) if total else 0
        return (f"{self.name}: WR={wr:.0f}% | {total} trades (local estimate — "
                f"ground truth in Supabase via resolver)")


# --------------------------------------------------------- SHARED DATA -------

_price_cache:    dict  = {"btc": 0.0, "eth": 0.0, "fetched_at": 0.0}
_funding_cache:  dict  = {"okx": 0.0, "binance": 0.0, "rate": 0.0, "fetched_at": 0.0}
_basis_cache:    dict  = {"spot": 0.0, "futures": 0.0, "fetched_at": 0.0}
_liq_cache:      dict  = {"long": 0.0, "short": 0.0, "fetched_at": 0.0}
_vol_cache:      dict  = {"range_pct": 0.0, "fetched_at": 0.0}
_ob_cache:       dict  = {"imbalance": 0.0, "bid_depth": 0.0, "ask_depth": 0.0,
                           "spread_pct": 0.0, "fetched_at": 0.0}
_regime:         dict  = {
    # Market microstructure regime — recomputed every poll cycle
    "momentum_label":  "NEUTRAL",   # TREND_UP | TREND_DOWN | NEUTRAL
    "volatility_label": "NORMAL",   # VOLATILE | NORMAL | DEAD
    "flow_label":      "BALANCED",  # LONG_CROWDED | SHORT_CROWDED | BALANCED
    "composite":       "CALM",      # TREND_UP | TREND_DOWN | CALM | VOLATILE | DEAD
    # Time-of-day / calendar regime — stable within a session
    "session":         "UNKNOWN",   # ASIA | LONDON | US | OVERLAP | OFFPEAK
    "day_type":        "WEEKDAY",   # WEEKDAY | WEEKEND
    "activity":        "NORMAL",    # HIGH | NORMAL | LOW | DEAD
}
_btc_history:    deque = deque(maxlen=12)  # last 60s of BTC prices (5s poll = 12 readings)
_volume_history: deque = deque(maxlen=25)
_volume_fetched: float = 0.0


def fetch_spot(symbol: str = BTC_SYMBOL) -> Optional[float]:
    """Coinbase primary, Kraken fallback."""
    coin_map   = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}
    kraken_map = {"BTCUSDT": "XBTUSD",  "ETHUSDT": "ETHUSD"}
    try:
        r = requests.get(
            f"https://api.coinbase.com/v2/prices/{coin_map.get(symbol,'BTC-USD')}/spot",
            timeout=5)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception:
        pass
    try:
        r = requests.get(
            f"https://api.kraken.com/0/public/Ticker?pair={kraken_map.get(symbol,'XBTUSD')}",
            timeout=5)
        r.raise_for_status()
        res = r.json()["result"]
        return float(res[list(res.keys())[0]]["c"][0])
    except Exception:
        return None


def refresh_shared_data():
    """Refresh all shared data using geo-unblocked sources."""
    now = time.time()

    # Spot prices
    if now - _price_cache["fetched_at"] >= POLL_SEC:
        btc = fetch_spot(BTC_SYMBOL)
        eth = fetch_spot(ETH_SYMBOL)
        if btc:
            _price_cache["btc"] = btc
            _btc_history.append({"price": btc, "ts": now})
        if eth: _price_cache["eth"] = eth
        _price_cache["fetched_at"] = now

    # Funding rate — OKX + Binance, both required for signal agreement
    if now - _funding_cache["fetched_at"] >= 60:
        try:
            r = requests.get(
                f"{OKX_BASE}/api/v5/public/funding-rate",
                params={"instId": OKX_BTC},
                timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [{}])[0]
            okx_rate = float(data.get("fundingRate", 0))
            _funding_cache["okx"] = okx_rate
        except Exception as e:
            log.warning(f"OKX funding fetch failed: {e}")

        try:
            r = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/contracts/BTC_USDT",
                timeout=5)
            r.raise_for_status()
            gate_rate = float(r.json().get("funding_rate", 0))
            _funding_cache["binance"] = gate_rate  # reusing key for compatibility
        except Exception as e:
            log.warning(f"Gate.io funding fetch failed: {e}")

        # Combined rate = average of OKX + Gate.io
        _funding_cache["rate"]       = (_funding_cache["okx"] + _funding_cache["binance"]) / 2
        _funding_cache["fetched_at"] = now
        log.info(f"Funding — OKX: {_funding_cache['okx']:+.6f} "
                 f"Gate.io: {_funding_cache['binance']:+.6f} "
                 f"avg: {_funding_cache['rate']:+.6f}")

    # Basis via OKX mark price vs spot
    if now - _basis_cache["fetched_at"] >= 10:
        try:
            r = requests.get(
                f"{OKX_BASE}/api/v5/public/mark-price",
                params={"instType": "SWAP", "instId": OKX_BTC},
                timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [{}])[0]
            _basis_cache["futures"]    = float(data.get("markPx", 0))
            _basis_cache["spot"]       = _price_cache["btc"]
            _basis_cache["fetched_at"] = now
        except Exception as e:
            log.warning(f"Basis fetch failed: {e}")

    # Liquidations via OKX — 2-minute window, refreshed every 60s
    # _liq_cache holds OKX-only values. Binance WebSocket data is combined
    # inside strategy_liquidation_cascade() with a matching 2-minute window.
    if now - _liq_cache["fetched_at"] >= 60:
        try:
            r = requests.get(
                f"{OKX_BASE}/api/v5/public/liquidation-orders",
                params={"instType": "SWAP", "uly": "BTC-USDT",
                        "state": "filled", "limit": "20"},
                timeout=5)
            r.raise_for_status()
            orders    = r.json().get("data", [{}])[0].get("details", [])
            cutoff_ms = (now - 120) * 1000   # 2-minute window matches Binance
            long_liqs = short_liqs = 0.0
            for o in orders:
                if float(o.get("ts", 0)) < cutoff_ms:
                    continue
                usd = float(o.get("sz", 0)) * float(o.get("bkPx", 0))
                if o.get("side") == "buy":  short_liqs += usd
                else:                       long_liqs  += usd
            _liq_cache["long"]       = long_liqs
            _liq_cache["short"]      = short_liqs
            _liq_cache["fetched_at"] = now
        except Exception as e:
            log.warning(f"Liquidation fetch failed: {e}")

    # 5-minute BTC price range for volatility-adjusted liquidation signal
    # Uses OKX 1m klines — high/low over last 5 candles
    global _vol_cache
    if now - _vol_cache["fetched_at"] >= 60:
        try:
            r = requests.get(
                f"{OKX_BASE}/api/v5/market/candles",
                params={"instId": OKX_BTC, "bar": "1m", "limit": "5"},
                timeout=5)
            r.raise_for_status()
            candles = r.json().get("data", [])
            if candles:
                highs = [float(c[2]) for c in candles]
                lows  = [float(c[3]) for c in candles]
                price_range = max(highs) - min(lows)
                mid         = (max(highs) + min(lows)) / 2
                _vol_cache["range_pct"]  = (price_range / mid * 100) if mid > 0 else 0.0
                _vol_cache["fetched_at"] = now
                log.debug(f"5min BTC range: {_vol_cache['range_pct']:.3f}%")
        except Exception as e:
            log.warning(f"Volatility range fetch failed: {e}")

    # Volume via OKX klines
    global _volume_fetched
    if now - _volume_fetched >= 60:
        try:
            r = requests.get(
                f"{OKX_BASE}/api/v5/market/candles",
                params={"instId": OKX_BTC, "bar": "1m", "limit": "22"},
                timeout=8)
            r.raise_for_status()
            candles = r.json().get("data", [])
            _volume_history.clear()
            for c in candles:
                close = float(c[4])
                open_ = float(c[1])
                vol   = float(c[5])
                _volume_history.append({
                    "volume":  vol,
                    "close":   close,
                    "open":    open_,
                    "buy_vol": vol if close >= open_ else 0.0,
                })
            _volume_fetched = now
        except Exception as e:
            log.warning(f"Volume fetch failed: {e}")

    # OKX futures order book imbalance — refreshed every poll cycle
    # Uses top 20 levels on each side, depth weighted by size
    try:
        r = requests.get(
            f"{OKX_BASE}/api/v5/market/books",
            params={"instId": OKX_BTC, "sz": "20"},
            timeout=4,
        )
        r.raise_for_status()
        data      = r.json().get("data", [{}])[0]
        bids_raw  = data.get("bids", [])   # each entry: [price, size, ...]
        asks_raw  = data.get("asks", [])
        bid_depth = sum(float(b[1]) for b in bids_raw)
        ask_depth = sum(float(a[1]) for a in asks_raw)
        total     = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0
        best_bid  = float(bids_raw[0][0]) if bids_raw else 0.0
        best_ask  = float(asks_raw[0][0]) if asks_raw else 0.0
        mid       = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
        spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0.0
        _ob_cache["imbalance"]  = round(imbalance, 4)
        _ob_cache["bid_depth"]  = round(bid_depth, 2)
        _ob_cache["ask_depth"]  = round(ask_depth, 2)
        _ob_cache["spread_pct"] = round(spread_pct, 4)
        _ob_cache["fetched_at"] = now
        log.debug(f"OB imbalance={imbalance:+.3f} bid={bid_depth:.0f} ask={ask_depth:.0f} "
                  f"spread={spread_pct:.4f}%")
    except Exception as e:
        log.warning(f"OB pressure fetch failed: {e}")
def compute_regime():
    """
    Compute market regime labels once per poll cycle.
    Writes to _regime dict so all strategies read a consistent snapshot
    rather than each computing their own momentum independently.

    Two axes:
      1. Microstructure regime  — what the market is doing RIGHT NOW
      2. Calendar regime        — what session/day we're in (structural baseline)
    """
    now = datetime.now(timezone.utc)

    # ── 1. CALENDAR REGIME ────────────────────────────────────────────────────
    hour    = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    is_weekend = weekday >= 5

    # Trading sessions (UTC)
    if 0 <= hour < 6:
        session = "ASIA" if hour >= 1 else "OFFPEAK"
    elif 6 <= hour < 8:
        session = "LONDON"
    elif 8 <= hour < 12:
        session = "OVERLAP"   # London + early US
    elif 12 <= hour < 17:
        session = "US"
    elif 17 <= hour < 21:
        session = "OFFPEAK"
    else:
        session = "ASIA"      # late UTC = early Asia

    day_type = "WEEKEND" if is_weekend else "WEEKDAY"

    # Activity level based on session + day
    if is_weekend:
        activity = "LOW"
    elif session in ("US", "OVERLAP"):
        activity = "HIGH"
    elif session == "LONDON":
        activity = "NORMAL"
    elif session == "ASIA":
        activity = "NORMAL"
    else:
        activity = "LOW"      # OFFPEAK

    _regime["session"]  = session
    _regime["day_type"] = day_type
    _regime["activity"] = activity

    # ── 2. MICROSTRUCTURE REGIME ──────────────────────────────────────────────
    momentum   = btc_momentum_pct(lookback_secs=30) or 0.0
    vol_range  = _vol_cache.get("range_pct", 0.0)
    ob_imbal   = _ob_cache.get("imbalance", 0.0)
    funding    = _funding_cache.get("rate", 0.0)

    # Momentum label
    if momentum > 0.08 and ob_imbal > 0.10:
        momentum_label = "TREND_UP"
    elif momentum < -0.08 and ob_imbal < -0.10:
        momentum_label = "TREND_DOWN"
    else:
        momentum_label = "NEUTRAL"

    # Volatility label
    if vol_range > 0.25:
        volatility_label = "VOLATILE"
    elif vol_range < 0.04:
        volatility_label = "DEAD"
    else:
        volatility_label = "NORMAL"

    # Flow/crowding label
    if funding > 0.0003 and momentum > 0:
        flow_label = "LONG_CROWDED"
    elif funding < -0.0003 and momentum < 0:
        flow_label = "SHORT_CROWDED"
    else:
        flow_label = "BALANCED"

    # Composite label — single string for logging and strategy gating
    if volatility_label == "DEAD":
        composite = "DEAD"
    elif volatility_label == "VOLATILE" and momentum_label == "NEUTRAL":
        composite = "VOLATILE"
    elif momentum_label == "TREND_UP":
        composite = "TREND_UP"
    elif momentum_label == "TREND_DOWN":
        composite = "TREND_DOWN"
    else:
        composite = "CALM"

    _regime["momentum_label"]   = momentum_label
    _regime["volatility_label"] = volatility_label
    _regime["flow_label"]       = flow_label
    _regime["composite"]        = composite

    log.debug(
        f"Regime: {composite} | session={session} ({day_type}/{activity}) | "
        f"momentum={momentum:+.3f}% | vol={vol_range:.3f}% | "
        f"ob={ob_imbal:+.3f} | flow={flow_label}"
    )


# ------------------------------------------------- CHAINLINK WEBSOCKET CACHE --
# Polymarket broadcasts the exact Chainlink price it uses for resolution
# via wss://ws-subscriptions-clob.polymarket.com/ws/ on topic
# crypto_prices_chainlink. We subscribe in a background thread and keep
# the latest price + timestamp in a shared dict.
# fetch_chainlink_price() reads from this cache instead of making HTTP calls.

_chainlink_cache: dict = {
    "price":      None,   # latest BTC/USD from Chainlink via Polymarket RTDS
    "updated_at": None,   # datetime when last updated
}
_POLYMARKET_WS = "wss://ws-live-data.polymarket.com"
_SUBSCRIBE_MSG = json.dumps({
    "action": "subscribe",
    "subscriptions": [
        {
            "topic":   "crypto_prices_chainlink",
            "type":    "*",
            "filters": "{\"symbol\":\"btc/usd\"}"
        }
    ]
})


def _on_ws_message(ws, message):
    global _chainlink_cache
    if not message:
        return
    try:
        data = json.loads(message)
        if data.get("topic") != "crypto_prices_chainlink":
            return
        payload = data.get("payload", {})
        if payload.get("symbol", "").lower() != "btc/usd":
            return
        price = float(payload["value"])
        ts_ms = payload.get("timestamp")
        if ts_ms:
            updated = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        else:
            updated = datetime.now(timezone.utc)
        _chainlink_cache["price"]      = price
        _chainlink_cache["updated_at"] = updated
        log.debug(f"[Chainlink WS] BTC/USD={price} updated_at={updated.isoformat()}")
    except Exception as e:
        log.warning(f"[Chainlink WS] message parse error: {e}")


def _on_ws_open(ws):
    log.info("[Chainlink WS] Connected to Polymarket RTDS — subscribing to btc/usd")
    ws.send(_SUBSCRIBE_MSG)


def _on_ws_error(ws, error):
    log.warning(f"[Chainlink WS] error: {error}")


def _on_ws_close(ws, close_status_code, close_msg):
    log.warning(f"[Chainlink WS] closed ({close_status_code}) — will reconnect")


def _chainlink_ws_thread():
    """Background thread: maintains persistent WebSocket to Polymarket RTDS."""
    while True:
        try:
            ws = websocket.WebSocketApp(
                _POLYMARKET_WS,
                on_open    = _on_ws_open,
                on_message = _on_ws_message,
                on_error   = _on_ws_error,
                on_close   = _on_ws_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.warning(f"[Chainlink WS] thread exception: {e}")
        log.info("[Chainlink WS] Reconnecting in 5s...")
        time.sleep(5)


def start_chainlink_ws():
    """Start the Chainlink WebSocket listener in a daemon background thread."""
    t = threading.Thread(target=_chainlink_ws_thread, daemon=True)
    t.start()
    log.info("[Chainlink WS] Background thread started")


def fetch_chainlink_price() -> tuple:
    """
    Returns (price, age_seconds) from the Polymarket RTDS Chainlink cache.
    This is the exact same BTC/USD feed Polymarket uses to resolve markets.
    Falls back to (None, None) if the WebSocket hasn't received a price yet
    or the last update is stale (> 60s).
    """
    price      = _chainlink_cache.get("price")
    updated_at = _chainlink_cache.get("updated_at")

    if price is None or updated_at is None:
        log.debug("[Chainlink WS] No price yet — WebSocket still connecting")
        return None, None

    age_sec = (datetime.now(timezone.utc) - updated_at).total_seconds()

    if age_sec > 60:
        log.warning(f"[Chainlink WS] Price stale ({age_sec:.0f}s) — possible disconnect")
        return None, None

    return price, age_sec


# ------------------------------------------------ BINANCE LIQUIDATION CACHE --
# Binance processes 3-5x more BTC liquidation volume than OKX.
# We subscribe to the Binance USDM futures forced liquidation stream
# in a background thread and accumulate liq USD over a rolling 5-minute
# window. The OKX REST poll and this WebSocket feed are combined in
# _liq_cache so the liquidation cascade strategy sees the full market picture.

_binance_liq_buffer: list = []   # raw events: {"side": "BUY"|"SELL", "usd": float, "ts": ms}
_binance_liq_lock = threading.Lock()
_BINANCE_LIQ_WS   = "wss://fstream.binance.com/ws/!forceOrder@arr"


def _on_binance_liq_message(ws, message):
    try:
        data  = json.loads(message)
        order = data.get("o", {})
        # Only care about BTCUSDT perpetual
        if order.get("s", "") != "BTCUSDT":
            return
        side   = order.get("S", "")   # "BUY" = short liq, "SELL" = long liq
        qty    = float(order.get("q", 0))
        price  = float(order.get("ap", 0)) or float(order.get("p", 0))
        usd    = qty * price
        ts_ms  = int(order.get("T", time.time() * 1000))
        with _binance_liq_lock:
            _binance_liq_buffer.append({"side": side, "usd": usd, "ts": ts_ms})
            # Keep only last 10 minutes of events to bound memory
            cutoff = (time.time() - 600) * 1000
            while _binance_liq_buffer and _binance_liq_buffer[0]["ts"] < cutoff:
                _binance_liq_buffer.pop(0)
        log.debug(f"[Binance Liq] {side} ${usd:,.0f} BTCUSDT")
    except Exception as e:
        log.warning(f"[Binance Liq] message parse error: {e}")


def _on_binance_liq_open(ws):
    log.info("[Binance Liq] Connected to Binance USDM liquidation stream")


def _on_binance_liq_error(ws, error):
    log.warning(f"[Binance Liq] error: {error}")


def _on_binance_liq_close(ws, close_status_code, close_msg):
    log.warning(f"[Binance Liq] closed ({close_status_code}) — will reconnect")


def _binance_liq_ws_thread():
    """Background thread: maintains persistent WebSocket to Binance liquidation feed."""
    while True:
        try:
            ws = websocket.WebSocketApp(
                _BINANCE_LIQ_WS,
                on_open    = _on_binance_liq_open,
                on_message = _on_binance_liq_message,
                on_error   = _on_binance_liq_error,
                on_close   = _on_binance_liq_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log.warning(f"[Binance Liq] thread exception: {e}")
        log.info("[Binance Liq] Reconnecting in 5s...")
        time.sleep(5)


def start_binance_liq_ws():
    """Start the Binance liquidation WebSocket listener in a daemon background thread."""
    t = threading.Thread(target=_binance_liq_ws_thread, daemon=True)
    t.start()
    log.info("[Binance Liq] Background thread started")


def get_binance_liq_5min() -> dict:
    """
    Returns combined long/short liquidation USD over the last 5 minutes
    from the Binance feed.
      BUY  side = short squeeze (shorts liquidated) → bullish
      SELL side = long liquidation (longs liquidated) → bearish
    """
    cutoff = (time.time() - 300) * 1000
    long_liqs = short_liqs = 0.0
    with _binance_liq_lock:
        for e in _binance_liq_buffer:
            if e["ts"] < cutoff:
                continue
            if e["side"] == "SELL":   # long liquidated
                long_liqs  += e["usd"]
            else:                     # short liquidated (BUY)
                short_liqs += e["usd"]
    return {"long": long_liqs, "short": short_liqs}


def get_binance_liq_2min() -> dict:
    """
    Returns combined long/short liquidation USD over the last 2 minutes.
    Shorter lookback = more recent signal, price has had less time to react.
    Used by Liquidation Cascade strategy for tighter timing.
    """
    cutoff = (time.time() - 120) * 1000
    long_liqs = short_liqs = 0.0
    with _binance_liq_lock:
        for e in _binance_liq_buffer:
            if e["ts"] < cutoff:
                continue
            if e["side"] == "SELL":
                long_liqs  += e["usd"]
            else:
                short_liqs += e["usd"]
    return {"long": long_liqs, "short": short_liqs}


def btc_momentum_pct(lookback_secs: float = 30.0) -> Optional[float]:
    """
    Returns BTC price change % over the last `lookback_secs` seconds.
    Uses the rolling _btc_history deque (sampled every POLL_SEC).
    Returns None if not enough history.
    Positive = BTC moved up, Negative = BTC moved down.
    """
    now = time.time()
    cutoff = now - lookback_secs
    history = list(_btc_history)
    old = next((h for h in history if h["ts"] >= cutoff), None)
    current = history[-1] if history else None
    if not old or not current or old["price"] == 0:
        return None
    return (current["price"] - old["price"]) / old["price"] * 100

def fetch_current_market() -> Optional[Market]:
    try:
        now        = datetime.now(timezone.utc)
        ts         = int(now.timestamp())
        current_5m = (ts // 300) * 300
        for window_ts in [current_5m + 300, current_5m + 600, current_5m + 900]:
            slug = f"btc-updown-5m-{window_ts}"
            try:
                r = requests.get(f"{GAMMA_API}/markets",
                                 params={"slug": slug}, timeout=10)
                r.raise_for_status()
                markets = r.json()
                if not markets:
                    continue
                m = markets[0]
                if m.get("closed"):
                    continue
                end_time = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
                if end_time <= now:
                    continue
                token_ids = json.loads(m["clobTokenIds"])
                outcomes  = m.get("outcomes")
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if outcomes and len(outcomes) == len(token_ids):
                    outcome_map   = {o.upper(): tid for o, tid in zip(outcomes, token_ids)}
                    up_token_id   = outcome_map.get("UP")
                    down_token_id = outcome_map.get("DOWN")
                else:
                    log.warning("No outcomes field in market response — falling back to index order")
                    up_token_id   = token_ids[0]
                    down_token_id = token_ids[1]
                if not up_token_id or not down_token_id:
                    log.warning(f"Could not map UP/DOWN tokens for {m['conditionId'][:12]}... — skipping")
                    continue
                return Market(
                    condition_id  = m["conditionId"],
                    up_token_id   = up_token_id,
                    down_token_id = down_token_id,
                    question      = m["question"],
                    end_time      = end_time,
                )
            except Exception:
                continue
        return None
    except Exception as e:
        log.error(f"Market fetch failed: {e}")
        return None


def _book_fill_price(book: dict, size_usdc: float) -> tuple:
    """
    Walk the ask ladder to estimate a size-weighted taker fill price.
    Returns (fill_price, slippage_vs_best_ask).
    """
    asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
    bids = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)

    if not asks:
        return 0.5, 0.0

    best_ask = float(asks[0]["price"])

    remaining  = size_usdc
    total_cost = 0.0
    total_qty  = 0.0
    for level in asks:
        px       = float(level["price"])
        qty      = float(level["size"])
        fill_qty = min(qty, remaining / px)
        total_cost += fill_qty * px
        total_qty  += fill_qty
        remaining  -= fill_qty * px
        if remaining <= 0:
            break

    if total_qty == 0:
        return best_ask, 0.0

    fill_price = total_cost / total_qty
    slippage   = fill_price - best_ask
    return round(fill_price, 6), round(slippage, 6)


def fetch_poly_prices(market: Market, size_usdc: float = 50.0) -> dict:
    """
    Returns executable fill prices (ask-side) for both tokens, plus spread.
    fill_up / fill_down are estimated taker prices for a buy of size_usdc.
    up_mid / down_mid are kept for reference/logging (e.g. snapshots) only —
    do not use as entry price.
    """
    try:
        up_book   = requests.get(f"{POLYMARKET_HOST}/book",
                                 params={"token_id": market.up_token_id},   timeout=8).json()
        down_book = requests.get(f"{POLYMARKET_HOST}/book",
                                 params={"token_id": market.down_token_id}, timeout=8).json()

        up_asks   = sorted(up_book.get("asks",   []), key=lambda x: float(x["price"]))
        up_bids   = sorted(up_book.get("bids",   []), key=lambda x: float(x["price"]), reverse=True)
        down_asks = sorted(down_book.get("asks", []), key=lambda x: float(x["price"]))
        down_bids = sorted(down_book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)

        spread = (float(up_asks[0]["price"]) - float(up_bids[0]["price"])
                  ) if up_asks and up_bids else 0.1

        fill_up,   slip_up   = _book_fill_price(up_book,   size_usdc)
        fill_down, slip_down = _book_fill_price(down_book, size_usdc)

        up_mid   = (float(up_asks[0]["price"])   + float(up_bids[0]["price"]))   / 2 \
                   if up_asks and up_bids else 0.5
        down_mid = (float(down_asks[0]["price"]) + float(down_bids[0]["price"])) / 2 \
                   if down_asks and down_bids else 0.5

        if slip_up > 0.005 or slip_down > 0.005:
            log.warning(f"High slippage estimate: UP +{slip_up:.4f} DOWN +{slip_down:.4f}")

        return {
            "up_mid":    up_mid,
            "down_mid":  down_mid,
            "fill_up":   fill_up,
            "fill_down": fill_down,
            "slip_up":   slip_up,
            "slip_down": slip_down,
            "spread":    spread,
        }
    except Exception as e:
        log.warning(f"fetch_poly_prices failed: {e}")
        return {
            "up_mid": 0.5, "down_mid": 0.5,
            "fill_up": 0.5, "fill_down": 0.5,
            "slip_up": 0.0, "slip_down": 0.0,
            "spread": 0.1,
        }


# --------------------------------------------------------- RESOLUTION --------

def fetch_market_winner(condition_id: str) -> Optional[str]:
    """
    Polls Gamma API every 5 seconds until a winner is found.
    Gives up after 10 minutes and returns "VOID".
    """
    POLL_INTERVAL    = 5
    MAX_WAIT         = 600
    VOID_THRESHOLD   = 120

    start_time       = time.time()
    zero_price_since = None

    attempt = 0
    while True:
        elapsed = time.time() - start_time
        attempt += 1

        if elapsed > MAX_WAIT:
            log.warning(f"[{condition_id}] No resolution after {MAX_WAIT}s — declaring VOID")
            return "VOID"

        try:
            r = requests.get(
                f"{GAMMA_API}/markets",
                params={"conditionId": condition_id},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()

            if not data:
                log.warning(f"[{condition_id}] Gamma returned empty (attempt {attempt}, "
                            f"{elapsed:.0f}s elapsed) — retrying")
                time.sleep(POLL_INTERVAL)
                continue

            m          = data[0]
            raw_prices = m.get("outcomePrices", '["0","0"]')
            prices     = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            up_price   = float(prices[0])
            down_price = float(prices[1])

            log.info(f"[{condition_id}] attempt={attempt} elapsed={elapsed:.0f}s | "
                     f"closed={m.get('closed')} up={up_price} down={down_price}")

            if up_price > 0.9:
                log.info(f"[{condition_id}] Resolved UP after {elapsed:.0f}s")
                return "UP"
            elif down_price > 0.9:
                log.info(f"[{condition_id}] Resolved DOWN after {elapsed:.0f}s")
                return "DOWN"

            if m.get("closed") and up_price == 0.0 and down_price == 0.0:
                if zero_price_since is None:
                    zero_price_since = time.time()
                zero_duration = time.time() - zero_price_since
                log.info(f"[{condition_id}] Closed with zero prices for {zero_duration:.0f}s")
                if zero_duration >= VOID_THRESHOLD:
                    log.warning(f"[{condition_id}] Prices stayed zero for {zero_duration:.0f}s "
                                f"— declaring VOID")
                    return "VOID"
            else:
                zero_price_since = None

        except Exception as e:
            log.warning(f"[{condition_id}] fetch error (attempt {attempt}, "
                        f"{elapsed:.0f}s elapsed): {e}")

        time.sleep(POLL_INTERVAL)


def resolve_positions(positions: dict, trackers: dict,
                      market: Market, portfolio: "SharedPortfolio") -> dict:
    """
    Resolves all open positions for an expired market.
    Updates local balance/win-loss counters only.
    Clears directional exposure tracking for this market.
    Supabase outcome writes happen in resolver.py using trade_id.
    """
    any_open = any(p is not None for p in positions.values())
    if not any_open:
        portfolio.clear_exposure(market.condition_id)
        return {k: None for k in positions}

    log.info(f"Resolving market locally: {market.question}")
    outcome = fetch_market_winner(market.condition_id)

    for key, pos in positions.items():
        if not pos:
            continue

        if outcome == "VOID":
            log.warning(f"[{trackers[key].name}] Market voided locally — refunding position")
            trackers[key].void(pos, market)

        elif outcome in ("UP", "DOWN"):
            exit_price = 1.0 if pos.side == outcome else 0.0
            reason     = "RESOLVED_WIN" if pos.side == outcome else "RESOLVED_LOSS"
            trackers[key].close(
                pos.side, pos.entry_price, exit_price, pos.size, market, reason
            )
        else:
            log.error(f"Unexpected outcome value: {outcome} — skipping close for {key}")

    portfolio.clear_exposure(market.condition_id)
    return {k: None for k in positions}


# --------------------------------------------------------- STRATEGIES --------

def strategy_chainlink_arb(market, secs_left, tracker, position, cl_history):
    """
    Strategy 1: Chainlink volatility-spike arbitrage.

    The real Polymarket RTDS Chainlink feed is fast — average divergence
    from spot is only 0.022%. So a fixed 0.15% threshold almost never fires.

    Instead we target volatility spikes: moments when BTC moves sharply
    and Chainlink briefly lags behind before catching up. These spikes
    reach 0.3-0.5% divergence and are confirmed by BTC momentum.

    Conditions to fire:
      1. Divergence > 0.08% (top 5% of observed divergence events)
      2. BTC moved > 0.08% in the last 30 seconds (confirms real move)
      3. Momentum direction matches divergence direction
      4. 3 consecutive readings all above threshold (filters noise)
      5. Chainlink age > 5s (it's lagging, not just updating)
      6. At least 90s left in market
    """
    try:
        cl_price, cl_age = fetch_chainlink_price()
        btc_price        = _price_cache["btc"]
        if not cl_price or not btc_price:
            return position

        divergence = (btc_price - cl_price) / cl_price * 100
        direction  = "UP" if divergence > 0 else "DOWN"
        cl_history.append({"div": divergence, "dir": direction})

        if len(cl_history) < 3:
            log.debug(f"[Chainlink] waiting for history ({len(cl_history)}/3)")
            return position

        recent = list(cl_history)[-3:]

        # All 3 recent readings must exceed 0.08% threshold
        THRESHOLD = 0.08
        if not all(abs(s["div"]) >= THRESHOLD for s in recent):
            _log_signal("Chainlink Arb", market, secs_left,
                        signal_value=abs(divergence), threshold=THRESHOLD,
                        reason="divergence_below_threshold")
            return position

        # All 3 readings must be in the same direction
        if not all(s["dir"] == direction for s in recent):
            _log_signal("Chainlink Arb", market, secs_left,
                        signal_value=abs(divergence), threshold=THRESHOLD,
                        reason="direction_inconsistent")
            return position

        # BTC must have moved meaningfully in the last 30s (confirms real spike)
        momentum = btc_momentum_pct(lookback_secs=30)
        if momentum is None:
            return position

        MOMENTUM_THRESHOLD = 0.08
        if abs(momentum) < MOMENTUM_THRESHOLD:
            _log_signal("Chainlink Arb", market, secs_left,
                        signal_value=abs(momentum), threshold=MOMENTUM_THRESHOLD,
                        reason="btc_momentum_too_weak")
            return position

        # Momentum direction must match divergence direction
        # If BTC went UP, spot > chainlink (divergence > 0), direction = UP
        momentum_dir = "UP" if momentum > 0 else "DOWN"
        if momentum_dir != direction:
            _log_signal("Chainlink Arb", market, secs_left,
                        signal_value=abs(momentum), threshold=MOMENTUM_THRESHOLD,
                        reason="momentum_direction_mismatch")
            return position

        # Chainlink must be at least 5s old (lagging, not just slow to update)
        if cl_age < 5:
            _log_signal("Chainlink Arb", market, secs_left,
                        signal_value=cl_age, threshold=5,
                        reason="chainlink_too_fresh")
            return position

        if secs_left < 90:
            _log_signal("Chainlink Arb", market, secs_left,
                        signal_value=secs_left, threshold=90,
                        reason="market_closing_soon")
            return position

        if position:
            return position

        prices = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            _log_signal("Chainlink Arb", market, secs_left,
                        signal_value=prices["spread"], threshold=0.06,
                        reason="spread_too_wide")
            return position

        price     = prices["fill_up"] if direction == "UP" else prices["fill_down"]
        intensity = min(abs(divergence) / 0.30, 1.0)
        size      = kelly_size(intensity, price, tracker.balance, "Chainlink Arb")
        if size == 0.0:
            return position  # Kelly says no edge at this price

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"divergence":  round(divergence, 4),
                         "cl_age":      round(cl_age, 1),
                         "momentum":    round(momentum, 4)},
                        market)
            if not trade_id:
                return position
            t = StrategyTrade("chainlink_arb", direction, size, price, market)
            t.trade_id = trade_id
            return t
    except Exception as e:
        log.warning(f"[Chainlink arb] error: {e}")
    return position


def strategy_funding_reversion(market, secs_left, tracker, position):
    """
    Strategy 2: Funding rate reversion.
    Extreme positive funding = longs overextended = bet DOWN.
    Extreme negative funding = shorts overextended = bet UP.
    Requires BOTH OKX and Gate.io to agree on direction.

    Regime filter: funding is a slow-moving signal that can be stale.
    Only fade when recent BTC momentum (30s) is ALREADY moving in the
    crowded direction — confirming the crowding is still active and
    price hasn't already digested it. This prevents fading a cold print.
    Example: positive funding → longs crowded → fade DOWN, but only if
    BTC has been ticking UP recently (confirming longs are still piling in).
    """
    try:
        okx_rate = _funding_cache["okx"]
        bnb_rate = _funding_cache["binance"]
        avg_rate = _funding_cache["rate"]

        THRESHOLD = 0.0003

        # Both exchanges must exceed threshold in the same direction
        okx_extreme = abs(okx_rate) >= THRESHOLD
        bnb_extreme = abs(bnb_rate) >= THRESHOLD
        same_sign   = (okx_rate > 0) == (bnb_rate > 0)

        if not (okx_extreme and bnb_extreme and same_sign):
            _log_signal("Funding Reversion", market, secs_left,
                        signal_value=abs(avg_rate), threshold=THRESHOLD,
                        reason="exchanges_disagree_or_not_extreme")
            return position

        # Regime filter: confirm crowding is still active via 30s momentum.
        # Positive funding = longs crowded = price should still be ticking UP.
        # Negative funding = shorts crowded = price should still be ticking DOWN.
        # If momentum has already reversed, the funding signal is stale — skip.
        momentum = btc_momentum_pct(lookback_secs=30)
        if momentum is None:
            return position

        crowded_direction = "UP" if avg_rate > 0 else "DOWN"
        momentum_dir      = "UP" if momentum > 0 else "DOWN"
        MOMENTUM_MIN      = 0.02  # price must have moved at least 0.02% in crowded direction

        if momentum_dir != crowded_direction or abs(momentum) < MOMENTUM_MIN:
            _log_signal("Funding Reversion", market, secs_left,
                        signal_value=abs(momentum), threshold=MOMENTUM_MIN,
                        reason="momentum_not_confirming_crowding")
            return position

        # Also skip if momentum is too strong — may still be accelerating, not ready to fade
        MOMENTUM_MAX = 0.15  # above this, don't fade yet
        if abs(momentum) > MOMENTUM_MAX:
            _log_signal("Funding Reversion", market, secs_left,
                        signal_value=abs(momentum), threshold=MOMENTUM_MAX,
                        reason="momentum_too_strong_to_fade")
            return position

        if secs_left < 60:
            _log_signal("Funding Reversion", market, secs_left,
                        signal_value=secs_left, threshold=60,
                        reason="market_closing_soon")
            return position

        if position:
            return position

        direction = "DOWN" if avg_rate > 0 else "UP"
        prices    = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            _log_signal("Funding Reversion", market, secs_left,
                        signal_value=prices["spread"], threshold=0.06,
                        reason="spread_too_wide")
            return position

        price     = prices["fill_up"] if direction == "UP" else prices["fill_down"]
        intensity = min(abs(avg_rate) / 0.001, 1.0)
        size      = kelly_size(intensity, price, tracker.balance, "Funding Reversion")
        if size == 0.0:
            return position  # Kelly says no edge at this price

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"okx_rate":  round(okx_rate, 6),
                         "bnb_rate":  round(bnb_rate, 6),
                         "avg_rate":  round(avg_rate, 6),
                         "momentum":  round(momentum, 4),
                         "intensity": round(intensity, 3)},
                        market)
            if not trade_id:
                return position
            t = StrategyTrade("funding_reversion", direction, size, price, market)
            t.trade_id = trade_id
            return t
    except Exception as e:
        log.warning(f"[Funding reversion] error: {e}")
    return position


def strategy_liquidation_cascade(market, secs_left, tracker, position):
    """
    Strategy 3: Liquidation cascade.
    Large long liquidations = forced selling = bet DOWN.
    Large short liquidations = short squeeze = bet UP.

    Data sources:
      - OKX REST poll: 2-minute window, stored in _liq_cache (OKX-only)
      - Binance WebSocket: 2-minute window, fetched fresh here via get_binance_liq_2min()
    Both use identical 2-minute lookback — no double-counting, no window mismatch.
    """
    try:
        # OKX 2min (from cache) + Binance 2min (from WebSocket buffer) — same window
        binance    = get_binance_liq_2min()
        long_liqs  = _liq_cache["long"]  + binance["long"]
        short_liqs = _liq_cache["short"] + binance["short"]
        total      = long_liqs + short_liqs

        if total < 10_000:
            _log_signal("Liquidation Cascade", market, secs_left,
                        signal_value=total, threshold=10_000,
                        reason="liquidation_volume_too_low")
            return position

        # Require meaningful imbalance — at least 65/35 split
        if total > 0:
            dominant_ratio = max(long_liqs, short_liqs) / total
            if dominant_ratio < 0.65:
                _log_signal("Liquidation Cascade", market, secs_left,
                            signal_value=dominant_ratio, threshold=0.65,
                            reason="liquidation_imbalance_too_low")
                return position

        if secs_left < 60:
            _log_signal("Liquidation Cascade", market, secs_left,
                        signal_value=secs_left, threshold=60,
                        reason="market_closing_soon")
            return position

        if position:
            return position

        direction = "DOWN" if long_liqs > short_liqs else "UP"

        # Volatility-adjusted intensity:
        # Normalize liquidation size by current 5-min BTC price range
        # High volatility = liquidations are expected = weaker signal
        # Low volatility = liquidations are surprising = stronger signal
        vol_range = _vol_cache.get("range_pct", 0.1)
        vol_range = max(vol_range, 0.05)  # floor to avoid division by zero
        raw_intensity    = min(total / 500_000, 1.0)
        vol_scalar       = max(0.3, min(1.0, 0.15 / vol_range))  # inverse vol
        adjusted_intensity = min(raw_intensity * vol_scalar, 1.0)

        prices = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            _log_signal("Liquidation Cascade", market, secs_left,
                        signal_value=prices["spread"], threshold=0.06,
                        reason="spread_too_wide")
            return position

        price = prices["fill_up"] if direction == "UP" else prices["fill_down"]
        size  = kelly_size(adjusted_intensity, price, tracker.balance, "Liquidation Cascade")
        if size == 0.0:
            return position  # Kelly says no edge at this price

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"long_liqs":   round(long_liqs),
                         "short_liqs":  round(short_liqs),
                         "vol_range":   round(vol_range, 4),
                         "intensity":   round(adjusted_intensity, 3),
                         "source":      "OKX+Binance"},
                        market)
            if not trade_id:
                return position
            t = StrategyTrade("liquidation_cascade", direction, size, price, market)
            t.trade_id = trade_id
            return t
    except Exception as e:
        log.warning(f"[Liquidation cascade] error: {e}")
    return position


def strategy_basis_arb(market, secs_left, tracker, position):
    """
    Strategy 4: Basis arbitrage.
    OKX mark price vs Coinbase spot.
    Futures at premium = convergence expected = bet DOWN.
    Futures at discount = convergence expected = bet UP.

    Regime filter: do not fade basis when short-term momentum is strong
    in the same direction as the premium/discount. A futures premium during
    a BTC rip is momentum, not mispricing — fading it means shorting a move
    that may still have legs. Only fade when momentum is weak or absent,
    meaning the premium is genuinely stale relative to price action.
    """
    try:
        spot    = _basis_cache["spot"]
        futures = _basis_cache["futures"]

        if spot == 0 or futures == 0:
            return position

        basis_pct = (futures - spot) / spot * 100

        if abs(basis_pct) < 0.10:
            _log_signal("Basis Arb", market, secs_left,
                        signal_value=abs(basis_pct), threshold=0.10,
                        reason="basis_too_small")
            return position

        # Regime filter: skip if 30s momentum is strong in the direction of the premium.
        # Positive basis (futures premium) → we'd fade DOWN. If momentum is also DOWN
        # (price already converging) that's fine. But if momentum is UP (premium growing),
        # we're fading an accelerating move — skip.
        momentum = btc_momentum_pct(lookback_secs=30)
        if momentum is not None:
            premium_direction = "UP" if basis_pct > 0 else "DOWN"
            momentum_dir      = "UP" if momentum > 0 else "DOWN"
            STRONG_MOMENTUM   = 0.08  # above this, premium may still be growing

            if momentum_dir == premium_direction and abs(momentum) > STRONG_MOMENTUM:
                _log_signal("Basis Arb", market, secs_left,
                            signal_value=abs(momentum), threshold=STRONG_MOMENTUM,
                            reason="momentum_confirms_premium_skip_fade")
                return position

        if secs_left < 60:
            _log_signal("Basis Arb", market, secs_left,
                        signal_value=secs_left, threshold=60,
                        reason="market_closing_soon")
            return position

        if position:
            return position

        direction = "DOWN" if basis_pct > 0 else "UP"
        intensity = min(abs(basis_pct) / 0.3, 1.0)
        prices    = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            _log_signal("Basis Arb", market, secs_left,
                        signal_value=prices["spread"], threshold=0.06,
                        reason="spread_too_wide")
            return position

        price = prices["fill_up"] if direction == "UP" else prices["fill_down"]
        size  = kelly_size(intensity, price, tracker.balance, "Basis Arb")
        if size == 0.0:
            return position  # Kelly says no edge at this price

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"basis_pct": round(basis_pct, 4),
                         "momentum":  round(momentum, 4) if momentum is not None else 0.0,
                         "spot":      round(spot, 2),
                         "futures":   round(futures, 2)},
                        market)
            if not trade_id:
                return position
            t = StrategyTrade("basis_arb", direction, size, price, market)
            t.trade_id = trade_id
            return t
    except Exception as e:
        log.warning(f"[Basis arb] error: {e}")
    return position


def strategy_odds_mispricing(market, secs_left, tracker, position):
    """
    Strategy 5: Polymarket odds mispricing.
    In first 2 minutes of market, if UP odds deviate 8%+ from 0.50,
    bet on reversion back toward 0.50.

    Regime filter: only fade odds when external BTC signals are neutral
    or conflicting. If Polymarket is 58/42 because BTC is genuinely
    ripping, that's correct pricing not mispricing. Only fade when
    30s BTC momentum is weak — meaning the deviation is likely a
    Polymarket-specific overreaction, not a real directional signal.
    """
    try:
        if secs_left < 120:
            _log_signal("Odds Mispricing", market, secs_left,
                        signal_value=secs_left, threshold=120,
                        reason="too_close_to_resolution")
            return position

        if secs_left > 270:
            _log_signal("Odds Mispricing", market, secs_left,
                        signal_value=secs_left, threshold=270,
                        reason="too_early_in_market")
            return position

        if position:
            return position

        prices    = fetch_poly_prices(market)
        up_mid    = prices["up_mid"]
        deviation = up_mid - 0.50

        if prices["spread"] > 0.04:
            _log_signal("Odds Mispricing", market, secs_left,
                        signal_value=prices["spread"], threshold=0.04,
                        reason="spread_too_wide")
            return position

        if abs(deviation) < 0.08:
            _log_signal("Odds Mispricing", market, secs_left,
                        signal_value=abs(deviation), threshold=0.08,
                        reason="deviation_too_small")
            return position

        # Regime filter: check if BTC momentum justifies the deviation.
        # If odds are high for UP and BTC is actually moving UP strongly,
        # that's real — don't fade it.
        # Only fade when momentum is weak (market is overreacting to noise).
        momentum = btc_momentum_pct(lookback_secs=30)
        if momentum is not None:
            deviation_direction = "UP" if deviation > 0 else "DOWN"
            momentum_dir        = "UP" if momentum > 0 else "DOWN"
            MOMENTUM_JUSTIFY    = 0.05  # above this, odds deviation may be justified

            if momentum_dir == deviation_direction and abs(momentum) > MOMENTUM_JUSTIFY:
                _log_signal("Odds Mispricing", market, secs_left,
                            signal_value=abs(momentum), threshold=MOMENTUM_JUSTIFY,
                            reason="btc_momentum_justifies_odds_deviation")
                return position

        direction = "DOWN" if deviation > 0 else "UP"
        intensity = min(abs(deviation) / 0.15, 1.0)
        price     = prices["fill_up"] if direction == "UP" else prices["fill_down"]
        size      = kelly_size(intensity, price, tracker.balance, "Odds Mispricing")
        if size == 0.0:
            return position  # Kelly says no edge at this price

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"up_mid":    round(up_mid, 4),
                         "deviation": round(deviation, 4),
                         "momentum":  round(momentum, 4) if momentum is not None else 0.0,
                         "secs_left": round(secs_left)},
                        market)
            if not trade_id:
                return position
            t = StrategyTrade("odds_mispricing", direction, size, price, market)
            t.trade_id = trade_id
            return t
    except Exception as e:
        log.warning(f"[Odds mispricing] error: {e}")
    return position


def strategy_volume_clock(market, secs_left, tracker, position):
    """
    Strategy 6: Volume clock.
    In last 90 seconds before resolution, strong buy/sell imbalance
    from OKX klines predicts direction.
    """
    try:
        if secs_left > 90:
            _log_signal("Volume Clock", market, secs_left,
                        signal_value=secs_left, threshold=90,
                        reason="not_in_volume_window_yet")
            return position

        if secs_left < 20:
            _log_signal("Volume Clock", market, secs_left,
                        signal_value=secs_left, threshold=20,
                        reason="too_close_to_resolution")
            return position

        if position:
            return position

        if len(_volume_history) < 5:
            _log_signal("Volume Clock", market, secs_left,
                        signal_value=len(_volume_history), threshold=5,
                        reason="insufficient_volume_history")
            return position

        candles   = list(_volume_history)
        recent    = candles[-3:]
        total_vol = sum(c["volume"] for c in recent)
        buy_vol   = sum(c["buy_vol"] for c in recent)

        if total_vol == 0:
            return position

        buy_ratio = buy_vol / total_vol

        if buy_ratio > 0.60:
            direction = "UP"
            intensity = min((buy_ratio - 0.60) / 0.20, 1.0)
        elif buy_ratio < 0.40:
            direction = "DOWN"
            intensity = min((0.40 - buy_ratio) / 0.20, 1.0)
        else:
            _log_signal("Volume Clock", market, secs_left,
                        signal_value=buy_ratio, threshold=0.60,
                        reason="buy_ratio_neutral")
            return position

        prices = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            _log_signal("Volume Clock", market, secs_left,
                        signal_value=prices["spread"], threshold=0.06,
                        reason="spread_too_wide")
            return position

        price = prices["fill_up"] if direction == "UP" else prices["fill_down"]
        size  = kelly_size(intensity, price, tracker.balance, "Volume Clock")
        if size == 0.0:
            return position  # Kelly says no edge at this price

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"buy_ratio": round(buy_ratio, 4), "secs_left": round(secs_left)},
                        market)
            if not trade_id:
                return position
            t = StrategyTrade("volume_clock", direction, size, price, market)
            t.trade_id = trade_id
            return t
    except Exception as e:
        log.warning(f"[Volume clock] error: {e}")
    return position


def strategy_ob_pressure(market, secs_left, tracker, position):
    """
    Strategy 7: Binance futures order book pressure.

    Short-term BTC direction is often determined by who is more aggressive
    right now — buyers lifting asks or sellers hitting bids. We measure this
    using depth-weighted imbalance across the top 20 levels of the Binance
    USDM futures book, which reflects real directional conviction unlike
    Polymarket's market-maker-dominated book.

    Signal is only meaningful in the last 90 seconds because:
      - Earlier, the market has time to revert before resolution
      - In the final 90s, BTC's current trajectory is the strongest predictor
        of whether it closes UP or DOWN from its opening price

    Conditions to fire:
      1. 20 <= secs_left <= 90 (final window, not too close to wire)
      2. |imbalance| >= 0.15  (meaningful directional skew, not noise)
      3. BTC momentum (last 30s) confirms imbalance direction
      4. Binance futures spread <= 0.003% (book is healthy, not dislocated)
      5. Polymarket spread <= 0.06 (executable fill available)
      6. No existing position
    """
    try:
        if secs_left > 90:
            _log_signal("OB Pressure", market, secs_left,
                        signal_value=secs_left, threshold=90,
                        reason="not_in_ob_window_yet")
            return position

        if secs_left < 20:
            _log_signal("OB Pressure", market, secs_left,
                        signal_value=secs_left, threshold=20,
                        reason="too_close_to_resolution")
            return position

        if position:
            return position

        imbalance  = _ob_cache.get("imbalance", 0.0)
        spread_pct = _ob_cache.get("spread_pct", 1.0)
        fetched_at = _ob_cache.get("fetched_at", 0.0)

        # Reject stale cache — OB data must be fresh (within 2 poll cycles)
        if time.time() - fetched_at > POLL_SEC * 2:
            _log_signal("OB Pressure", market, secs_left,
                        signal_value=0.0, threshold=0.15,
                        reason="ob_cache_stale")
            return position

        IMBALANCE_THRESHOLD = 0.15
        if abs(imbalance) < IMBALANCE_THRESHOLD:
            _log_signal("OB Pressure", market, secs_left,
                        signal_value=abs(imbalance), threshold=IMBALANCE_THRESHOLD,
                        reason="imbalance_too_weak")
            return position

        # Binance spread must be healthy — wide spread means dislocation not pressure
        SPREAD_THRESHOLD = 0.003
        if spread_pct > SPREAD_THRESHOLD:
            _log_signal("OB Pressure", market, secs_left,
                        signal_value=spread_pct, threshold=SPREAD_THRESHOLD,
                        reason="binance_spread_too_wide")
            return position

        # BTC momentum must confirm — imbalance without price movement is noise
        momentum = btc_momentum_pct(lookback_secs=30)
        if momentum is None:
            return position

        MOMENTUM_THRESHOLD = 0.03
        if abs(momentum) < MOMENTUM_THRESHOLD:
            _log_signal("OB Pressure", market, secs_left,
                        signal_value=abs(momentum), threshold=MOMENTUM_THRESHOLD,
                        reason="momentum_too_weak")
            return position

        # Direction from imbalance; momentum must agree
        direction    = "UP" if imbalance > 0 else "DOWN"
        momentum_dir = "UP" if momentum > 0 else "DOWN"
        if direction != momentum_dir:
            _log_signal("OB Pressure", market, secs_left,
                        signal_value=abs(imbalance), threshold=IMBALANCE_THRESHOLD,
                        reason="momentum_direction_conflict")
            return position

        prices = fetch_poly_prices(market, 50.0)
        if prices["spread"] > 0.06:
            _log_signal("OB Pressure", market, secs_left,
                        signal_value=prices["spread"], threshold=0.06,
                        reason="polymarket_spread_too_wide")
            return position

        # Size scales with imbalance strength
        intensity = min(abs(imbalance) / 0.40, 1.0)
        price     = prices["fill_up"] if direction == "UP" else prices["fill_down"]
        slip      = prices["slip_up"] if direction == "UP" else prices["slip_down"]
        size      = kelly_size(intensity, price, tracker.balance, "OB Pressure")
        if size == 0.0:
            return position  # Kelly says no edge at this price

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"imbalance":   round(imbalance, 4),
                         "momentum":    round(momentum, 4),
                         "spread_pct":  round(spread_pct, 4),
                         "slip":        round(slip, 4),
                         "secs_left":   round(secs_left)},
                        market)
            log.info(f"[OB Pressure] {direction} | imbalance={imbalance:+.3f} | "
                     f"momentum={momentum:+.3f}% | slip={slip:+.4f} | "
                     f"size={size:.2f} @ {price:.4f}")
            if not trade_id:
                return position
            t = StrategyTrade("ob_pressure", direction, size, price, market)
            t.trade_id = trade_id
            return t

    except Exception as e:
        log.warning(f"[OB Pressure] error: {e}")
    return position


def restore_state_from_supabase(portfolio: "SharedPortfolio") -> dict:
    """
    On startup, query Supabase for any OPEN trades that are not yet resolved.
    Re-populate the portfolio's exposure tracking and deduct committed capital
    so the exposure cap is correct from the first poll cycle.

    Also reconstructs positions dict keyed by strategy name so the main loop
    knows which strategies already have an open position on the current market.

    Returns: {condition_id: {strategy_name: StrategyTrade}} for open positions.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning("No Supabase credentials — skipping state restore")
        return {}

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "application/json",
                "Range":         "0-999",
            },
            params={
                "action":           "eq.OPEN",
                "resolved_outcome": "is.null",
                "select":           "id,trade_id,strategy,side,price,size,fee,"
                                    "condition_id,question,market_end_time",
            },
            timeout=10,
        )
        resp.raise_for_status()
        open_trades = resp.json()
    except Exception as e:
        log.warning(f"State restore failed — could not fetch open trades: {e}")
        return {}

    if not open_trades:
        log.info("State restore: no open trades found — starting fresh")
        return {}

    log.info(f"State restore: found {len(open_trades)} open trade(s) — restoring exposure")

    restored: dict = {}   # condition_id → {strategy_key: StrategyTrade}

    strategy_key_map = {
        "Chainlink Arb":       "chainlink",
        "Funding Reversion":   "funding",
        "Liquidation Cascade": "liquidation",
        "Basis Arb":           "basis",
        "Odds Mispricing":     "odds",
        "Volume Clock":        "volume",
        "OB Pressure":         "ob_pressure",
    }

    now = datetime.now(timezone.utc)

    for t in open_trades:
        condition_id  = t.get("condition_id", "")
        strategy_name = t.get("strategy", "")
        side          = t.get("side", "")
        size          = float(t.get("size", 0))
        price         = float(t.get("price", 0.5))
        fee           = float(t.get("fee", 0))
        trade_id      = t.get("trade_id", "")
        question      = t.get("question", "")
        end_time_str  = t.get("market_end_time")

        if not condition_id or not strategy_name or not side or size <= 0:
            continue

        # Parse market end time
        try:
            end_time = datetime.fromisoformat(
                end_time_str.replace("Z", "+00:00")) if end_time_str else now
        except Exception:
            end_time = now

        # Skip if market already expired — resolver will handle it
        if end_time <= now:
            log.info(f"State restore: skipping expired trade {trade_id[:8]}... "
                     f"({strategy_name} on {condition_id[:12]}...)")
            continue

        # Re-register exposure in portfolio — deduct size+fee from balance
        # Use _lock-safe debit to restore the pre-restart state
        portfolio.debit(size, fee, condition_id, side)

        # Reconstruct a minimal StrategyTrade so the main loop knows
        # this strategy already has a position on this market
        market = Market(
            condition_id  = condition_id,
            up_token_id   = "",   # not needed for position tracking
            down_token_id = "",
            question      = question,
            end_time      = end_time,
        )
        st = StrategyTrade(
            strategy    = strategy_name,
            side        = side,
            size        = size,
            entry_price = price,
            market      = market,
        )
        st.trade_id = trade_id

        strategy_key = strategy_key_map.get(strategy_name)
        if not strategy_key:
            log.warning(f"State restore: unknown strategy name '{strategy_name}' — skipping")
            continue

        if condition_id not in restored:
            restored[condition_id] = {}
        restored[condition_id][strategy_key] = st

        log.info(f"State restore: {strategy_name} {side} size={size:.2f} @ {price:.4f} "
                 f"on {condition_id[:12]}... | portfolio=${portfolio.available():.2f}")

    log.info(f"State restore complete | portfolio=${portfolio.available():.2f} | "
             f"markets with open positions: {len(restored)}")
    return restored


# ------------------------------------------------------------------ MAIN -----

def run():
    log.info("Multi-Strategy Mechanical Edge Simulator v3.2")
    log.info("Strategies: Chainlink | Funding | Liquidation | Basis | Odds | Volume | OB Pressure")
    log.info("Data: Coinbase + Kraken + OKX (all geo-unblocked)")
    log.info("Supabase: OPEN rows only — resolver.py writes all outcomes")

    # Start Chainlink WebSocket before anything else
    start_chainlink_ws()
    start_binance_liq_ws()
    log.info("Waiting 3s for WebSocket connections to establish...")
    time.sleep(3)

    portfolio = SharedPortfolio(STARTING_BALANCE)
    trackers = {
        "chainlink":   StrategyTracker("Chainlink Arb",        portfolio),
        "funding":     StrategyTracker("Funding Reversion",    portfolio),
        "liquidation": StrategyTracker("Liquidation Cascade",  portfolio),
        "basis":       StrategyTracker("Basis Arb",            portfolio),
        "odds":        StrategyTracker("Odds Mispricing",      portfolio),
        "volume":      StrategyTracker("Volume Clock",         portfolio),
        "ob_pressure": StrategyTracker("OB Pressure",         portfolio),
    }

    # Restore open positions and exposure from Supabase before first poll cycle.
    # This ensures the exposure cap is correct even after a redeploy mid-market.
    restored_state   = restore_state_from_supabase(portfolio)

    # Load real per-strategy win rates for Kelly sizing.
    # Strategies with < MIN_KELLY_SAMPLES resolved trades fall back to priors.
    _win_rates.update(fetch_win_rates())

    positions        = {k: None for k in trackers}
    cl_history       = deque(maxlen=6)
    current_market   = None
    last_market_fetch  = 0
    last_summary       = 0
    last_winrate_fetch = time.time()

    while True:
        try:
            now  = datetime.now(timezone.utc)
            hour = now.hour

            if hour in SKIP_HOURS_UTC:
                log.info(f"Skipping hour {hour:02d}:00 UTC")
                time.sleep(30)
                continue

            refresh_shared_data()
            compute_regime()

            # Refresh market every 60s
            if time.time() - last_market_fetch > 60:
                market = fetch_current_market()
                if market and (not current_market or
                               market.condition_id != current_market.condition_id):
                    log.info(f"New market: {market.question}")
                    if current_market:
                        portfolio.clear_exposure(current_market.condition_id)
                    current_market = market
                    positions      = {k: None for k in trackers}
                    cl_history.clear()
                    # Restore any open positions from pre-restart state
                    if market.condition_id in restored_state:
                        for strategy_key, st in restored_state[market.condition_id].items():
                            positions[strategy_key] = st
                            log.info(f"Restored position: {strategy_key} {st.side} "
                                     f"size={st.size:.2f} on current market")
                last_market_fetch = time.time()

            if not current_market:
                log.warning("No market — retrying in 15s")
                time.sleep(15)
                continue

            secs_left = (current_market.end_time - now).total_seconds()

            # Market expired — resolve all open positions locally
            if secs_left <= 0:
                log.info(f"Market expired: {current_market.question} — resolving positions locally")
                positions         = resolve_positions(positions, trackers, current_market, portfolio)
                current_market    = None
                last_market_fetch = 0
                time.sleep(5)
                continue

            spot    = _price_cache["btc"]
            futures = _basis_cache["futures"]
            basis   = (futures - spot) / spot * 100 if spot > 0 and futures > 0 else 0.0

            log.info(f"BTC=${spot:.2f} | {secs_left:.0f}s left | "
                     f"funding={_funding_cache['rate']:+.6f} | basis={basis:+.3f}% | "
                     f"regime={_regime['composite']} | session={_regime['session']} | "
                     f"activity={_regime['activity']}")

            # Run all 7 strategies
            positions["chainlink"]   = strategy_chainlink_arb(
                current_market, secs_left, trackers["chainlink"],
                positions["chainlink"], cl_history)

            positions["funding"]     = strategy_funding_reversion(
                current_market, secs_left, trackers["funding"],
                positions["funding"])

            positions["liquidation"] = strategy_liquidation_cascade(
                current_market, secs_left, trackers["liquidation"],
                positions["liquidation"])

            positions["basis"]       = strategy_basis_arb(
                current_market, secs_left, trackers["basis"],
                positions["basis"])

            positions["odds"]        = strategy_odds_mispricing(
                current_market, secs_left, trackers["odds"],
                positions["odds"])

            positions["volume"]      = strategy_volume_clock(
                current_market, secs_left, trackers["volume"],
                positions["volume"])

            positions["ob_pressure"] = strategy_ob_pressure(
                current_market, secs_left, trackers["ob_pressure"],
                positions["ob_pressure"])

            # Refresh win rates every 30 minutes so Kelly sizing
            # automatically improves as more trades resolve
            if time.time() - last_winrate_fetch > 1800:
                new_rates = fetch_win_rates()
                _win_rates.clear()
                _win_rates.update(new_rates)
                last_winrate_fetch = time.time()

            # Print summary every 30 minutes
            if time.time() - last_summary > 1800:
                log.info("=" * 60)
                log.info("STRATEGY PERFORMANCE SUMMARY (local estimates)")
                log.info("Ground truth win rates available via resolver.py")
                for line in portfolio.summary_lines():
                    log.info(line)
                for t in trackers.values():
                    log.info(t.summary())
                log.info("=" * 60)
                last_summary = time.time()

            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            log.info("Stopped.")
            log.info("FINAL LOCAL SUMMARY:")
            for line in portfolio.summary_lines():
                log.info(line)
            for t in trackers.values():
                log.info(t.summary())
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    print("MULTI-STRATEGY BOT STARTING v3", flush=True)
    run()
