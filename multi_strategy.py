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
MAX_BET          = 50.0
MIN_BET          = 5.0
POLL_SEC         = 5

SKIP_HOURS_UTC   = {0, 1, 2, 3, 4, 5}


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


def supabase_snapshot(snapshot: dict):
    """Insert a signal snapshot into Supabase every 5 minutes."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/signal_snapshots",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=snapshot,
            timeout=5,
        )
        if r.status_code == 201:
            log.info("Snapshot saved to Supabase")
        else:
            log.warning(f"Snapshot insert status: {r.status_code} {r.text[:100]}")
    except Exception as e:
        log.warning(f"Snapshot insert failed: {e}")


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
    }
    log.debug(f"[SIGNAL] {strategy} | {reason} | value={signal_value:.6f} threshold={threshold:.6f}")
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


# --------------------------------------------------------- STRATEGY TRACKER --

class StrategyTracker:
    def __init__(self, name: str):
        self.name      = name
        self.balance   = STARTING_BALANCE
        self.start_bal = STARTING_BALANCE
        self.wins      = 0
        self.losses    = 0
        self.trades    = []
        self.logfile   = f"strategy_{name.lower().replace(' ', '_')}.jsonl"

    def open(self, side: str, price: float, size: float,
             signal_data: dict, market: Market) -> str:
        """
        Records an open position locally and writes ONE row to Supabase.
        Returns the trade_id so the caller can store it on StrategyTrade.
        The resolver will later patch this row with the real outcome.
        """
        fee = size * TAKER_FEE
        self.balance -= (size + fee)
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
            "trade_id":       trade_id,
            "strategy":       self.name,
            "action":         "OPEN",
            "side":           side,
            "price":          price,
            "size":           size,
            "fee":            round(size * TAKER_FEE, 4),
            "condition_id":   market.condition_id,
            "question":       market.question,
            "market_end_time": market.end_time.isoformat(),
            "signal_data":    signal_data,
        })
        log.info(f"[{self.name}] OPEN {side} | size={size:.2f} @ {price:.4f} | "
                 f"balance=${self.balance:.2f} | trade_id={trade_id[:8]}... | "
                 f"{json.dumps(signal_data)}")
        return trade_id

    def close(self, side: str, entry_price: float, exit_price: float,
              size: float, market: Market, reason: str = "RESOLVED"):
        """
        Updates local balance and win/loss counters only.
        Does NOT write to Supabase — resolver.py owns all outcome writes.
        """
        if side == "UP":
            pnl = (exit_price - entry_price) * size / entry_price
        else:
            pnl = (entry_price - exit_price) * size / entry_price
        fee = size * TAKER_FEE
        pnl -= fee
        self.balance += size + pnl
        if pnl > 0:
            self.wins   += 1
        else:
            self.losses += 1

        entry = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "action":      "CLOSE_LOCAL",
            "strategy":    self.name,
            "reason":      reason,
            "side":        side,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "size":        size,
            "pnl":         round(pnl, 4),
            "fee":         round(fee, 4),
            "balance":     round(self.balance, 2),
            "condition_id": market.condition_id,
        }
        self.trades.append(entry)
        with open(self.logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # No supabase_insert here — resolver writes the real outcome to the OPEN row.
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total else 0
        log.info(f"[{self.name}] CLOSE_LOCAL {side} ({reason}) | exit={exit_price:.2f} | "
                 f"PnL={pnl:+.3f} | balance=${self.balance:.2f} | "
                 f"WR={wr:.0f}% ({self.wins}W/{self.losses}L)")

    def void(self, pos: "StrategyTrade", market: Market):
        """
        Market was voided — refund the position size (minus fees already paid).
        Does not count as win or loss. Does not write to Supabase —
        resolver.py will mark the row VOID when it confirms the market outcome.
        """
        self.balance += pos.size  # refund stake, fees are lost
        entry = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "action":       "VOID_LOCAL",
            "strategy":     self.name,
            "side":         pos.side,
            "entry_price":  pos.entry_price,
            "size":         pos.size,
            "pnl":          0.0,
            "balance":      round(self.balance, 2),
            "condition_id": market.condition_id,
        }
        self.trades.append(entry)
        with open(self.logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")
        log.info(f"[{self.name}] VOID_LOCAL {pos.side} | size refunded={pos.size:.2f} | "
                 f"balance=${self.balance:.2f}")

    def summary(self) -> str:
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total else 0
        pnl   = self.balance - self.start_bal
        return (f"{self.name}: ${self.balance:.2f} | PnL={pnl:+.2f} | "
                f"WR={wr:.0f}% | {total} trades (local estimate — "
                f"ground truth in Supabase via resolver)")


# --------------------------------------------------------- SHARED DATA -------

_price_cache:    dict  = {"btc": 0.0, "eth": 0.0, "fetched_at": 0.0}
_funding_cache:  dict  = {"okx": 0.0, "binance": 0.0, "rate": 0.0, "fetched_at": 0.0}
_basis_cache:    dict  = {"spot": 0.0, "futures": 0.0, "fetched_at": 0.0}
_liq_cache:      dict  = {"long": 0.0, "short": 0.0, "fetched_at": 0.0}
_vol_cache:      dict  = {"range_pct": 0.0, "fetched_at": 0.0}
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

    # Liquidations via OKX
    if now - _liq_cache["fetched_at"] >= 60:
        try:
            r = requests.get(
                f"{OKX_BASE}/api/v5/public/liquidation-orders",
                params={"instType": "SWAP", "uly": "BTC-USDT",
                        "state": "filled", "limit": "20"},
                timeout=5)
            r.raise_for_status()
            orders    = r.json().get("data", [{}])[0].get("details", [])
            cutoff_ms = (now - 300) * 1000
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

        # Merge Binance liquidations (WebSocket feed) into cache
        # Binance handles 3-5x more BTC liq volume than OKX
        try:
            bnb = get_binance_liq_5min()
            _liq_cache["long"]  += bnb["long"]
            _liq_cache["short"] += bnb["short"]
            if bnb["long"] + bnb["short"] > 0:
                log.info(f"Liquidations — OKX+Binance combined | "
                         f"long=${_liq_cache['long']:,.0f} short=${_liq_cache['short']:,.0f}")
        except Exception as e:
            log.warning(f"Binance liq merge failed: {e}")

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
                return Market(
                    condition_id  = m["conditionId"],
                    up_token_id   = token_ids[0],
                    down_token_id = token_ids[1],
                    question      = m["question"],
                    end_time      = end_time,
                )
            except Exception:
                continue
        return None
    except Exception as e:
        log.error(f"Market fetch failed: {e}")
        return None


def fetch_poly_prices(market: Market) -> dict:
    try:
        up_mid   = float(requests.get(f"{POLYMARKET_HOST}/midpoint",
                         params={"token_id": market.up_token_id},
                         timeout=8).json().get("mid", 0.5))
        down_mid = float(requests.get(f"{POLYMARKET_HOST}/midpoint",
                         params={"token_id": market.down_token_id},
                         timeout=8).json().get("mid", 0.5))
        book     = requests.get(f"{POLYMARKET_HOST}/book",
                                params={"token_id": market.up_token_id},
                                timeout=8).json()
        bids     = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
        asks     = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
        spread   = (float(asks[0]["price"]) - float(bids[0]["price"])) if bids and asks else 0.1
        return {"up_mid": up_mid, "down_mid": down_mid, "spread": spread}
    except Exception:
        return {"up_mid": 0.5, "down_mid": 0.5, "spread": 0.1}


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


def resolve_positions(positions: dict, trackers: dict, market: Market) -> dict:
    """
    Resolves all open positions for an expired market.
    Updates local balance/win-loss counters only.
    Supabase outcome writes happen in resolver.py using trade_id.
    """
    any_open = any(p is not None for p in positions.values())
    if not any_open:
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

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        # Size scales with divergence strength — stronger divergence = bigger bet
        intensity = min(abs(divergence) / 0.30, 1.0)
        size      = min(MAX_BET * intensity, tracker.balance * 0.4)
        size      = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"divergence":  round(divergence, 4),
                         "cl_age":      round(cl_age, 1),
                         "momentum":    round(momentum, 4)},
                        market)
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
    Requires BOTH OKX and Binance to agree on direction — filters false signals.
    Threshold lowered to 0.0003 (was 0.0005) for more frequent firing.
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

        price     = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        intensity = min(abs(avg_rate) / 0.001, 1.0)
        size      = min(MAX_BET * intensity, tracker.balance * 0.3)
        size      = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"okx_rate":  round(okx_rate, 6),
                         "bnb_rate":  round(bnb_rate, 6),
                         "avg_rate":  round(avg_rate, 6),
                         "intensity": round(intensity, 3)},
                        market)
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
    Data from OKX (REST poll) + Binance (WebSocket) — combined for full market picture.

    Improvements:
    - 2-minute lookback (was 5min) — more recent signal, price hasn't already moved
    - Volatility-adjusted intensity — $50M in calm market != $50M in volatile market
    - Minimum imbalance ratio required — filters balanced liquidation events
    """
    try:
        binance    = get_binance_liq_2min()
        okx_long   = _liq_cache["long"]
        okx_short  = _liq_cache["short"]
        long_liqs  = okx_long  + binance["long"]
        short_liqs = okx_short + binance["short"]
        total      = long_liqs + short_liqs

        if total < 300_000:
            _log_signal("Liquidation Cascade", market, secs_left,
                        signal_value=total, threshold=300_000,
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
        raw_intensity    = min(total / 2_000_000, 1.0)
        vol_scalar       = max(0.3, min(1.0, 0.15 / vol_range))  # inverse vol
        adjusted_intensity = min(raw_intensity * vol_scalar, 1.0)

        prices = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            _log_signal("Liquidation Cascade", market, secs_left,
                        signal_value=prices["spread"], threshold=0.06,
                        reason="spread_too_wide")
            return position

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * adjusted_intensity, tracker.balance * 0.3)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"long_liqs":   round(long_liqs),
                         "short_liqs":  round(short_liqs),
                         "vol_range":   round(vol_range, 4),
                         "intensity":   round(adjusted_intensity, 3),
                         "source":      "OKX+Binance"},
                        market)
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

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * intensity, tracker.balance * 0.3)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"basis_pct": round(basis_pct, 4),
                         "spot": round(spot, 2), "futures": round(futures, 2)},
                        market)
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

        direction = "DOWN" if deviation > 0 else "UP"
        intensity = min(abs(deviation) / 0.15, 1.0)
        price     = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size      = min(MAX_BET * intensity, tracker.balance * 0.3)
        size      = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"up_mid": round(up_mid, 4), "deviation": round(deviation, 4),
                         "secs_left": round(secs_left)},
                        market)
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

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * intensity, tracker.balance * 0.3)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            trade_id = tracker.open(direction, price, size,
                        {"buy_ratio": round(buy_ratio, 4), "secs_left": round(secs_left)},
                        market)
            t = StrategyTrade("volume_clock", direction, size, price, market)
            t.trade_id = trade_id
            return t
    except Exception as e:
        log.warning(f"[Volume clock] error: {e}")
    return position


# ------------------------------------------------------------------ MAIN -----

def run():
    log.info("Multi-Strategy Mechanical Edge Simulator v3.1")
    log.info("Strategies: Chainlink | Funding | Liquidation | Basis | Odds | Volume")
    log.info("Data: Coinbase + Kraken + OKX (all geo-unblocked)")
    log.info("Supabase: OPEN rows only — resolver.py writes all outcomes")

    # Start Chainlink WebSocket before anything else
    start_chainlink_ws()
    start_binance_liq_ws()
    log.info("Waiting 3s for WebSocket connections to establish...")
    time.sleep(3)

    trackers = {
        "chainlink":   StrategyTracker("Chainlink Arb"),
        "funding":     StrategyTracker("Funding Reversion"),
        "liquidation": StrategyTracker("Liquidation Cascade"),
        "basis":       StrategyTracker("Basis Arb"),
        "odds":        StrategyTracker("Odds Mispricing"),
        "volume":      StrategyTracker("Volume Clock"),
    }

    positions         = {k: None for k in trackers}
    cl_history        = deque(maxlen=6)
    current_market    = None
    last_market_fetch = 0
    last_summary      = 0
    last_snapshot     = 0

    while True:
        try:
            now  = datetime.now(timezone.utc)
            hour = now.hour

            if hour in SKIP_HOURS_UTC:
                log.info(f"Skipping hour {hour:02d}:00 UTC")
                time.sleep(30)
                continue

            refresh_shared_data()

            # Refresh market every 60s
            if time.time() - last_market_fetch > 60:
                market = fetch_current_market()
                if market and (not current_market or
                               market.condition_id != current_market.condition_id):
                    log.info(f"New market: {market.question}")
                    current_market = market
                    positions      = {k: None for k in trackers}
                    cl_history.clear()
                last_market_fetch = time.time()

            if not current_market:
                log.warning("No market — retrying in 15s")
                time.sleep(15)
                continue

            secs_left = (current_market.end_time - now).total_seconds()

            # Market expired — resolve all open positions locally
            if secs_left <= 0:
                log.info(f"Market expired: {current_market.question} — resolving positions locally")
                positions         = resolve_positions(positions, trackers, current_market)
                current_market    = None
                last_market_fetch = 0
                time.sleep(5)
                continue

            spot    = _price_cache["btc"]
            futures = _basis_cache["futures"]
            basis   = (futures - spot) / spot * 100 if spot > 0 and futures > 0 else 0.0

            log.info(f"BTC=${spot:.2f} | {secs_left:.0f}s left | "
                     f"funding={_funding_cache['rate']:+.6f} | basis={basis:+.3f}%")

            # Save signal snapshot every 5 minutes
            if time.time() - last_snapshot > 300:
                cl_price, cl_age = fetch_chainlink_price()
                cl_div      = round((spot - cl_price) / cl_price * 100, 4) if cl_price else 0.0
                candles     = list(_volume_history)
                recent_vol  = candles[-3:] if len(candles) >= 3 else []
                total_vol   = sum(c["volume"] for c in recent_vol)
                buy_vol     = sum(c["buy_vol"] for c in recent_vol)
                buy_ratio   = round(buy_vol / total_vol, 4) if total_vol > 0 else 0.5
                liq_2min    = get_binance_liq_2min()
                long_liq    = round(_liq_cache["long"]  + liq_2min["long"], 2)
                short_liq   = round(_liq_cache["short"] + liq_2min["short"], 2)
                prices      = fetch_poly_prices(current_market) if current_market else {}
                supabase_snapshot({
                    "btc_price":        round(spot, 2),
                    "eth_price":        round(_price_cache["eth"], 2),
                    "funding_rate":     round(_funding_cache["rate"], 6),
                    "okx_funding":      round(_funding_cache["okx"], 6),
                    "gateio_funding":   round(_funding_cache["binance"], 6),
                    "basis_pct":        round(basis, 4),
                    "chainlink_div":    cl_div,
                    "chainlink_age":    round(cl_age, 1) if cl_age else 30.0,
                    "up_mid":           round(prices.get("up_mid", 0.5), 4),
                    "spread":           round(prices.get("spread", 0.1), 4),
                    "volume_buy_ratio": buy_ratio,
                    "long_liq_usd":     long_liq,
                    "short_liq_usd":    short_liq,
                    "liq_total_usd":    round(long_liq + short_liq, 2),
                    "vol_range_pct":    round(_vol_cache.get("range_pct", 0.0), 4),
                    "secs_left":        round(secs_left),
                    "market_question":  current_market.question if current_market else "",
                    "condition_id":     current_market.condition_id if current_market else "",
                })
                last_snapshot = time.time()

            # Run all 6 strategies
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

            # Print summary every 30 minutes
            if time.time() - last_summary > 1800:
                log.info("=" * 60)
                log.info("STRATEGY PERFORMANCE SUMMARY (local estimates)")
                log.info("Ground truth win rates available via resolver.py")
                for t in trackers.values():
                    log.info(t.summary())
                log.info("=" * 60)
                last_summary = time.time()

            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            log.info("Stopped.")
            log.info("FINAL LOCAL SUMMARY:")
            for t in trackers.values():
                log.info(t.summary())
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    print("MULTI-STRATEGY BOT STARTING v3", flush=True)
    run()
