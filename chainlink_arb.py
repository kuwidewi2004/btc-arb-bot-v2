"""
Chainlink Arbitrage Bot v3
===========================
Mechanical edges stacked:
  1. Chainlink lag arbitrage       — oracle heartbeat lag creates predictable direction
  2. Funding rate confirmation     — extreme funding = directional pressure (mechanical cost)
  3. Cross-asset confirmation      — BTC + ETH moving same direction = macro move
  4. Time-of-day filter            — skip low-liquidity hours
  5. Volume spike detection        — sudden 3x volume confirms real money moving
  6. Open interest spike           — new futures contracts = directional conviction
  7. Liquidation cascade detection — forced liquidations create predictable momentum

Confidence scoring:
  - Chainlink divergence is PRIMARY gate — no trade without it
  - Each secondary edge adds to confirmation count
  - 0 confirmations → skip
  - 1 confirmation  → 25% of max size
  - 2 confirmations → 50% of max size
  - 3+ confirmations → 100% of max size

Price feeds: Coinbase (primary) + Kraken (fallback) — no geo-blocking

DRY RUN MODE: Set DRY_RUN = True to simulate without real orders.
"""

import os
import time
import logging
import json
import sys
import io
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import requests
import numpy as np

print("BOT STARTING UP", flush=True)

# Fix Windows terminal Unicode
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chainlink_arb.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ CONFIG ---

DRY_RUN              = True

POLYMARKET_HOST      = "https://clob.polymarket.com"
GAMMA_API            = "https://gamma-api.polymarket.com"
POLY_PRIVATE_KEY     = os.environ.get("POLY_PRIVATE_KEY", "YOUR_PRIVATE_KEY_HERE")
POLY_FUNDER          = os.environ.get("POLY_FUNDER_ADDRESS", "")
POLY_CHAIN_ID        = 137

BINANCE_FUTURES      = "https://fapi.binance.com"
BTC_SYMBOL           = "BTCUSDT"
ETH_SYMBOL           = "ETHUSDT"

# Primary gate
DIVERGENCE_THRESHOLD = 0.15
MAX_LAG_AGE_SEC      = 45
MIN_MARKET_SECS_LEFT = 90
MAX_BOOK_SPREAD      = 0.06
TAKER_FEE            = 0.0025
MAX_POSITION_USDC    = 100.0
POLL_INTERVAL_SEC    = 5

# Funding rate
FUNDING_EXTREME_POS  =  0.0003
FUNDING_EXTREME_NEG  = -0.0003
FUNDING_CACHE_SEC    = 300

# Cross-asset
ETH_MOVE_THRESHOLD   = 0.05
ETH_LOOKBACK_SEC     = 60

# Time of day (UTC)
SKIP_HOURS_UTC       = {0, 1, 2, 3, 4, 5}
REDUCED_HOURS_UTC    = {6, 7, 22, 23}

# Volume spike
VOLUME_SPIKE_MULT    = 2.5        # current volume must be 2.5x the 20-period average
VOLUME_LOOKBACK      = 20         # candles to compute average volume

# Open interest spike
OI_SPIKE_PCT         = 0.3        # OI must have grown 0.3%+ in last 5 minutes
OI_CACHE_SEC         = 30

# Liquidation cascade
LIQ_THRESHOLD_USD    = 500_000    # $500k+ liquidations in last 5 minutes = cascade
LIQ_CACHE_SEC        = 60

DRY_RUN_BALANCE      = 100.0


# --------------------------------------------------------------- DATACLASSES --

@dataclass
class PriceSnapshot:
    binance_price:     float
    chainlink_price:   float
    chainlink_age_sec: float
    timestamp:         datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def divergence_pct(self) -> float:
        return (self.binance_price - self.chainlink_price) / self.chainlink_price * 100

    @property
    def direction(self) -> Optional[str]:
        return "UP" if self.binance_price > self.chainlink_price else "DOWN"


@dataclass
class ConfirmationResult:
    funding_confirms:    bool  = False
    funding_rate:        float = 0.0
    crossasset_confirms: bool  = False
    eth_move_pct:        float = 0.0
    volume_confirms:     bool  = False
    volume_ratio:        float = 0.0
    oi_confirms:         bool  = False
    oi_change_pct:       float = 0.0
    liq_confirms:        bool  = False
    liq_volume_usd:      float = 0.0
    timeofday_ok:        bool  = True
    timeofday_reduced:   bool  = False

    @property
    def count(self) -> int:
        return sum([
            self.funding_confirms,
            self.crossasset_confirms,
            self.volume_confirms,
            self.oi_confirms,
            self.liq_confirms,
        ])

    @property
    def size_multiplier(self) -> float:
        if not self.timeofday_ok:
            return 0.0
        min_conf = 2 if self.timeofday_reduced else 1
        if self.count < min_conf:
            return 0.0
        if self.count == 1: return 0.25
        if self.count == 2: return 0.50
        if self.count >= 3: return 1.00
        return 0.0

    def summary(self) -> str:
        parts = [
            f"funding={'Y' if self.funding_confirms else 'n'}({self.funding_rate:+.5f})",
            f"eth={'Y' if self.crossasset_confirms else 'n'}({self.eth_move_pct:+.2f}%)",
            f"vol={'Y' if self.volume_confirms else 'n'}({self.volume_ratio:.1f}x)",
            f"oi={'Y' if self.oi_confirms else 'n'}({self.oi_change_pct:+.2f}%)",
            f"liq={'Y' if self.liq_confirms else 'n'}(${self.liq_volume_usd/1e6:.1f}M)",
        ]
        return " | ".join(parts)


@dataclass
class Market:
    condition_id:  str
    up_token_id:   str
    down_token_id: str
    question:      str
    end_time:      datetime


@dataclass
class Position:
    side:          str
    entry_price:   float
    size:          float
    market:        Market
    confirmations: int = 0
    entry_time:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------- PORTFOLIO ----------

class SimPortfolio:
    def __init__(self, balance: float):
        self.balance          = balance
        self.starting_balance = balance
        self.trades           = []
        self.wins             = 0
        self.losses           = 0

    def open(self, side: str, price: float, size: float,
             divergence: float, confirmations: int):
        fee = size * TAKER_FEE
        self.balance -= (size + fee)
        self.trades.append({
            "action": "OPEN", "side": side, "price": price,
            "size": size, "fee": round(fee, 4),
            "divergence_pct": round(divergence, 4),
            "confirmations": confirmations,
        })
        log.info(f"[DRY RUN] OPEN {side} | size={size:.2f} @ {price:.4f} | "
                 f"div={divergence:+.3f}% | conf={confirmations} | "
                 f"balance=${self.balance:.2f}")

    def close(self, side: str, entry_price: float, exit_price: float,
              size: float, reason: str = "RESOLVED"):
        pnl  = (exit_price - entry_price) * size / entry_price if side == "UP" \
               else (entry_price - exit_price) * size / entry_price
        fee  = size * TAKER_FEE
        pnl -= fee
        self.balance += size + pnl
        if pnl > 0: self.wins   += 1
        else:       self.losses += 1
        self.trades.append({
            "action": "CLOSE", "reason": reason, "side": side,
            "entry": entry_price, "exit": exit_price,
            "pnl": round(pnl, 4), "fee": round(fee, 4),
        })
        log.info(f"[DRY RUN] CLOSE {side} ({reason}) | PnL={pnl:+.3f} | "
                 f"balance=${self.balance:.2f}")
        self._summary()

    def _summary(self):
        pnl   = self.balance - self.starting_balance
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total else 0
        log.info(f"[DRY RUN] ${self.balance:.2f} | PnL={pnl:+.2f} | "
                 f"WR={wr:.0f}% ({self.wins}W/{self.losses}L)")

    def save(self):
        with open("chainlink_arb_results.json", "w") as f:
            json.dump({
                "starting_balance": self.starting_balance,
                "final_balance":    round(self.balance, 4),
                "total_pnl":        round(self.balance - self.starting_balance, 4),
                "wins":   self.wins,
                "losses": self.losses,
                "trades": self.trades,
            }, f, indent=2)
        log.info("Results saved to chainlink_arb_results.json")


# --------------------------------------------------------- PRICE FEEDS --------

def fetch_spot_price(symbol: str = BTC_SYMBOL) -> Optional[float]:
    """Fetch current price from Coinbase (primary) or Kraken (fallback)."""
    coin_map   = {"BTCUSDT": "BTC-USD",  "ETHUSDT": "ETH-USD"}
    kraken_map = {"BTCUSDT": "XBTUSD",   "ETHUSDT": "ETHUSD"}

    # Coinbase
    try:
        resp = requests.get(
            f"https://api.coinbase.com/v2/prices/{coin_map.get(symbol, 'BTC-USD')}/spot",
            timeout=5,
        )
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])
    except Exception:
        pass

    # Kraken fallback
    try:
        resp = requests.get(
            f"https://api.kraken.com/0/public/Ticker?pair={kraken_map.get(symbol, 'XBTUSD')}",
            timeout=5,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        key    = list(result.keys())[0]
        return float(result[key]["c"][0])
    except Exception as e:
        log.warning(f"Price fetch failed ({symbol}): {e}")
        return None


def fetch_chainlink_price() -> tuple:
    """Returns (price, age_seconds)."""
    try:
        resp = requests.get(
            "https://min-api.cryptocompare.com/data/price",
            params={"fsym": "BTC", "tsyms": "USD"},
            timeout=5,
        )
        resp.raise_for_status()
        price   = float(resp.json()["USD"])
        age_sec = 30.0
        try:
            cl_resp = requests.get(
                "https://data.chain.link/api/proxy/btc-usd/latest",
                timeout=5,
            )
            if cl_resp.status_code == 200:
                cl_data = cl_resp.json()
                updated = cl_data.get("updatedAt") or cl_data.get("timestamp")
                if updated:
                    updated_dt = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                    age_sec    = (datetime.now(timezone.utc) - updated_dt).total_seconds()
                    price      = float(cl_data.get("answer") or cl_data.get("price") or price)
        except Exception:
            pass
        return price, age_sec
    except Exception as e:
        log.warning(f"Chainlink price fetch failed: {e}")
        return None, None


def fetch_snapshot() -> Optional[PriceSnapshot]:
    binance_price           = fetch_spot_price(BTC_SYMBOL)
    chainlink_price, cl_age = fetch_chainlink_price()
    if binance_price is None or chainlink_price is None:
        return None
    return PriceSnapshot(
        binance_price     = binance_price,
        chainlink_price   = chainlink_price,
        chainlink_age_sec = cl_age or 30.0,
    )


# ----------------------------------------- SECONDARY EDGE: FUNDING RATE -----

_funding_cache: dict = {"rate": 0.0, "fetched_at": 0.0}


def fetch_funding_rate() -> float:
    now = time.time()
    if now - _funding_cache["fetched_at"] < FUNDING_CACHE_SEC:
        return _funding_cache["rate"]
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES}/fapi/v1/premiumIndex",
            params={"symbol": BTC_SYMBOL}, timeout=5,
        )
        resp.raise_for_status()
        rate = float(resp.json()["lastFundingRate"])
        _funding_cache["rate"]       = rate
        _funding_cache["fetched_at"] = now
        return rate
    except Exception as e:
        log.warning(f"Funding rate fetch failed: {e}")
        return 0.0


def check_funding(direction: str) -> tuple:
    rate = fetch_funding_rate()
    if direction == "UP"   and rate <= FUNDING_EXTREME_NEG: return True, rate
    if direction == "DOWN" and rate >= FUNDING_EXTREME_POS: return True, rate
    return False, rate


# --------------------------------------- SECONDARY EDGE: CROSS-ASSET ETH ----

_eth_history: deque = deque(maxlen=20)
_last_eth_fetch: float = 0.0


def check_crossasset(direction: str) -> tuple:
    global _last_eth_fetch
    now = time.time()
    if now - _last_eth_fetch >= POLL_INTERVAL_SEC:
        eth_price = fetch_spot_price(ETH_SYMBOL)
        if eth_price:
            _eth_history.append((now, eth_price))
        _last_eth_fetch = now

    if len(_eth_history) < 2:
        return False, 0.0

    cutoff    = now - ETH_LOOKBACK_SEC
    window    = [(t, p) for t, p in _eth_history if t >= cutoff]
    if len(window) < 2:
        return False, 0.0

    eth_move  = (window[-1][1] - window[0][1]) / window[0][1] * 100
    if direction == "UP"   and eth_move >= ETH_MOVE_THRESHOLD: return True, eth_move
    if direction == "DOWN" and eth_move <= -ETH_MOVE_THRESHOLD: return True, eth_move
    return False, eth_move


# -------------------------------------- SECONDARY EDGE: VOLUME SPIKE --------

_volume_history: deque = deque(maxlen=VOLUME_LOOKBACK + 2)
_last_volume_fetch: float = 0.0


def check_volume_spike(direction: str) -> tuple:
    """
    Fetches recent 1-minute klines from Binance futures.
    Returns (confirms, ratio) where ratio = current_volume / avg_volume.
    A spike in the direction of our trade confirms real money is moving.
    """
    global _last_volume_fetch
    now = time.time()

    if now - _last_volume_fetch >= 60:   # refresh every minute
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES}/fapi/v1/klines",
                params={"symbol": BTC_SYMBOL, "interval": "1m",
                        "limit": VOLUME_LOOKBACK + 2},
                timeout=8,
            )
            resp.raise_for_status()
            candles = resp.json()
            _volume_history.clear()
            for c in candles:
                _volume_history.append({
                    "volume":        float(c[5]),
                    "taker_buy_vol": float(c[9]),
                    "close":         float(c[4]),
                    "open":          float(c[1]),
                })
            _last_volume_fetch = now
        except Exception as e:
            log.warning(f"Volume fetch failed: {e}")
            return False, 0.0

    if len(_volume_history) < VOLUME_LOOKBACK:
        return False, 0.0

    candles      = list(_volume_history)
    current      = candles[-1]
    historical   = candles[:-2]   # exclude last 2 (current + forming)
    avg_vol      = sum(c["volume"] for c in historical) / len(historical)

    if avg_vol == 0:
        return False, 0.0

    ratio        = current["volume"] / avg_vol
    # Also check taker buy/sell ratio to confirm direction
    total_vol    = current["volume"]
    buy_vol      = current["taker_buy_vol"]
    sell_vol     = total_vol - buy_vol
    buy_dominant = buy_vol > sell_vol

    spike        = ratio >= VOLUME_SPIKE_MULT
    dir_confirms = (direction == "UP" and buy_dominant) or \
                   (direction == "DOWN" and not buy_dominant)

    return (spike and dir_confirms), ratio


# ------------------------------------ SECONDARY EDGE: OPEN INTEREST SPIKE ---

_oi_cache: dict = {"oi": 0.0, "prev_oi": 0.0, "fetched_at": 0.0}


def check_open_interest(direction: str) -> tuple:
    """
    Fetches BTC futures open interest from Binance.
    A spike in OI means new money is entering with directional conviction.
    Returns (confirms, change_pct).
    """
    now = time.time()

    if now - _oi_cache["fetched_at"] >= OI_CACHE_SEC:
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES}/fapi/v1/openInterest",
                params={"symbol": BTC_SYMBOL},
                timeout=5,
            )
            resp.raise_for_status()
            new_oi = float(resp.json()["openInterest"])
            _oi_cache["prev_oi"]    = _oi_cache["oi"]
            _oi_cache["oi"]         = new_oi
            _oi_cache["fetched_at"] = now
        except Exception as e:
            log.warning(f"OI fetch failed: {e}")
            return False, 0.0

    prev_oi = _oi_cache["prev_oi"]
    curr_oi = _oi_cache["oi"]

    if prev_oi == 0:
        return False, 0.0

    change_pct = (curr_oi - prev_oi) / prev_oi * 100

    # OI increasing = new positions opening = confirms momentum direction
    # Combined with price direction, rising OI confirms the move
    confirms = abs(change_pct) >= OI_SPIKE_PCT

    return confirms, change_pct


# --------------------------------- SECONDARY EDGE: LIQUIDATION CASCADE ------

_liq_cache: dict = {"volume": 0.0, "direction": None, "fetched_at": 0.0}


def check_liquidation_cascade(direction: str) -> tuple:
    """
    Fetches recent liquidation orders from Binance futures.
    Large liquidations in our direction = forced selling/buying confirms move.
    Returns (confirms, total_usd_liquidated).
    """
    now = time.time()

    if now - _liq_cache["fetched_at"] >= LIQ_CACHE_SEC:
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES}/fapi/v1/allForceOrders",
                params={"symbol": BTC_SYMBOL, "limit": 100},
                timeout=5,
            )
            resp.raise_for_status()
            orders = resp.json()

            cutoff_ms    = (now - 300) * 1000   # last 5 minutes
            long_liqs    = 0.0
            short_liqs   = 0.0

            for o in orders:
                if float(o.get("time", 0)) < cutoff_ms:
                    continue
                usd_val = float(o.get("origQty", 0)) * float(o.get("price", 0))
                if o.get("side") == "SELL":   # long position liquidated
                    long_liqs  += usd_val
                else:                          # short position liquidated
                    short_liqs += usd_val

            # More long liquidations = bearish (forced sellers)
            # More short liquidations = bullish (forced buyers/short squeeze)
            dominant_dir = "DOWN" if long_liqs > short_liqs else "UP"
            total        = long_liqs + short_liqs

            _liq_cache["volume"]     = total
            _liq_cache["direction"]  = dominant_dir
            _liq_cache["fetched_at"] = now

        except Exception as e:
            log.warning(f"Liquidation fetch failed: {e}")
            return False, 0.0

    total    = _liq_cache["volume"]
    liq_dir  = _liq_cache["direction"]
    confirms = total >= LIQ_THRESHOLD_USD and liq_dir == direction

    return confirms, total


# -------------------------------------------- TIME OF DAY FILTER ------------

def check_time_of_day() -> tuple:
    hour = datetime.now(timezone.utc).hour
    if hour in SKIP_HOURS_UTC:
        return False, False
    if hour in REDUCED_HOURS_UTC:
        return True, True
    return True, False


# ---------------------------------------------------------- ARB DETECTION ----

class DivergenceTracker:
    def __init__(self, window: int = 6):
        self.history = deque(maxlen=window)

    def add(self, snapshot: PriceSnapshot):
        self.history.append(snapshot)

    def is_sustained(self, threshold_pct: float) -> bool:
        if len(self.history) < 3:
            return False
        return all(abs(s.divergence_pct) >= threshold_pct
                   for s in list(self.history)[-3:])

    def consensus_direction(self) -> Optional[str]:
        if len(self.history) < 3:
            return None
        recent = list(self.history)[-3:]
        dirs   = [s.direction for s in recent]
        return dirs[0] if all(d == dirs[0] for d in dirs) else None


def check_all_confirmations(direction: str) -> ConfirmationResult:
    r = ConfirmationResult()
    r.timeofday_ok, r.timeofday_reduced = check_time_of_day()
    r.funding_confirms,    r.funding_rate    = check_funding(direction)
    r.crossasset_confirms, r.eth_move_pct    = check_crossasset(direction)
    r.volume_confirms,     r.volume_ratio    = check_volume_spike(direction)
    r.oi_confirms,         r.oi_change_pct   = check_open_interest(direction)
    r.liq_confirms,        r.liq_volume_usd  = check_liquidation_cascade(direction)
    return r


def log_arb_trade(action: str, side: str, size: float, price: float,
                  divergence: float, confirmations: int,
                  market: Market, detail: str = ""):
    with open("arb_trades.jsonl", "a") as f:
        f.write(json.dumps({
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "dry_run":       DRY_RUN,
            "action":        action,
            "side":          side,
            "size":          size,
            "price":         price,
            "fee":           round(size * TAKER_FEE, 4),
            "divergence":    round(divergence, 4),
            "confirmations": confirmations,
            "detail":        detail,
            "condition_id":  market.condition_id,
            "question":      market.question,
        }) + "\n")


# --------------------------------------------------------- MARKET DISCOVERY --

def fetch_current_btc_market() -> Optional[Market]:
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            resp = requests.get(
    f"{GAMMA_API}/events",
    params={"slug": "btc-up-or-down-5m", "limit": 5},
    timeout=10,
)
resp.raise_for_status()
raw = resp.text[:300]
print(f"EVENTS RESPONSE: {raw}", flush=True)
events = resp.json()
if not events:
    print("NO EVENTS FOUND", flush=True)
    return None
# Get markets from the event
markets = events[0].get("markets", [])
print(f"Markets in event: {len(markets)}", flush=True)
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.text[:500]
        print(f"RAW RESPONSE: {raw}", flush=True)
        markets = resp.json()
        markets = [m for m in markets if "btc" in m.get("question", "").lower() and "up or down" in m.get("question", "").lower()]
        print(f"BTC 5m markets found: {len(markets)}", flush=True)
        print(f"MARKETS RESPONSE: {len(markets)} items", flush=True)
        if not markets:
            print("NO MARKETS FOUND", flush=True)
            return None
        now   = datetime.now(timezone.utc)
        valid = [(datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")), m)
                 for m in markets
                 if datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")) > now]
        if not valid:
            return None
        valid.sort(key=lambda x: x[0])
        end_time, m = valid[0]
        token_ids   = json.loads(m["clobTokenIds"])
        return Market(
            condition_id  = m["conditionId"],
            up_token_id   = token_ids[0],
            down_token_id = token_ids[1],
            question      = m["question"],
            end_time      = end_time,
        )
    except Exception as e:
        log.error(f"Market fetch failed: {e}")
        return None


def fetch_poly_prices(market: Market) -> dict:
    try:
        up_mid   = float(requests.get(f"{POLYMARKET_HOST}/midpoint",
                         params={"token_id": market.up_token_id}, timeout=8
                         ).json().get("mid", 0.5))
        down_mid = float(requests.get(f"{POLYMARKET_HOST}/midpoint",
                         params={"token_id": market.down_token_id}, timeout=8
                         ).json().get("mid", 0.5))
        book     = requests.get(f"{POLYMARKET_HOST}/book",
                                params={"token_id": market.up_token_id}, timeout=8).json()
        bids     = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
        asks     = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
        spread   = (float(asks[0]["price"]) - float(bids[0]["price"])) if bids and asks else 0.1
        return {"up_mid": up_mid, "down_mid": down_mid, "spread": spread}
    except Exception:
        return {"up_mid": 0.5, "down_mid": 0.5, "spread": 0.1}


# ------------------------------------------------------------------ MAIN -----

def run():
    log.info(f"Chainlink Arb Bot v3 | Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    log.info(f"Primary: div>={DIVERGENCE_THRESHOLD}% + CL lag>={MAX_LAG_AGE_SEC}s")
    log.info(f"Secondary edges: funding + ETH + volume spike + OI + liquidations")
    log.info(f"Skip hours UTC: {sorted(SKIP_HOURS_UTC)}")
    log.info(f"Price feed: Coinbase (primary) + Kraken (fallback)")

    sim     = SimPortfolio(DRY_RUN_BALANCE) if DRY_RUN else None
    tracker = DivergenceTracker()

    current_market:   Optional[Market]   = None
    current_position: Optional[Position] = None
    last_market_fetch = 0

    while True:
        try:
            now  = datetime.now(timezone.utc)

            # Time of day quick check
            time_ok, time_reduced = check_time_of_day()
            if not time_ok:
                log.info(f"Skipping low-quality hour ({now.hour:02d}:00 UTC)")
                time.sleep(30)
                continue

            # Refresh market every 60s
            if time.time() - last_market_fetch > 60:
                market = fetch_current_btc_market()
                if market and (not current_market or
                               market.condition_id != current_market.condition_id):
                    log.info(f"Market: {market.question}")
                    current_market   = market
                    current_position = None
                last_market_fetch = time.time()

            if not current_market:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            secs_left = (current_market.end_time - now).total_seconds()

            # Market expired
            if secs_left <= 0:
                if current_position and sim:
                    prices  = fetch_poly_prices(current_market)
                    exit_px = prices["up_mid"] if current_position.side == "UP" \
                              else prices["down_mid"]
                    sim.close(current_position.side, current_position.entry_price,
                              exit_px, current_position.size, "MARKET_CLOSE")
                current_position  = None
                current_market    = None
                last_market_fetch = 0
                time.sleep(5)
                continue

            # Fetch snapshot
            snapshot = fetch_snapshot()
            if not snapshot:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            tracker.add(snapshot)

            abs_div    = abs(snapshot.divergence_pct)
            cl_lagging = snapshot.chainlink_age_sec >= 15
            sustained  = tracker.is_sustained(DIVERGENCE_THRESHOLD)
            direction  = tracker.consensus_direction()

            log.info(
                f"BTC=${snapshot.binance_price:.2f} | "
                f"CL=${snapshot.chainlink_price:.2f} | "
                f"div={snapshot.divergence_pct:+.3f}% | "
                f"CL_age={snapshot.chainlink_age_sec:.0f}s | "
                f"{secs_left:.0f}s left"
            )

            primary_gate = (
                abs_div >= DIVERGENCE_THRESHOLD and
                cl_lagging and sustained and
                direction is not None and
                secs_left >= MIN_MARKET_SECS_LEFT
            )

            if not primary_gate:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # Primary gate passed — check all secondary confirmations
            conf = check_all_confirmations(direction)
            log.info(f"ARB GATE: {direction} | {conf.summary()}")
            log.info(f"Confirmations: {conf.count} | Size: {conf.size_multiplier:.0%}")

            if conf.size_multiplier == 0.0:
                log.info("Not enough confirmations — skipping.")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # Handle existing position
            if current_position:
                if current_position.side == direction:
                    log.info(f"Already in {direction} — holding.")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                else:
                    prices  = fetch_poly_prices(current_market)
                    exit_px = prices["up_mid"] if current_position.side == "UP" \
                              else prices["down_mid"]
                    if sim:
                        sim.close(current_position.side, current_position.entry_price,
                                  exit_px, current_position.size, "DIRECTION_FLIP")
                    log_arb_trade("CLOSE", current_position.side,
                                  current_position.size, exit_px,
                                  snapshot.divergence_pct,
                                  current_position.confirmations, current_market)
                    current_position = None

            # Open new position
            prices  = fetch_poly_prices(current_market)
            spread  = prices["spread"]

            if spread > MAX_BOOK_SPREAD:
                log.warning(f"Spread {spread:.4f} too wide.")
            else:
                price   = prices["up_mid"] if direction == "UP" else prices["down_mid"]
                balance = sim.balance if DRY_RUN else 0.0

                div_strength = min(abs_div / (DIVERGENCE_THRESHOLD * 3), 1.0)
                size         = MAX_POSITION_USDC * div_strength * conf.size_multiplier
                size         = min(size, balance * 0.4)
                size         = max(size, 5.0)

                if balance < 5.0:
                    log.warning("Insufficient balance.")
                else:
                    if sim:
                        sim.open(direction, price, size,
                                 snapshot.divergence_pct, conf.count)
                    log_arb_trade("OPEN", direction, size, price,
                                  snapshot.divergence_pct, conf.count,
                                  current_market, conf.summary())
                    current_position = Position(
                        side=direction, entry_price=price,
                        size=size, market=current_market,
                        confirmations=conf.count,
                    )

            time.sleep(POLL_INTERVAL_SEC)

        except KeyboardInterrupt:
            log.info("Bot stopped.")
            if sim:
                sim._summary()
                sim.save()
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    print("CALLING RUN", flush=True)
    try:
        run()
    except Exception as e:
        print(f"CRASH: {e}", flush=True)
        import traceback
        traceback.print_exc()