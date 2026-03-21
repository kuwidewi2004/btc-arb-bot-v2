"""
Multi-Strategy Mechanical Edge Simulator
==========================================
Runs 6 independent mechanical edge strategies simultaneously in dry run.
Each strategy logs its own trades independently so you can compare them.

Strategies:
  1. Chainlink lag arb          — Binance vs Chainlink price divergence
  2. Funding rate reversion     — extreme funding predicts reversion
  3. Liquidation cascade        — large liquidations predict direction
  4. Basis arbitrage            — futures premium/discount mean reversion
  5. Odds mispricing            — Polymarket odds deviating from 50/50
  6. Volume clock               — aggressive order flow before resolution

Each strategy:
  - Tracks its own simulated balance starting at $100
  - Logs every trade to its own .jsonl file
  - Prints its own win rate summary

Run alongside resolver.py to track actual outcomes.

Usage:
    python multi_strategy.py
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

TAKER_FEE        = 0.0025
STARTING_BALANCE = 100.0
MAX_BET          = 50.0
MIN_BET          = 5.0
POLL_SEC         = 5

# Skip low quality hours
SKIP_HOURS_UTC   = {0, 1, 2, 3, 4, 5}


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
    signal_data: dict = field(default_factory=dict)
    entry_time:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------- STRATEGY TRACKER --

class StrategyTracker:
    """Tracks performance of a single strategy independently."""

    def __init__(self, name: str):
        self.name     = name
        self.balance  = STARTING_BALANCE
        self.start_bal = STARTING_BALANCE
        self.wins     = 0
        self.losses   = 0
        self.trades   = []
        self.logfile  = f"strategy_{name.lower().replace(' ', '_')}.jsonl"

    def open(self, side: str, price: float, size: float, signal_data: dict, market: Market):
        fee = size * TAKER_FEE
        self.balance -= (size + fee)
        entry = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "action":      "OPEN",
            "strategy":    self.name,
            "side":        side,
            "price":       price,
            "size":        size,
            "fee":         round(fee, 4),
            "balance":     round(self.balance, 2),
            "condition_id": market.condition_id,
            "question":    market.question,
            **signal_data,
        }
        self.trades.append(entry)
        with open(self.logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")
        log.info(f"[{self.name}] OPEN {side} | size={size:.2f} @ {price:.4f} | "
                 f"balance=${self.balance:.2f} | {json.dumps(signal_data)}")

    def close(self, side: str, entry_price: float, exit_price: float,
              size: float, market: Market, reason: str = "RESOLVED"):
        if side == "UP":
            pnl = (exit_price - entry_price) * size / entry_price
        else:
            pnl = (entry_price - exit_price) * size / entry_price
        fee = size * TAKER_FEE
        pnl -= fee
        self.balance += size + pnl
        if pnl > 0: self.wins   += 1
        else:       self.losses += 1

        entry = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "action":      "CLOSE",
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

        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total else 0
        log.info(f"[{self.name}] CLOSE {side} ({reason}) | PnL={pnl:+.3f} | "
                 f"balance=${self.balance:.2f} | WR={wr:.0f}% ({self.wins}W/{self.losses}L)")

    def summary(self) -> str:
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total else 0
        pnl   = self.balance - self.start_bal
        return (f"{self.name}: ${self.balance:.2f} | PnL={pnl:+.2f} | "
                f"WR={wr:.0f}% | {total} trades")


# --------------------------------------------------------- SHARED DATA -------

_price_cache:   dict  = {"btc": 0.0, "eth": 0.0, "fetched_at": 0.0}
_funding_cache: dict  = {"rate": 0.0, "fetched_at": 0.0}
_basis_cache:   dict  = {"spot": 0.0, "futures": 0.0, "fetched_at": 0.0}
_oi_cache:      dict  = {"oi": 0.0, "prev_oi": 0.0, "fetched_at": 0.0}
_liq_cache:     dict  = {"long": 0.0, "short": 0.0, "fetched_at": 0.0}
_volume_history: deque = deque(maxlen=25)
_volume_fetched: float = 0.0


def fetch_spot(symbol: str = BTC_SYMBOL) -> Optional[float]:
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
    """Refresh all shared data caches."""
    now = time.time()

    # Spot prices
    if now - _price_cache["fetched_at"] >= POLL_SEC:
        btc = fetch_spot(BTC_SYMBOL)
        eth = fetch_spot(ETH_SYMBOL)
        if btc: _price_cache["btc"] = btc
        if eth: _price_cache["eth"] = eth
        _price_cache["fetched_at"] = now

    # Funding rate
    if now - _funding_cache["fetched_at"] >= 60:
        try:
            r = requests.get(f"{BINANCE_FUTURES}/fapi/v1/premiumIndex",
                             params={"symbol": BTC_SYMBOL}, timeout=5)
            r.raise_for_status()
            _funding_cache["rate"]       = float(r.json()["lastFundingRate"])
            print(f"FUNDING RATE: {_funding_cache['rate']}", flush=True)
            _funding_cache["fetched_at"] = now
        except Exception:
            pass

    # Basis (spot vs futures)
    if now - _basis_cache["fetched_at"] >= 10:
        try:
            r = requests.get(f"{BINANCE_FUTURES}/fapi/v1/ticker/price",
                             params={"symbol": BTC_SYMBOL}, timeout=5)
            r.raise_for_status()
            _basis_cache["futures"]    = float(r.json()["price"])
            _basis_cache["spot"]       = _price_cache["btc"]
            _basis_cache["fetched_at"] = now
        except Exception:
            pass

    # Open interest
    if now - _oi_cache["fetched_at"] >= 30:
        try:
            r = requests.get(f"{BINANCE_FUTURES}/fapi/v1/openInterest",
                             params={"symbol": BTC_SYMBOL}, timeout=5)
            r.raise_for_status()
            _oi_cache["prev_oi"]    = _oi_cache["oi"]
            _oi_cache["oi"]         = float(r.json()["openInterest"])
            _oi_cache["fetched_at"] = now
        except Exception:
            pass

    # Liquidations
    if now - _liq_cache["fetched_at"] >= 60:
        try:
            r = requests.get(f"{BINANCE_FUTURES}/fapi/v1/allForceOrders",
                             params={"symbol": BTC_SYMBOL, "limit": 100}, timeout=5)
            r.raise_for_status()
            orders    = r.json()
            cutoff_ms = (now - 300) * 1000
            long_liqs = short_liqs = 0.0
            for o in orders:
                if float(o.get("time", 0)) < cutoff_ms:
                    continue
                usd = float(o.get("origQty", 0)) * float(o.get("price", 0))
                if o.get("side") == "SELL": long_liqs  += usd
                else:                       short_liqs += usd
            _liq_cache["long"]       = long_liqs
            _liq_cache["short"]      = short_liqs
            _liq_cache["fetched_at"] = now
        except Exception:
            pass

    # Volume
    global _volume_fetched
    if now - _volume_fetched >= 60:
        try:
            r = requests.get(f"{BINANCE_FUTURES}/fapi/v1/klines",
                             params={"symbol": BTC_SYMBOL, "interval": "1m", "limit": 22},
                             timeout=8)
            r.raise_for_status()
            _volume_history.clear()
            for c in r.json():
                _volume_history.append({
                    "volume":    float(c[5]),
                    "buy_vol":   float(c[9]),
                    "close":     float(c[4]),
                    "open":      float(c[1]),
                })
            _volume_fetched = now
        except Exception:
            pass


# --------------------------------------------------------- MARKET DISCOVERY --

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


# --------------------------------------------------------- STRATEGIES --------

def strategy_chainlink_arb(market: Market, secs_left: float,
                            tracker: StrategyTracker,
                            position: Optional[StrategyTrade],
                            cl_history: deque) -> Optional[StrategyTrade]:
    """Strategy 1: Chainlink lag arbitrage."""
    try:
        cl_price, cl_age = fetch_chainlink_price()
        btc_price        = _price_cache["btc"]
        if not cl_price or not btc_price:
            return position

        divergence = (btc_price - cl_price) / cl_price * 100
        direction  = "UP" if divergence > 0 else "DOWN"
        cl_history.append({"div": divergence, "dir": direction})

        # Require 3 sustained snapshots
        if len(cl_history) < 3:
            return position
        recent = list(cl_history)[-3:]
        if not all(abs(s["div"]) >= 0.15 for s in recent):
            return position
        if not all(s["dir"] == direction for s in recent):
            return position
        if cl_age < 15 or secs_left < 90:
            return position

        if position:
            return position

        prices = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            return position

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * min(abs(divergence) / 0.45, 1.0), tracker.balance * 0.4)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"divergence": round(divergence, 4), "cl_age": round(cl_age, 1)},
                        market)
            return StrategyTrade("chainlink_arb", direction, size, price, market,
                               {"divergence": divergence})
    except Exception as e:
        log.warning(f"[Chainlink arb] error: {e}")
    return position


def fetch_chainlink_price() -> tuple:
    try:
        r = requests.get("https://min-api.cryptocompare.com/data/price",
                        params={"fsym": "BTC", "tsyms": "USD"}, timeout=5)
        r.raise_for_status()
        price   = float(r.json()["USD"])
        age_sec = 30.0
        try:
            cl = requests.get("https://data.chain.link/api/proxy/btc-usd/latest", timeout=5)
            if cl.status_code == 200:
                d       = cl.json()
                updated = d.get("updatedAt") or d.get("timestamp")
                if updated:
                    dt      = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                    age_sec = (datetime.now(timezone.utc) - dt).total_seconds()
                    price   = float(d.get("answer") or d.get("price") or price)
        except Exception:
            pass
        return price, age_sec
    except Exception:
        return None, None


def strategy_funding_reversion(market: Market, secs_left: float,
                                tracker: StrategyTracker,
                                position: Optional[StrategyTrade]) -> Optional[StrategyTrade]:
    """
    Strategy 2: Funding rate reversion.
    Extreme positive funding = overextended longs = bet DOWN
    Extreme negative funding = overextended shorts = bet UP
    Threshold: >0.05% or <-0.05% (10x normal)
    """
    try:
        rate = _funding_cache["rate"]
        if abs(rate) < 0.0005:   # not extreme enough
            return position
        if secs_left < 60:
            return position
        if position:
            return position

        direction = "DOWN" if rate > 0.0005 else "UP"
        prices    = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            return position

        price     = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        # Size proportional to how extreme the funding is
        intensity = min(abs(rate) / 0.001, 1.0)
        size      = min(MAX_BET * intensity, tracker.balance * 0.3)
        size      = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"funding_rate": round(rate, 6), "intensity": round(intensity, 3)},
                        market)
            return StrategyTrade("funding_reversion", direction, size, price, market,
                               {"funding_rate": rate})
    except Exception as e:
        log.warning(f"[Funding reversion] error: {e}")
    return position


def strategy_liquidation_cascade(market: Market, secs_left: float,
                                  tracker: StrategyTracker,
                                  position: Optional[StrategyTrade]) -> Optional[StrategyTrade]:
    """
    Strategy 3: Liquidation cascade.
    Large liquidations predict continuation in cascade direction.
    Long liquidations (SELL side) = bearish cascade = bet DOWN
    Short liquidations (BUY side) = bullish cascade = bet UP
    """
    try:
        long_liqs  = _liq_cache["long"]
        short_liqs = _liq_cache["short"]
        total      = long_liqs + short_liqs

        if total < 300_000:   # need at least $300k in liquidations
            return position
        if secs_left < 60:
            return position
        if position:
            return position

        direction = "DOWN" if long_liqs > short_liqs else "UP"
        intensity = min(total / 2_000_000, 1.0)   # scale up to $2M
        prices    = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            return position

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * intensity, tracker.balance * 0.3)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"long_liqs": round(long_liqs), "short_liqs": round(short_liqs),
                         "total_usd": round(total)},
                        market)
            return StrategyTrade("liquidation_cascade", direction, size, price, market,
                               {"total_liqs": total})
    except Exception as e:
        log.warning(f"[Liquidation cascade] error: {e}")
    return position


def strategy_basis_arb(market: Market, secs_left: float,
                        tracker: StrategyTracker,
                        position: Optional[StrategyTrade]) -> Optional[StrategyTrade]:
    """
    Strategy 4: Basis arbitrage.
    Futures trading at large premium to spot = bet DOWN (futures will revert)
    Futures trading at large discount to spot = bet UP (futures will revert)
    Threshold: >0.1% premium/discount
    """
    try:
        spot    = _basis_cache["spot"]
        futures = _basis_cache["futures"]

        if spot == 0 or futures == 0:
            return position

        basis_pct = (futures - spot) / spot * 100

        if abs(basis_pct) < 0.10:   # not enough basis
            return position
        if secs_left < 60:
            return position
        if position:
            return position

        # Futures at premium = spot will catch up UP or futures revert DOWN
        # For 5m window: bet that price reverts toward spot
        direction = "DOWN" if basis_pct > 0 else "UP"
        intensity = min(abs(basis_pct) / 0.3, 1.0)
        prices    = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            return position

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * intensity, tracker.balance * 0.3)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"basis_pct": round(basis_pct, 4), "spot": round(spot, 2),
                         "futures": round(futures, 2)},
                        market)
            return StrategyTrade("basis_arb", direction, size, price, market,
                               {"basis_pct": basis_pct})
    except Exception as e:
        log.warning(f"[Basis arb] error: {e}")
    return position


def strategy_odds_mispricing(market: Market, secs_left: float,
                              tracker: StrategyTracker,
                              position: Optional[StrategyTrade]) -> Optional[StrategyTrade]:
    """
    Strategy 5: Polymarket odds mispricing.
    When UP odds deviate significantly from 0.50 early in the market,
    they tend to revert toward 0.50 — bet against the extreme.
    Only enter in first 2 minutes of market (180-300 seconds left)
    """
    try:
        # Only trade early in the market window
        if secs_left < 120 or secs_left > 270:
            return position
        if position:
            return position

        prices  = fetch_poly_prices(market)
        up_mid  = prices["up_mid"]
        spread  = prices["spread"]

        if spread > 0.04:
            return position

        deviation = up_mid - 0.50

        if abs(deviation) < 0.08:   # need at least 8% deviation
            return position

        # Bet against the extreme — if UP is at 0.62, bet DOWN (reversion to 0.50)
        direction = "DOWN" if deviation > 0 else "UP"
        intensity = min(abs(deviation) / 0.15, 1.0)
        price     = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size      = min(MAX_BET * intensity, tracker.balance * 0.3)
        size      = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"up_mid": round(up_mid, 4), "deviation": round(deviation, 4),
                         "secs_left": round(secs_left)},
                        market)
            return StrategyTrade("odds_mispricing", direction, size, price, market,
                               {"deviation": deviation})
    except Exception as e:
        log.warning(f"[Odds mispricing] error: {e}")
    return position


def strategy_volume_clock(market: Market, secs_left: float,
                           tracker: StrategyTracker,
                           position: Optional[StrategyTrade]) -> Optional[StrategyTrade]:
    """
    Strategy 6: Volume clock.
    In the last 90 seconds before market closes, aggressive order flow
    on Binance futures predicts the resolution direction.
    Only enter 60-90 seconds before resolution.
    """
    try:
        if secs_left > 90 or secs_left < 20:
            return position
        if position:
            return position
        if len(_volume_history) < 5:
            return position

        candles   = list(_volume_history)
        recent    = candles[-3:]   # last 3 minutes
        total_vol = sum(c["volume"] for c in recent)
        buy_vol   = sum(c["buy_vol"] for c in recent)

        if total_vol == 0:
            return position

        buy_ratio = buy_vol / total_vol
        # Strong buying = UP, strong selling = DOWN
        if buy_ratio > 0.60:
            direction = "UP"
            intensity = (buy_ratio - 0.60) / 0.20
        elif buy_ratio < 0.40:
            direction = "DOWN"
            intensity = (0.40 - buy_ratio) / 0.20
        else:
            return position   # no clear signal

        intensity = min(intensity, 1.0)
        prices    = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            return position

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * intensity, tracker.balance * 0.3)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"buy_ratio": round(buy_ratio, 4),
                         "secs_left": round(secs_left),
                         "intensity": round(intensity, 3)},
                        market)
            return StrategyTrade("volume_clock", direction, size, price, market,
                               {"buy_ratio": buy_ratio})
    except Exception as e:
        log.warning(f"[Volume clock] error: {e}")
    return position


# ------------------------------------------------------------------ MAIN -----

def run():
    log.info("Multi-Strategy Mechanical Edge Simulator")
    log.info("Strategies: Chainlink arb | Funding reversion | Liquidation cascade")
    log.info("            Basis arb | Odds mispricing | Volume clock")

    # Initialize trackers
    trackers = {
        "chainlink":    StrategyTracker("Chainlink Arb"),
        "funding":      StrategyTracker("Funding Reversion"),
        "liquidation":  StrategyTracker("Liquidation Cascade"),
        "basis":        StrategyTracker("Basis Arb"),
        "odds":         StrategyTracker("Odds Mispricing"),
        "volume":       StrategyTracker("Volume Clock"),
    }

    # Current positions per strategy
    positions = {k: None for k in trackers}

    # Chainlink history for arb strategy
    cl_history = deque(maxlen=6)

    current_market    = None
    last_market_fetch = 0
    last_summary      = 0

    while True:
        try:
            now  = datetime.now(timezone.utc)
            hour = now.hour

            # Skip low quality hours
            if hour in SKIP_HOURS_UTC:
                log.info(f"Skipping hour {hour:02d}:00 UTC")
                time.sleep(30)
                continue

            # Refresh shared data
            refresh_shared_data()

            # Refresh market
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
                log.warning("No market found — retrying in 15s")
                time.sleep(15)
                continue

            secs_left = (current_market.end_time - now).total_seconds()

            # Market expired — close all positions
            if secs_left <= 0:
                prices = fetch_poly_prices(current_market)
                for key, pos in positions.items():
                    if pos:
                        exit_px = prices["up_mid"] if pos.side == "UP" else prices["down_mid"]
                        trackers[key].close(pos.side, pos.entry_price, exit_px,
                                           pos.size, current_market, "MARKET_CLOSE")
                positions      = {k: None for k in trackers}
                current_market = None
                last_market_fetch = 0
                time.sleep(5)
                continue

            log.info(f"BTC=${_price_cache['btc']:.2f} | {secs_left:.0f}s left | "
                     f"funding={_funding_cache['rate']:+.5f} | "
                     f"basis={(_basis_cache['futures']-_basis_cache['spot'])/_max(_basis_cache['spot'],1)*100:+.3f}%")

            # Run all strategies
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
                log.info("STRATEGY PERFORMANCE SUMMARY")
                for t in trackers.values():
                    log.info(t.summary())
                log.info("=" * 60)
                last_summary = time.time()

            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            log.info("Stopped.")
            log.info("FINAL SUMMARY:")
            for t in trackers.values():
                log.info(t.summary())
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            time.sleep(10)


def _max(a, b):
    return a if a > b else b


if __name__ == "__main__":
    print("MULTI-STRATEGY BOT STARTING", flush=True)
    run()
