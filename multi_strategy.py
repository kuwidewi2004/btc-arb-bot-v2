"""
Multi-Strategy Mechanical Edge Simulator v2
=============================================
Runs 6 independent mechanical edge strategies simultaneously in dry run.

Strategies:
  1. Chainlink lag arb     — Coinbase (real-time) vs Chainlink (lagging oracle)
  2. Funding rate reversion — extreme funding predicts reversion (OKX data)
  3. Liquidation cascade   — large liquidations predict direction
  4. Basis arbitrage       — futures premium/discount mean reversion (OKX data)
  5. Odds mispricing       — Polymarket odds deviating from 50/50
  6. Volume clock          — aggressive order flow before resolution

Data sources (all work on Railway US servers):
  - Spot prices:    Coinbase + Kraken fallback
  - Funding/Basis:  OKX public API (no geo-blocking)
  - Volume:         OKX klines (no geo-blocking)
  - Liquidations:   OKX liquidation feed (no geo-blocking)
  - Chainlink:      CryptoCompare + Chainlink API
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
    def __init__(self, name: str):
        self.name      = name
        self.balance   = STARTING_BALANCE
        self.start_bal = STARTING_BALANCE
        self.wins      = 0
        self.losses    = 0
        self.trades    = []
        self.logfile   = f"strategy_{name.lower().replace(' ', '_')}.jsonl"

    def open(self, side: str, price: float, size: float,
             signal_data: dict, market: Market):
        fee = size * TAKER_FEE
        self.balance -= (size + fee)
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
        supabase_insert({
            "timestamp": entry["timestamp"], "strategy": self.name,
            "action": "OPEN", "side": side, "price": price,
            "size": size, "fee": round(size * TAKER_FEE, 4),
            "balance": round(self.balance, 2),
            "condition_id": market.condition_id,
            "question": market.question,
            "signal_data": signal_data,
        })
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
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "action":       "CLOSE",
            "strategy":     self.name,
            "reason":       reason,
            "side":         side,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "size":         size,
            "pnl":          round(pnl, 4),
            "fee":          round(fee, 4),
            "balance":      round(self.balance, 2),
            "condition_id": market.condition_id,
        }
        self.trades.append(entry)
        with open(self.logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")
        supabase_insert({
            "timestamp": entry["timestamp"], "strategy": self.name,
            "action": "CLOSE", "side": side, "price": exit_price,
            "size": size, "pnl": round(pnl, 4), "fee": round(fee, 4),
            "balance": round(self.balance, 2),
            "condition_id": market.condition_id,
            "signal_data": {"reason": reason, "entry_price": entry_price},
        })
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

_price_cache:    dict  = {"btc": 0.0, "eth": 0.0, "fetched_at": 0.0}
_funding_cache:  dict  = {"rate": 0.0, "fetched_at": 0.0}
_basis_cache:    dict  = {"spot": 0.0, "futures": 0.0, "fetched_at": 0.0}
_liq_cache:      dict  = {"long": 0.0, "short": 0.0, "fetched_at": 0.0}
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
        if btc: _price_cache["btc"] = btc
        if eth: _price_cache["eth"] = eth
        _price_cache["fetched_at"] = now

    # Funding rate via OKX public API
    if now - _funding_cache["fetched_at"] >= 60:
        try:
            r = requests.get(
                f"{OKX_BASE}/api/v5/public/funding-rate",
                params={"instId": OKX_BTC},
                timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [{}])[0]
            rate = float(data.get("fundingRate", 0))
            _funding_cache["rate"]       = rate
            _funding_cache["fetched_at"] = now
            log.info(f"Funding rate (OKX): {rate:+.6f}")
        except Exception as e:
            log.warning(f"Funding fetch failed: {e}")

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
                if o.get("side") == "buy":  short_liqs += usd  # short being liquidated
                else:                       long_liqs  += usd  # long being liquidated
            _liq_cache["long"]       = long_liqs
            _liq_cache["short"]      = short_liqs
            _liq_cache["fetched_at"] = now
        except Exception as e:
            log.warning(f"Liquidation fetch failed: {e}")

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
                # OKX format: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
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


def fetch_chainlink_price() -> tuple:
    """Returns (price, age_seconds)."""
    try:
        r = requests.get("https://min-api.cryptocompare.com/data/price",
                         params={"fsym": "BTC", "tsyms": "USD"}, timeout=5)
        r.raise_for_status()
        price   = float(r.json()["USD"])
        age_sec = 30.0
        try:
            cl = requests.get("https://data.chain.link/api/proxy/btc-usd/latest",
                              timeout=5)
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

def strategy_chainlink_arb(market, secs_left, tracker, position, cl_history):
    """
    Strategy 1: Chainlink lag arbitrage.
    Coinbase price = real-time truth.
    Chainlink price = lagging oracle Polymarket uses to resolve.
    When they diverge 0.15%+, bet in Chainlink's catch-up direction.
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
        if not all(abs(s["div"]) >= 0.15 for s in recent):
            log.info(f"[Chainlink] div too small: {[round(s['div'],4) for s in recent]}")
            return position
        if not all(s["dir"] == direction for s in recent):
            log.info(f"[Chainlink] direction inconsistent")
            return position
        if cl_age < 15:
            log.info(f"[Chainlink] CL too fresh: {cl_age:.0f}s")
            return position
        if secs_left < 90:
            log.info(f"[Chainlink] market closing soon: {secs_left:.0f}s")
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
            return StrategyTrade("chainlink_arb", direction, size, price, market)
    except Exception as e:
        log.warning(f"[Chainlink arb] error: {e}")
    return position


def strategy_funding_reversion(market, secs_left, tracker, position):
    """
    Strategy 2: Funding rate reversion.
    Extreme positive funding = longs overextended = bet DOWN.
    Extreme negative funding = shorts overextended = bet UP.
    Data from OKX public API.
    """
    try:
        rate = _funding_cache["rate"]
        if abs(rate) < 0.0005:
            log.info(f"[Funding] rate not extreme: {rate:+.6f} (need ±0.0005)")
            return position
        if secs_left < 60 or position:
            return position

        direction = "DOWN" if rate > 0.0005 else "UP"
        prices    = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            return position

        price     = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        intensity = min(abs(rate) / 0.001, 1.0)
        size      = min(MAX_BET * intensity, tracker.balance * 0.3)
        size      = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"funding_rate": round(rate, 6), "intensity": round(intensity, 3)},
                        market)
            return StrategyTrade("funding_reversion", direction, size, price, market)
    except Exception as e:
        log.warning(f"[Funding reversion] error: {e}")
    return position


def strategy_liquidation_cascade(market, secs_left, tracker, position):
    """
    Strategy 3: Liquidation cascade.
    Large long liquidations = forced selling = bet DOWN.
    Large short liquidations = short squeeze = bet UP.
    Data from OKX public API.
    """
    try:
        long_liqs  = _liq_cache["long"]
        short_liqs = _liq_cache["short"]
        total      = long_liqs + short_liqs

        if total < 300_000 or secs_left < 60 or position:
            return position

        direction = "DOWN" if long_liqs > short_liqs else "UP"
        intensity = min(total / 2_000_000, 1.0)
        prices    = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            return position

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * intensity, tracker.balance * 0.3)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"long_liqs": round(long_liqs), "short_liqs": round(short_liqs)},
                        market)
            return StrategyTrade("liquidation_cascade", direction, size, price, market)
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
        if spot == 0 or futures == 0 or secs_left < 60 or position:
            return position

        basis_pct = (futures - spot) / spot * 100
        if abs(basis_pct) < 0.05:
            log.info(f"[Basis] too small: {basis_pct:+.4f}% (need ±0.05%)")
            return position

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
                        {"basis_pct": round(basis_pct, 4),
                         "spot": round(spot, 2), "futures": round(futures, 2)},
                        market)
            return StrategyTrade("basis_arb", direction, size, price, market)
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
        if secs_left < 120 or secs_left > 270 or position:
            return position

        prices    = fetch_poly_prices(market)
        up_mid    = prices["up_mid"]
        deviation = up_mid - 0.50

        if prices["spread"] > 0.04:
            log.info(f"[Odds] spread too wide: {prices['spread']:.4f}")
            return position
        if abs(deviation) < 0.04:
            log.info(f"[Odds] deviation too small: {deviation:+.4f} (need ±0.04)")
            return position

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
            return StrategyTrade("odds_mispricing", direction, size, price, market)
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
        if secs_left > 90 or secs_left < 20 or position:
            return position
        if len(_volume_history) < 5:
            return position

        candles   = list(_volume_history)
        recent    = candles[-3:]
        total_vol = sum(c["volume"] for c in recent)
        buy_vol   = sum(c["buy_vol"] for c in recent)

        if total_vol == 0:
            return position

        buy_ratio = buy_vol / total_vol

        if buy_ratio > 0.58:
            direction = "UP"
            intensity = min((buy_ratio - 0.58) / 0.20, 1.0)
        elif buy_ratio < 0.42:
            direction = "DOWN"
            intensity = min((0.42 - buy_ratio) / 0.20, 1.0)
        else:
            return position

        prices = fetch_poly_prices(market)
        if prices["spread"] > 0.06:
            return position

        price = prices["up_mid"] if direction == "UP" else prices["down_mid"]
        size  = min(MAX_BET * intensity, tracker.balance * 0.3)
        size  = max(size, MIN_BET)

        if tracker.balance >= MIN_BET:
            tracker.open(direction, price, size,
                        {"buy_ratio": round(buy_ratio, 4), "secs_left": round(secs_left)},
                        market)
            return StrategyTrade("volume_clock", direction, size, price, market)
    except Exception as e:
        log.warning(f"[Volume clock] error: {e}")
    return position


# ------------------------------------------------------------------ MAIN -----

def run():
    log.info("Multi-Strategy Mechanical Edge Simulator v2")
    log.info("Strategies: Chainlink | Funding | Liquidation | Basis | Odds | Volume")
    log.info("Data: Coinbase + Kraken + OKX (all geo-unblocked)")

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

            # Market expired — close all positions
            if secs_left <= 0:
                prices = fetch_poly_prices(current_market)
                for key, pos in positions.items():
                    if pos:
                        exit_px = prices["up_mid"] if pos.side == "UP" else prices["down_mid"]
                        trackers[key].close(pos.side, pos.entry_price, exit_px,
                                            pos.size, current_market, "MARKET_CLOSE")
                positions         = {k: None for k in trackers}
                current_market    = None
                last_market_fetch = 0
                time.sleep(5)
                continue

            spot    = _price_cache["btc"]
            futures = _basis_cache["futures"]
            basis   = (futures - spot) / spot * 100 if spot > 0 and futures > 0 else 0.0

            log.info(f"BTC=${spot:.2f} | {secs_left:.0f}s left | "
                     f"funding={_funding_cache['rate']:+.6f} | basis={basis:+.3f}%")

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


if __name__ == "__main__":
    print("MULTI-STRATEGY BOT STARTING", flush=True)
    run()
