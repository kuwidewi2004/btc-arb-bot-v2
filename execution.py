"""
dYdX v4 BTC Perpetual Futures Execution Layer
===============================================
Places orders on dYdX Chain (Cosmos-based) BTC-USD perpetual.

Authentication: 24-word mnemonic (Cosmos wallet)
Min order: 0.0001 BTC (~$6-7)
Fees: 0.05% taker, 0.02% maker

Environment variables:
  DYDX_MNEMONIC       — 24-word seed phrase
  LIVE_TRADING         — "true" to enable real orders
  TRADE_SIZE           — USD per trade (default 10.0 — min ~$7 for 0.0001 BTC)
  MAX_DAILY_LOSS       — stop trading after this much loss (default 10.0)
  MAX_CONSECUTIVE_LOSSES — stop after N losses in a row (default 5)
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime, timezone

log = logging.getLogger(__name__)

LIVE_TRADING   = os.environ.get("LIVE_TRADING", "false").lower() == "true"
TRADE_SIZE     = float(os.environ.get("TRADE_SIZE", "10.0"))
MAX_DAILY_LOSS = float(os.environ.get("MAX_DAILY_LOSS", "10.0"))
MAX_CONSECUTIVE_LOSSES = int(os.environ.get("MAX_CONSECUTIVE_LOSSES", "15"))


class DydxExecutor:

    def __init__(self):
        self.mnemonic   = os.environ.get("DYDX_MNEMONIC", "")
        self.client     = None
        self.indexer    = None
        self.wallet     = None
        self.address    = None
        self.market     = None
        self.enabled    = False
        self.trade_log  = []
        self.position   = None
        self.daily_pnl  = 0.0
        self.consecutive_losses = 0
        self.circuit_broken = False

        if self.mnemonic and LIVE_TRADING:
            try:
                self._init_async()
                self.enabled = True
                log.info(f"[Executor] dYdX LIVE trading enabled (${TRADE_SIZE}/trade)")
            except Exception as e:
                log.warning(f"[Executor] dYdX init failed: {e}")
                log.info("[Executor] Running in PAPER mode")
        else:
            if not self.mnemonic:
                log.warning("[Executor] DYDX_MNEMONIC not set — paper trading only")
            else:
                log.info("[Executor] dYdX connected — PAPER mode (LIVE_TRADING=false)")

    def _init_async(self):
        """Initialize async dYdX clients synchronously for startup."""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._async_init())
        loop.close()

    async def _async_init(self):
        from dydx_v4_client.key_pair import KeyPair
        from dydx_v4_client.wallet import Wallet
        from dydx_v4_client.node.client import NodeClient
        from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
        from dydx_v4_client.node.market import Market
        from dydx_v4_client.network import make_mainnet

        # make_mainnet returns a partial — call it with the endpoints
        network = make_mainnet(
            node_url="dydx-grpc.publicnode.com:443",
            rest_indexer="https://indexer.dydx.trade/v4",
            websocket_indexer="wss://indexer.dydx.trade/v4/ws",
        )
        node_config = network.node
        log.info(f"[Executor] Step 1: Connecting to node...")
        self.node = await NodeClient.connect(node_config)
        log.info(f"[Executor] Step 2: Node connected")

        rest_url = network.rest_indexer
        if callable(rest_url):
            rest_url = rest_url()
        self.indexer = IndexerClient(rest_url)
        log.info(f"[Executor] Step 3: Indexer connected")

        log.info(f"[Executor] Step 3b: KeyPair type={type(KeyPair)}, from_mnemonic type={type(KeyPair.from_mnemonic)}")
        log.info(f"[Executor] Step 3c: Mnemonic words={len(self.mnemonic.split())}")
        kp = KeyPair.from_mnemonic(self.mnemonic)
        log.info(f"[Executor] Step 3d: KeyPair created, type={type(kp)}")
        log.info(f"[Executor] Step 3e: Wallet class={Wallet}, address method={type(getattr(Wallet, 'address', None))}")
        w = Wallet(kp, 0, 0)
        log.info(f"[Executor] Step 3f: Wallet instance created, type={type(w)}")
        log.info(f"[Executor] Step 3g: address attr type={type(w.address)}")
        self.address = w.address if isinstance(w.address, str) else w.address()
        log.info(f"[Executor] Step 4: Address derived: {self.address}")

        self.wallet = await Wallet.from_mnemonic(self.node, self.mnemonic, self.address)
        log.info(f"[Executor] Step 5: Wallet created")

        markets = await self.indexer.markets.get_perpetual_markets("BTC-USD")
        self.market = Market(markets["markets"]["BTC-USD"])
        log.info(f"[Executor] Step 6: Market loaded — dYdX ready!")

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def get_price(self) -> float:
        """Get BTC-USD mark price from indexer."""
        try:
            import requests
            r = requests.get("https://indexer.dydx.trade/v4/perpetualMarkets?ticker=BTC-USD", timeout=5)
            data = r.json()
            return float(data["markets"]["BTC-USD"]["oraclePrice"])
        except Exception:
            return 0.0

    def open_position(self, side: str, size_usd: float,
                      score: float = 0.0, reason: str = "") -> dict:
        """
        Open a BTC-USD perpetual position.
        side: "LONG" or "SHORT"
        size_usd: position size in USD
        """
        if self.position is not None:
            log.warning("[Executor] Already have open position — skipping")
            return {}

        if self.circuit_broken:
            log.warning(f"[Executor] CIRCUIT BROKEN — daily=${self.daily_pnl:+.2f} "
                        f"streak={self.consecutive_losses}")
            return {}

        decision_ts = time.time()
        price = self.get_price()
        if price <= 0:
            log.warning("[Executor] Can't get price — skipping")
            return {}

        # Calculate BTC quantity (min 0.0001)
        quantity = round(size_usd / price, 4)
        if quantity < 0.0001:
            quantity = 0.0001

        record = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "action":         "OPEN",
            "side":           side,
            "size_usd":       round(size_usd, 2),
            "quantity_btc":   quantity,
            "expected_price": round(price, 2),
            "score":          round(score, 4),
            "reason":         reason,
            "decision_ts":    decision_ts,
        }

        if not self.enabled:
            record["actual_price"]    = round(price, 2)
            record["slippage"]        = 0.0
            record["fill_latency_ms"] = 0.0
            record["order_id"]        = "PAPER"
            record["status"]          = "PAPER_FILL"
            log.info(f"[Executor] PAPER {side} {quantity} BTC @ ${price:.2f} "
                     f"(${size_usd:.2f}) | score={score:.3f}")
        else:
            try:
                submit_ts = time.time()
                tx = self._run_async(self._place_order(side, quantity, price))
                fill_ts = time.time()

                record["actual_price"]    = round(price, 2)
                record["slippage"]        = 0.0
                record["fill_latency_ms"] = round((fill_ts - submit_ts) * 1000, 1)
                record["order_id"]        = str(tx) if tx else "unknown"
                record["status"]          = "SUBMITTED"

                log.info(f"[Executor] OPEN {side} {quantity} BTC @ ${price:.2f} "
                         f"(${size_usd:.2f}) | latency={record['fill_latency_ms']:.0f}ms")
            except Exception as e:
                record["status"] = f"FAILED: {e}"
                record["actual_price"] = None
                log.error(f"[Executor] Order FAILED: {e}")
                self._log_trade(record)
                return {}

        self.position = record
        self._log_trade(record)
        return record

    async def _place_order(self, side: str, quantity: float, price: float):
        """Async order placement on dYdX chain."""
        import random
        from dydx_v4_client import MAX_CLIENT_ID, OrderFlags
        from v4_proto.dydxprotocol.clob.order_pb2 import Order
        from dydx_v4_client.indexer.rest.constants import OrderType

        order_side = Order.Side.SIDE_BUY if side == "LONG" else Order.Side.SIDE_SELL

        # Safety price bound: 5% above/below oracle for market orders
        if side == "LONG":
            limit_price = price * 1.05
        else:
            limit_price = price * 0.95

        order_id = self.market.order_id(
            self.address, 0, random.randint(0, MAX_CLIENT_ID), OrderFlags.SHORT_TERM
        )
        current_block = await self.node.latest_block_height()

        new_order = self.market.order(
            order_id=order_id,
            order_type=OrderType.MARKET,
            side=order_side,
            size=quantity,
            price=limit_price,
            time_in_force=Order.TimeInForce.TIME_IN_FORCE_UNSPECIFIED,
            reduce_only=False,
            good_til_block=current_block + 10,
        )

        tx = await self.node.place_order(wallet=self.wallet, order=new_order)
        self.wallet.sequence += 1
        return tx

    def close_position(self) -> dict:
        """Close current position with opposing market order."""
        if self.position is None:
            return {}

        decision_ts = time.time()
        open_rec = self.position
        close_side = "SHORT" if open_rec["side"] == "LONG" else "LONG"
        price = self.get_price()
        quantity = open_rec["quantity_btc"]

        record = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "action":         "CLOSE",
            "side":           open_rec["side"],
            "quantity_btc":   quantity,
            "expected_price": round(price, 2),
            "open_price":     open_rec.get("actual_price", 0),
            "decision_ts":    decision_ts,
        }

        if not self.enabled:
            record["actual_price"]    = round(price, 2)
            record["slippage"]        = 0.0
            record["fill_latency_ms"] = 0.0
            record["order_id"]        = "PAPER"
            record["status"]          = "PAPER_FILL"
        else:
            try:
                submit_ts = time.time()
                tx = self._run_async(self._close_order(close_side, quantity, price))
                fill_ts = time.time()

                record["actual_price"]    = round(price, 2)
                record["slippage"]        = 0.0
                record["fill_latency_ms"] = round((fill_ts - submit_ts) * 1000, 1)
                record["order_id"]        = str(tx) if tx else "unknown"
                record["status"]          = "CLOSED"
            except Exception as e:
                record["status"] = f"CLOSE FAILED: {e}"
                record["actual_price"] = price
                log.error(f"[Executor] Close FAILED: {e}")

        # PnL calculation
        open_price  = open_rec.get("actual_price", 0) or 0
        close_price = record.get("actual_price", 0) or 0
        if open_price > 0 and close_price > 0:
            if open_rec["side"] == "LONG":
                pnl_pct = (close_price - open_price) / open_price * 100
            else:
                pnl_pct = (open_price - close_price) / open_price * 100
            pnl_usd = pnl_pct / 100 * open_rec["size_usd"]
            fee_usd = open_rec["size_usd"] * 0.001  # dYdX: 0.05% taker × 2 sides = 0.10% round trip
            net_pnl = pnl_usd - fee_usd
        else:
            pnl_pct = net_pnl = fee_usd = 0.0

        record["pnl_pct"]  = round(pnl_pct, 4)
        record["pnl_usd"]  = round(net_pnl, 4)
        record["fee_usd"]  = round(fee_usd, 4)
        record["hold_secs"] = round(decision_ts - open_rec.get("decision_ts", time.time()), 1)
        record["score"]    = open_rec.get("score", 0)

        # Circuit breaker
        self.daily_pnl += net_pnl
        if net_pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        if self.daily_pnl <= -MAX_DAILY_LOSS:
            self.circuit_broken = True
            log.warning(f"[Executor] CIRCUIT BREAKER — daily ${self.daily_pnl:.2f}")
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self.circuit_broken = True
            log.warning(f"[Executor] CIRCUIT BREAKER — {self.consecutive_losses} consecutive losses")

        win = "WIN" if net_pnl > 0 else "LOSS"
        log.info(f"[Executor] CLOSE {open_rec['side']} | {win} "
                 f"PnL=${net_pnl:+.4f} ({pnl_pct:+.2f}%) "
                 f"held={record['hold_secs']:.0f}s | "
                 f"daily=${self.daily_pnl:+.2f} streak={self.consecutive_losses}")

        self.position = None
        self._log_trade(record)
        return record

    async def _close_order(self, side: str, quantity: float, price: float):
        """Async close order — reduce only."""
        import random
        from dydx_v4_client import MAX_CLIENT_ID, OrderFlags
        from v4_proto.dydxprotocol.clob.order_pb2 import Order
        from dydx_v4_client.indexer.rest.constants import OrderType

        order_side = Order.Side.SIDE_BUY if side == "LONG" else Order.Side.SIDE_SELL

        if side == "LONG":
            limit_price = price * 1.05
        else:
            limit_price = price * 0.95

        order_id = self.market.order_id(
            self.address, 0, random.randint(0, MAX_CLIENT_ID), OrderFlags.SHORT_TERM
        )
        current_block = await self.node.latest_block_height()

        new_order = self.market.order(
            order_id=order_id,
            order_type=OrderType.MARKET,
            side=order_side,
            size=quantity,
            price=limit_price,
            time_in_force=Order.TimeInForce.TIME_IN_FORCE_UNSPECIFIED,
            reduce_only=True,
            good_til_block=current_block + 10,
        )

        tx = await self.node.place_order(wallet=self.wallet, order=new_order)
        self.wallet.sequence += 1
        return tx

    def _log_trade(self, record: dict):
        self.trade_log.append(record)
        try:
            with open("execution_log.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def summary(self) -> str:
        closes = [t for t in self.trade_log if t.get("action") == "CLOSE"]
        if not closes:
            return "[Executor] No completed trades this session"
        total_pnl = sum(t.get("pnl_usd", 0) for t in closes)
        wins = sum(1 for t in closes if t.get("pnl_usd", 0) > 0)
        return (f"[Executor] {len(closes)} trades | PnL=${total_pnl:+.4f} | "
                f"WR={wins}/{len(closes)}")
