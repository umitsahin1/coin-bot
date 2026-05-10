"""Paper trading state. Single JSON file, idempotent operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_state() -> dict:
    return {
        "created_at": _now_iso(),
        "cash": config.INITIAL_CAPITAL_USDT,
        "initial_capital": config.INITIAL_CAPITAL_USDT,
        "positions": {},      # symbol -> {qty, entry_price, entry_time, peak_price, cost_usdt}
        "trades": [],         # list of {time, side, symbol, qty, price, pnl_usdt, reason}
    }


def load() -> dict:
    p: Path = config.STATE_FILE
    if not p.exists():
        s = _empty_state()
        save(s)
        return s
    with p.open("r") as f:
        return json.load(f)


def save(state: dict) -> None:
    with config.STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2, default=str)


def reset() -> dict:
    s = _empty_state()
    save(s)
    return s


def open_position(state: dict, symbol: str, price: float, alloc_usdt: float, reason: str) -> dict:
    """Buy with TAKER_FEE deducted from notional. Returns the trade record."""
    if symbol in state["positions"]:
        raise ValueError(f"{symbol} already held")
    if alloc_usdt > state["cash"] + 1e-9:
        raise ValueError(f"insufficient cash: need {alloc_usdt:.2f}, have {state['cash']:.2f}")
    fee = alloc_usdt * config.TAKER_FEE
    qty = (alloc_usdt - fee) / price
    state["cash"] -= alloc_usdt
    state["positions"][symbol] = {
        "qty": qty,
        "entry_price": price,
        "entry_time": _now_iso(),
        "peak_price": price,
        "cost_usdt": alloc_usdt,
    }
    trade = {
        "time": _now_iso(), "side": "BUY", "symbol": symbol,
        "qty": qty, "price": price, "fee_usdt": fee,
        "pnl_usdt": 0.0, "reason": reason,
    }
    state["trades"].append(trade)
    return trade


def close_position(state: dict, symbol: str, price: float, reason: str) -> dict:
    pos = state["positions"].get(symbol)
    if not pos:
        raise ValueError(f"{symbol} not held")
    gross = pos["qty"] * price
    fee = gross * config.TAKER_FEE
    proceeds = gross - fee
    pnl = proceeds - pos["cost_usdt"]
    state["cash"] += proceeds
    del state["positions"][symbol]
    trade = {
        "time": _now_iso(), "side": "SELL", "symbol": symbol,
        "qty": pos["qty"], "price": price, "fee_usdt": fee,
        "pnl_usdt": pnl, "reason": reason,
    }
    state["trades"].append(trade)
    return trade


def update_peak(state: dict, symbol: str, price: float) -> None:
    pos = state["positions"].get(symbol)
    if not pos:
        return
    if price > pos["peak_price"]:
        pos["peak_price"] = price


def position_pnl_pct(pos: dict, price: float) -> float:
    return (price - pos["entry_price"]) / pos["entry_price"]


def trailing_stop_hit(pos: dict, price: float) -> bool:
    """Trailing stop: only active once peak gain >= TP/2; then sell on retrace."""
    peak_gain = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
    if peak_gain < config.TAKE_PROFIT_PCT / 2:
        return False
    drop = (price - pos["peak_price"]) / pos["peak_price"]
    return drop <= -config.TRAILING_STOP_PCT


def equity(state: dict, prices: dict[str, float]) -> dict[str, Any]:
    holdings = 0.0
    rows = []
    for sym, pos in state["positions"].items():
        px = prices.get(sym, pos["entry_price"])
        val = pos["qty"] * px
        pnl = val - pos["cost_usdt"]
        pnl_pct = position_pnl_pct(pos, px) * 100
        holdings += val
        rows.append({"symbol": sym, "qty": pos["qty"], "entry": pos["entry_price"],
                     "now": px, "value_usdt": val, "pnl_usdt": pnl, "pnl_pct": pnl_pct})
    total = state["cash"] + holdings
    realized = sum(t.get("pnl_usdt", 0.0) for t in state["trades"] if t["side"] == "SELL")
    return {
        "cash": state["cash"],
        "holdings_value": holdings,
        "total_equity": total,
        "initial": state["initial_capital"],
        "total_return_pct": (total / state["initial_capital"] - 1) * 100,
        "realized_pnl": realized,
        "positions": rows,
    }
