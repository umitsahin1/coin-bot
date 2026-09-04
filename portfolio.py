"""Paper trading state. Single JSON file, idempotent operations."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
        "cooldowns": {},      # symbol -> ISO time until which re-entry is blocked
    }


def load() -> dict:
    p: Path = config.STATE_FILE
    if not p.exists():
        s = _empty_state()
        save(s)
        return s
    with p.open("r") as f:
        state = json.load(f)
    # Forward-compat: states written before cooldowns existed.
    state.setdefault("cooldowns", {})
    return state


def save(state: dict) -> None:
    with config.STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2, default=str)


def reset() -> dict:
    s = _empty_state()
    save(s)
    return s


# --- re-entry cooldown after a stop-out -------------------------------------

def set_cooldown(state: dict, symbol: str, hours: float | None = None) -> None:
    h = config.STOP_COOLDOWN_HOURS if hours is None else hours
    until = datetime.now(timezone.utc) + timedelta(hours=h)
    state.setdefault("cooldowns", {})[symbol] = until.isoformat(timespec="seconds")


def cooldown_remaining_h(state: dict, symbol: str) -> float:
    """Hours left on the block, 0.0 if the symbol is tradable."""
    raw = state.get("cooldowns", {}).get(symbol)
    if not raw:
        return 0.0
    try:
        until = datetime.fromisoformat(raw)
    except ValueError:
        return 0.0
    left = (until - datetime.now(timezone.utc)).total_seconds() / 3600.0
    return max(0.0, left)


def is_blocked(state: dict, symbol: str) -> bool:
    return cooldown_remaining_h(state, symbol) > 0.0


def prune_cooldowns(state: dict) -> None:
    cds = state.get("cooldowns", {})
    for sym in [s for s in cds if cooldown_remaining_h(state, s) <= 0.0]:
        del cds[sym]


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
    if reason.startswith("STOP_LOSS"):
        set_cooldown(state, symbol)
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
    """Trailing stop: only active once peak gain >= TRAIL_ACTIVATE_PCT; then sell on retrace."""
    peak_gain = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
    if peak_gain < config.TRAIL_ACTIVATE_PCT:
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
