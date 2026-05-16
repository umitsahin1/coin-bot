"""Bybit v5 spot market data fetching. No API key required.

We use Bybit instead of Binance because Binance returns HTTP 451 to
GitHub-hosted runners (US IPs). Bybit's public market API is reachable
from US IPs and exposes the same USDT spot pairs.
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests

import config


BYBIT_BASE = "https://api.bybit.com"

# We use Bybit USDT-margined linear perpetuals (`linear`) instead of spot
# because Bybit's spot order book is thin -- only a handful of pairs clear
# our volume filter. Perp prices track spot via funding arbitrage, so for
# a paper-trading bot reading OHLCV it makes no practical difference.
_CATEGORY = "linear"

# Map our human-readable timeframe to Bybit's interval string.
_INTERVAL_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W", "1M": "M",
}

# Bybit interval -> milliseconds, used to compute close_time so we can
# drop the in-progress candle deterministically.
_INTERVAL_MS = {
    "1": 60_000, "3": 180_000, "5": 300_000, "15": 900_000, "30": 1_800_000,
    "60": 3_600_000, "120": 7_200_000, "240": 14_400_000,
    "360": 21_600_000, "720": 43_200_000,
    "D": 86_400_000, "W": 604_800_000,
}


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "coin-paper-bot/0.2"})


def _get(path: str, params: dict | None = None, timeout: int = 15) -> dict:
    url = f"{BYBIT_BASE}{path}"
    r = _SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if js.get("retCode") != 0:
        raise RuntimeError(f"Bybit error {js.get('retCode')}: {js.get('retMsg')}")
    return js


def _is_leveraged_token(symbol: str) -> bool:
    # Bybit leveraged tokens look like BTC3LUSDT, ETH2SUSDT, etc.
    for tag in ("2L", "2S", "3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR"):
        if f"{tag}USDT" in symbol and symbol.endswith("USDT"):
            return True
    return False


def list_universe() -> list[dict]:
    """Top-N USDT linear perpetuals by 24h quote turnover."""
    js = _get("/v5/market/tickers", {"category": _CATEGORY})
    rows = []
    for t in js["result"]["list"]:
        sym = t["symbol"]
        if not sym.endswith(config.QUOTE_ASSET):
            continue
        if sym in config.EXCLUDE_STABLES:
            continue
        if _is_leveraged_token(sym):
            continue
        # Skip perp multiplier tickers like 1000PEPEUSDT, 10000LADYSUSDT --
        # the price reference is the multiplied token, which complicates
        # interpretation. The base coin is still tradable elsewhere.
        if sym[0].isdigit():
            continue
        try:
            turnover = float(t.get("turnover24h", "0"))
            last = float(t.get("lastPrice", "0"))
        except (TypeError, ValueError):
            continue
        if turnover < config.MIN_QUOTE_VOLUME_USDT or last <= 0:
            continue
        rows.append({"symbol": sym, "quote_volume": turnover, "last": last})
    rows.sort(key=lambda r: r["quote_volume"], reverse=True)
    return rows[: config.UNIVERSE_SIZE]


def fetch_klines(symbol: str, interval: str = config.TIMEFRAME,
                 limit: int = config.KLINE_LIMIT) -> pd.DataFrame:
    """Return closed OHLCV bars only. Drops the in-progress (latest) candle."""
    bybit_iv = _INTERVAL_MAP.get(interval, interval)
    js = _get(
        "/v5/market/kline",
        {"category": _CATEGORY, "symbol": symbol, "interval": bybit_iv, "limit": limit},
    )
    raw = js["result"]["list"]
    if not raw:
        return pd.DataFrame()

    # Bybit returns newest-first; we want chronological (oldest-first).
    rows = list(reversed(raw))
    cols = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_numeric(df["open_time"]).astype("int64")
    for c in ("open", "high", "low", "close", "volume", "turnover"):
        df[c] = df[c].astype(float)

    # Compute close_time = open_time + interval_ms; drop in-progress.
    step = _INTERVAL_MS.get(bybit_iv, 0)
    df["close_time"] = df["open_time"] + step
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] <= now_ms].reset_index(drop=True)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["quote_volume"] = df["turnover"]
    return df


def fetch_last_price(symbol: str) -> float:
    js = _get("/v5/market/tickers", {"category": _CATEGORY, "symbol": symbol})
    lst = js["result"]["list"]
    if not lst:
        raise RuntimeError(f"no ticker for {symbol}")
    return float(lst[0]["lastPrice"])


def fetch_many_klines(symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = fetch_klines(s)
            if len(df) >= 60:
                out[s] = df
        except requests.HTTPError:
            continue
        except Exception:
            continue
    return out
