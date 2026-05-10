"""Binance public market data fetching. No API key required."""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests

import config


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "coin-paper-bot/0.1"})


def _get(path: str, params: dict | None = None) -> list | dict:
    url = f"{config.EXCHANGE_BASE}{path}"
    r = _SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def list_universe() -> list[dict]:
    """Top-N USDT pairs by 24h quote volume, with leveraged tokens & stables filtered."""
    tickers = _get("/api/v3/ticker/24hr")
    rows = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith(config.QUOTE_ASSET):
            continue
        if sym in config.EXCLUDE_STABLES:
            continue
        if any(sym.endswith(suf) for suf in config.EXCLUDE_QUOTES):
            continue
        try:
            qv = float(t["quoteVolume"])
        except (TypeError, ValueError):
            continue
        if qv < config.MIN_QUOTE_VOLUME_USDT:
            continue
        rows.append({"symbol": sym, "quote_volume": qv, "last": float(t["lastPrice"])})
    rows.sort(key=lambda r: r["quote_volume"], reverse=True)
    return rows[: config.UNIVERSE_SIZE]


def fetch_klines(symbol: str, interval: str = config.TIMEFRAME,
                 limit: int = config.KLINE_LIMIT) -> pd.DataFrame:
    """Return closed OHLCV bars only. Drops the in-progress (last) candle."""
    data = _get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not data:
        return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Drop the in-progress bar: any bar whose close_time is in the future is unfinished.
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"].astype("int64") // 10**6 <= now_ms].reset_index(drop=True)
    return df


def fetch_last_price(symbol: str) -> float:
    data = _get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])


def fetch_many_klines(symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = fetch_klines(s)
            if len(df) >= 60:    # need enough bars for ADX/MACD/etc.
                out[s] = df
        except requests.HTTPError:
            continue
        except Exception:
            continue
    return out
