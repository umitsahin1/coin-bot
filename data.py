"""OKX v5 spot market data fetching. No API key required.

We use OKX (not Binance / Bybit) because both of those geo-block public
market endpoints from GitHub-hosted runners (US IPs). OKX's market data
API is globally reachable, and its USDT spot universe is comparable.

Internal symbols stay in `BTCUSDT` form (the format stored in state.json
and used by the rest of the codebase). Conversion to OKX's `BTC-USDT`
happens at the boundary inside this module only.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import requests

import config


OKX_BASE = "https://www.okx.com"

# Map our human-readable timeframe to OKX's bar string (uppercase H/D).
_BAR_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
    "1d": "1D", "1w": "1W", "1M": "1M",
}

# Suffixes that mark leveraged / structured tokens we want to skip.
# These match exactly before "USDT" so they don't false-positive on
# legit coins like JUPUSDT (contains "UP" but ends with "JUPUSDT").
_LEVERAGE_SUFFIXES = ("2L", "2S", "3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "coin-paper-bot/0.3"})


def _to_okx(symbol: str) -> str:
    """BTCUSDT -> BTC-USDT (OKX format)."""
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USDT"
    if symbol.endswith("USDC"):
        return f"{symbol[:-4]}-USDC"
    return symbol


def _from_okx(inst_id: str) -> str:
    """BTC-USDT -> BTCUSDT (internal format)."""
    return inst_id.replace("-", "")


def _get(path: str, params: dict | None = None, timeout: int = 15) -> dict:
    url = f"{OKX_BASE}{path}"
    r = _SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if str(js.get("code", "")) != "0":
        raise RuntimeError(f"OKX error code={js.get('code')} msg={js.get('msg')}")
    return js


def _is_leveraged_token(symbol: str) -> bool:
    # symbol is in internal form, e.g. BTCUSDT or BTC3LUSDT.
    if not symbol.endswith("USDT"):
        return False
    base = symbol[:-4]   # e.g. "BTC", "BTC3L", "JUP"
    return any(base.endswith(suf) for suf in _LEVERAGE_SUFFIXES)


def list_universe() -> list[dict]:
    """Top-N USDT spot pairs by 24h quote volume."""
    js = _get("/api/v5/market/tickers", {"instType": "SPOT"})
    rows = []
    for t in js.get("data", []):
        inst = t.get("instId", "")
        if not inst.endswith("-USDT"):
            continue
        sym = _from_okx(inst)
        if sym in config.EXCLUDE_STABLES:
            continue
        if _is_leveraged_token(sym):
            continue
        try:
            qv = float(t.get("volCcy24h", "0") or 0)
            last = float(t.get("last", "0") or 0)
        except (TypeError, ValueError):
            continue
        if qv < config.MIN_QUOTE_VOLUME_USDT or last <= 0:
            continue
        rows.append({"symbol": sym, "quote_volume": qv, "last": last})
    rows.sort(key=lambda r: r["quote_volume"], reverse=True)
    return rows[: config.UNIVERSE_SIZE]


def fetch_klines(symbol: str, interval: str = config.TIMEFRAME,
                 limit: int = config.KLINE_LIMIT) -> pd.DataFrame:
    """Return closed OHLCV bars only. OKX marks in-progress bars with confirm=0."""
    bar = _BAR_MAP.get(interval, interval)
    js = _get("/api/v5/market/candles",
              {"instId": _to_okx(symbol), "bar": bar, "limit": str(limit)})
    raw = js.get("data", [])
    if not raw:
        return pd.DataFrame()

    # OKX returns newest-first; we want chronological (oldest-first).
    rows = list(reversed(raw))
    # OKX candle: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    cols = ["open_time", "open", "high", "low", "close",
            "volume", "vol_ccy", "vol_ccy_quote", "confirm"]
    df = pd.DataFrame(rows, columns=cols)

    # Drop in-progress bar (confirm == "0").
    df = df[df["confirm"] == "1"].reset_index(drop=True)
    if df.empty:
        return df

    df["open_time"] = pd.to_numeric(df["open_time"]).astype("int64")
    for c in ("open", "high", "low", "close", "volume", "vol_ccy_quote"):
        df[c] = df[c].astype(float)
    df["quote_volume"] = df["vol_ccy_quote"]      # USDT volume of the bar
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def fetch_last_price(symbol: str) -> float:
    js = _get("/api/v5/market/ticker", {"instId": _to_okx(symbol)})
    data = js.get("data", [])
    if not data:
        raise RuntimeError(f"no ticker for {symbol}")
    return float(data[0]["last"])


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
