"""Technical indicators. Uses the `ta` library for the well-tested ones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, MACD, EMAIndicator, SMAIndicator
from ta.volume import MFIIndicator


@dataclass
class Snapshot:
    """All indicator readings for the latest closed bar of a symbol."""
    close: float
    high: float
    low: float
    open_: float

    rsi: float
    adx: float
    plus_di: float
    minus_di: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    mfi: float

    ma_fast: float          # EMA20
    ma_mid: float           # EMA50
    ma_slow: float          # SMA200 (or shorter when bars < 200)
    ma_slope_fast: float    # %change of EMA20 over last 5 bars

    stoch_k: float
    stoch_d: float
    kdj_k: float
    kdj_d: float
    kdj_j: float

    candle: str             # 'bull_engulf', 'bear_engulf', 'hammer', 'shooting_star', 'doji', 'none'


def _last(s: pd.Series) -> float:
    v = s.iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _detect_candle(df: pd.DataFrame) -> str:
    """Identify a single common pattern on the last closed bar."""
    if len(df) < 2:
        return "none"
    o, h, l, c = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    po, pc = df["open"].iloc[-2], df["close"].iloc[-2]
    body = abs(c - o)
    rng = max(h - l, 1e-12)
    upper = h - max(c, o)
    lower = min(c, o) - l

    # Doji: body very small relative to range
    if body / rng < 0.1:
        return "doji"

    # Bullish engulfing
    if pc < po and c > o and c >= po and o <= pc:
        return "bull_engulf"
    # Bearish engulfing
    if pc > po and c < o and c <= po and o >= pc:
        return "bear_engulf"

    # Hammer: small body near top, long lower wick (>= 2x body), bullish close preferred
    if lower >= 2 * body and upper <= body and c >= o:
        return "hammer"
    # Shooting star: small body near bottom, long upper wick
    if upper >= 2 * body and lower <= body and c <= o:
        return "shooting_star"

    return "none"


def compute(df: pd.DataFrame) -> Snapshot | None:
    """Compute the full indicator snapshot. Returns None if not enough data."""
    if len(df) < 60:
        return None
    high, low, close, vol, open_ = df["high"], df["low"], df["close"], df["volume"], df["open"]

    rsi = RSIIndicator(close=close, window=14).rsi()
    adx_obj = ADXIndicator(high=high, low=low, close=close, window=14)
    adx = adx_obj.adx()
    plus_di = adx_obj.adx_pos()
    minus_di = adx_obj.adx_neg()
    macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd()
    macd_sig = macd_obj.macd_signal()
    macd_hist = macd_obj.macd_diff()
    mfi = MFIIndicator(high=high, low=low, close=close, volume=vol, window=14).money_flow_index()

    ema20 = EMAIndicator(close=close, window=20).ema_indicator()
    ema50 = EMAIndicator(close=close, window=50).ema_indicator()
    sma_slow_window = 200 if len(df) >= 200 else min(100, max(50, len(df) // 2))
    sma_slow = SMAIndicator(close=close, window=sma_slow_window).sma_indicator()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()

    # KDJ from raw stochastic %K with EMA-style smoothing (Chinese-style: 1/3 weighting)
    rsv = 100 * (close - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min())
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3 * k - 2 * d

    ma_slope_fast = float((ema20.iloc[-1] / ema20.iloc[-6] - 1) * 100) if len(ema20.dropna()) >= 6 else 0.0

    return Snapshot(
        close=_last(close), high=_last(high), low=_last(low), open_=_last(open_),
        rsi=_last(rsi),
        adx=_last(adx), plus_di=_last(plus_di), minus_di=_last(minus_di),
        macd=_last(macd_line), macd_signal=_last(macd_sig),
        macd_hist=_last(macd_hist),
        macd_hist_prev=float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 and pd.notna(macd_hist.iloc[-2]) else 0.0,
        mfi=_last(mfi),
        ma_fast=_last(ema20), ma_mid=_last(ema50), ma_slow=_last(sma_slow),
        ma_slope_fast=ma_slope_fast,
        stoch_k=_last(stoch_k), stoch_d=_last(stoch_d),
        kdj_k=_last(k), kdj_d=_last(d), kdj_j=_last(j),
        candle=_detect_candle(df),
    )
