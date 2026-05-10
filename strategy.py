"""Group-weighted scoring engine.

Each indicator group emits a -1..+1 directional vote. ADX gates momentum:
when there is no trend, momentum signals are unreliable so we damp them.
The final score is mapped to 0..100 and a coarse signal label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import config
from indicators import Snapshot


Signal = Literal["BUY", "WEAK_BUY", "NEUTRAL", "WEAK_SELL", "SELL"]


# Sum = 1.0
WEIGHTS = {
    "trend":    0.35,
    "momentum": 0.35,
    "volume":   0.15,
    "pattern":  0.15,
}


@dataclass
class Score:
    symbol: str
    score: int                  # 0..100
    signal: Signal
    direction: float            # -1..+1
    breakdown: dict = field(default_factory=dict)
    snapshot: Snapshot | None = None
    notes: list[str] = field(default_factory=list)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _trend_vote(s: Snapshot) -> tuple[float, list[str]]:
    notes = []
    vote = 0.0

    # ADX magnitude with DI direction
    if s.adx >= config.ADX_STRONG:
        notes.append(f"ADX {s.adx:.1f} strong")
        if s.plus_di > s.minus_di:
            vote += 0.6
        else:
            vote -= 0.6
    elif s.adx >= config.ADX_TREND_MIN:
        notes.append(f"ADX {s.adx:.1f} mild")
        vote += 0.3 if s.plus_di > s.minus_di else -0.3
    else:
        notes.append(f"ADX {s.adx:.1f} no trend")

    # MA stack: fast > mid > slow = bull, opposite = bear
    if s.ma_fast > s.ma_mid > s.ma_slow:
        vote += 0.3
        notes.append("MA stack bull")
    elif s.ma_fast < s.ma_mid < s.ma_slow:
        vote -= 0.3
        notes.append("MA stack bear")

    # EMA20 slope
    if s.ma_slope_fast > 1.0:
        vote += 0.2
    elif s.ma_slope_fast < -1.0:
        vote -= 0.2

    # Price vs EMA50
    if s.close > s.ma_mid:
        vote += 0.1
    else:
        vote -= 0.1

    return _clip(vote), notes


def _momentum_vote(s: Snapshot) -> tuple[float, list[str]]:
    notes = []
    votes = []

    # RSI: 50 is neutral; flag overbought/oversold
    if s.rsi < 30:
        votes.append(0.7); notes.append(f"RSI {s.rsi:.0f} oversold")
    elif s.rsi < 45:
        votes.append(0.2)
    elif s.rsi > 70:
        votes.append(-0.7); notes.append(f"RSI {s.rsi:.0f} overbought")
    elif s.rsi > 55:
        votes.append(-0.2)
    else:
        votes.append(0.0)

    # MACD: line vs signal AND histogram direction
    macd_v = 0.0
    if s.macd > s.macd_signal:
        macd_v += 0.4
    else:
        macd_v -= 0.4
    if s.macd_hist > s.macd_hist_prev:
        macd_v += 0.2
    else:
        macd_v -= 0.2
    votes.append(_clip(macd_v))
    if s.macd > s.macd_signal:
        notes.append("MACD bullish cross")
    else:
        notes.append("MACD bearish")

    # Stoch
    if s.stoch_k < 20 and s.stoch_k > s.stoch_d:
        votes.append(0.6); notes.append("Stoch oversold turning up")
    elif s.stoch_k > 80 and s.stoch_k < s.stoch_d:
        votes.append(-0.6); notes.append("Stoch overbought turning down")
    else:
        votes.append(_clip((s.stoch_k - s.stoch_d) / 50.0))

    # KDJ J line — extreme reversal cues
    if s.kdj_j < 0:
        votes.append(0.5); notes.append("KDJ J<0 reversal up")
    elif s.kdj_j > 100:
        votes.append(-0.5); notes.append("KDJ J>100 reversal down")
    else:
        votes.append(_clip((s.kdj_k - s.kdj_d) / 30.0))

    avg = sum(votes) / len(votes)

    # ADX gate: no trend = momentum is noise, damp it down
    if s.adx < config.ADX_TREND_MIN:
        avg *= 0.4
        notes.append("momentum damped (no trend)")

    return _clip(avg), notes


def _volume_vote(s: Snapshot) -> tuple[float, list[str]]:
    notes = []
    if s.mfi < 20:
        notes.append(f"MFI {s.mfi:.0f} oversold flow")
        return 0.7, notes
    if s.mfi > 80:
        notes.append(f"MFI {s.mfi:.0f} overbought flow")
        return -0.7, notes
    if s.mfi > 50:
        return 0.2, notes
    return -0.2, notes


def _pattern_vote(s: Snapshot) -> tuple[float, list[str]]:
    bull = {"bull_engulf": 0.8, "hammer": 0.6}
    bear = {"bear_engulf": -0.8, "shooting_star": -0.6}
    if s.candle in bull:
        return bull[s.candle], [f"candle {s.candle}"]
    if s.candle in bear:
        return bear[s.candle], [f"candle {s.candle}"]
    if s.candle == "doji":
        return 0.0, ["doji (indecision)"]
    return 0.0, []


def score_symbol(symbol: str, snap: Snapshot) -> Score:
    t_v, t_n = _trend_vote(snap)
    m_v, m_n = _momentum_vote(snap)
    v_v, v_n = _volume_vote(snap)
    p_v, p_n = _pattern_vote(snap)

    direction = (
        WEIGHTS["trend"] * t_v
        + WEIGHTS["momentum"] * m_v
        + WEIGHTS["volume"] * v_v
        + WEIGHTS["pattern"] * p_v
    )
    direction = _clip(direction)

    score = int(round((direction + 1) * 50))    # -1..+1 -> 0..100

    if score >= config.BUY_SCORE_MIN + 15:
        signal: Signal = "BUY"
    elif score >= config.BUY_SCORE_MIN:
        signal = "WEAK_BUY"
    elif score <= 100 - (config.BUY_SCORE_MIN + 15):
        signal = "SELL"
    elif score <= 100 - config.BUY_SCORE_MIN:
        signal = "WEAK_SELL"
    else:
        signal = "NEUTRAL"

    return Score(
        symbol=symbol,
        score=score,
        signal=signal,
        direction=direction,
        breakdown={"trend": t_v, "momentum": m_v, "volume": v_v, "pattern": p_v},
        snapshot=snap,
        notes=t_n + m_n + v_n + p_n,
    )
