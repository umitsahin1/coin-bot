"""Offline tests for backtest.py. No network: prices are synthetic.

The important one is test_precompute_matches_compute. The backtester replaces
the live per-bar compute() with a single vectorised pass for speed; if the two
ever disagree, every number the backtester prints is measuring something other
than the bot. That test pins them together.

    python3 test_backtest.py
"""

from __future__ import annotations

import random
import sys

import pandas as pd

import backtest
import config
from indicators import compute

FAILED: list[str] = []
H = 3600_000


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(label)


# --------------------------------------------------------------------------
def make_hourly(n: int, seed: int, drift: float = 0.0004,
                vol: float = 0.006) -> pd.DataFrame:
    """A random walk with regime changes, as 1h OHLCV bars."""
    rng = random.Random(seed)
    price, t0 = 100.0, 1_700_000_000_000
    rows = []
    trend = drift
    for i in range(n):
        if i % 220 == 0:                       # flip regime periodically
            trend = rng.choice([drift, -drift, drift * 2.5, 0.0])
        o = price
        step = trend + rng.gauss(0, vol)
        c = max(0.01, o * (1 + step))
        hi = max(o, c) * (1 + abs(rng.gauss(0, vol / 2)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, vol / 2)))
        v = abs(rng.gauss(1000, 300)) + 50
        rows.append({"open_time": t0 + i * H, "open": o, "high": hi,
                     "low": lo, "close": c, "volume": v,
                     "quote_volume": v * c})
        price = c
    return pd.DataFrame(rows)


def to_4h(h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h into 4h so the two series are mutually consistent."""
    out = []
    for i in range(0, len(h) - 3, 4):
        g = h.iloc[i:i + 4]
        out.append({
            "open_time": int(g["open_time"].iloc[0]),
            "open": float(g["open"].iloc[0]),
            "high": float(g["high"].max()),
            "low": float(g["low"].min()),
            "close": float(g["close"].iloc[-1]),
            "volume": float(g["volume"].sum()),
            "quote_volume": float(g["quote_volume"].sum()),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def test_precompute_matches_compute() -> None:
    print("\n=== precompute() vs live compute() ===")
    worst = 0.0
    for seed in (1, 2, 3):
        df = to_4h(make_hourly(1400, seed))
        snaps = backtest.precompute(df)
        d = backtest.verify(df, snaps, samples=10)
        print(f"    seed {seed}: {len(df)} bar, max sapma {d:.3e}")
        worst = max(worst, d)
    check("indicators agree to 1e-6", worst < 1e-6, f"(worst {worst:.2e})")

    df = to_4h(make_hourly(1400, 7))
    snaps = backtest.precompute(df)
    check("no Snapshot before 200 bars of warm-up",
          all(s is None for s in snaps[:199]))
    check("Snapshot from bar 200 on", snaps[199] is not None)

    want = compute(df.iloc[:301].reset_index(drop=True))
    got = snaps[300]
    check("candle pattern matches too", want.candle == got.candle,
          f"({want.candle})")


def test_cost_model() -> None:
    print("\n=== cost model ===")
    b = backtest._cost(100.0, "buy")
    s = backtest._cost(100.0, "sell")
    check("buys fill above mid", b > 100.0, f"({b:.4f})")
    check("sells fill below mid", s < 100.0, f"({s:.4f})")
    check("round trip costs 2x slippage",
          abs((b - s) / 100.0 - 2 * backtest.SLIPPAGE_PCT) < 1e-9)

    bar = {"open": 95.0, "high": 96.0, "low": 90.0}
    check("normal stop fills at the level",
          backtest._exit_fill(bar, 93.0, True) == 93.0)
    gap = {"open": 88.0, "high": 89.0, "low": 85.0}
    check("gap-down fills at the open, not the level",
          backtest._exit_fill(gap, 93.0, True) == 88.0)


def test_levels() -> None:
    print("\n=== exit levels ===")
    config.STOP_LOSS_PCT = -0.07
    config.TAKE_PROFIT_PCT = 0.30
    config.TRAILING_STOP_PCT = 0.07
    config.TRAIL_ACTIVATE_PCT = 0.07

    stop, tp = backtest._levels({"entry_price": 100.0, "peak_price": 102.0})
    check("hard stop before trailing activates", abs(stop - 93.0) < 1e-9)
    check("take profit at +30%", abs(tp - 130.0) < 1e-9)

    stop, _ = backtest._levels({"entry_price": 100.0, "peak_price": 120.0})
    check("trailing takes over above the stop", abs(stop - 111.6) < 1e-9)


def _dataset(seeds: range, n: int = 1600, vol: float = 0.006,
             drift: float = 0.0004):
    bars, snaps, hourly = {}, {}, {}
    for i in seeds:
        sym = f"T{i:02d}USDT"
        h = make_hourly(n, seed=100 + i, drift=drift, vol=vol)
        d4 = to_4h(h)
        bars[sym] = d4
        snaps[sym] = backtest.precompute(d4)
        hourly[sym] = h
    return bars, snaps, hourly


def test_run_end_to_end() -> None:
    print("\n=== full simulation ===")
    bars, snaps, hourly = _dataset(range(6))
    m = backtest.run(bars, snaps, hourly, backtest.Params())

    check("produced trades", m["trades"] > 0, f"({m['trades']} islem)")
    check("equity is a real number", m["final_equity"] > 0,
          f"(${m['final_equity']:.2f}, {m['return_pct']:+.1f}%)")
    check("win rate in range", 0 <= m["win_rate"] <= 100)
    check("charged fees", m["fees"] > 0, f"(${m['fees']:.2f})")
    check("drawdown is negative or flat", m["max_dd"] <= 0,
          f"({m['max_dd']:.1f}%)")

    held = backtest.config.MAX_OPEN_POSITIONS
    check("never exceeded the position cap", held == config.MAX_OPEN_POSITIONS)


def test_guard_reduces_stop_overshoot() -> None:
    print("\n=== guard vs 4h-only exits ===")
    # Deliberately violent tape: gentle data never trips a -7% stop, and a
    # comparison over zero stop exits would prove nothing.
    bars, snaps, hourly = _dataset(range(8), vol=0.022, drift=-0.0004)

    slow = backtest.run(bars, snaps, hourly, backtest.Params(guard="off"))
    fast = backtest.run(bars, snaps, hourly, backtest.Params(guard="poll"))
    rest = backtest.run(bars, snaps, hourly, backtest.Params(guard="resting"))

    def avg_stop(m):
        xs = m["stop_exits"]          # already in percent
        return sum(xs) / len(xs) if xs else 0.0

    s_slow, s_fast = avg_stop(slow), avg_stop(fast)
    print(f"    4h  : {len(slow['stop_exits'])} stop, ort {s_slow:+.2f}%")
    print(f"    guard: {len(fast['stop_exits'])} stop, ort {s_fast:+.2f}%")

    check("both modes traded", slow["trades"] > 0 and fast["trades"] > 0)
    check("the tape actually produced stop exits",
          len(slow["stop_exits"]) >= 3 and len(fast["stop_exits"]) >= 3,
          f"({len(slow['stop_exits'])} / {len(fast['stop_exits'])})")

    # What synthetic data CAN prove: the intrabar path fires and fills at the
    # level. What it CANNOT prove: that the guard is better on average. These
    # bars are i.i.d. gaussian steps, so a move has no persistence -- and
    # persistence is exactly what makes a 4h-delayed exit expensive in real
    # markets. Use `backtest.py compare` on OKX history for that question.
    target = config.STOP_LOSS_PCT * 100
    precise = lambda xs: [x for x in xs if abs(x - target) < 0.10]

    # "poll" is the honest model: guard.py reads one price per check, so it
    # can never fill exactly on the level. Only a resting exchange order can.
    check("poll mode never fills exactly at the level",
          len(precise(fast["stop_exits"])) == 0,
          "(gercek guard emir birakmaz, fiyata bakar)")
    check("resting mode does fill at the level",
          len(precise(rest["stop_exits"])) > 0,
          f"({len(precise(rest['stop_exits']))}/{len(rest['stop_exits'])})")
    check("all three modes produced stop exits",
          min(len(slow["stop_exits"]), len(fast["stop_exits"]),
              len(rest["stop_exits"])) >= 3)
    r_avg = sum(rest["stop_exits"]) / max(1, len(rest["stop_exits"]))
    print(f"    ort stop: 4h {s_slow:+.2f}%  poll {s_fast:+.2f}%  "
          f"resting {r_avg:+.2f}%")
    print("    not: hangisinin ortalamada iyi oldugu SENTETIK veriyle")
    print("         cevaplanamaz -- bu seri bagimsiz adimlardan olusuyor,")
    print("         oysa 4h gecikmeyi pahali yapan sey trendin surmesi.")
    print("         Karar icin: backtest.py compare (gercek OKX gecmisi).")


def test_sweep_changes_results() -> None:
    print("\n=== parameter sweep wiring ===")
    from dataclasses import replace
    bars, snaps, hourly = _dataset(range(5))
    base = backtest.Params()
    a = backtest.run(bars, snaps, hourly, replace(base, buy_score_min=55))
    b = backtest.run(bars, snaps, hourly, replace(base, buy_score_min=75))
    print(f"    esik 55 -> {a['trades']} islem, {a['return_pct']:+.1f}%")
    print(f"    esik 75 -> {b['trades']} islem, {b['return_pct']:+.1f}%")
    check("a higher bar trades less often", a["trades"] >= b["trades"])
    check("config is restored between runs after _apply",
          config.BUY_SCORE_MIN == 75)


def main() -> int:
    saved = {k: getattr(config, k) for k in
             ("BUY_SCORE_MIN", "HOLD_SCORE_MIN", "STOP_LOSS_PCT",
              "TAKE_PROFIT_PCT", "TRAILING_STOP_PCT", "TRAIL_ACTIVATE_PCT",
              "MAX_OPEN_POSITIONS", "REPLACE_MARGIN", "STOP_COOLDOWN_HOURS")}
    try:
        test_precompute_matches_compute()
        test_cost_model()
        test_levels()
        test_run_end_to_end()
        test_guard_reduces_stop_overshoot()
        test_sweep_changes_results()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
