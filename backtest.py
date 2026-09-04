"""Backtest harness for the live strategy.

It imports strategy.py and indicators.py unchanged, so what it measures is
the bot itself, not a re-implementation. The portfolio maths comes from
portfolio.py for the same reason.

    python3 backtest.py fetch                 # download + cache OKX history
    python3 backtest.py run                   # one run with config.py defaults
    python3 backtest.py run --intrabar        # ...with the 1h guard simulation
    python3 backtest.py sweep --param BUY_SCORE_MIN --values 50,55,60,65,70
    python3 backtest.py compare               # 4h-only exits vs guard exits

Honest limits, read these before trusting a number
--------------------------------------------------
* Survivorship bias. The symbol list is chosen from *today's* liquid pairs,
  so the test never sees a coin that was liquid in March and is dead now.
  Real-world results will be worse than anything printed here.
* Slippage. Modelled as a flat SLIPPAGE_PCT per side on top of the taker fee.
  Thin alts fill worse than that.
* Intrabar order. When one bar's range covers both the stop and the target,
  the stop is assumed to hit first. Pessimistic on purpose.
* Sample size. 53 live trades were not enough to prove an edge; a backtest
  over the same few months is not independent evidence. Treat the output as
  "does this change help or hurt", not as a forecast.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

import config
import portfolio
from indicators import Snapshot, compute, _detect_candle
from strategy import score_symbol

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, MACD, EMAIndicator, SMAIndicator
from ta.volume import MFIIndicator

OKX = "https://www.okx.com"
CACHE = config.PROJECT_DIR / "bt_cache"
UA = {"User-Agent": "coin-bot-backtest/1.0"}

# Extra cost per side, on top of config.TAKER_FEE. 0.0005 = 5 bps.
SLIPPAGE_PCT = 0.0005

_BAR = {"4h": "4H", "1h": "1H", "1d": "1D"}


# ==========================================================================
# data
# ==========================================================================
def _http(url: str, tries: int = 4) -> dict:
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=20
            ) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def fetch_ohlcv(symbol: str, bar: str, days: int) -> pd.DataFrame:
    """Closed candles, oldest first. Cached on disk so sweeps are free."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"{symbol}_{bar}_{days}.json"
    if cache_file.exists():
        raw = json.loads(cache_file.read_text())
    else:
        inst = f"{symbol[:-4]}-USDT"
        need = int(days * 24 / int(bar[:-1])) + 260      # +warmup for MA200
        raw, cursor = [], ""
        while len(raw) < need:
            q = {"instId": inst, "bar": _BAR[bar], "limit": "100"}
            if cursor:
                q["after"] = cursor
            js = _http(f"{OKX}/api/v5/market/history-candles?"
                       + urllib.parse.urlencode(q))
            page = [c for c in js.get("data", []) if c[-1] == "1"]
            if not page:
                break
            raw.extend(page)
            cursor = page[-1][0]
            time.sleep(0.12)                              # OKX rate limit
        cache_file.write_text(json.dumps(raw))

    if not raw:
        return pd.DataFrame()
    rows = sorted(raw, key=lambda c: int(c[0]))
    df = pd.DataFrame({
        "open_time": [int(c[0]) for c in rows],
        "open": [float(c[1]) for c in rows],
        "high": [float(c[2]) for c in rows],
        "low": [float(c[3]) for c in rows],
        "close": [float(c[4]) for c in rows],
        "volume": [float(c[5]) for c in rows],
        "quote_volume": [float(c[7]) for c in rows],
    }).drop_duplicates("open_time").reset_index(drop=True)
    return df


def pick_universe(n: int) -> list[str]:
    """Today's liquid, moving pairs -- the survivorship caveat lives here."""
    import data as live_data
    return [r["symbol"] for r in live_data.list_universe()][:n]


# ==========================================================================
# indicators, precomputed once per symbol
# ==========================================================================
def precompute(df: pd.DataFrame) -> list[Snapshot | None]:
    """One vectorised pass producing the same Snapshot compute() would return.

    compute() is called per-bar in the live bot; doing that here would be
    O(n^2) and make sweeps unusable. `backtest.py verify` proves the two
    agree bar for bar.
    """
    n = len(df)
    out: list[Snapshot | None] = [None] * n
    if n < 200:
        return out

    high, low, close, vol, open_ = (df["high"], df["low"], df["close"],
                                    df["volume"], df["open"])
    rsi = RSIIndicator(close=close, window=14).rsi()
    adx_o = ADXIndicator(high=high, low=low, close=close, window=14)
    adx, pdi, mdi = adx_o.adx(), adx_o.adx_pos(), adx_o.adx_neg()
    macd_o = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_l, macd_s, macd_h = macd_o.macd(), macd_o.macd_signal(), macd_o.macd_diff()
    mfi = MFIIndicator(high=high, low=low, close=close, volume=vol,
                       window=14).money_flow_index()
    ema20 = EMAIndicator(close=close, window=20).ema_indicator()
    ema50 = EMAIndicator(close=close, window=50).ema_indicator()
    sma200 = SMAIndicator(close=close, window=200).sma_indicator()
    stoch = StochasticOscillator(high=high, low=low, close=close,
                                 window=14, smooth_window=3)
    k_, d_ = stoch.stoch(), stoch.stoch_signal()
    rsv = (100 * (close - low.rolling(9).min())
           / (high.rolling(9).max() - low.rolling(9).min())).fillna(50)
    kdj_k = rsv.ewm(alpha=1/3, adjust=False).mean()
    kdj_d = kdj_k.ewm(alpha=1/3, adjust=False).mean()
    kdj_j = 3 * kdj_k - 2 * kdj_d

    def f(s, i):
        v = s.iloc[i]
        return float(v) if pd.notna(v) else float("nan")

    for i in range(199, n):
        slope = ((ema20.iloc[i] / ema20.iloc[i - 5] - 1) * 100
                 if i >= 5 and pd.notna(ema20.iloc[i - 5]) else 0.0)
        out[i] = Snapshot(
            close=f(close, i), high=f(high, i), low=f(low, i), open_=f(open_, i),
            rsi=f(rsi, i), adx=f(adx, i), plus_di=f(pdi, i), minus_di=f(mdi, i),
            macd=f(macd_l, i), macd_signal=f(macd_s, i), macd_hist=f(macd_h, i),
            macd_hist_prev=(float(macd_h.iloc[i - 1])
                            if pd.notna(macd_h.iloc[i - 1]) else 0.0),
            mfi=f(mfi, i),
            ma_fast=f(ema20, i), ma_mid=f(ema50, i), ma_slow=f(sma200, i),
            ma_slope_fast=float(slope),
            stoch_k=f(k_, i), stoch_d=f(d_, i),
            kdj_k=f(kdj_k, i), kdj_d=f(kdj_d, i), kdj_j=f(kdj_j, i),
            candle=_detect_candle(df.iloc[: i + 1]),
        )
    return out


def verify(df: pd.DataFrame, snaps: list[Snapshot | None],
           samples: int = 12) -> float:
    """Max relative deviation between precompute() and the live compute()."""
    idx = [i for i in range(len(df)) if snaps[i] is not None]
    if not idx:
        return float("nan")
    step = max(1, len(idx) // samples)
    worst = 0.0
    for i in idx[::step]:
        want = compute(df.iloc[: i + 1].reset_index(drop=True))
        got = snaps[i]
        if want is None:
            continue
        for field in ("rsi", "adx", "macd", "mfi", "ma_fast", "ma_mid",
                      "stoch_k", "kdj_j"):
            a, b = getattr(want, field), getattr(got, field)
            if math.isnan(a) or math.isnan(b):
                continue
            denom = max(abs(a), 1e-9)
            worst = max(worst, abs(a - b) / denom)
    return worst


# ==========================================================================
# simulation
# ==========================================================================
@dataclass
class Params:
    buy_score_min: int = config.BUY_SCORE_MIN
    hold_score_min: int = config.HOLD_SCORE_MIN
    replace_margin: int = config.REPLACE_MARGIN
    stop_loss_pct: float = config.STOP_LOSS_PCT
    take_profit_pct: float = config.TAKE_PROFIT_PCT
    trailing_stop_pct: float = config.TRAILING_STOP_PCT
    trail_activate_pct: float = config.TRAIL_ACTIVATE_PCT
    max_positions: int = config.MAX_OPEN_POSITIONS
    stop_cooldown_h: float = config.STOP_COOLDOWN_HOURS
    intrabar: bool = False          # simulate the 30-min guard using 1h bars


def _apply(p: Params) -> None:
    """portfolio.py reads these off config, so the sweep sets them there."""
    config.BUY_SCORE_MIN = p.buy_score_min
    config.HOLD_SCORE_MIN = p.hold_score_min
    config.REPLACE_MARGIN = p.replace_margin
    config.STOP_LOSS_PCT = p.stop_loss_pct
    config.TAKE_PROFIT_PCT = p.take_profit_pct
    config.TRAILING_STOP_PCT = p.trailing_stop_pct
    config.TRAIL_ACTIVATE_PCT = p.trail_activate_pct
    config.MAX_OPEN_POSITIONS = p.max_positions
    config.STOP_COOLDOWN_HOURS = p.stop_cooldown_h


def _cost(price: float, side: str) -> float:
    """Fill price after slippage; the taker fee is charged by portfolio.py."""
    return price * (1 + SLIPPAGE_PCT) if side == "buy" else price * (1 - SLIPPAGE_PCT)


def _hard_exit(pos: dict, price: float) -> str | None:
    pnl = portfolio.position_pnl_pct(pos, price)
    if pnl <= config.STOP_LOSS_PCT:
        return f"STOP_LOSS {pnl*100:.2f}%"
    if config.TAKE_PROFIT_PCT > 0 and pnl >= config.TAKE_PROFIT_PCT:
        return f"TAKE_PROFIT {pnl*100:.2f}%"
    if portfolio.trailing_stop_hit(pos, price):
        return f"TRAILING_STOP from peak {pos['peak_price']:.6g}"
    return None


def _exit_fill(bar, level: float, side_down: bool) -> float:
    """Where a guard sitting on `level` would actually get out.

    Filling at the bar's extreme would flatter or punish the guard for noise
    it never saw. The realistic fill is the level itself -- unless the bar
    opened past it, which is a gap the guard cannot beat.
    """
    o = float(bar["open"])
    if side_down:
        return min(o, level)
    return max(o, level)


def _levels(pos: dict) -> tuple[float, float]:
    """(downside exit level, take-profit level) for an open position."""
    stop = pos["entry_price"] * (1 + config.STOP_LOSS_PCT)
    gain = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
    if gain >= config.TRAIL_ACTIVATE_PCT:
        stop = max(stop, pos["peak_price"] * (1 - config.TRAILING_STOP_PCT))
    tp = (pos["entry_price"] * (1 + config.TAKE_PROFIT_PCT)
          if config.TAKE_PROFIT_PCT > 0 else float("inf"))
    return stop, tp


def run(bars: dict[str, pd.DataFrame], snaps: dict[str, list],
        hourly: dict[str, pd.DataFrame], p: Params) -> dict:
    _apply(p)
    state = portfolio._empty_state()
    curve: list[tuple[int, float]] = []

    # O(1) lookups instead of scanning the frame for every timestamp.
    index = {s: {int(t): i for i, t in enumerate(df["open_time"])}
             for s, df in bars.items()}
    hbars = {s: df.sort_values("open_time").to_dict("records")
             for s, df in hourly.items()}
    hpos = {s: 0 for s in hbars}

    # Warm-up: start only once every symbol can produce a Snapshot.
    warm = max(int(df["open_time"].iloc[199]) for df in bars.values()
               if len(df) > 199)
    times = sorted({int(t) for df in bars.values() for t in df["open_time"]
                    if int(t) > warm})

    for t in times:
        # ---- 1. intrabar hard exits (what the 30-minute guard would catch)
        if p.intrabar:
            for sym in list(state["positions"]):
                rows = hbars.get(sym)
                if not rows:
                    continue
                i = hpos[sym]
                while i < len(rows) and rows[i]["open_time"] < t - 4 * 3600_000:
                    i += 1
                j = i
                while j < len(rows) and rows[j]["open_time"] < t:
                    pos = state["positions"].get(sym)
                    if pos is None:
                        break
                    b = rows[j]
                    stop, tp = _levels(pos)
                    if float(b["low"]) <= stop:                 # adverse first
                        px = _exit_fill(b, stop, True)
                        why = _hard_exit(pos, px) or f"STOP_LOSS forced"
                        portfolio.close_position(state, sym,
                                                 _cost(px, "sell"), why)
                        break
                    portfolio.update_peak(state, sym, float(b["high"]))
                    if float(b["high"]) >= tp:
                        px = _exit_fill(b, tp, False)
                        portfolio.close_position(
                            state, sym, _cost(px, "sell"),
                            f"TAKE_PROFIT {portfolio.position_pnl_pct(pos, px)*100:.2f}%")
                        break
                    j += 1
                hpos[sym] = i

        # ---- 2. the 4h close: score, hard exits, then signal exits
        scores = {}
        for sym, df in bars.items():
            i = index[sym].get(t)
            if i is None or snaps[sym][i] is None:
                continue
            scores[sym] = score_symbol(sym, snaps[sym][i])

        for sym in list(state["positions"]):
            sc = scores.get(sym)
            if sc is None:
                continue
            px = sc.snapshot.close
            portfolio.update_peak(state, sym, px)
            why = _hard_exit(state["positions"][sym], px)
            if why:
                portfolio.close_position(state, sym, _cost(px, "sell"), why)
                continue
            if sc.signal in ("SELL", "WEAK_SELL") or sc.score < config.HOLD_SCORE_MIN:
                portfolio.close_position(
                    state, sym, _cost(px, "sell"),
                    f"signal={sc.signal} score={sc.score}")

        # ---- 3. fill empty slots
        slots = config.MAX_OPEN_POSITIONS - len(state["positions"])
        if slots > 0 and state["cash"] > 10:
            held_min = min((scores[s].score for s in state["positions"]
                            if s in scores), default=0)
            picks = []
            for sc in sorted(scores.values(), key=lambda x: x.score, reverse=True):
                if sc.symbol in state["positions"]:
                    continue
                if sc.score < config.BUY_SCORE_MIN:
                    continue
                if held_min and sc.score < held_min + config.REPLACE_MARGIN:
                    continue
                if portfolio.cooldown_remaining_h(state, sc.symbol) > 0:
                    continue
                picks.append(sc)
                if len(picks) >= slots:
                    break
            if picks:
                alloc = state["cash"] / len(picks)
                for sc in picks:
                    if alloc >= 10:
                        portfolio.open_position(
                            state, sc.symbol, _cost(sc.snapshot.close, "buy"),
                            alloc, f"score={sc.score}")

        mtm = sum(pos["qty"] * scores[s].snapshot.close
                  for s, pos in state["positions"].items() if s in scores)
        curve.append((t, state["cash"] + mtm))

    return _metrics(state, curve, extra_trades=state["trades"])


def _row(df: pd.DataFrame, t: int) -> int | None:
    hit = df.index[df["open_time"] == t]
    return int(hit[0]) if len(hit) else None


def _stop_pcts(trades: list[dict]) -> list[float]:
    """Realised % move on stop-loss exits, entry->exit, fees excluded.

    Pairs each SELL with the BUY that opened it so the number answers the
    only question that matters here: how far past the configured stop did
    the position actually get out?
    """
    entry: dict[str, float] = {}
    out = []
    for t in trades:
        if t["side"] == "BUY":
            entry[t["symbol"]] = t["price"]
        else:
            e = entry.pop(t["symbol"], None)
            if e and t["reason"].startswith("STOP_LOSS"):
                out.append((t["price"] / e - 1) * 100)
    return out


def _metrics(state: dict, curve: list[tuple[int, float]],
             extra_trades=None) -> dict:
    sells = [x for x in state["trades"] if x["side"] == "SELL"]
    pnl = [x["pnl_usdt"] for x in sells]
    eq = [v for _, v in curve] or [config.INITIAL_CAPITAL_USDT]
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = min(mdd, (v - peak) / peak * 100)
    wins = [x for x in pnl if x > 0]
    sd = statistics.stdev(pnl) if len(pnl) > 1 else 0.0
    se = sd / len(pnl) ** 0.5 if pnl else 0.0
    exp = sum(pnl) / len(pnl) if pnl else 0.0
    top3 = sum(sorted(pnl, reverse=True)[3:]) if len(pnl) > 3 else 0.0
    return {
        "final_equity": eq[-1],
        "return_pct": (eq[-1] / config.INITIAL_CAPITAL_USDT - 1) * 100,
        "trades": len(sells),
        "win_rate": 100 * len(wins) / len(pnl) if pnl else 0.0,
        "expectancy": exp,
        "t_stat": exp / se if se else 0.0,
        "max_dd": mdd,
        "fees": sum(x.get("fee_usdt", 0.0) for x in state["trades"]),
        "pnl_ex_top3": top3,
        "stop_exits": _stop_pcts(state["trades"]),
        "trade_log": sells,
    }


# ==========================================================================
# reporting
# ==========================================================================
HDR = (f"{'label':<22}{'getiri%':>9}{'islem':>7}{'isabet%':>9}"
       f"{'bek.deger':>11}{'t':>7}{'maxDD%':>9}{'komisyon':>10}{'top3siz':>10}")


def line(label: str, m: dict) -> str:
    return (f"{label:<22}{m['return_pct']:>+9.2f}{m['trades']:>7}"
            f"{m['win_rate']:>9.0f}{m['expectancy']:>+11.2f}{m['t_stat']:>7.2f}"
            f"{m['max_dd']:>9.1f}{m['fees']:>10.2f}{m['pnl_ex_top3']:>+10.2f}")


def load_all(symbols: list[str], days: int, intrabar: bool):
    bars, snaps, hourly = {}, {}, {}
    for i, s in enumerate(symbols, 1):
        df = fetch_ohlcv(s, "4h", days)
        if len(df) < 260:
            print(f"  [{i}/{len(symbols)}] {s}: yetersiz veri ({len(df)} mum)")
            continue
        bars[s] = df
        snaps[s] = precompute(df)
        if intrabar:
            hourly[s] = fetch_ohlcv(s, "1h", days)
        print(f"  [{i}/{len(symbols)}] {s}: {len(df)} adet 4h mum")
    return bars, snaps, hourly


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "sweep", "compare", "verify"])
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--symbols", type=int, default=25)
    ap.add_argument("--intrabar", action="store_true")
    ap.add_argument("--param")
    ap.add_argument("--values")
    a = ap.parse_args(argv[1:])

    syms = pick_universe(a.symbols)
    print(f"Evren ({len(syms)}): {', '.join(s[:-4] for s in syms)}\n")
    need_hourly = a.intrabar or a.cmd == "compare"
    bars, snaps, hourly = load_all(syms, a.days, need_hourly)
    if not bars:
        print("Veri yok."); return 1
    print(f"\n{len(bars)} sembol yuklendi.\n")

    if a.cmd == "fetch":
        return 0

    if a.cmd == "verify":
        worst = 0.0
        for s in list(bars)[:5]:
            d = verify(bars[s], snaps[s])
            print(f"  {s}: max sapma {d:.2e}")
            worst = max(worst, d)
        ok = worst < 1e-6
        print(f"\n{'OK' if ok else 'FAIL'}: precompute vs compute max sapma {worst:.2e}")
        return 0 if ok else 1

    if a.cmd == "run":
        m = run(bars, snaps, hourly, Params(intrabar=a.intrabar))
        print(HDR); print(line("varsayilan", m))
        return 0

    if a.cmd == "compare":
        print(HDR)
        print(line("4h cikislar (eski)", run(bars, snaps, hourly, Params(intrabar=False))))
        print(line("guard (1h cozunurluk)", run(bars, snaps, hourly, Params(intrabar=True))))
        return 0

    if a.cmd == "sweep":
        if not a.param or not a.values:
            print("--param ve --values gerekli"); return 2
        field = a.param.lower()
        if field not in Params.__dataclass_fields__:
            print(f"bilinmeyen parametre. secenekler: "
                  f"{', '.join(Params.__dataclass_fields__)}"); return 2
        typ = type(getattr(Params(), field))
        print(HDR)
        for raw in a.values.split(","):
            val = typ(raw) if typ is not bool else raw.lower() == "true"
            m = run(bars, snaps, hourly,
                    replace(Params(intrabar=a.intrabar), **{field: val}))
            print(line(f"{a.param}={raw}", m))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
