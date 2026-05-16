"""Command-line interface for the paper trading bot.

Usage:
  python3 cli.py auto      -> scan-or-check (used by GitHub Actions cron)
  python3 cli.py scan      -> pick top 2 coins, open paper positions
  python3 cli.py check     -> re-evaluate held positions, recommend HOLD/SELL/REPLACE
  python3 cli.py status    -> show portfolio P&L
  python3 cli.py reset     -> wipe state.json (start over)
"""

from __future__ import annotations

import sys
from typing import Iterable

import config
import data
import notify
import portfolio
from indicators import compute
from strategy import score_symbol, Score


# ---------- pretty printing ----------

def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


def _print_score_line(rank: int, sc: Score) -> None:
    snap = sc.snapshot
    print(f"  {rank:>2}. {sc.symbol:<12} score={sc.score:>3}  {sc.signal:<10} "
          f"price={snap.close:.6g}  ADX={snap.adx:.0f}  RSI={snap.rsi:.0f}  "
          f"MFI={snap.mfi:.0f}  candle={snap.candle}")


def _print_breakdown(sc: Score) -> None:
    b = sc.breakdown
    print(f"     trend={b['trend']:+.2f}  momentum={b['momentum']:+.2f}  "
          f"volume={b['volume']:+.2f}  pattern={b['pattern']:+.2f}")
    if sc.notes:
        print(f"     notes: {', '.join(sc.notes)}")


# ---------- core operations ----------

def _scan_universe(symbols: Iterable[str]) -> list[Score]:
    syms = list(symbols)
    print(f"Fetching klines for {len(syms)} symbols ...")
    klines = data.fetch_many_klines(syms)
    print(f"Got {len(klines)} valid series. Scoring ...")
    scores: list[Score] = []
    for sym, df in klines.items():
        snap = compute(df)
        if snap is None:
            continue
        scores.append(score_symbol(sym, snap))
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def _do_scan(state: dict) -> tuple[list[dict], list[Score]]:
    """Open positions to fill empty slots. Returns (actions, top_scores)."""
    actions: list[dict] = []
    universe = data.list_universe()
    print(f"Universe: top {len(universe)} USDT pairs by 24h volume.")
    syms = [u["symbol"] for u in universe]
    scores = _scan_universe(syms)
    if not scores:
        print("No scored candidates.")
        return actions, []

    print("\nTop 10:")
    for i, sc in enumerate(scores[:10], 1):
        _print_score_line(i, sc)

    open_slots = config.MAX_OPEN_POSITIONS - len(state["positions"])
    if open_slots <= 0:
        print(f"\nAll {config.MAX_OPEN_POSITIONS} slots filled.")
        return actions, scores[:5]

    candidates = [s for s in scores if s.symbol not in state["positions"]
                  and s.score >= config.BUY_SCORE_MIN][:open_slots]
    if not candidates:
        print(f"\nNo candidate scored >= {config.BUY_SCORE_MIN}. No trades.")
        return actions, scores[:5]

    alloc_each = state["cash"] / len(candidates)
    print(f"\nOpening {len(candidates)} paper position(s):")
    for sc in candidates:
        if alloc_each < 10:
            print(f"  {sc.symbol}: skipping (alloc {_fmt_money(alloc_each)} too small)")
            continue
        price = sc.snapshot.close
        trade = portfolio.open_position(state, sc.symbol, price, alloc_each,
                                        reason=f"score={sc.score} {sc.signal}")
        print(f"  BUY {sc.symbol}  qty={trade['qty']:.6g}  @ {price:.6g}  "
              f"alloc={_fmt_money(alloc_each)}  score={sc.score}")
        _print_breakdown(sc)
        actions.append({"side": "BUY", "symbol": sc.symbol, "price": price,
                        "score": sc.score, "signal": sc.signal,
                        "alloc_usdt": alloc_each, "reason": "open"})
    return actions, scores[:5]


def _do_check(state: dict) -> list[dict]:
    """Re-score holdings, enforce SL/TP, replace if better candidate found."""
    actions: list[dict] = []
    held = list(state["positions"].keys())
    print(f"Holding: {held}")

    klines = data.fetch_many_klines(held)
    held_scores: dict[str, Score] = {}
    current_prices: dict[str, float] = {}
    for sym in held:
        df = klines.get(sym)
        snap = compute(df) if (df is not None and len(df) >= 60) else None
        if snap is None:
            try:
                current_prices[sym] = data.fetch_last_price(sym)
            except Exception:
                current_prices[sym] = state["positions"][sym]["entry_price"]
            continue
        sc = score_symbol(sym, snap)
        held_scores[sym] = sc
        current_prices[sym] = snap.close

    # 1. Hard rules: SL / TP / trailing stop
    for sym in list(state["positions"].keys()):
        pos = state["positions"][sym]
        price = current_prices.get(sym, pos["entry_price"])
        portfolio.update_peak(state, sym, price)
        pnl_pct = portfolio.position_pnl_pct(pos, price)
        forced = None
        if pnl_pct <= config.STOP_LOSS_PCT:
            forced = f"STOP_LOSS {pnl_pct*100:.2f}%"
        elif config.TAKE_PROFIT_PCT > 0 and pnl_pct >= config.TAKE_PROFIT_PCT:
            forced = f"TAKE_PROFIT {pnl_pct*100:.2f}%"
        elif portfolio.trailing_stop_hit(pos, price):
            forced = f"TRAILING_STOP from peak {pos['peak_price']:.6g}"
        if forced:
            t = portfolio.close_position(state, sym, price, forced)
            print(f"  SELL {sym} @ {price:.6g}  reason={forced}  "
                  f"pnl={_fmt_money(t['pnl_usdt'])}")
            actions.append({"side": "SELL", "symbol": sym, "price": price,
                            "pnl_usdt": t["pnl_usdt"], "pnl_pct": pnl_pct * 100,
                            "reason": forced})

    # 2. Signal-based exits on remaining positions
    for sym, pos in list(state["positions"].items()):
        sc = held_scores.get(sym)
        price = current_prices.get(sym, pos["entry_price"])
        pnl_pct = portfolio.position_pnl_pct(pos, price) * 100
        if sc is None:
            print(f"  {sym}: no signal data, holding. price={price:.6g}  pnl={_fmt_pct(pnl_pct)}")
            continue
        _print_score_line(0, sc)
        _print_breakdown(sc)
        print(f"     pnl={_fmt_pct(pnl_pct)}")
        if sc.signal in ("SELL", "WEAK_SELL") or sc.score < config.HOLD_SCORE_MIN:
            t = portfolio.close_position(state, sym, price,
                                         reason=f"signal={sc.signal} score={sc.score}")
            print(f"  -> SELL {sym} @ {price:.6g}  pnl={_fmt_money(t['pnl_usdt'])}")
            actions.append({"side": "SELL", "symbol": sym, "price": price,
                            "pnl_usdt": t["pnl_usdt"], "pnl_pct": pnl_pct,
                            "reason": f"weak signal score={sc.score}"})
        else:
            print(f"  -> HOLD {sym}  score={sc.score}")
            actions.append({"side": "HOLD", "symbol": sym, "price": price,
                            "pnl_pct": pnl_pct, "score": sc.score,
                            "reason": "score still strong"})

    # 3. Replacement scan: fill open slots
    open_slots = config.MAX_OPEN_POSITIONS - len(state["positions"])
    if open_slots > 0:
        print(f"\n{open_slots} slot(s) open — scanning universe for replacements ...")
        universe = data.list_universe()
        syms = [u["symbol"] for u in universe if u["symbol"] not in state["positions"]]
        scores = _scan_universe(syms)
        held_min = min((held_scores[s].score for s in state["positions"] if s in held_scores),
                       default=0)
        cands: list[Score] = []
        for sc in scores:
            if sc.score < config.BUY_SCORE_MIN:
                continue
            if held_min and sc.score < held_min + config.REPLACE_MARGIN:
                continue
            cands.append(sc)
            if len(cands) >= open_slots:
                break
        if not cands:
            margin_note = (f", and beat held by {config.REPLACE_MARGIN}"
                           if held_min else "")
            print(f"  No candidate clears bar (score >= {config.BUY_SCORE_MIN}{margin_note}).")
        else:
            alloc_each = state["cash"] / len(cands)
            for sc in cands:
                if alloc_each < 10:
                    continue
                price = sc.snapshot.close
                portfolio.open_position(state, sc.symbol, price, alloc_each,
                                        reason=f"REPLACE score={sc.score} {sc.signal}")
                print(f"  BUY {sc.symbol} @ {price:.6g}  alloc={_fmt_money(alloc_each)}  "
                      f"score={sc.score}")
                _print_breakdown(sc)
                actions.append({"side": "BUY", "symbol": sc.symbol, "price": price,
                                "score": sc.score, "signal": sc.signal,
                                "alloc_usdt": alloc_each, "reason": "replacement"})
    return actions


def _print_status_short(state: dict) -> dict:
    prices = {}
    for sym in state["positions"]:
        try:
            prices[sym] = data.fetch_last_price(sym)
        except Exception:
            prices[sym] = state["positions"][sym]["entry_price"]
    eq = portfolio.equity(state, prices)
    print(f"\n  cash={_fmt_money(eq['cash'])}  "
          f"holdings={_fmt_money(eq['holdings_value'])}  "
          f"equity={_fmt_money(eq['total_equity'])}  "
          f"return={_fmt_pct(eq['total_return_pct'])}")
    return eq


# ---------- Telegram message formatting ----------

_SIGNAL_TR = {
    "BUY": "GÜÇLÜ AL",
    "WEAK_BUY": "ZAYIF AL",
    "NEUTRAL": "NÖTR",
    "WEAK_SELL": "ZAYIF SAT",
    "SELL": "GÜÇLÜ SAT",
}


def _tr_signal(sig: str) -> str:
    return _SIGNAL_TR.get(sig, sig)


def _tr_reason(reason: str) -> str:
    """Translate internal reason strings to Turkish at the display boundary.
    Kept separate so state.json keeps stable English keys for debugging."""
    if reason.startswith("STOP_LOSS"):
        return reason.replace("STOP_LOSS", "ZARAR KES")
    if reason.startswith("TAKE_PROFIT"):
        return reason.replace("TAKE_PROFIT", "KÂR AL")
    if reason.startswith("TRAILING_STOP from peak"):
        return reason.replace("TRAILING_STOP from peak", "TRAILING zirveden")
    if reason == "open":
        return "yeni pozisyon"
    if reason == "replacement":
        return "yer değiştirme"
    if reason == "score still strong":
        return "skor güçlü"
    if reason.startswith("weak signal"):
        return reason.replace("weak signal", "zayıf sinyal").replace("score=", "skor=")
    return reason


def _format_telegram(actions: list[dict], eq: dict,
                    top_scores: list[Score] | None = None) -> str:
    """Build a Telegram message in Turkish. Always returns a string (heartbeat on quiet runs)."""
    actionable = [a for a in actions if a["side"] in ("BUY", "SELL")]
    holds = [a for a in actions if a["side"] == "HOLD"]

    lines = []
    if actionable:
        lines.append("<b>🔔 Coin Bot — Sinyal</b>")
    elif holds:
        lines.append("<b>📊 Coin Bot — Durum (Tut)</b>")
    else:
        lines.append("<b>⏳ Coin Bot — Tarama (İşlem Yok)</b>")

    for a in actions:
        sym = a["symbol"]
        if a["side"] == "BUY":
            lines.append(f"🟢 AL <b>{sym}</b> @ {a['price']:.6g}  "
                         f"skor={a['score']} {_tr_signal(a['signal'])}  "
                         f"tutar=${a['alloc_usdt']:.2f}  ({_tr_reason(a['reason'])})")
        elif a["side"] == "SELL":
            pnl = a["pnl_usdt"]
            emoji = "🔴" if pnl < 0 else "🟢"
            lines.append(f"{emoji} SAT <b>{sym}</b> @ {a['price']:.6g}  "
                         f"kâr=${pnl:+.2f} ({a['pnl_pct']:+.2f}%)  ({_tr_reason(a['reason'])})")
        elif a["side"] == "HOLD":
            lines.append(f"⏸ TUT <b>{sym}</b> @ {a['price']:.6g}  "
                         f"skor={a['score']}  kâr={a['pnl_pct']:+.2f}%")

    # If there were no actions, show the top candidates so user sees
    # what the bot is "looking at" and why it's sitting still.
    if not actions and top_scores:
        lines.append("<i>en yüksek skorlar (AL eşiği=" + str(config.BUY_SCORE_MIN) + "):</i>")
        for sc in top_scores[:5]:
            lines.append(f"  • {sc.symbol}  skor={sc.score}  {_tr_signal(sc.signal)}  "
                         f"@{sc.snapshot.close:.6g}")

    lines.append("")
    lines.append(
        f"💼 varlık=${eq['total_equity']:.2f}  "
        f"nakit=${eq['cash']:.2f}  "
        f"getiri={eq['total_return_pct']:+.2f}%"
    )
    return "\n".join(lines)


# ---------- commands ----------

def cmd_scan() -> int:
    state = portfolio.load()
    print(f"=== SCAN ===  cash={_fmt_money(state['cash'])}  "
          f"holding={list(state['positions'].keys()) or 'none'}")
    actions, top = _do_scan(state)
    portfolio.save(state)
    eq = _print_status_short(state)
    msg = _format_telegram(actions, eq, top_scores=top)
    if notify.is_configured():
        notify.send(msg)
    return 0


def cmd_check() -> int:
    state = portfolio.load()
    if not state["positions"]:
        print("No open positions. Run 'scan' first (or 'auto').")
        return 0
    print("=== CHECK ===")
    actions = _do_check(state)
    portfolio.save(state)
    eq = _print_status_short(state)
    msg = _format_telegram(actions, eq)
    if notify.is_configured():
        notify.send(msg)
    if not [a for a in actions if a["side"] != "HOLD"]:
        print("\nNo actions this round.")
    return 0


def cmd_auto() -> int:
    """One command for the cron: scan if empty, otherwise check."""
    state = portfolio.load()
    top: list[Score] = []
    if not state["positions"]:
        print("=== AUTO (initial scan) ===")
        actions, top = _do_scan(state)
    else:
        print("=== AUTO (check) ===")
        actions = _do_check(state)
    portfolio.save(state)
    eq = _print_status_short(state)
    msg = _format_telegram(actions, eq, top_scores=top)
    if notify.is_configured():
        notify.send(msg)
    else:
        print("\n[notify] Telegram not configured — skipped.")
    return 0


def cmd_status() -> int:
    state = portfolio.load()
    prices = {}
    for sym in state["positions"]:
        try:
            prices[sym] = data.fetch_last_price(sym)
        except Exception:
            prices[sym] = state["positions"][sym]["entry_price"]
    eq = portfolio.equity(state, prices)
    print("=== STATUS ===")
    print(f"  initial:        {_fmt_money(eq['initial'])}")
    print(f"  cash:           {_fmt_money(eq['cash'])}")
    print(f"  holdings value: {_fmt_money(eq['holdings_value'])}")
    print(f"  total equity:   {_fmt_money(eq['total_equity'])}")
    print(f"  total return:   {_fmt_pct(eq['total_return_pct'])}")
    print(f"  realized pnl:   {_fmt_money(eq['realized_pnl'])}")
    if eq["positions"]:
        print("\n  open positions:")
        for p in eq["positions"]:
            print(f"    {p['symbol']:<12} qty={p['qty']:.6g}  entry={p['entry']:.6g}  "
                  f"now={p['now']:.6g}  value={_fmt_money(p['value_usdt'])}  "
                  f"pnl={_fmt_money(p['pnl_usdt'])} ({_fmt_pct(p['pnl_pct'])})")
    else:
        print("\n  no open positions.")
    trades = state["trades"]
    if trades:
        print(f"\n  recent trades ({min(len(trades), 8)} of {len(trades)}):")
        for t in trades[-8:]:
            tail = (f" pnl={_fmt_money(t['pnl_usdt'])}" if t["side"] == "SELL" else "")
            print(f"    {t['time']}  {t['side']:<4} {t['symbol']:<12} "
                  f"qty={t['qty']:.6g} @ {t['price']:.6g}{tail}  ({t['reason']})")
    return 0


def cmd_reset() -> int:
    portfolio.reset()
    print(f"State reset. Starting capital: {_fmt_money(config.INITIAL_CAPITAL_USDT)}")
    return 0


COMMANDS = {
    "auto": cmd_auto,
    "scan": cmd_scan,
    "check": cmd_check,
    "status": cmd_status,
    "reset": cmd_reset,
}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in COMMANDS:
        print(__doc__)
        return 2
    return COMMANDS[argv[1]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
