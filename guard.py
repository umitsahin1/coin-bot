"""Fast stop-loss / take-profit / trailing guard.

Why this file exists
--------------------
`cli.py auto` runs every 4 hours and prices positions off the last *closed*
4h candle. A -7% stop therefore only fires at the next 4h boundary, and the
realised exits drifted far past the intended level:

    target -7.00%   realised avg -10.56%   worst -20.29%
    6 of 15 stops closed below -10%

This module re-checks the same hard exit rules against the *live* ticker every
few minutes. It deliberately does NOT score, fetch candles, or open positions --
that stays in cli.py. Keeping it narrow is what makes it cheap enough to run
often: stdlib only, no pandas / ta / requests, so the workflow needs no
`pip install` step at all.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

import config
import portfolio

OKX_BASE = "https://www.okx.com"
_UA = {"User-Agent": "coin-paper-bot-guard/1.0"}
_TIMEOUT = 12


# --------------------------------------------------------------------------
# tiny stdlib HTTP helpers (no requests dependency)
# --------------------------------------------------------------------------
def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _to_okx(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USDT"
    if symbol.endswith("USDC"):
        return f"{symbol[:-4]}-USDC"
    return symbol


def fetch_last_price(symbol: str) -> float | None:
    """Live ticker price, or None if OKX is unreachable / unknown symbol."""
    try:
        js = _get_json(f"{OKX_BASE}/api/v5/market/ticker?instId={_to_okx(symbol)}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"  [warn] {symbol}: price fetch failed ({e})")
        return None
    if str(js.get("code", "")) != "0" or not js.get("data"):
        print(f"  [warn] {symbol}: OKX code={js.get('code')} msg={js.get('msg')}")
        return None
    try:
        return float(js["data"][0]["last"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        print("[notify] Telegram not configured — skipped.")
        return False
    payload = json.dumps({
        "chat_id": chat, "text": text,
        "parse_mode": "HTML", "disable_web_page_preview": True,
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload, headers={"Content-Type": "application/json", **_UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            r.read()
        return True
    except Exception as e:                       # noqa: BLE001
        print(f"[notify] failed: {e}")
        return False


# --------------------------------------------------------------------------
def exit_level(pos: dict) -> tuple[float, str]:
    """Nearest downside exit for an open position.

    Two levels can close a position on the way down: the hard stop measured
    from entry, and -- once the peak gain has cleared TRAIL_ACTIVATE_PCT --
    the trailing level measured from the peak. The higher of the two is the
    one price reaches first, so that is the one worth warning about.
    """
    stop = pos["entry_price"] * (1 + config.STOP_LOSS_PCT)
    peak_gain = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
    if peak_gain >= config.TRAIL_ACTIVATE_PCT:
        trail = pos["peak_price"] * (1 - config.TRAILING_STOP_PCT)
        if trail > stop:
            # Still worth warning about even when the guard does not act on
            # it: the 4h cycle will, at the next close.
            return trail, "trailing"
    return stop, "stop"


def _tr_reason(reason: str) -> str:
    if reason.startswith("STOP_LOSS"):
        return reason.replace("STOP_LOSS", "ZARAR KES")
    if reason.startswith("TAKE_PROFIT"):
        return reason.replace("TAKE_PROFIT", "KÂR AL")
    if reason.startswith("TRAILING_STOP from peak"):
        return reason.replace("TRAILING_STOP from peak", "TRAILING zirveden")
    return reason


def main() -> int:
    state = portfolio.load()
    if not state["positions"]:
        print("No open positions — nothing to guard.")
        return 0

    print(f"Guarding {len(state['positions'])} position(s): "
          f"{list(state['positions'])}")

    sold: list[dict] = []
    warned: list[dict] = []
    changed = False

    for sym in list(state["positions"].keys()):
        pos = state["positions"][sym]
        price = fetch_last_price(sym)
        if price is None:
            # Never act on a missing price: no exit, no peak update.
            continue

        peak_before = pos["peak_price"]
        portfolio.update_peak(state, sym, price)
        if pos["peak_price"] != peak_before:
            changed = True

        pnl_pct = portfolio.position_pnl_pct(pos, price)
        reason = None
        if pnl_pct <= config.STOP_LOSS_PCT:
            reason = f"STOP_LOSS {pnl_pct*100:.2f}%"
        elif config.TAKE_PROFIT_PCT > 0 and pnl_pct >= config.TAKE_PROFIT_PCT:
            reason = f"TAKE_PROFIT {pnl_pct*100:.2f}%"
        elif (getattr(config, "GUARD_ENFORCES_TRAILING", True)
              and portfolio.trailing_stop_hit(pos, price)):
            # Off by default: see the measurement in config.py. cli.py still
            # applies the trailing stop at every 4h close.
            reason = f"TRAILING_STOP from peak {pos['peak_price']:.6g}"

        if reason:
            # close_position arms the re-entry cooldown for STOP_LOSS exits.
            t = portfolio.close_position(state, sym, price, reason)
            changed = True
            sold.append({"symbol": sym, "price": price, "reason": reason,
                         "pnl_usdt": t["pnl_usdt"], "pnl_pct": pnl_pct * 100})
            print(f"  SELL {sym} @ {price:.6g}  {reason}  "
                  f"pnl=${t['pnl_usdt']:+.2f}")
        else:
            level, kind = exit_level(pos)
            distance = (price - level) / price * 100 if price else 0.0
            print(f"  hold {sym} @ {price:.6g}  pnl={pnl_pct*100:+.2f}%  "
                  f"peak={pos['peak_price']:.6g}  "
                  f"{kind}={level:.6g} ({distance:+.2f}% away)")

            already = pos.get("warned_level")
            if 0 < distance <= config.WARN_PROXIMITY_PCT:
                # Warn once per level. If the trailing level has climbed with
                # the peak, that is a new situation and earns a fresh warning.
                if already is None or level > already * 1.005:
                    pos["warned_level"] = level
                    changed = True
                    warned.append({"symbol": sym, "price": price, "level": level,
                                   "kind": kind, "distance": distance,
                                   "pnl_pct": pnl_pct * 100,
                                   "peak": pos["peak_price"]})
            elif already is not None and distance > 2 * config.WARN_PROXIMITY_PCT:
                # Price pulled clear -- re-arm so a second approach warns again.
                del pos["warned_level"]
                changed = True

    if changed:
        portfolio.prune_cooldowns(state)
        portfolio.save(state)

    if sold:
        lines = ["<b>🛡 Coin Bot — Hızlı Çıkış</b>"]
        for a in sold:
            emoji = "🔴" if a["pnl_usdt"] < 0 else "🟢"
            lines.append(
                f"{emoji} SAT <b>{a['symbol']}</b> @ {a['price']:.6g}  "
                f"kâr=${a['pnl_usdt']:+.2f} ({a['pnl_pct']:+.2f}%)  "
                f"({_tr_reason(a['reason'])})"
            )
        lines.append("")
        lines.append(f"💰 nakit=${state['cash']:.2f}  "
                     f"açık pozisyon={len(state['positions'])}")
        telegram("\n".join(lines))
    elif warned:
        lines = ["<b>⚠️ Coin Bot — Yaklaşma Uyarısı</b>"]
        for w in warned:
            etiket = "trailing" if w["kind"] == "trailing" else "zarar kes"
            lines.append(
                f"🟠 <b>{w['symbol']}</b> @ {w['price']:.6g}  "
                f"kâr={w['pnl_pct']:+.2f}%\n"
                f"    {etiket} {w['level']:.6g} seviyesine "
                f"%{w['distance']:.2f} kaldı"
                + (f"  (zirve {w['peak']:.6g})" if w["kind"] == "trailing" else "")
            )
        telegram("\n".join(lines))
        print(f"Warned on {len(warned)} position(s).")
    else:
        print("No exit triggered.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
