"""Tests for the three changes: guard, universe filter, stop cooldown.

Run with:  python3 test_changes.py
No pytest, no network, no pandas/ta needed for the guard and cooldown groups.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import config
import portfolio

FAILED: list[str] = []


def check(label: str, cond: bool) -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILED.append(label)


# --------------------------------------------------------------------------
def test_leverage_filter() -> None:
    print("\n=== leveraged-token filter ===")
    from data import _is_leveraged_token as lev

    # Regression: the old endswith(("UP", ...)) test ate these two.
    check("JUPUSDT is tradable", not lev("JUPUSDT"))
    check("SUPUSDT is tradable", not lev("SUPUSDT"))
    check("PUMPUSDT is tradable", not lev("PUMPUSDT"))
    check("BTCUSDT is tradable", not lev("BTCUSDT"))
    check("NEARUSDT is tradable", not lev("NEARUSDT"))

    check("BTC3LUSDT filtered", lev("BTC3LUSDT"))
    check("ETH5SUSDT filtered", lev("ETH5SUSDT"))
    check("XRPUPUSDT filtered", lev("XRPUPUSDT"))
    check("BTCDOWNUSDT filtered", lev("BTCDOWNUSDT"))
    check("ETHBULLUSDT filtered", lev("ETHBULLUSDT"))


def test_universe() -> None:
    print("\n=== universe: liquidity + ADR + deny-list ===")
    import data

    fixture = {"data": [
        {"instId": "BTC-USDT",  "volCcy24h": "1730000000", "last": "79500", "high24h": "81800",  "low24h": "78300"},
        {"instId": "ETH-USDT",  "volCcy24h": "1116000000", "last": "2450",  "high24h": "2530",   "low24h": "2430"},
        {"instId": "BNB-USDT",  "volCcy24h": "114000000",  "last": "717",   "high24h": "728",    "low24h": "706"},
        {"instId": "USDC-USDT", "volCcy24h": "4200000000", "last": "1.0",   "high24h": "1.0",    "low24h": "1.0"},
        {"instId": "USDG-USDT", "volCcy24h": "40000000",   "last": "1.0",   "high24h": "1.0",    "low24h": "1.0"},
        {"instId": "RLUSD-USDT","volCcy24h": "70000000",   "last": "1.0",   "high24h": "1.001",  "low24h": "0.999"},
        {"instId": "XAUT-USDT", "volCcy24h": "19700000",   "last": "4354",  "high24h": "4400",   "low24h": "4288"},
        {"instId": "XTSLA-USDT","volCcy24h": "3300000",    "last": "400",   "high24h": "430",    "low24h": "390"},
        {"instId": "XMU-USDT",  "volCcy24h": "3100000",    "last": "200",   "high24h": "215",    "low24h": "195"},
        {"instId": "XRP-USDT",  "volCcy24h": "86300000",   "last": "1.39",  "high24h": "1.47",   "low24h": "1.38"},
        {"instId": "XLM-USDT",  "volCcy24h": "3300000",    "last": "0.17",  "high24h": "0.181",  "low24h": "0.170"},
        {"instId": "JUP-USDT",  "volCcy24h": "5000000",    "last": "0.30",  "high24h": "0.33",   "low24h": "0.30"},
        {"instId": "BTC3L-USDT","volCcy24h": "9000000",    "last": "5",     "high24h": "6",      "low24h": "5"},
        {"instId": "TINY-USDT", "volCcy24h": "900000",     "last": "1",     "high24h": "1.3",    "low24h": "1.0"},
        {"instId": "DOGE-USDT", "volCcy24h": "93000000",   "last": "0.084", "high24h": "0.088",  "low24h": "0.0835"},
        {"instId": "ETH-BTC",   "volCcy24h": "99000000",   "last": "0.03",  "high24h": "0.032",  "low24h": "0.030"},
    ]}
    original = data._get
    data._get = lambda path, params=None, timeout=15: fixture
    try:
        got = {r["symbol"] for r in data.list_universe()}
    finally:
        data._get = original

    check("stablecoins dropped (USDC/USDG/RLUSD)",
          not ({"USDCUSDT", "USDGUSDT", "RLUSDUSDT"} & got))
    check("tokenised gold dropped (XAUT)", "XAUTUSDT" not in got)
    check("tokenised stocks dropped (XTSLA/XMU)",
          not ({"XTSLAUSDT", "XMUUSDT"} & got))
    check("leveraged dropped (BTC3L)", "BTC3LUSDT" not in got)
    check("thin volume dropped (TINY)", "TINYUSDT" not in got)
    check("non-USDT dropped (ETH-BTC)", "ETHBTC" not in got)
    check("flat coin dropped by ADR (BNB at 3.1%)", "BNBUSDT" not in got)
    check("JUP now present", "JUPUSDT" in got)
    check("real coins kept", {"BTCUSDT", "ETHUSDT", "XRPUSDT", "XLMUSDT",
                             "DOGEUSDT"} <= got)


# --------------------------------------------------------------------------
def _fresh(positions: dict) -> dict:
    st = portfolio._empty_state()
    for sym, (entry, peak, alloc) in positions.items():
        portfolio.open_position(st, sym, entry, alloc, "test")
        st["positions"][sym]["peak_price"] = peak
    portfolio.save(st)
    return st


def test_cooldown() -> None:
    print("\n=== stop-loss re-entry cooldown ===")
    st = _fresh({"AAAUSDT": (100.0, 100.0, 500.0)})
    portfolio.close_position(st, "AAAUSDT", 92.0, "STOP_LOSS -8.00%")
    check("armed after STOP_LOSS", portfolio.is_blocked(st, "AAAUSDT"))
    check("duration matches config",
          abs(portfolio.cooldown_remaining_h(st, "AAAUSDT")
              - config.STOP_COOLDOWN_HOURS) < 0.05)

    st = _fresh({"BBBUSDT": (10.0, 10.0, 500.0)})
    portfolio.close_position(st, "BBBUSDT", 13.5, "TAKE_PROFIT 35.00%")
    check("NOT armed after TAKE_PROFIT", not portfolio.is_blocked(st, "BBBUSDT"))

    st = _fresh({"CCCUSDT": (1.0, 1.2, 500.0)})
    portfolio.close_position(st, "CCCUSDT", 1.1, "TRAILING_STOP from peak 1.2")
    check("NOT armed after TRAILING_STOP", not portfolio.is_blocked(st, "CCCUSDT"))

    st = portfolio._empty_state()
    portfolio.set_cooldown(st, "OLDUSDT", hours=-1)
    portfolio.set_cooldown(st, "NEWUSDT", hours=5)
    portfolio.prune_cooldowns(st)
    check("expired entry pruned", "OLDUSDT" not in st["cooldowns"])
    check("live entry kept", "NEWUSDT" in st["cooldowns"])

    legacy = {"created_at": "x", "cash": 0.0, "initial_capital": 1000.0,
              "positions": {}, "trades": []}
    config.STATE_FILE.write_text(__import__("json").dumps(legacy))
    check("old state.json without 'cooldowns' still loads",
          portfolio.load().get("cooldowns") == {})


def test_guard() -> None:
    print("\n=== guard: hard exits against the live ticker ===")
    import guard
    guard.telegram = lambda text: True

    def run(prices: dict) -> dict:
        guard.fetch_last_price = lambda s: prices.get(s)
        guard.main()
        return portfolio.load()

    _fresh({"AAAUSDT": (100.0, 100.0, 500.0)})
    st = run({"AAAUSDT": 92.0})
    check("-8% closes the position", "AAAUSDT" not in st["positions"])
    check("-8% arms the cooldown", portfolio.is_blocked(st, "AAAUSDT"))

    _fresh({"BBBUSDT": (10.0, 10.0, 500.0)})
    st = run({"BBBUSDT": 13.5})
    check("+35% takes profit", "BBBUSDT" not in st["positions"])

    # Trailing is deliberately NOT enforced by the guard (config.py explains
    # the measurement). It must hold, and cli.py exits it at the 4h close.
    _fresh({"CCCUSDT": (1.0, 1.12, 500.0)})
    st = run({"CCCUSDT": 1.04})
    check("guard does not trail out (left to the 4h cycle)",
          "CCCUSDT" in st["positions"])
    check("but the peak is still tracked",
          abs(st["positions"]["CCCUSDT"]["peak_price"] - 1.12) < 1e-9)

    saved = config.GUARD_ENFORCES_TRAILING
    config.GUARD_ENFORCES_TRAILING = True
    _fresh({"CCCUSDT": (1.0, 1.12, 500.0)})
    st = run({"CCCUSDT": 1.04})
    config.GUARD_ENFORCES_TRAILING = saved
    check("...and still trails out when the flag is on",
          "CCCUSDT" not in st["positions"])

    _fresh({"DDDUSDT": (1.0, 1.0, 500.0)})
    st = run({})                      # price fetch returns None
    check("unreachable price never closes a position",
          "DDDUSDT" in st["positions"])

    _fresh({"EEEUSDT": (1.0, 1.0, 500.0)})
    st = run({"EEEUSDT": 0.97})
    check("-3% is held", "EEEUSDT" in st["positions"])

    _fresh({"FFFUSDT": (1.0, 1.0, 500.0)})
    st = run({"FFFUSDT": 1.05})
    check("new high updates the peak",
          abs(st["positions"]["FFFUSDT"]["peak_price"] - 1.05) < 1e-9)


# --------------------------------------------------------------------------
def test_proximity_warning() -> None:
    print("\n=== guard: proximity warning ===")
    import guard

    sent: list[str] = []
    guard.telegram = lambda text: (sent.append(text), True)[1]

    def run(prices: dict) -> dict:
        guard.fetch_last_price = lambda s: prices.get(s)
        guard.main()
        return portfolio.load()

    # entry 100 -> hard stop at 93. Peak never cleared +7%, so trailing is off.
    lvl, kind = guard.exit_level({"entry_price": 100.0, "peak_price": 102.0})
    check("stop is the level before trailing activates",
          abs(lvl - 93.0) < 1e-9 and kind == "stop")

    # peak 120 (+20%) -> trailing at 111.6, which is above the 93 stop.
    lvl, kind = guard.exit_level({"entry_price": 100.0, "peak_price": 120.0})
    check("trailing takes over once it is above the stop",
          abs(lvl - 111.6) < 1e-9 and kind == "trailing")

    # 93.9 is 0.96% above the 93 stop -> inside the 1.5% band.
    _fresh({"AAAUSDT": (100.0, 100.0, 500.0)})
    sent.clear()
    st = run({"AAAUSDT": 93.9})
    check("warns near the stop", len(sent) == 1 and "Yaklaşma" in sent[0])
    check("position stays open", "AAAUSDT" in st["positions"])
    check("warned level recorded", "warned_level" in st["positions"]["AAAUSDT"])

    # Same price again -> no second message for the same level.
    sent.clear()
    st = run({"AAAUSDT": 93.85})
    check("does not repeat at the same level", not sent)

    # Price pulls clear (>3% away) -> warning re-arms.
    sent.clear()
    st = run({"AAAUSDT": 97.0})
    check("re-arms when price pulls away",
          "warned_level" not in st["positions"]["AAAUSDT"])

    # Approaching again warns a second time.
    sent.clear()
    st = run({"AAAUSDT": 93.8})
    check("warns again on a second approach", len(sent) == 1)

    # Comfortably far -> silence.
    _fresh({"BBBUSDT": (100.0, 100.0, 500.0)})
    sent.clear()
    st = run({"BBBUSDT": 99.0})
    check("silent when far from the level", not sent)

    # A real exit must send the sale message, never the warning.
    _fresh({"CCCUSDT": (100.0, 100.0, 500.0)})
    sent.clear()
    st = run({"CCCUSDT": 92.0})
    check("selling sends the exit message, not a warning",
          len(sent) == 1 and "Hızlı Çıkış" in sent[0])


def main() -> int:
    config.STATE_FILE = pathlib.Path(tempfile.mkdtemp()) / "state.json"

    try:
        test_leverage_filter()
        test_universe()
    except ImportError as e:
        print(f"\n  [SKIP] universe tests need pandas ({e})")

    test_cooldown()
    test_guard()
    test_proximity_warning()

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

