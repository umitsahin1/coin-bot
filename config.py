
"""Configuration constants for the paper trading bot."""

from pathlib import Path

# --- Capital & risk ---
INITIAL_CAPITAL_USDT = 1000.0
MAX_OPEN_POSITIONS = 2
PER_POSITION_PCT = 0.50          # split capital across the 2 picks
TAKER_FEE = 0.001                # Binance spot taker fee, applied per side
STOP_LOSS_PCT = -0.07            # -7% hard stop
TAKE_PROFIT_PCT = 0.30           # +30% hard take profit (DENGELI)
TRAILING_STOP_PCT = 0.07         # sell when price drops 7% from peak
TRAIL_ACTIVATE_PCT = 0.07        # trailing kicks in once peak >= +7%

# --- Market & data ---
EXCHANGE_BASE = "https://api.binance.com"   # unused; data.py talks to OKX
QUOTE_ASSET = "USDT"
TIMEFRAME = "4h"
KLINE_LIMIT = 200                # ~33 days of 4h bars, enough for ADX/MA200 ~ish
UNIVERSE_SIZE = 50               # top N USDT pairs by 24h quote volume
MIN_QUOTE_VOLUME_USDT = 3_000_000
EXCLUDE_QUOTES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")  # leveraged tokens
EXCLUDE_STABLES = ("USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT", "USDPUSDT")

# A coin that cannot move cannot pay. ADR = (24h high - 24h low) / 24h low.
# At 4% this also removes every stablecoin and tokenised-metal pair for free
# (USDG/RLUSD/USDC sit at 0.0%, XAUT/PAXG around 2.6%), which is why they are
# not repeated in a deny-list below.
MIN_ADR_PCT = 4.0

# OKX lists tokenised US equities as X<TICKER>-USDT. They follow stock-market
# hours, gap over weekends, and are not what this bot is modelling, so they are
# excluded by name. A regex is not safe here: real crypto shares the shape
# (XRP, XLM, XPL) and XMU is Micron. Re-check this list when OKX adds more.
EXCLUDE_SYMBOLS = frozenset({
    "XMRVLUSDT", "XHOODUSDT", "XSPCXUSDT", "XSKHYUSDT", "XINTCUSDT",
    "XMSTRUSDT", "XMUUSDT", "XAVGOUSDT", "XTSLAUSDT", "XCBRSUSDT",
    "XCRCLUSDT", "XLITEUSDT", "XSNDKUSDT", "XSOXLUSDT",
})

# --- Strategy thresholds ---
BUY_SCORE_MIN = 60               # 0..100; need >= this to enter
HOLD_SCORE_MIN = 45               # below this on a held coin -> sell candidate
REPLACE_MARGIN = 15               # candidate must beat held score by this margin to replace
ADX_TREND_MIN = 20                # below = no trend, suppress momentum signals
ADX_STRONG = 25

# Should the fast guard also enforce the trailing stop, or only the hard stop
# and take-profit?
#
# Measured on 180 days of OKX history, 24 symbols (backtest.py compare):
#     guard off .................. +37.0%   bek.deger +6.14
#     guard, trailing enforced .... -5.4%   bek.deger -0.68
#     guard, trailing left to 4h .. +54.8%  bek.deger +6.49
#
# Checking a trailing stop every 30 minutes turns ordinary noise into an exit
# and cuts winners short; checking it every 4 hours means the trend really did
# turn. The hard stop is the opposite: the sooner it fires, the better. So the
# guard keeps the disaster brake and hands the trailing decision back to the
# 4h cycle. It still updates the peak on every check, which makes that 4h
# decision more accurate than it was before the guard existed.
GUARD_ENFORCES_TRAILING = False

# guard.py warns once when an open position drifts within this many percent of
# its nearest downside exit (hard stop, or the trailing level when active).
# The warning re-arms if price pulls away by more than twice this distance.
WARN_PROXIMITY_PCT = 1.5

# After a stop-loss on a symbol, refuse to re-enter it for this long. Without
# it the scanner re-bought the same falling coin hours later (ALLOUSDT went
# through 9 round trips in 8 days).
STOP_COOLDOWN_HOURS = 24

# --- Files ---
PROJECT_DIR = Path(__file__).resolve().parent
STATE_FILE = PROJECT_DIR / "state.json"
