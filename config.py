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
EXCHANGE_BASE = "https://api.binance.com"
QUOTE_ASSET = "USDT"
TIMEFRAME = "4h"
KLINE_LIMIT = 200                # ~33 days of 4h bars, enough for ADX/MA200 ~ish
UNIVERSE_SIZE = 50               # top N USDT pairs by 24h quote volume
MIN_QUOTE_VOLUME_USDT = 10_000_000
EXCLUDE_QUOTES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")  # leveraged tokens
EXCLUDE_STABLES = ("USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT", "USDPUSDT")

# --- Strategy thresholds ---
BUY_SCORE_MIN = 60               # 0..100; need >= this to enter
HOLD_SCORE_MIN = 45               # below this on a held coin -> sell candidate
REPLACE_MARGIN = 15               # candidate must beat held score by this margin to replace
ADX_TREND_MIN = 20                # below = no trend, suppress momentum signals
ADX_STRONG = 25

# --- Files ---
PROJECT_DIR = Path(__file__).resolve().parent
STATE_FILE = PROJECT_DIR / "state.json"
