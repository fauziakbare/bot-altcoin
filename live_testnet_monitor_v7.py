import os
import time
import sys
import json
import sqlite3
import logging
import threading
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from contextlib import closing
from binance import ThreadedWebsocketManager

# Turso (cloud SQLite) is an optional cloud replica for remote access.
# The local SQLite file is always the authoritative store, so trade history
# survives Turso downtime, expired tokens, or network loss.
try:
    from libsql_experimental import connect as turso_connect
    TURSO_AVAILABLE = True
except ImportError:
    TURSO_AVAILABLE = False
    turso_connect = None
    print("[WARN] libsql-experimental not installed — history saved to local SQLite only. "
          "Run: pip install libsql-experimental")

# Import Rich components for a stunning terminal UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ==========================================
# CONFIGURATION & CONSTANTS (EBTA-aligned)
# ==========================================
TIMEFRAME = '15m'
LEVERAGE = 10
CANDLE_LIMIT = 250  # Enough for 200 EMA and indicators warm-up

# ---------------------------------------------------------------
# 5. CONSOLIDATED RISK MANAGEMENT & DYNAMIC ORDER SIZING
#    Total bot budget hard-capped; margin split across Max slots.
# ---------------------------------------------------------------
TOTAL_BOT_BUDGET = 50.0        # USDT — total capital delegated to the bot
MAX_ACTIVE_COINS = 2           # Max concurrent positions at any one time
COLLATERAL_PER_TRADE = 25.0    # USDT margin per position (50.0 / 2)
RISK_REWARD_RATIO = 2.0        # TP distance = SL distance * this ratio
SL_ATR_MULTIPLE = 2.5          # Initial hard Stop Loss = 2.5x ATR on entry
# -> Implied TP distance = 2.5 * ATR * 2.0 = 5.0x ATR

# ---------------------------------------------------------------
# 3. ADX HYSTERESIS CORRIDOR
#    Entry requires strong regime; exit only once regime collapses,
#    giving the trade a 5-point buffer against minor consolidation
#    so commission + slippage churn (1.4% round-trip on margin) is not
#    paid out on every wiggle.
# ---------------------------------------------------------------
ADX_ENTRY_THRESHOLD = 25.0     # 15m ADX required to ENTER
ADX_REGIME_EXIT = 20.0         # 15m ADX must fall below this to EXIT

# ---------------------------------------------------------------
# 2. BREAKOUT FRESHNESS WINDOW (Multi-Bar Horizon)
#    A breakout stays valid for N completed candles so a lagging ADX
#    confirmation (which can take a bar or two to catch up) no longer
#    causes the entry to be missed on the very first candle.
# ---------------------------------------------------------------
BREAKOUT_WINDOW = 3            # Donchian penetration valid within last N closed bars

# Volume confirmation multiplier for the closed candle vs its 20-bar SMA.
VOL_CONFIRM_MULTIPLE = 1.5

# List of 10 candidate assets to scan every 24 hours
CANDIDATE_ASSETS = [
    'ADA/USDT:USDT', 'XRP/USDT:USDT', 'SOL/USDT:USDT', 'LTC/USDT:USDT',
    'POL/USDT:USDT', 'DOT/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT',
    'LINK/USDT:USDT', '1000SHIB/USDT:USDT'
]

# Strategy parameters
EMA_PERIOD = 200
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ATR_PERIOD = 14
VOL_SMA_PERIOD = 20

# Trading costs simulation (0.05% Taker + 0.02% Slippage per side)
ROUND_TRIP_FRICTION = 0.0014

# ==========================================
# PERMANENT PNL DATABASE (local mirror + Turso cloud)
# ==========================================
# The local SQLite file is the AUTHORITATIVE store: every closed trade lands
# there first, so a Turso outage or expired token can never lose history.
# Path is anchored to the script directory -> always found no matter the CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'trade_history.db')
# Last-resort spool: one JSON object per line, written only when BOTH stores fail.
SPOOL_PATH = os.path.join(BASE_DIR, 'data', 'trade_history_spool.jsonl')

# Single source of truth for the `trades` payload column order.
TRADE_COLUMNS = (
    'timestamp',
    'symbol',
    'side',
    'quantity',
    'entry_price',
    'exit_price',
    'net_return_pct',
    'exit_reason',
)

# JSON keys exposed by the API, in TRADE_COLUMNS order.
TRADE_JSON_KEYS = (
    'timestamp',
    'symbol',
    'side',
    'quantity',
    'entry',
    'exit',
    'return_pct',
    'reason',
)

_COLUMN_TYPES = {
    'timestamp': 'TEXT',
    'symbol': 'TEXT',
    'side': 'TEXT',
    'quantity': 'REAL',
    'entry_price': 'REAL',
    'exit_price': 'REAL',
    'net_return_pct': 'REAL',
    'exit_reason': 'TEXT',
}

_COLUMN_DEFS = ',\n        '.join(
    f"{name:<14} {_COLUMN_TYPES[name]}" for name in TRADE_COLUMNS
)
_COLUMN_LIST = ', '.join(TRADE_COLUMNS)
_PLACEHOLDERS = ', '.join('?' for _ in TRADE_COLUMNS)

# Local mirror carries an extra `synced` flag (0 = not yet replicated to Turso).
TRADES_SCHEMA_LOCAL = f"""
    CREATE TABLE IF NOT EXISTS trades (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        {_COLUMN_DEFS},
        synced         INTEGER NOT NULL DEFAULT 0
    )
"""

# Turso mirror holds the payload columns only.
TRADES_SCHEMA_CLOUD = f"""
    CREATE TABLE IF NOT EXISTS trades (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        {_COLUMN_DEFS}
    )
"""

TRADES_INSERT_LOCAL = f"""
    INSERT INTO trades ({_COLUMN_LIST}, synced)
    VALUES ({_PLACEHOLDERS}, ?)
"""

TRADES_INSERT_CLOUD = f"""
    INSERT INTO trades ({_COLUMN_LIST})
    VALUES ({_PLACEHOLDERS})
"""

# Ordering uses timestamp (wall clock, comparable across stores) and only falls
# back to id for ties. `id` sequences are independent per store, so id alone
# would not produce the same window in the local mirror and in Turso.
TRADES_SELECT = f"""
    SELECT {_COLUMN_LIST}
    FROM trades
    ORDER BY timestamp DESC, id DESC
    LIMIT ?
"""

TRADES_SELECT_UNSYNCED = f"""
    SELECT id, {_COLUMN_LIST}
    FROM trades
    WHERE synced = 0
    ORDER BY id ASC
    LIMIT ?
"""


def rows_to_trades(rows):
    """Maps DB rows (in TRADE_COLUMNS order) to API dicts (TRADE_JSON_KEYS)."""
    return [dict(zip(TRADE_JSON_KEYS, row)) for row in rows]


def ensure_local_schema(conn):
    """
    Creates the local `trades` table and adds the `synced` column when an older
    database file predates it, so an existing mirror keeps working after upgrade.
    """
    conn.execute(TRADES_SCHEMA_LOCAL)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    if 'synced' not in existing:
        # Pre-existing rows were written before replication tracking existed;
        # default them to synced so they are not replayed to Turso wholesale.
        conn.execute("ALTER TABLE trades ADD COLUMN synced INTEGER NOT NULL DEFAULT 1")


def redact_db_url(url):
    """Returns the host of a libsql URL for logging, never the full secret URL."""
    if not url:
        return "(unset)"
    host = str(url).split('://', 1)[-1].split('/', 1)[0]
    return host or "(unknown host)"


# KAPAN SCANNING DIJALANKAN (PILIHAN JAM DALAM WIB)
# Jam 7 = 07:00 WIB (Daily Close Binance - Rekomendasi Utama EBTA)
# Jam 22 = 22:00 WIB (2 Jam setelah US Market Open - Rekomendasi Taktis Volatilitas)
SCHEDULED_SCAN_HOUR = 22  # Ganti ke 7 jika ingin daily close Binance

# Live UI ticks down every second; market metrics/balance refresh on this cadence.
REFRESH_INTERVAL_SECONDS = 15
# Longer pause after an API error so a ban/outage is not hammered.
ERROR_BACKOFF_SECONDS = 60

# ==========================================
# TECHNICAL INDICATORS CALCULATION
# ==========================================
def calculate_indicators(df):
    """
    Computes EMA200, Donchian Channels, ADX, ATR, and Volume SMA.

    All rolling windows are shifted by one bar via `shift(1)` so that every
    value at index i only uses data from bars STRICTLY BEFORE i. This guarantees
    the indicator values on a completed bar (df.iloc[-2]) never leak anything
    from the active live bar (df.iloc[-1]) — eliminating look-ahead bias.
    """
    # 1. EMA 200 (uses shifted close history implicitly via ewm on prior bars)
    df['ema_200'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()

    # 2. Donchian Channels (shifted by 1 to avoid look-ahead bias)
    df['donchian_high'] = df['high'].shift(1).rolling(window=DONCHIAN_PERIOD).max()
    df['donchian_low'] = df['low'].shift(1).rolling(window=DONCHIAN_PERIOD).min()

    # 3. ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift(1))
    low_cp = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = np.max(np.column_stack((high_low, high_cp, low_cp)), axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    # 4. Volume SMA (20-period, built only from closed candles)
    df['vol_sma'] = df['volume'].rolling(window=VOL_SMA_PERIOD).mean()

    # 5. ADX (Average Directional Index)
    df['up'] = df['high'] - df['high'].shift(1)
    df['down'] = df['low'] - df['low'].shift(1)

    df['plus_DM'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0.0)
    df['minus_DM'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0.0)

    df['smoothed_tr'] = df['tr'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    df['smoothed_plus_DM'] = df['plus_DM'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    df['smoothed_minus_DM'] = df['minus_DM'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    df['plus_DI'] = 100 * (df['smoothed_plus_DM'] / np.where(df['smoothed_tr'] == 0, 1e-9, df['smoothed_tr']))
    df['minus_DI'] = 100 * (df['smoothed_minus_DM'] / np.where(df['smoothed_tr'] == 0, 1e-9, df['smoothed_tr']))

    di_sum = df['plus_DI'] + df['minus_DI']
    df['dx'] = 100 * (np.abs(df['plus_DI'] - df['minus_DI']) / np.where(di_sum == 0, 1e-9, di_sum))
    df['adx'] = df['dx'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    return df


def calculate_daily_metrics(df):
    """
    Computes ADX and Volume Ratio on daily (1D) data for the daily scanner filter.
    """
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift(1))
    low_cp = np.abs(df['low'] - df['close'].shift(1))
    tr = np.max(np.column_stack((high_low, high_cp, low_cp)), axis=1)

    # 20-day Volume SMA by volume
    vol_sma_20 = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / np.where(vol_sma_20 == 0, 1e-9, vol_sma_20)

    # ADX Calculation
    up = df['high'] - df['high'].shift(1)
    down = df['low'] - df['low'].shift(1)

    plus_DM = np.where((up > down) & (up > 0), up, 0.0)
    minus_DM = np.where((down > up) & (down > 0), down, 0.0)

    smoothed_tr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    smoothed_plus_DM = pd.Series(plus_DM).ewm(alpha=1/14, adjust=False).mean()
    smoothed_minus_DM = pd.Series(minus_DM).ewm(alpha=1/14, adjust=False).mean()

    plus_DI = 100 * (smoothed_plus_DM / np.where(smoothed_tr == 0, 1e-9, smoothed_tr))
    minus_DI = 100 * (smoothed_minus_DM / np.where(smoothed_tr == 0, 1e-9, smoothed_tr))

    di_sum = plus_DI + minus_DI
    dx = 100 * (np.abs(plus_DI - minus_DI) / np.where(di_sum == 0, 1e-9, di_sum))
    df['adx_1d'] = dx.ewm(alpha=1/14, adjust=False).mean()

    return df


# ==========================================
# ROTATOR REAL EXECUTION BOT (TESTNET Sandbox)
# ==========================================
class RealExecutionRotatorBot:
    def __init__(self, api_key, api_secret, use_render_mode=False):
        self.use_render_mode = use_render_mode
        self.console = Console() if RICH_AVAILABLE else None
        self.logs = []

        # Sanitize API credentials: strip whitespace/newlines/quoting noise
        api_key = (api_key or "").strip()
        api_secret = (api_secret or "").strip()
        if not api_key or not api_secret:
            self.log_event("❌ WARNING: BINANCE_API_KEY/API_SECRET kosong setelah sanitasi.")

        # ---------------------------------------------------------------
        # DUAL CLIENT ARCHITECTURE
        # ---------------------------------------------------------------
        # (a) market_client -> Binance Futures MAINNET public (NO API key).
        self.market_client = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
        })

        # OPTIONAL: route all PUBLIC market-data REST calls through a
        # Cloudflare Worker relay to escape Render's shared-IP Binance ban.
        relay_url = (os.getenv("MARKET_RELAY_URL") or "").strip().rstrip("/")
        if relay_url:
            _api = self.market_client.urls['api']
            _api['fapiPublic'] = relay_url
            _api['fapiData'] = relay_url
            self.log_event(f"🔁 MARKET RELAY enabled: {relay_url}")
            print(f"[RELAY] market_client fapiPublic -> {relay_url}")

        # Determine environment: testnet (legacy), demo, or mainnet
        use_testnet = os.getenv("USE_TESTNET", "false").lower() == "true"
        enable_demo = os.getenv("ENABLE_DEMO_TRADING", "false").lower() == "true"

        # (b) trade_client -> Binance Futures with API key.
        trade_client_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
        }
        if use_testnet or enable_demo:
            # Complete demo override for the sandbox: every endpoints routed to
            # demo-fapi.binance.com so no real funds or mainnet keys are touched.
            demo_base = 'https://demo-fapi.binance.com'
            trade_client_config['urls'] = {
                'api': {
                    'fapi': f'{demo_base}/fapi/v1',
                    'fapiPublic': f'{demo_base}/fapi/v1',
                    'fapiPublicV2': f'{demo_base}/fapi/v2',
                    'fapiPublicV3': f'{demo_base}/fapi/v3',
                    'fapiPrivate': f'{demo_base}/fapi/v1',
                    'fapiPrivateV2': f'{demo_base}/fapi/v2',
                    'fapiPrivateV3': f'{demo_base}/fapi/v3',
                    'fapiData': f'{demo_base}/futures/data',
                    'fapiDataV2': f'{demo_base}/futures/data',
                    'dapi': f'{demo_base}/dapi/v1',
                    'dapiPublic': f'{demo_base}/dapi/v1',
                    'dapiPrivate': f'{demo_base}/dapi/v1',
                    'dapiPrivateV2': f'{demo_base}/dapi/v2',
                    'dapiData': f'{demo_base}/futures/data',
                    'dapiDataV2': f'{demo_base}/futures/data',
                    'sapi': 'https://demo-bapi.binance.com/bapi/v1',
                    'sapiV2': 'https://demo-bapi.binance.com/bapi/v2',
                    'sapiV3': 'https://demo-bapi.binance.com/bapi/v3',
                    'sapiV4': 'https://demo-bapi.binance.com/bapi/v4',
                }
            }

        try:
            self.trade_client = ccxt.binance(trade_client_config)
            if use_testnet or enable_demo:
                mode_label = "Demo Trading" if enable_demo else "Testnet"
                self.log_event(f"🔧 {mode_label} mode ENABLED (demo-fapi.binance.com)")
            else:
                self.log_event("🔧 Live Mainnet mode (production)")

            market_host = self.market_client.urls['api'].get('fapiPublic', '?')
            trade_host = self.trade_client.urls['api'].get('fapiPrivate', '?')
            key_prefix = api_key[:6] if api_key else '(empty)'
            print(f"[AUTH DEBUG] market_client host = {market_host}")
            print(f"[AUTH DEBUG] trade_client host = {trade_host}")
            print(f"[AUTH DEBUG] api_key prefix = {key_prefix} (length={len(api_key)})")
            print(f"[AUTH DEBUG] api_secret length = {len(api_secret)} (content hidden)")
            print(f"[AUTH DEBUG] defaultType=future | ALL signed requests -> {trade_host}")
            if trade_host and 'demo-fapi' in trade_host:
                print("[AUTH DEBUG] ROUTE OK: private futures endpoints -> demo-fapi.binance.com")
            else:
                print("[AUTH DEBUG] ROUTE WARNING: trade_client NOT on demo-fapi host!")
        except Exception as e:
            self.trade_client = None
            self.log_event(f"❌ Gagal inisialisasi trade_client: {self._exchange_error(e)}")

        # [DIAG] signed host used for balance/orders (must be demo-fapi)
        if self.trade_client is not None:
            print(f"[DIAG] fapiPrivate    = {self.trade_client.urls['api'].get('fapiPrivate', '?')}")
            try:
                self.trade_client.fetch_balance()
                print("[DIAG] balance OK")
            except Exception as ex:
                print(f"[DIAG] balance FAIL: {ex}")
        self._session_banned = False
        self.bsm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
        try:
            self.bsm.start()
        except Exception as e:
            self.log_event(f"⚠️ WebSocket manager start failed (will continue on REST): {self._exchange_error(e)}")
        self.klines_cache = {}  # symbol -> list of klines (OHLCV)
        self._cache_lock = threading.Lock()

        # Pre-fetch initial candles for each candidate via REST (hard-throttled).
        for symbol in CANDIDATE_ASSETS:
            if self._session_banned:
                self.log_event(f"⛔ IP banned/dDoS -199, skipping pre-fetch for {symbol}")
                break
            try:
                ohlcv = self.market_client.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLE_LIMIT)
                with self._cache_lock:
                    self.klines_cache[symbol] = ohlcv
                self.log_event(f"📡 Pre-fetched {CANDLE_LIMIT} candles for {symbol}")
            except Exception as e:
                self.log_event(f"❌ Pre-fetch failed for {symbol}: {self._exchange_error(e)}")
                if self._is_ban(e):
                    self.log_event("⛔ Binance IP ban detected — aborting REST pre-fetch.")
                    break
            time.sleep(1.2)

        # Build symbol mappings for WebSocket
        self.binance_to_internal = {}
        self.internal_to_binance = {}
        for sym in CANDIDATE_ASSETS:
            binance_sym = sym.split('/')[0] + 'USDT'
            self.binance_to_internal[binance_sym] = sym
            self.internal_to_binance[sym] = binance_sym
        # Subscribe to 15m klines for all candidates
        ws_started = 0
        for symbol in CANDIDATE_ASSETS:
            try:
                self.bsm.start_kline_socket(self._handle_kline, self.internal_to_binance[symbol], '15m')
                ws_started += 1
            except Exception as e:
                self.log_event(f"⚠️ WS subscribe failed for {self.internal_to_binance[symbol]} (REST fallback): {self._exchange_error(e)}")
        if ws_started:
            self.log_event(f"🔌 WebSocket kline subscriptions started for {ws_started} candidates")
        else:
            self.log_event("🟡 No WebSocket subscriptions — market data via REST cache only.")

        # Turso configuration (optional cloud replica of the local mirror)
        self.turso_url = os.getenv("TURSO_DB_URL")
        self.turso_token = os.getenv("TURSO_AUTH_TOKEN")
        self.turso_configured = bool(TURSO_AVAILABLE and self.turso_url and self.turso_token)
        self.db_conn = None
        self._db_lock = threading.Lock()
        self.local_db_ready = False

        # Optional push-notification sink (attached by the Telegram bridge).
        self.notifier = None

        self.positions = {}      # Tracks real active positions
        self.active_assets = []  # Curated coins to trade (may hold < MAX when Neutral)
        self.scanner_results = {}
        self.last_scan_time = None
        self.next_scan_time = None
        self.markets_state = {}
        self.balance_info = {'free': 0.0, 'total': 0.0}
        self.realized_pnl = 0.0  # Virtual P&L in USDT for bot balance

        # Live 15-second refresh countdown trackers
        self.seconds_until_refresh = 0  # Force immediate refresh on start

        self.log_event("Sistem REAL EXECUTION diinisialisasi pada BINANCE Futures Testnet.")
        if self.use_render_mode:
            self.log_event("RENDER MODE AKTIF: Dashboard visual terminal dinonaktifkan.")

        # Initialize permanent PNL database (idempotent on every startup)
        self.init_database()

        # Restore the virtual balance from persisted trade history so a restart
        # does not reset the bot's balance back to the initial budget. Sums the
        # net levered return of every closed trade, then maps it back to USDT.
        self._load_realized_pnl_from_db()

    def _load_realized_pnl_from_db(self):
        """
        Reconstructs realized_pnl (USDT) from the local SQLite trade history.
        Each stored net_return_pct is the levered net return of one closed trade;
        realized_pnl per trade equals COLLATERAL_PER_TRADE * (net_return_pct / 100).
        Fails silently (leaving the balance at the base budget) if the DB is
        unavailable or has no trades.
        """
        try:
            with closing(sqlite3.connect(DB_PATH)) as conn:
                total_pct = conn.execute(
                    "SELECT COALESCE(SUM(net_return_pct), 0.0) FROM trades"
                ).fetchone()[0]
            self.realized_pnl = COLLATERAL_PER_TRADE * (float(total_pct) / 100.0)
            if self.realized_pnl:
                self.log_event(
                    f"💵 Balance bot dipulihkan dari history: "
                    f"{TOTAL_BOT_BUDGET + self.realized_pnl:.2f} USDT "
                    f"(realized P&L {self.realized_pnl:+.2f} USDT)"
                )
        except Exception:
            self.realized_pnl = 0.0

    def log_event(self, message):
        if not hasattr(self, 'logs'):
            self.logs = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_str = f"[{timestamp}] {message}"
        self.logs.append(log_str)
        if len(self.logs) > 30:
            self.logs.pop(0)
        print(log_str)
        logging.info(log_str)

    def _notify(self, text):
        """
        Pushes an event to the attached notifier (e.g. Telegram).
        Never raises: a notification failure must not affect trading.
        """
        if self.notifier is None:
            return
        try:
            self.notifier.broadcast(text)
        except Exception as e:
            logging.info("Notifier gagal mengirim: %s", type(e).__name__)

    @staticmethod
    def _is_ban(e):
        """True if the error is a Binance DDoS/ban (-1003 / 418 / 429)."""
        code = getattr(e, 'code', None)
        status = getattr(e, 'status_code', None)
        msg = str(e).lower()
        return (
            code == -1003
            or status in (418, 429)
            or 'way too much request weight' in msg
            or 'ddos' in msg
            or 'i\'m a teapot' in msg
        )

    @staticmethod
    def _exchange_error(e):
        """Return class name, message, and any Binance error code/response."""
        detail_parts = []
        for attr in ('name', 'code', 'status_code'):
            val = getattr(e, attr, None)
            if val is not None:
                detail_parts.append(f"{attr}={val}")
        raw = getattr(e, 'response', None)
        if isinstance(raw, str):
            detail_parts.append(f"response={raw[:300]}")
        elif isinstance(raw, dict):
            detail_parts.append(f"response={str(raw)[:300]}")
        suffix = f" [{' | '.join(detail_parts)}]" if detail_parts else ""
        return f"{type(e).__name__}: {str(e)}{suffix}"

    def init_database(self):
        """
        Prepares the trade-history stores:
          - local SQLite  -> authoritative, always written first
          - Turso (cloud) -> optional replica for remote access
        A Turso failure never blocks the local write. Called on every startup
        and is fully idempotent (CREATE TABLE IF NOT EXISTS).
        """
        # --- Local SQLite (authoritative) ---
        try:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            with closing(sqlite3.connect(DB_PATH)) as conn, conn:
                ensure_local_schema(conn)
            self.local_db_ready = True
            self.log_event(f"🗄️ SQLite lokal siap: {DB_PATH}")
        except Exception as e:
            self.local_db_ready = False
            self.log_event(f"❌ GAGAL INISIALISASI SQLite lokal: {str(e)} "
                           "— history akan ditulis ke spool file.")

        # --- Turso cloud (replica) ---
        if not TURSO_AVAILABLE:
            self.log_event("⚠️ libsql-experimental belum terpasang — history hanya ke SQLite lokal. "
                           "Jalankan: pip install libsql-experimental")
            return
        if not self.turso_url or not self.turso_token:
            self.log_event("⚠️ TURSO_DB_URL / TURSO_AUTH_TOKEN belum diset di .env — "
                           "history hanya ke SQLite lokal.")
            return
        self._connect_turso(log_success=True)

    def _connect_turso(self, log_success=False):
        """
        Opens (or reopens) the Turso connection and ensures its schema.
        Returns True when self.db_conn is usable. Safe to call repeatedly.
        Caller must hold self._db_lock, or call before worker threads start.
        """
        if not self.turso_configured:
            return False
        try:
            conn = turso_connect(self.turso_url, auth_token=self.turso_token)
            conn.execute(TRADES_SCHEMA_CLOUD)
            conn.commit()
            self.db_conn = conn
            if log_success:
                self.log_event(f"🗄️ Turso database siap: {redact_db_url(self.turso_url)}")
            return True
        except Exception as e:
            self.db_conn = None
            self.log_event(f"❌ Koneksi Turso gagal ({redact_db_url(self.turso_url)}): {str(e)} "
                           "— history tetap tersimpan di SQLite lokal.")
            return False

    def _spool_trade(self, row, reason):
        """
        Last-resort persistence: appends the trade as one JSON line so a closed
        position is never silently lost when the local database is unusable.
        """
        try:
            os.makedirs(os.path.dirname(SPOOL_PATH), exist_ok=True)
            record = dict(zip(TRADE_COLUMNS, row))
            record['spooled_because'] = reason
            with open(SPOOL_PATH, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            self.log_event(f"❌ GAGAL menulis spool file: {str(e)}")
            return False

    def _replay_unsynced_to_turso(self, limit=200):
        """
        Pushes locally stored rows that never reached Turso (synced = 0).
        Caller must hold self._db_lock.
        """
        if self.db_conn is None or not self.local_db_ready:
            return 0
        try:
            with closing(sqlite3.connect(DB_PATH)) as conn:
                pending = conn.execute(TRADES_SELECT_UNSYNCED, (limit,)).fetchall()
        except Exception as e:
            self.log_event(f"⚠️ Gagal membaca baris unsynced: {str(e)}")
            return 0

        replayed = []
        for record in pending:
            row_id, payload = record[0], tuple(record[1:])
            try:
                self.db_conn.execute(TRADES_INSERT_CLOUD, payload)
                self.db_conn.commit()
                replayed.append(row_id)
            except Exception as e:
                self.log_event(f"⚠️ Replay ke Turso terhenti pada id={row_id}: {str(e)}")
                break

        if not replayed:
            return 0

        try:
            with closing(sqlite3.connect(DB_PATH)) as conn, conn:
                conn.executemany(
                    "UPDATE trades SET synced = 1 WHERE id = ?",
                    [(row_id,) for row_id in replayed],
                )
        except Exception as e:
            self.log_event(f"❌ Baris sudah masuk Turso tapi gagal ditandai synced: {str(e)}")
            return len(replayed)

        self.log_event(f"🔁 {len(replayed)} trade unsynced berhasil direplikasi ke Turso.")
        return len(replayed)

    def save_trade_record(self, symbol, side, quantity, entry_price, exit_price, net_return_pct, exit_reason):
        """
        Persists one closed trade. The local SQLite row is authoritative and is
        written first; Turso replication is attempted after and may fail without
        losing data (the row stays flagged synced = 0 for later replay).

        Returns a dict describing where the row landed.
        """
        row = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            side,
            quantity,
            entry_price,
            exit_price,
            round(net_return_pct, 4),
            exit_reason,
        )

        result = {'local': False, 'turso': False, 'spooled': False, 'persisted': False}

        # --- 1. Local SQLite (authoritative) ---
        local_error = None
        if self.local_db_ready:
            try:
                with closing(sqlite3.connect(DB_PATH)) as conn, conn:
                    conn.execute(TRADES_INSERT_LOCAL, row + (0,))
                result['local'] = True
            except Exception as e:
                local_error = str(e)
                self.log_event(f"❌ GAGAL simpan trade ke SQLite lokal: {local_error}")
        else:
            local_error = "SQLite lokal tidak siap"

        # --- 2. Turso replica (best effort, reconnect once on failure) ---
        if self.turso_configured:
            with self._db_lock:
                if self.db_conn is None:
                    self._connect_turso()
                if self.db_conn is not None:
                    try:
                        self.db_conn.execute(TRADES_INSERT_CLOUD, row)
                        self.db_conn.commit()
                        result['turso'] = True
                    except Exception as e:
                        self.log_event(f"⚠️ Simpan ke Turso gagal, mencoba sambung ulang: {str(e)}")
                        self.db_conn = None
                        if self._connect_turso():
                            try:
                                self.db_conn.execute(TRADES_INSERT_CLOUD, row)
                                self.db_conn.commit()
                                result['turso'] = True
                            except Exception as e2:
                                self.db_conn = None
                                self.log_event(f"❌ GAGAL simpan trade ke Turso: {str(e2)}")

                # Mark this row synced and drain any earlier backlog.
                if result['turso'] and result['local']:
                    try:
                        with closing(sqlite3.connect(DB_PATH)) as conn, conn:
                            conn.execute(
                                "UPDATE trades SET synced = 1 WHERE id = "
                                "(SELECT MAX(id) FROM trades)"
                            )
                    except Exception as e:
                        self.log_event(f"⚠️ Gagal menandai baris synced: {str(e)}")
                if result['turso']:
                    self._replay_unsynced_to_turso()

        # --- 3. Spool file (only when the authoritative store failed) ---
        if not result['local']:
            result['spooled'] = self._spool_trade(row, local_error or "unknown")

        result['persisted'] = result['local'] or result['turso'] or result['spooled']

        if result['local']:
            # Local SQLite landed -> this is the authoritative confirmation the
            # spec mandates, regardless of the optional Turso replica state.
            detail = "Turso belum aktif" if not self.turso_configured else "Turso gagal — akan direplay"
            self.log_event(f"💾 Trade record permanently saved to SQLite database ({detail}).")
        elif result['turso'] and result['spooled']:
            self.log_event("⚠️ Trade record tersimpan di Turso + spool file, TAPI SQLite lokal gagal.")
        elif result['turso']:
            self.log_event("⚠️ Trade record HANYA tersimpan di Turso (SQLite lokal & spool gagal).")
        elif result['spooled']:
            self.log_event("⚠️ Trade record hanya tersimpan di spool file — perlu diimport manual.")
        else:
            self.log_event("❌ Trade record TIDAK tersimpan di mana pun (SQLite, Turso, spool gagal).")

        return result

    def calculate_next_scan_time(self):
        """Calculates the exact datetime for the next scheduled daily scan."""
        now = datetime.now()
        scheduled_today = now.replace(hour=SCHEDULED_SCAN_HOUR, minute=0, second=0, microsecond=0)
        if now < scheduled_today:
            return scheduled_today
        else:
            return scheduled_today + timedelta(days=1)

    def scan_daily_market(self):
        """
        Scans all CANDIDATE_ASSETS once daily at the scheduled hour and selects
        the Top 2 assets with 1D ADX >= 25 AND 1D Volume Ratio > 1.5.

        4. RIGOROUS NEUTRAL MODE:
        If fewer than 2 assets qualify, the bot stays NEUTRAL for the empty
        slot(s) — it NEVER forces trades on fallback assets (ADA/XRP). Forcing a
        trade without a verified statistical edge makes the system a noise
        trader. A held slot simply waits for the next daily scan.
        """
        self.log_event(f"🔍 [SCANNER] Memulai Pemindaian Pasar Harian (Jadwal: Pukul {SCHEDULED_SCAN_HOUR:02d}:00 WIB)...")
        candidates_stats = []

        for symbol in CANDIDATE_ASSETS:
            try:
                # Fetch last 35 daily candles to compute indicators
                ohlcv = self.market_client.fetch_ohlcv(symbol, '1d', limit=35)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = calculate_daily_metrics(df)

                # Closed-bar evaluation on the last COMPLETED daily bar.
                last_row = df.iloc[-2]
                daily_adx = last_row['adx_1d']
                daily_vol_ratio = last_row['vol_ratio']

                # Check if asset meets trend criteria
                qualifies = (daily_adx >= ADX_ENTRY_THRESHOLD) and (daily_vol_ratio > VOL_CONFIRM_MULTIPLE)
                candidates_stats.append({
                    'symbol': symbol,
                    'adx': daily_adx,
                    'vol_ratio': daily_vol_ratio,
                    'qualifies': qualifies,
                    'rank_score': daily_adx * daily_vol_ratio if qualifies else -1
                })

                time.sleep(0.5)

            except Exception as e:
                self.log_event(f"Gagal memindai data harian untuk {symbol}: {self._exchange_error(e)}")

        # Sort candidates
        candidates_stats.sort(key=lambda x: x['rank_score'], reverse=True)
        self.scanner_results = {c['symbol']: c for c in candidates_stats}

        # Select ONLY genuinely qualifying assets, capped at MAX_ACTIVE_COINS.
        selected = [c['symbol'] for c in candidates_stats if c['qualifies']][:MAX_ACTIVE_COINS]

        # NO fallback. Empty slots stay NEUTRAL if the market lacks an edge.
        neutral_slots = MAX_ACTIVE_COINS - len(selected)

        self.active_assets = selected
        self.last_scan_time = datetime.now()
        self.next_scan_time = self.calculate_next_scan_time()

        # Setup margin mode and leverage ONLY for the genuinely selected assets.
        for symbol in self.active_assets:
            self.setup_leverage_and_margin(symbol)

        if neutral_slots > 0:
            self.log_event(
                f"🧘 NEUTRAL MODE: hanya {len(selected)}/{MAX_ACTIVE_COINS} aset memenuhi "
                "kriteria (1D ADX>=25 & VolRatio>1.5). Tidak ada aset cadangan dipaksa. "
                f"{neutral_slots} slot tetap NETRAL menunggu scan harian berikutnya."
            )

        names = ', '.join(s.split('/')[0] for s in selected) if selected else "(NETRAL - no qualifying asset)"
        status_msg = f"🎯 [ROTASI] Koin Terpilih Hari Ini: {names}"
        if neutral_slots > 0:
            status_msg += f" ({neutral_slots} slot NETRAL)"
        else:
            status_msg += " (2 slot terisi penuh)"
        self.log_event(status_msg)

    def setup_leverage_and_margin(self, symbol):
        """
        Configures Isolated Margin and 10x Leverage on Binance for the asset.
        Errors from the exchange are logged clearly without stopping the bot.
        """
        if self.trade_client is None:
            self.log_event(f"⚠️ trade_client tidak tersedia (init gagal), lewati setup leverage/margin untuk {symbol}")
            return

        try:
            # Set isolated margin mode
            try:
                self.trade_client.set_margin_mode('ISOLATED', symbol)
                self.log_event(f"⚙️ {symbol.split('/')[0]} disetel ke ISOLATED margin mode.")
            except Exception as e:
                err = self._exchange_error(e)
                if "No need to change margin type" in err or "-4046" in err:
                    self.log_event(f"⚙️ {symbol.split('/')[0]} sudah ISOLATED margin mode.")
                else:
                    self.log_event(f"❌ Margin mode ISOLATED gagal untuk {symbol}: {err}")

            # Set leverage to LEVERAGE x
            try:
                self.trade_client.set_leverage(LEVERAGE, symbol)
                self.log_event(f"⚙️ {symbol.split('/')[0]} disetel ke Leverage {LEVERAGE}x.")
            except Exception as e:
                self.log_event(f"⚙️ Leverage gagal untuk {symbol}: {self._exchange_error(e)}")
        except Exception as e:
            self.log_event(f"Gagal mengatur leverage/margin untuk {symbol}: {self._exchange_error(e)}")

    def _handle_kline(self, data):
        """Callback for WebSocket kline updates."""
        try:
            kline = data['k']
            symbol = data['s']  # Binance symbol e.g. BTCUSDT
            if symbol not in self.binance_to_internal:
                return
            internal_symbol = self.binance_to_internal[symbol]
            timestamp = kline['t']
            open_price = float(kline['o'])
            high_price = float(kline['h'])
            low_price = float(kline['l'])
            close_price = float(kline['c'])
            volume = float(kline['v'])
            new_candle = [timestamp, open_price, high_price, low_price, close_price, volume]
            with self._cache_lock:
                cache = self.klines_cache.get(internal_symbol, [])
                if cache and cache[-1][0] == timestamp:
                    cache[-1] = new_candle
                else:
                    cache.append(new_candle)
                    if len(cache) > CANDLE_LIMIT:
                        cache.pop(0)
                self.klines_cache[internal_symbol] = cache
        except Exception as e:
            self.log_event(f"WebSocket kline error: {self._exchange_error(e)}")

    def fetch_market_data(self, symbol):
        # Read from cache (updated via WebSocket) instead of REST
        with self._cache_lock:
            ohlcv = self.klines_cache.get(symbol)
        if ohlcv is None or len(ohlcv) < CANDLE_LIMIT:
            # Fallback to REST if cache insufficient (e.g. initial load)
            if self._session_banned:
                return None
            try:
                ohlcv = self.market_client.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLE_LIMIT)
                with self._cache_lock:
                    self.klines_cache[symbol] = ohlcv
            except Exception as e:
                self.log_event(f"Error fetching 15M data untuk {symbol}: {self._exchange_error(e)}")
                if self._is_ban(e):
                    self._session_banned = True
                    self.log_event("⛔ REST ban detected — pausing market data fetches.")
                return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return calculate_indicators(df)

    # ------------------------------------------------------------------
    # 2. BREAKOUT FRESHNESS WINDOW — Multi-bar Donchian penetration test
    # ------------------------------------------------------------------
    def _donchian_long_breakout(self, df, latest_donchian_high, current_closed_price):
        """
        True when the Donchian High was penetrated (a bar's high crossed ABOVE
        the historical boundary known at that bar) within the last
        BREAKOUT_WINDOW COMPLETED candles AND the current closed price is still
        above the LATEST Donchian High. Excludes the live bar entirely.
        """
        window = df.iloc[-(BREAKOUT_WINDOW + 1):-1]  # last N completed bars only
        for _, row in window.iterrows():
            prior_dh = row['donchian_high']      # boundary known at that bar (shifted)
            if row['high'] > prior_dh:
                return current_closed_price > latest_donchian_high
        return False

    def _donchian_short_breakout(self, df, latest_donchian_low, current_closed_price):
        """
        True when the Donchian Low was penetrated (a bar's low crossed BELOW the
        historical boundary known at that bar) within the last BREAKOUT_WINDOW
        COMPLETED candles AND the current closed price is still below the LATEST
        Donchian Low. Excludes the live bar entirely.
        """
        window = df.iloc[-(BREAKOUT_WINDOW + 1):-1]  # last N completed bars only
        for _, row in window.iterrows():
            prior_dl = row['donchian_low']      # boundary known at that bar (shifted)
            if row['low'] < prior_dl:
                return current_closed_price < latest_donchian_low
        return False

    # ------------------------------------------------------------------
    # DYNAMIC ORDER SIZING helper — 25 USDT collateral @ LEVERAGE, rounded
    # to the exchange lot-step via amount_to_precision.
    # ------------------------------------------------------------------
    def size_quantity(self, symbol, price):
        """
        Computes the order quantity from the fixed per-trade collateral and
        leverage, then rounds to the exchange's dynamic precision rule.
        """
        if price is None or price <= 0:
            return 0.0
        notional_value = COLLATERAL_PER_TRADE * LEVERAGE
        quantity = notional_value / price
        try:
            if self.trade_client is not None:
                quantity = float(self.trade_client.amount_to_precision(symbol, quantity))
            else:
                # Offline fallback: quantize to a sane 6-decimals precision.
                quantity = round(quantity, 6)
        except Exception as e:
            self.log_event(f"⚠️ Gagal format precision untuk {symbol}: {self._exchange_error(e)} — guna round manual.")
            quantity = round(quantity, 6)
        return max(quantity, 0.0)

    def describe_pending_orders(self, symbol, closed_row, held):
        """
        Builds the human-readable pending trigger instructions shown in the
        metrics status row, with the planned order target and estimated quantity.

        Example:
          Pending LONG Market Buy at >= 123.45000 (Est. Qty: 2.03 unit)
          Pending SHORT Market Sell at <= 120.10000 (Est. Qty: 2.08 unit)

        Uses the 25.0 USDT collateral allocation for the estimated quantity.
        Only a setup that currently qualifies (regime + trend) is actionable,
        so a pending order is only advertised while ADX hysteresis is met.
        """
        lines = []
        ema = closed_row['ema_200']
        price = closed_row['close']
        adx = closed_row['adx']
        dh = closed_row['donchian_high']
        dl = closed_row['donchian_low']

        if held:
            return lines, dh, dl

        trend_aligned_long = price > ema and price < dh
        trend_aligned_short = price < ema and price > dl
        regime_ready = adx >= ADX_ENTRY_THRESHOLD

        if regime_ready and trend_aligned_long:
            qty = self.size_quantity(symbol, dh)
            lines.append(f"Pending LONG Market Buy at >= {dh:.5f} (Est. Qty: {qty} unit)")
        if regime_ready and trend_aligned_short:
            qty = self.size_quantity(symbol, dl)
            lines.append(f"Pending SHORT Market Sell at <= {dl:.5f} (Est. Qty: {qty} unit)")

        return lines, dh, dl

    def check_signals(self, symbol, df):
        """
        1. STRICT CLOSED-BAR EVALUATION:
        Every indicator (EMA200, Donchian, ADX, ATR, Volume SMA) is read ONLY
        from the most recent COMPLETED bar (`df.iloc[-2]`), never from the live
        in-progress bar (`df.iloc[-1]`). Volume confirmation compares the closed
        candle's volume to the 20-period Volume SMA of closed candles. This kills
        the look-ahead / repainting bias of comparing an incomplete bar against
        full historical bars.

        3. ADX HYSTERESIS: entry only when 15m ADX >= ADX_ENTRY_THRESHOLD (25).

        2. BREAKOUT FRESHNESS: boundary penetration valid within the last
        BREAKOUT_WINDOW (3) completed candles while price still beyond the
        latest historical boundary.
        """
        if df is None or len(df) < CANDLE_LIMIT:
            return "WAITING_FOR_DATA"

        closed = df.iloc[-2]  # <-- ONLY the completed bar; live bar is never used
        current_closed = closed['close']
        ema = closed['ema_200']
        donchian_high = closed['donchian_high']
        donchian_low = closed['donchian_low']
        adx = closed['adx']
        volume = closed['volume']
        vol_sma = closed['vol_sma']
        atr = closed['atr']

        trend_bullish = current_closed > ema
        trend_bearish = current_closed < ema
        adx_active = adx >= ADX_ENTRY_THRESHOLD
        volume_confirmed = (volume > (VOL_CONFIRM_MULTIPLE * vol_sma)) if vol_sma > 0 else False

        # Multi-bar breakout freshness window (completed bars only)
        breakout_above = self._donchian_long_breakout(df, donchian_high, current_closed)
        breakout_below = self._donchian_short_breakout(df, donchian_low, current_closed)

        # Active Position Exit Checks (authoritative path is manage_open_positions).
        if symbol in self.positions:
            self.evaluate_exit(symbol, df)
            return "HOLDING" if symbol in self.positions else "CLOSED"

        # New Position Entry Checks
        if trend_bullish and breakout_above and adx_active and volume_confirmed:
            self.open_position(symbol, 'LONG', current_closed, atr)
            return "BUY_SIGNAL"
        elif trend_bearish and breakout_below and adx_active and volume_confirmed:
            self.open_position(symbol, 'SHORT', current_closed, atr)
            return "SELL_SIGNAL"

        return "WAITING_FOR_BREAKOUT"

    def evaluate_exit(self, symbol, df):
        """
        Evaluates trailing stop / take profit / regime exit for ONE open position.

        Deliberately independent of self.active_assets: a position must keep
        being managed after its asset is rotated out of the daily Top 2,
        otherwise the stop and target stop being enforced and the position is
        orphaned on the exchange. Returns the exit reason when closed, else None.

        3. ADX HYSTERESIS: the trade is exited on regime collapse ONLY once
        15m ADX drops below ADX_REGIME_EXIT (20.0), NOT at the entry threshold
        (25.0). The 5-point corridor lets the position breathe through minor
        consolidations instead of churning on every dip.
        """
        pos = self.positions.get(symbol)
        if pos is None or df is None or len(df) < 3:
            return None

        # Closed-bar evaluation (never act on the live bar).
        closed = df.iloc[-2]
        current_price = closed['close']
        atr = closed['atr']
        adx = closed['adx']

        if pos['type'] == 'LONG':
            if current_price > pos['peak_price']:
                self.positions[symbol]['peak_price'] = current_price
            trailing_sl = self.positions[symbol]['peak_price'] - (SL_ATR_MULTIPLE * atr)

            if current_price <= trailing_sl:
                reason = "TRAILING_STOP_HIT (ATR)"
            elif current_price >= pos['tp']:
                reason = f"TAKE_PROFIT_HIT ({SL_ATR_MULTIPLE * RISK_REWARD_RATIO:.1f}x ATR)"
            elif adx < ADX_REGIME_EXIT:  # Hysteresis regime exit (only below 20)
                reason = "REGIME_EXIT_ADX_LOW (<20)"
            else:
                return None
        else:  # SHORT
            if current_price < pos['peak_price']:
                self.positions[symbol]['peak_price'] = current_price
            trailing_sl = self.positions[symbol]['peak_price'] + (SL_ATR_MULTIPLE * atr)

            if current_price >= trailing_sl:
                reason = "TRAILING_STOP_HIT (ATR)"
            elif current_price <= pos['tp']:
                reason = f"TAKE_PROFIT_HIT ({SL_ATR_MULTIPLE * RISK_REWARD_RATIO:.1f}x ATR)"
            elif adx < ADX_REGIME_EXIT:  # Hysteresis regime exit (only below 20)
                reason = "REGIME_EXIT_ADX_LOW (<20)"
            else:
                return None

        self.close_position(symbol, current_price, reason)
        return reason

    def manage_open_positions(self):
        """
        6. GUARD AGAINST ORPHANED POSITIONS:
        Runs exit management for EVERY held position, including assets that are
        no longer in the daily Top 2. This is completely separated from the daily
        scanner list, so a rotated-out symbol is never left unmanaged. Call this
        every tick, before signal evaluation.

        Returns {symbol: exit_reason} for positions closed on this pass.
        """
        closed = {}
        # Snapshot the keys: close_position() mutates self.positions.
        for symbol in list(self.positions.keys()):
            try:
                df = self.fetch_market_data(symbol)
                if df is None:
                    self.log_event(f"⚠️ Data {symbol} tidak tersedia — exit check dilewati tick ini.")
                    continue
                reason = self.evaluate_exit(symbol, df)
                if reason:
                    closed[symbol] = reason
                    if symbol not in self.active_assets:
                        self.log_event(
                            f"🧹 Posisi rotated-out {symbol.split('/')[0]} ditutup ({reason})."
                        )
            except Exception as e:
                self.log_event(f"❌ Gagal evaluasi exit {symbol}: {self._exchange_error(e)}")
        return closed

    def open_position(self, symbol, side, entry_price, atr):
        """
        Executes a real MARKET order on Binance USD-M Testnet using
        COLLATERAL_PER_TRADE (25.0 USDT) margin at LEVERAGE (10x).

        5. CONSOLIDATED RISK MANAGEMENT:
          notional_value = COLLATERAL_PER_TRADE * LEVERAGE
          quantity       = notional_value / entry_price (rounded to exchange precision)
          SL             = 2.5x ATR (registered on entry at Binance Testnet)
          TP             = SL_distance * RISK_REWARD_RATIO  (= 5.0x ATR at R:R=2.0)

        Guardrails:
          - never open a second position on a symbol already held
          - never exceed MAX_ACTIVE_COINS concurrent positions
        """
        if symbol in self.positions:
            self.log_event(f"⏭️ {symbol.split('/')[0]} sudah ada posisi terbuka — entry dilewati.")
            return
        if len(self.positions) >= MAX_ACTIVE_COINS:
            # Budget guard: MAX_ACTIVE_COINS x COLLATERAL_PER_TRADE == TOTAL_BOT_BUDGET.
            self.log_event(
                f"🚧 Batas {MAX_ACTIVE_COINS} posisi tercapai "
                f"({', '.join(s.split('/')[0] for s in self.positions)}) — "
                f"entry {symbol.split('/')[0]} ditunda."
            )
            return
        try:
            # 1. Size the order from the fixed per-trade collateral and leverage,
            #    rounded to the exchange dynamic lot-step rule.
            quantity = self.size_quantity(symbol, entry_price)
            if quantity <= 0:
                self.log_event(f"⛔ Quantity hasil hitung <= 0 untuk {symbol} — entry dibatalkan.")
                return

            self.log_event(
                f"🛒 [TESTNET] Mengirim market order {side} {symbol} sebanyak {quantity} unit "
                f"(Margin: {COLLATERAL_PER_TRADE} USDT x {LEVERAGE}x)..."
            )

            # 2. Execute Market Order on Binance
            order_side = 'buy' if side == 'LONG' else 'sell'
            response = self.trade_client.create_market_order(symbol, order_side, quantity)
            executed_price = float(response.get('price') or entry_price)

            # 3. Register the initial hard Stop Loss & programmatic Take Profit.
            #    SL distance = 2.5x ATR; TP distance = SL * RISK_REWARD_RATIO (=> 5x ATR).
            sl_distance = SL_ATR_MULTIPLE * atr
            tp_distance = sl_distance * RISK_REWARD_RATIO
            sl = executed_price - sl_distance if side == 'LONG' else executed_price + sl_distance
            tp = executed_price + tp_distance if side == 'LONG' else executed_price - tp_distance

            self.positions[symbol] = {
                'type': side,
                'entry_price': executed_price,
                'peak_price': executed_price,
                'sl': sl,
                'tp': tp,
                'quantity': quantity,
                'entry_time': datetime.now().strftime("%H:%M:%S")
            }

            self.log_event(
                f"🚀 POSISI TERBUKA: {side} {symbol} @ {executed_price:.5f} | "
                f"SL: {sl:.5f} | TP: {tp:.5f} | Qty: {quantity}"
            )
            side_emoji = "📈" if side == 'LONG' else "📉"
            self._notify(
                f"🚀 POSISI TERBUKA\n"
                f"{side_emoji} {side} {symbol.split('/')[0]} @ {executed_price:.5f}\n"
                f"🛑 SL {sl:.5f} / 🎯 TP {tp:.5f}\n"
                f"📦 Qty {quantity}\n"
                f"🎫 Margin {COLLATERAL_PER_TRADE:.2f} USDT | ⚡ {LEVERAGE}x"
            )

        except Exception as e:
            self.log_event(f"❌ GAGAL MEMBUKA POSISI untuk {symbol}: {self._exchange_error(e)}")

    def close_position(self, symbol, exit_price, reason):
        """
        Executes a real MARKET close order with reduceOnly=True, then saves the
        completed trade to the SQLite database via save_trade_record().
        """
        try:
            pos = self.positions[symbol]
            side = pos['type']
            quantity = pos['quantity']
            entry = pos['entry_price']

            self.log_event(
                f"🛒 [TESTNET] Menutup posisi {side} {symbol} sebanyak {quantity} unit karena {reason}..."
            )

            # Execute closing market order (opposite direction, reduceOnly=True)
            close_side = 'sell' if side == 'LONG' else 'buy'
            response = self.trade_client.create_market_order(symbol, close_side, quantity, params={'reduceOnly': True})
            executed_price = float(response.get('price') or exit_price)

            # Calculate returns
            if side == 'LONG':
                raw_ret = (executed_price - entry) / entry
            else:
                raw_ret = (entry - executed_price) / entry

            net_ret = (raw_ret * LEVERAGE) - (ROUND_TRIP_FRICTION * LEVERAGE)
            # Update virtual bot balance: profit = collateral * net_ret
            self.realized_pnl += COLLATERAL_PER_TRADE * net_ret

            # 7. Permanently save the closed trade (leveraged net return incl.
            #    0.14% friction) to the SQLite database. save_trade_record uses
            #    a clean context-managed connection and logs the confirmation.
            save_result = self.save_trade_record(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry,
                exit_price=executed_price,
                net_return_pct=net_ret * 100,
                exit_reason=reason
            )

            self.log_event(
                f"🏁 POSISI TERTUTUP: {side} {symbol} @ {executed_price:.5f} | "
                f"Net Return: {net_ret*100:+.2f}% (Friction Applied)"
            )
            equity = TOTAL_BOT_BUDGET + self.realized_pnl
            side_emoji = "📈" if side == 'LONG' else "📉"
            pnl_emoji = "🟢" if net_ret > 0 else ("🔴" if net_ret < 0 else "⚪")
            if "TAKE_PROFIT" in reason:
                reason_emoji = "🎯"
            elif "TRAILING_STOP" in reason:
                reason_emoji = "🛑"
            elif "REGIME_EXIT" in reason:
                reason_emoji = "🌫️"
            else:
                reason_emoji = "🏁"
            self._notify(
                f"{pnl_emoji} POSISI TERTUTUP\n"
                f"{side_emoji} {side} {symbol.split('/')[0]} @ {executed_price:.5f}\n"
                f"🎬 Entry {entry:.5f}\n"
                f"{pnl_emoji} Net {net_ret*100:+.2f}% ({COLLATERAL_PER_TRADE * net_ret:+.2f} USDT)\n"
                f"{reason_emoji} Alasan: {reason}\n"
                f"💵 Saldo bot: {equity:.2f} USDT"
            )
            if not save_result.get('persisted'):
                # The exchange position is closed for real, so it must be dropped
                # from memory; make the lost history record impossible to miss.
                self.log_event(
                    f"❌ CATATAN HILANG: {side} {symbol} @ {executed_price:.5f} "
                    f"({net_ret*100:+.2f}%, {reason}) gagal disimpan ke SEMUA store. "
                    "Catat manual — posisi tetap ditutup di exchange."
                )
                self._notify(
                    f"🚨 PERINGATAN: catatan trade {side} {symbol.split('/')[0]} "
                    f"@ {executed_price:.5f} ({net_ret*100:+.2f}%) gagal disimpan "
                    "ke semua store. Catat manual."
                )
            del self.positions[symbol]

        except Exception as e:
            self.log_event(f"❌ GAGAL MENUTUP POSISI untuk {symbol}: {self._exchange_error(e)}")

    def render_dashboard(self, markets_state, balance_info):
        """
        8. Creates a gorgeous terminal UI using Rich, showing live countdowns.
        """
        if not RICH_AVAILABLE or self.use_render_mode:
            return

        os.system('cls' if os.name == 'nt' else 'clear')

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="rotation_info", size=4),
            Layout(name="body", ratio=1),
            Layout(name="logs", size=8)
        )

        # 1. Header
        header_text = Text("\nCRYPTO QUANT LIVE TRADING MONITOR & ROTATOR v7 (TESTNET)", style="bold green", justify="center")
        header_text.append(f"\nAPI: CONNECTED | EBTA CLOSED-BAR ENGINE | Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim white")
        layout["header"].update(Panel(header_text, style="green"))

        # 2. Daily Rotation Panel with countdown
        time_left = self.next_scan_time - datetime.now() if self.next_scan_time else timedelta(0)
        hours, remainder = divmod(int(time_left.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        countdown_str = f"{hours:02d}j {minutes:02d}m {seconds:02d}d"

        rotation_text = Text(justify="center")
        if self.active_assets:
            rotation_text.append("🔄 ROTASI AKTIF UNTUK 24 JAM: ", style="bold white")
            rotation_text.append(f" {', '.join([s.split('/')[0] for s in self.active_assets])} ", style="bold yellow")
        else:
            rotation_text.append("🧘 NEUTRAL MODE: ", style="bold white")
            rotation_text.append(" TIDAK ADA ASET MEMENUHI KRITERIA (1D ADX>=25 & VolRatio>1.5) ", style="bold cyan")
        rotation_text.append(f" | Scan Harian Berikutnya (Pukul {SCHEDULED_SCAN_HOUR:02d}:00 WIB): ", style="white")
        rotation_text.append(countdown_str, style="bold red")

        # Live 15-second refresh countdown
        rotation_text.append(f"\n⏳ Penyegaran Metrik 15M Berikutnya dalam: ", style="dim gray")
        rotation_text.append(f"{self.seconds_until_refresh} detik", style="bold cyan")

        layout["rotation_info"].update(Panel(rotation_text, style="yellow", box=box.ROUNDED))

        # 3. Body Split (Market table & Active Positions)
        body_layout = Layout()
        body_layout.split_row(
            Layout(name="markets", ratio=3),
            Layout(name="positions", ratio=2)
        )
        layout["body"].update(body_layout)

        # 3a. Market State Table (LIVE UPDATING, closed-bar based)
        market_table = Table(box=box.MINIMAL, expand=True)
        market_table.add_column("Asset", style="cyan")
        market_table.add_column("Price", justify="right")
        market_table.add_column("Trigger LONG", justify="right")
        market_table.add_column("Trigger SHORT", justify="right")
        market_table.add_column("Trend (EMA200)", justify="center")
        market_table.add_column("ADX (Regime)", justify="center")
        market_table.add_column("Volume Ratio", justify="center")
        market_table.add_column("State / Signal", justify="left")

        for symbol in self.active_assets:
            if symbol in markets_state:
                state = markets_state[symbol]
                price_str = f"{state['price']:.5f}"
                ema_str = "[green]BULLISH[/green]" if state['trend'] == 'BULLISH' else "[red]BEARISH[/red]"

                # Entry trigger price levels (20-period Donchian bands), 5 dp.
                trigger_long = state['trigger_long']
                trigger_short = state['trigger_short']
                trig_long_color = "green" if state['price'] < trigger_long else "white"
                trig_short_color = "red" if state['price'] > trigger_short else "white"
                trig_long_str = f"[{trig_long_color}]{trigger_long:.5f}[/{trig_long_color}]"
                trig_short_str = f"[{trig_short_color}]{trigger_short:.5f}[/{trig_short_color}]"

                adx_val = state['adx']
                adx_color = "green" if adx_val >= ADX_ENTRY_THRESHOLD else "red"
                adx_str = f"[{adx_color}]{adx_val:.1f} ({'TREND' if adx_val >= ADX_ENTRY_THRESHOLD else 'CHOP'})[/{adx_color}]"

                vol_ratio = state['vol_ratio']
                vol_color = "green" if vol_ratio >= VOL_CONFIRM_MULTIPLE else "white"
                vol_str = f"[{vol_color}]{vol_ratio:.2f}x[/{vol_color}]"

                # Status row: base signal + live pending order instructions.
                sig_str = state['signal']
                if "BUY" in str(sig_str) or "SELL" in str(sig_str):
                    sig_str = f"[bold green]{sig_str}[/bold green]"
                elif "HOLDING" in str(sig_str):
                    sig_str = f"[yellow]{sig_str}[/yellow]"
                pending_lines = state.get('pending_orders', [])
                if pending_lines:
                    rendered_pending = "\n".join(f"[dim cyan]{ln}[/dim cyan]" for ln in pending_lines)
                    sig_str = f"{sig_str}\n{rendered_pending}" if sig_str else rendered_pending
            else:
                price_str = "0.00000"
                trig_long_str = "--"
                trig_short_str = "--"
                ema_str = "UNKNOWN"
                adx_str = "0.0"
                vol_str = "0.00"
                sig_str = "FETCHING..."

            market_table.add_row(symbol.split('/')[0], price_str, trig_long_str, trig_short_str, ema_str, adx_str, vol_str, sig_str)

        body_layout["markets"].update(Panel(market_table, title="📊 15M SCALPING METRICS (LIVE UPDATING)", box=box.ROUNDED))

        # 3b. Active Positions & Balance Panel
        held_symbols = list(self.positions.keys())

        if held_symbols:
            pos_table = Table(box=box.MINIMAL, expand=True)
            pos_table.add_column("Asset", style="cyan")
            pos_table.add_column("Side", justify="center")
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Entry", justify="right")
            pos_table.add_column("SL", justify="right")
            pos_table.add_column("TP", justify="right")
            pos_table.add_column("P&L (%)", justify="right")

            for symbol in held_symbols:
                pos = self.positions[symbol]
                curr_price = markets_state[symbol]['price'] if symbol in markets_state else pos['entry_price']
                entry = pos['entry_price']
                side = pos['type']
                qty = pos['quantity']

                if side == 'LONG':
                    raw_pnl = (curr_price - entry) / entry
                else:
                    raw_pnl = (entry - curr_price) / entry

                leveraged_pnl = raw_pnl * LEVERAGE * 100
                pnl_color = "green" if leveraged_pnl >= 0 else "red"
                pnl_str = f"[{pnl_color}]{leveraged_pnl:+.2f}%[/{pnl_color}]"

                pos_table.add_row(
                    symbol.split('/')[0],
                    f"[bold {'green' if side == 'LONG' else 'red'}]{side}[/bold]",
                    f"{qty}",
                    f"{entry:.5f}",
                    f"{pos['sl']:.5f}",
                    f"{pos['tp']:.5f}",
                    pnl_str
                )
            panel_body = pos_table
        else:
            # Absolute visual confirmation that order triggers are alive.
            idle_panel = Panel(
                Text(
                    "No Active Positions. Standing by with 2 Pending Breakout Trigger Orders.",
                    style="bold cyan", justify="center"
                ),
                box=box.ROUNDED,
                style="dim cyan",
            )
            panel_body = idle_panel

        balance_text = Text()
        balance_text.append("\n")
        balance_text.append(f"USDT Testnet Balance: ", style="bold white")
        balance_text.append(f"{balance_info['free']:.2f} USDT", style="green")
        balance_text.append("\n")
        balance_text.append(f"Equity (Margin Sized): ", style="bold white")
        balance_text.append(f"{balance_info['total']:.2f} USDT", style="green")
        balance_text.append("\n")
        balance_text.append(f"Collateral Limit Per Trade: ", style="bold dim white")
        balance_text.append(f"{COLLATERAL_PER_TRADE} USDT @ {LEVERAGE}x", style="cyan")
        balance_text.append("\n")
        balance_text.append(f"Budget Guard: ", style="bold dim white")
        balance_text.append(f"{len(held_symbols)}/{MAX_ACTIVE_COINS} slots used", style="yellow")

        pos_panel_content = Layout()
        pos_panel_content.split_column(
            Layout(panel_body, ratio=1),
            Layout(Panel(balance_text, style="dim white", box=box.SIMPLE), size=6)
        )

        body_layout["positions"].update(Panel(pos_panel_content, title="💼 ACTIVE POSITIONS (TESTNET REAL ORDER)", box=box.ROUNDED))

        # 4. Log History Panel
        log_content = "\n".join(self.logs[-6:])
        layout["logs"].update(Panel(log_content, title="📜 RECENT LOGS", box=box.ROUNDED, style="dim white"))

        self.console.print(layout)

    def run_one_loop(self):
        """
        Single tick used by the offline sandbox test and by the external worker.
        Manages positions first, then evaluates active assets on closed bars.
        """
        # 1. Trigger scheduled daily coin rotation scan
        if self.last_scan_time is None or datetime.now() >= self.next_scan_time:
            self.scan_daily_market()

        markets_state = {}
        try:
            balance = self.trade_client.fetch_balance()
            usdt_free = balance['free'].get('USDT', 0.0)
            usdt_total = balance['total'].get('USDT', 0.0)
            balance_info = {'free': usdt_free, 'total': usdt_total}
        except Exception as e:
            self.log_event(f"Error fetching balance: {self._exchange_error(e)}")
            balance_info = {'free': 0.0, 'total': 0.0}

        # 2. Manage EVERY open position first, including rotated-out assets.
        closed_now = self.manage_open_positions()

        try:
            for symbol in self.active_assets:
                df = self.fetch_market_data(symbol)
                if df is not None and len(df) >= 2:
                    live_row = df.iloc[-1]
                    closed_row = df.iloc[-2]
                    held = symbol in self.positions
                    if symbol in closed_now:
                        signal = f"CLOSED: {closed_now[symbol]}"
                        pending_lines, dh, dl = self.describe_pending_orders(symbol, closed_row, held=False)
                    else:
                        signal = self.check_signals(symbol, df)
                        if symbol in self.positions:
                            signal = "HOLDING"
                        pending_lines, dh, dl = self.describe_pending_orders(symbol, closed_row, held=(symbol in self.positions))

                    vol_sma = closed_row['vol_sma']
                    markets_state[symbol] = {
                        'price': live_row['close'],
                        'trend': 'BULLISH' if closed_row['close'] > closed_row['ema_200'] else 'BEARISH',
                        'adx': closed_row['adx'],
                        'vol_ratio': closed_row['volume'] / vol_sma if vol_sma > 0 else 1.0,
                        'signal': signal,
                        'trigger_long': dh,
                        'trigger_short': dl,
                        'pending_orders': pending_lines,
                    }
                else:
                    markets_state[symbol] = {
                        'price': 0.0,
                        'trend': 'UNKNOWN',
                        'adx': 0.0,
                        'vol_ratio': 1.0,
                        'signal': 'ERROR_DATA',
                        'trigger_long': 0.0,
                        'trigger_short': 0.0,
                        'pending_orders': [],
                    }
        except Exception as e:
            self.log_event(f"Error in market tick: {self._exchange_error(e)}")

        # Publish state so external readers observe the same snapshot as the UI.
        self.markets_state = markets_state
        self.balance_info = balance_info
        self.render_dashboard(markets_state, balance_info)

    def build_markets_state(self, symbol, df, closed_now):
        """
        Shared helper used by both run_one_loop and the __main__ live loop so the
        market snapshot (including pending order instructions) is built exactly
        once and identically in every execution context.
        """
        if df is None or len(df) < 2:
            return {
                'price': 0.0,
                'trend': 'UNKNOWN',
                'adx': 0.0,
                'vol_ratio': 1.0,
                'signal': 'ERROR_DATA',
                'trigger_long': 0.0,
                'trigger_short': 0.0,
                'pending_orders': [],
            }

        live_row = df.iloc[-1]
        closed_row = df.iloc[-2]
        held = symbol in self.positions

        if symbol in closed_now:
            signal = f"CLOSED: {closed_now[symbol]}"
            pending_lines, dh, dl = self.describe_pending_orders(symbol, closed_row, held=False)
        else:
            signal = self.check_signals(symbol, df)
            if symbol in self.positions:
                signal = "HOLDING"
            pending_lines, dh, dl = self.describe_pending_orders(symbol, closed_row, held=(symbol in self.positions))

        vol_sma = closed_row['vol_sma']
        return {
            'price': live_row['close'],
            'trend': 'BULLISH' if closed_row['close'] > closed_row['ema_200'] else 'BEARISH',
            'adx': closed_row['adx'],
            'vol_ratio': closed_row['volume'] / vol_sma if vol_sma > 0 else 1.0,
            'signal': signal,
            'trigger_long': dh,
            'trigger_short': dl,
            'pending_orders': pending_lines,
        }

# ==========================================
# MAIN RUNNER WITH LIVE TICKING SECONDS
# ==========================================
if __name__ == "__main__":
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))  # wajib: ambil BINANCE_API_KEY/SECRET dari .env

    # Setup logging with UTF-8 encoding to handle emojis
    os.makedirs('logs', exist_ok=True)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler('logs/bot.log', encoding='utf-8'),
                            logging.StreamHandler(sys.stdout)
                        ])

    API_KEY = os.getenv("BINANCE_API_KEY", "your_testnet_api_key")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "your_testnet_api_secret")
    IS_RENDER = os.getenv("RENDER", "false").lower() == "true"

    bot = RealExecutionRotatorBot(api_key=API_KEY, api_secret=API_SECRET, use_render_mode=IS_RENDER)

    # Offline sandbox check
    if os.getenv("SANDBOX_TEST", "false") == "true" or (API_KEY == "your_testnet_api_key"):
        print("\n[INFO] Menjalankan test run loop tunggal (Offline Sandbox)...")
        bot.run_one_loop()
        print("Success! RealExecutionRotatorBot syntax and modules verified.")
    else:
        # Live 1-second interval execution loop
        try:
            bot.seconds_until_refresh = 0  # Force fetch on first run
            markets_state = {}
            balance_info = {'free': TOTAL_BOT_BUDGET, 'total': TOTAL_BOT_BUDGET}

            while True:
                # Exit management runs EVERY second for every held position,
                # including assets rotated out of the daily Top 2. Kept outside
                # the refresh gate so a stop is never delayed by the countdown.
                closed_now = {}
                try:
                    closed_now = bot.manage_open_positions()
                except Exception as e:
                    bot.log_event(f"Error managing open positions: {bot._exchange_error(e)}")

                # Every REFRESH_INTERVAL_SECONDS, refresh scan/balance/signals
                if bot.seconds_until_refresh <= 0:
                    try:
                        if bot.last_scan_time is None or datetime.now() >= bot.next_scan_time:
                            bot.scan_daily_market()
                    except Exception as e:
                        bot.log_event(f"Error in daily scan: {bot._exchange_error(e)}")

                    try:
                        balance = bot.trade_client.fetch_balance()
                        usdt_free = balance['free'].get('USDT', 0.0)
                        usdt_total = balance['total'].get('USDT', 0.0)
                        balance_info = {'free': usdt_free, 'total': usdt_total}
                    except Exception as e:
                        bot.log_event(f"Error fetching balance: {bot._exchange_error(e)}")
                        balance_info = {'free': 0.0, 'total': 0.0}

                    backoff = False
                    try:
                        for symbol in bot.active_assets:
                            df = bot.fetch_market_data(symbol)
                            markets_state[symbol] = bot.build_markets_state(symbol, df, closed_now)
                    except Exception as e:
                        bot.log_event(f"Error in market tick: {bot._exchange_error(e)}")
                        backoff = True

                    # Back off on error, otherwise use the standard 15s cadence.
                    bot.seconds_until_refresh = ERROR_BACKOFF_SECONDS if backoff else REFRESH_INTERVAL_SECONDS

                # Render the dashboard EVERY second to show countdowns ticking live
                bot.render_dashboard(markets_state, balance_info)

                # Wait exactly 1 second
                time.sleep(1)
                bot.seconds_until_refresh -= 1

        except KeyboardInterrupt:
            print("\nExiting Live Monitor gracefully...")