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
# CONFIGURATION & CONSTANTS
# ==========================================
TIMEFRAME = '15m'
LEVERAGE = 10
RISK_REWARD_RATIO = 2.0
CANDLE_LIMIT = 250  # Enough for 200 EMA and indicators warm-up

# Jaminan Margin per Posisi Transaksi (Sesuai Batas Aman Anda)
# Total Alokasi Nominal Modal untuk seluruh operasional Bot ini
TOTAL_BOT_BUDGET = 50.0  # USDT (Total saldo yang didelegasikan ke bot)
MAX_ACTIVE_COINS = 2
COLLATERAL_PER_TRADE = TOTAL_BOT_BUDGET / MAX_ACTIVE_COINS  # Margin per trade (25.0 USDT)

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
# Turso is a cloud replica for remote access. Rows that fail to reach Turso are
# marked synced=0 locally and replayed on the next successful connection.
# Path is anchored to the script directory -> always found no matter the CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'trade_history.db')
# Last-resort spool: one JSON object per line, written only when BOTH stores fail.
SPOOL_PATH = os.path.join(BASE_DIR, 'data', 'trade_history_spool.jsonl')

# Single source of truth for the `trades` payload column order.
# Every schema, INSERT, SELECT, row tuple, and JSON key below is derived from
# this tuple, so a column can never drift between writer and reader.
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
    """
    # 1. EMA 200
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
    
    # 4. Volume SMA
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
    Computes ADX and Volume Ratio on daily (1D) data for filtering.
    """
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift(1))
    low_cp = np.abs(df['low'] - df['close'].shift(1))
    tr = np.max(np.column_stack((high_low, high_cp, low_cp)), axis=1)
    
    # 20-day Volume SMA
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
        # that .env files or copy-paste can easily introduce.
        api_key = (api_key or "").strip()
        api_secret = (api_secret or "").strip()
        if not api_key or not api_secret:
            self.log_event("❌ WARNING: BINANCE_API_KEY/API_SECRET kosong setelah sanitasi.")
        
        # ---------------------------------------------------------------
        # DUAL CLIENT ARCHITECTURE
        # ---------------------------------------------------------------
        # (a) market_client -> Binance Futures MAINNET public (NO API key).
        #     Only used for fetch_ohlcv so historical candles and indicators
        #     are complete/accurate (testnet candles can be thin or missing).
        self.market_client = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
        })

        # OPTIONAL: route all PUBLIC market-data REST calls through a
        # Cloudflare Worker relay to escape Render's shared-IP Binance ban.
        # Set MARKET_RELAY_URL (no trailing slash) e.g. https://your-worker.workers.dev
        # Only 'fapiPublic'/'fapiData' are overridden: those host the klines,
        # OI and funding endpoints this bot actually calls. Public API-key-less,
        # so the relay needs no auth and Binance sees the Worker's IP edge.
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
        #     Only used for balance, positions, leverage/margin, and orders.
        #     Demo/testnet mode overrides the REST endpoints to demo-fapi.binance.com
        #     because CCXT deprecated set_sandbox_mode for the old testnet.
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
            # Complete demo override: every USDT-M (fapi) and COIN-M (dapi)
            # endpoint is routed to Binance Futures Demo Trading host.
            # Keys mirror ccxt.binance.describe()['urls']['api'] for futures.
            demo_base = 'https://demo-fapi.binance.com'
            trade_client_config['urls'] = {
                'api': {
                    # USDT-M Futures (fapi) — public + private v1/v2/v3 + data
                    'fapi': f'{demo_base}/fapi/v1',
                    'fapiPublic': f'{demo_base}/fapi/v1',
                    'fapiPublicV2': f'{demo_base}/fapi/v2',
                    'fapiPublicV3': f'{demo_base}/fapi/v3',
                    'fapiPrivate': f'{demo_base}/fapi/v1',
                    'fapiPrivateV2': f'{demo_base}/fapi/v2',
                    'fapiPrivateV3': f'{demo_base}/fapi/v3',
                    'fapiData': f'{demo_base}/futures/data',
                    'fapiDataV2': f'{demo_base}/futures/data',
                    # COIN-M Futures (dapi) — public + private v1/v2 + data
                    'dapi': f'{demo_base}/dapi/v1',
                    'dapiPublic': f'{demo_base}/dapi/v1',
                    'dapiPrivate': f'{demo_base}/dapi/v1',
                    'dapiPrivateV2': f'{demo_base}/dapi/v2',
                    'dapiData': f'{demo_base}/futures/data',
                    'dapiDataV2': f'{demo_base}/futures/data',
                    # USDT-M Wallet/Account (sapi) — used by fetch_balance for futures mode.
                    # Must be overridden too, or signed balance calls hit mainnet api.binance.com
                    # and error with -2008 Invalid Api-Key. Demo wallet host = demo-bapi.
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

            # [AUTH DEBUG] host + key prefix — never log full key/secret
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
        # Pre-fetch initial 250 candles for each candidate via REST.
        # Throttle hard and stop on DDoS/ban (-1003) to avoid exceeding weight
        # and getting the shared Render IP banned.
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
            # 1.2s gap between klines requests: 10 symbols x weight stays well
            # under Binance's 2400/min futures weight budget.
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
        # db_conn is touched by the bot worker thread and by Turso replay, so
        # every use is serialized through this lock.
        self._db_lock = threading.Lock()
        self.local_db_ready = False

        # Optional push-notification sink (attached by the Telegram bridge).
        # Left as None so the bot runs unchanged when Telegram is not used.
        self.notifier = None
        
        self.positions = {}  # Tracks real active positions
        self.active_assets = []  # Curated koin to trade
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
        A Turso failure never blocks the local write.
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
        Returns True when self.db_conn is usable. Safe to call repeatedly:
        used both at startup and to recover after a dropped connection.
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
        Returns True when the row reached the spool file.
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
        Called after a successful Turso write so a recovered connection
        automatically closes the replication gap instead of leaving it forever.
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
            # Rows are in Turso but still flagged unsynced locally: the next
            # replay would duplicate them, so this is surfaced loudly.
            self.log_event(f"❌ Baris sudah masuk Turso tapi gagal ditandai synced: {str(e)}")
            return len(replayed)

        self.log_event(f"🔁 {len(replayed)} trade unsynced berhasil direplikasi ke Turso.")
        return len(replayed)

    def save_trade_record(self, symbol, side, quantity, entry_price, exit_price, net_return_pct, exit_reason):
        """
        Persists one closed trade. The local SQLite row is authoritative and is
        written first; Turso replication is attempted after and may fail without
        losing data (the row stays flagged synced = 0 for later replay).

        Returns a dict describing where the row landed:
            {'local': bool, 'turso': bool, 'spooled': bool, 'persisted': bool}
        `persisted` is False only when the trade reached no store at all.
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

        if result['local'] and result['turso']:
            self.log_event("💾 Trade record permanently saved to SQLite database + Turso.")
        elif result['local']:
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
        """
        Calculates the exact datetime for the next scheduled daily scan.
        """
        now = datetime.now()
        # Scheduled time for today
        scheduled_today = now.replace(hour=SCHEDULED_SCAN_HOUR, minute=0, second=0, microsecond=0)
        
        if now < scheduled_today:
            return scheduled_today
        else:
            # If scheduled time has already passed today, schedule for tomorrow
            return scheduled_today + timedelta(days=1)

    def scan_daily_market(self):
        """
        Scans all CANDIDATE_ASSETS once daily at the scheduled hour.
        Selects Top 2 with Daily ADX >= 25 and Volume Ratio > 1.5.
        Falls back to ADA and XRP if criteria aren't met.
        """
        self.log_event(f"🔍 [SCANNER] Memulai Pemindaian Pasar Harian (Jadwal: Pukul {SCHEDULED_SCAN_HOUR:02d}:00 WIB)...")
        candidates_stats = []
        
        for symbol in CANDIDATE_ASSETS:
            try:
                # Fetch last 35 daily candles to compute indicators
                ohlcv = self.market_client.fetch_ohlcv(symbol, '1d', limit=35)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = calculate_daily_metrics(df)
                
                last_row = df.iloc[-1]
                daily_adx = last_row['adx_1d']
                daily_vol_ratio = last_row['vol_ratio']
                
                # Check if asset meets trend criteria
                qualifies = (daily_adx >= 25) and (daily_vol_ratio > 1.5)
                candidates_stats.append({
                    'symbol': symbol,
                    'adx': daily_adx,
                    'vol_ratio': daily_vol_ratio,
                    'qualifies': qualifies,
                    'rank_score': daily_adx * daily_vol_ratio if qualifies else -1
                })
                
                # Sleep delay to respect API limit
                time.sleep(0.5)
                
            except Exception as e:
                self.log_event(f"Gagal memindai data harian untuk {symbol}: {self._exchange_error(e)}")
                
        # Sort candidates
        candidates_stats.sort(key=lambda x: x['rank_score'], reverse=True)
        self.scanner_results = {c['symbol']: c for c in candidates_stats}
        
        # Select top 2 qualifying assets
        selected = [c['symbol'] for c in candidates_stats if c['qualifies']][:2]
        
        # Fallback if less than 2 qualify
        fallback_used = False
        if len(selected) < 2:
            fallback_used = True
            for fallback in ['ADA/USDT:USDT', 'XRP/USDT:USDT']:
                if fallback not in selected:
                    selected.append(fallback)
            selected = selected[:2]
            
        self.active_assets = selected
        self.last_scan_time = datetime.now()
        self.next_scan_time = self.calculate_next_scan_time()
        
        # Setup margin mode and leverage for new selected assets on Binance
        for symbol in self.active_assets:
            self.setup_leverage_and_margin(symbol)
            
        status_msg = f"🎯 [ROTASI] Koin Terpilih Hari Ini: {', '.join([s.split('/')[0] for s in selected])}"
        if fallback_used:
            status_msg += " (Kombinasi Fallback diaktifkan)"
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
                
            # Set leverage to 10x
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

    def evaluate_exit(self, symbol, df):
        """
        Evaluates trailing stop / take profit / regime exit for ONE open position.

        Deliberately independent of self.active_assets: a position must keep being
        managed after its asset is rotated out of the daily Top 2, otherwise the
        stop and target stop being enforced and the position is orphaned on the
        exchange. Returns the exit reason when closed, else None.
        """
        pos = self.positions.get(symbol)
        if pos is None or df is None or len(df) < 2:
            return None

        last_row = df.iloc[-1]
        current_price = last_row['close']
        atr = last_row['atr']
        adx = last_row['adx']
        adx_active = adx >= 25

        if pos['type'] == 'LONG':
            if current_price > pos['peak_price']:
                self.positions[symbol]['peak_price'] = current_price
            trailing_sl = self.positions[symbol]['peak_price'] - (2.5 * atr)

            if current_price <= trailing_sl:
                reason = "TRAILING_STOP_HIT (ATR)"
            elif current_price >= pos['tp']:
                reason = "TAKE_PROFIT_HIT (5.0x ATR)"
            elif not adx_active:  # Tri-State ADX Exit
                reason = "REGIME_EXIT_ADX_LOW (<25)"
            else:
                return None
        else:  # SHORT
            if current_price < pos['peak_price']:
                self.positions[symbol]['peak_price'] = current_price
            trailing_sl = self.positions[symbol]['peak_price'] + (2.5 * atr)

            if current_price >= trailing_sl:
                reason = "TRAILING_STOP_HIT (ATR)"
            elif current_price <= pos['tp']:
                reason = "TAKE_PROFIT_HIT (5.0x ATR)"
            elif not adx_active:
                reason = "REGIME_EXIT_ADX_LOW (<25)"
            else:
                return None

        self.close_position(symbol, current_price, reason)
        return reason

    def manage_open_positions(self):
        """
        Runs exit management for EVERY held position, including assets that are
        no longer in the daily Top 2. Call this on every tick, before signal
        evaluation, so a rotated-out position is never left unmanaged.

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

    def check_signals(self, symbol, df):
        if df is None or len(df) < CANDLE_LIMIT:
            return "WAITING_FOR_DATA"

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        current_price = last_row['close']
        ema = last_row['ema_200']
        donchian_high = last_row['donchian_high']
        donchian_low = last_row['donchian_low']
        adx = last_row['adx']
        volume = last_row['volume']
        vol_sma = last_row['vol_sma']
        atr = last_row['atr']

        trend_bullish = current_price > ema
        trend_bearish = current_price < ema
        adx_active = adx >= 25
        volume_confirmed = volume > (1.5 * vol_sma)
        
        breakout_above = prev_row['close'] <= prev_row['donchian_high'] and current_price > donchian_high
        breakout_below = prev_row['close'] >= prev_row['donchian_low'] and current_price < donchian_low

        # Active Position Exit Checks.
        # manage_open_positions() is the authoritative exit path (it covers
        # rotated-out assets too); this keeps exits responsive for assets that
        # are still active without duplicating the rules.
        if symbol in self.positions:
            self.evaluate_exit(symbol, df)
            return "HOLDING" if symbol in self.positions else "CLOSED"

        # New Position Entry Checks
        if trend_bullish and breakout_above and adx_active and volume_confirmed:
            self.open_position(symbol, 'LONG', current_price, atr)
            return "BUY_SIGNAL"
        elif trend_bearish and breakout_below and adx_active and volume_confirmed:
            self.open_position(symbol, 'SHORT', current_price, atr)
            return "SELL_SIGNAL"

        return "WAITING_FOR_BREAKOUT"

    def open_position(self, symbol, side, entry_price, atr):
        """
        Executes a real MARKET order on Binance USD-M Testnet using
        COLLATERAL_PER_TRADE margin at LEVERAGE.

        Guardrails enforced before any order is sent:
          - never open a second position on a symbol already held
          - never exceed MAX_ACTIVE_COINS concurrent positions, so total
            committed margin stays within TOTAL_BOT_BUDGET
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
            # 1. Size the order from the fixed per-trade collateral and leverage
            notional_value = COLLATERAL_PER_TRADE * LEVERAGE
            quantity = notional_value / entry_price
            
            # Format quantity to match Binance lot step size rules
            # We fetch precision rules from the symbol info dynamically
            market_info = self.trade_client.market(symbol)
            precision = market_info['precision']['amount']
            quantity = self.trade_client.amount_to_precision(symbol, quantity)
            quantity = float(quantity)
            
            self.log_event(f"🛒 [TESTNET] Mengirim market order {side} {symbol} sebanyak {quantity} unit (Margin: {COLLATERAL_PER_TRADE} USDT)...")
            
            # 2. Execute Market Order on Binance
            order_side = 'buy' if side == 'LONG' else 'sell'
            response = self.trade_client.create_market_order(symbol, order_side, quantity)
            executed_price = response.get('price', entry_price)
            
            # Calculate SL and TP levels
            sl_distance = 2.5 * atr
            tp_distance = 5.0 * atr
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
            
            self.log_event(f"🚀 POSISI TERBUKA: {side} {symbol} @ {executed_price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Qty: {quantity}")
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
        Executes a real MARKET close order with reduceOnly=True on Binance USD-M Testnet.
        """
        try:
            pos = self.positions[symbol]
            side = pos['type']
            quantity = pos['quantity']
            entry = pos['entry_price']
            
            self.log_event(f"🛒 [TESTNET] Menutup posisi {side} {symbol} sebanyak {quantity} unit karena {reason}...")
            
            # Execute closing market order (opposite direction, reduceOnly=True)
            close_side = 'sell' if side == 'LONG' else 'buy'
            response = self.trade_client.create_market_order(symbol, close_side, quantity, params={'reduceOnly': True})
            executed_price = response.get('price', exit_price)
            
            # Calculate returns
            if side == 'LONG':
                raw_ret = (executed_price - entry) / entry
            else:
                raw_ret = (entry - executed_price) / entry
                
            net_ret = (raw_ret * LEVERAGE) - (ROUND_TRIP_FRICTION * LEVERAGE)
            # Update virtual bot balance: profit = collateral * net_ret (net_ret is fraction, e.g., 0.05 for 5%)
            self.realized_pnl += COLLATERAL_PER_TRADE * net_ret

            # Permanently save the closed trade (leveraged net return incl. 0.14% friction)
            save_result = self.save_trade_record(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry,
                exit_price=executed_price,
                net_return_pct=net_ret * 100,
                exit_reason=reason
            )

            self.log_event(f"🏁 POSISI TERTUTUP: {side} {symbol} @ {executed_price:.5f} | Net Return: {net_ret*100:+.2f}% (Friction Applied)")
            equity = TOTAL_BOT_BUDGET + self.realized_pnl
            # Emoji mirror the sign/reason so the outcome is readable at a glance
            # on a phone notification; the numbers are always printed too.
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
        Creates a gorgeous terminal UI using Rich, showing live countdowns.
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
        header_text = Text("\nCRYPTO QUANT LIVE TRADING MONITOR & ROTATOR (TESTNET)", style="bold green", justify="center")
        header_text.append(f"\nAPI: CONNECTED | Sandbox: REAL EXECUTION | Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim white")
        layout["header"].update(Panel(header_text, style="green"))

        # 2. Daily Rotation Panel with countdown
        time_left = self.next_scan_time - datetime.now() if self.next_scan_time else timedelta(0)
        hours, remainder = divmod(int(time_left.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        countdown_str = f"{hours:02d}j {minutes:02d}m {seconds:02d}d"
        
        rotation_text = Text(justify="center")
        rotation_text.append("🔄 ROTASI AKTIF UNTUK 24 JAM: ", style="bold white")
        rotation_text.append(f" {', '.join([s.split('/')[0] for s in self.active_assets])} ", style="bold yellow")
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

        # 3a. Market State Table
        market_table = Table(box=box.MINIMAL, expand=True)
        market_table.add_column("Asset", style="cyan")
        market_table.add_column("Price", justify="right")
        market_table.add_column("Trigger LONG", justify="right")
        market_table.add_column("Trigger SHORT", justify="right")
        market_table.add_column("Trend (EMA200)", justify="center")
        market_table.add_column("ADX (Regime)", justify="center")
        market_table.add_column("Volume Ratio", justify="center")
        market_table.add_column("State / Signal", justify="center")

        for symbol in self.active_assets:
            if symbol in markets_state:
                state = markets_state[symbol]
                price_str = f"{state['price']:.5f}"
                ema_str = "[green]BULLISH[/green]" if state['trend'] == 'BULLISH' else "[red]BEARISH[/red]"
                
                # Entry trigger price levels (20-period Donchian bands)
                trigger_long = state['trigger_long']
                trigger_short = state['trigger_short']
                trig_long_color = "green" if state['price'] < trigger_long else "white"
                trig_short_color = "red" if state['price'] > trigger_short else "white"
                trig_long_str = f"[{trig_long_color}]{trigger_long:.5f}[/{trig_long_color}]"
                trig_short_str = f"[{trig_short_color}]{trigger_short:.5f}[/{trig_short_color}]"
                
                adx_val = state['adx']
                adx_color = "green" if adx_val >= 25 else "red"
                adx_str = f"[{adx_color}]{adx_val:.1f} ({'TREND' if adx_val >= 25 else 'CHOP'})[/{adx_color}]"
                
                vol_ratio = state['vol_ratio']
                vol_color = "green" if vol_ratio >= 1.5 else "white"
                vol_str = f"[{vol_color}]{vol_ratio:.2f}x[/{vol_color}]"
                
                sig_str = state['signal']
                if "BUY" in sig_str or "SELL" in sig_str:
                    sig_str = f"[bold green]{sig_str}[/bold green]"
                elif "HOLDING" in sig_str:
                    sig_str = f"[yellow]{sig_str}[/yellow]"
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
        pos_table = Table(box=box.MINIMAL, expand=True)
        pos_table.add_column("Asset", style="cyan")
        pos_table.add_column("Side", justify="center")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("P&L (%)", justify="right")

        for symbol, pos in self.positions.items():
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
            
            pos_table.add_row(symbol.split('/')[0], f"[bold {'green' if side == 'LONG' else 'red'}]{side}[/bold]", f"{qty}", pnl_str)

        balance_text = f"\n[bold white]USDT Testnet Balance:[/bold white] [green]{balance_info['free']:.2f} USDT[/green]\n"
        balance_text += f"[bold white]Equity (Margin Sized):[/bold white] [green]{balance_info['total']:.2f} USDT[/green]\n"
        balance_text += f"[bold dim white]Collateral Limit Per Trade:[/bold dim white] [cyan]{COLLATERAL_PER_TRADE} USDT[/cyan]"
        
        pos_panel_content = Layout()
        pos_panel_content.split_column(
            Layout(pos_table, ratio=1),
            Layout(Panel(balance_text, style="dim white", box=box.SIMPLE), size=5)
        )

        body_layout["positions"].update(Panel(pos_panel_content, title="💼 ACTIVE POSITIONS (TESTNET REAL ORDER)", box=box.ROUNDED))

        # 4. Log History Panel
        log_content = "\n".join(self.logs[-6:])
        layout["logs"].update(Panel(log_content, title="📜 RECENT LOGS", box=box.ROUNDED, style="dim white"))

        self.console.print(layout)

    def run_one_loop(self):
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

        # 2. Manage EVERY open position first, including assets rotated out of
        #    the daily Top 2. Runs before signal evaluation so stops are never
        #    skipped for an orphaned symbol.
        closed_now = self.manage_open_positions()

        try:
            for symbol in self.active_assets:
                df = self.fetch_market_data(symbol)
                if df is not None:
                    last_row = df.iloc[-1]
                    if symbol in closed_now:
                        # Just exited on this tick: skip entry evaluation so the
                        # same breakout cannot immediately re-open the position
                        # and pay round-trip friction twice.
                        signal = f"CLOSED: {closed_now[symbol]}"
                    else:
                        signal = self.check_signals(symbol, df)

                    vol_sma = last_row['vol_sma']
                    markets_state[symbol] = {
                        'price': last_row['close'],
                        'trend': 'BULLISH' if last_row['close'] > last_row['ema_200'] else 'BEARISH',
                        'adx': last_row['adx'],
                        'vol_ratio': last_row['volume'] / vol_sma if vol_sma > 0 else 1.0,
                        'signal': signal,
                        'trigger_long': last_row['donchian_high'],
                        'trigger_short': last_row['donchian_low']
                    }
                else:
                    markets_state[symbol] = {
                        'price': 0.0,
                        'trend': 'UNKNOWN',
                        'adx': 0.0,
                        'vol_ratio': 1.0,
                        'signal': 'ERROR_DATA',
                        'trigger_long': 0.0,
                        'trigger_short': 0.0
                    }
        except Exception as e:
            self.log_event(f"Error in market tick: {self._exchange_error(e)}")

        # Publish state so external readers (Telegram /market, web dashboard)
        # observe the same snapshot the terminal UI renders.
        self.markets_state = markets_state
        self.balance_info = balance_info
        self.render_dashboard(markets_state, balance_info)

# ==========================================
# MAIN RUNNER WITH LIVE TICKING SECONDS
# ==========================================
if __name__ == "__main__":
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))  # wajib: ambil BINANCE_API_KEY/SECRET dari .env

    # Setup logging with UTF-8 encoding to handle emojis
    os.makedirs('logs', exist_ok=True)
    import sys
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
                        # Update Daily Scan scheduled times
                        if bot.last_scan_time is None or datetime.now() >= bot.next_scan_time:
                            bot.scan_daily_market()
                    except Exception as e:
                        bot.log_event(f"Error in daily scan: {bot._exchange_error(e)}")

                    try:
                        # Fetch balance
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
                            if df is not None:
                                last_row = df.iloc[-1]
                                if symbol in closed_now:
                                    # Exited this tick: skip entry so the same
                                    # breakout cannot immediately re-enter.
                                    signal = f"CLOSED: {closed_now[symbol]}"
                                else:
                                    signal = bot.check_signals(symbol, df)
                                vol_sma = last_row['vol_sma']
                                markets_state[symbol] = {
                                    'price': last_row['close'],
                                    'trend': 'BULLISH' if last_row['close'] > last_row['ema_200'] else 'BEARISH',
                                    'adx': last_row['adx'],
                                    'vol_ratio': last_row['volume'] / vol_sma if vol_sma > 0 else 1.0,
                                    'signal': signal,
                                    'trigger_long': last_row['donchian_high'],
                                    'trigger_short': last_row['donchian_low']
                                }
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
