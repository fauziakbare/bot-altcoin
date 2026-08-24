import os
import time
import sys
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

# Turso (SQLite-compatible cloud DB) - fallback if not available
try:
    from libsql_experimental import connect as turso_connect
    TURSO_AVAILABLE = True
except ImportError:
    TURSO_AVAILABLE = False
    # Fallback to local SQLite if Turso not installed
    import sqlite3
    def turso_connect(url, auth_token=None):
        # Return a SQLite connection to local DB path
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'trade_history.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path)
    print("[WARN] libsql-experimental not available, falling back to local SQLite")

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
# PERMANENT PNL DATABASE (SQLite)
# ==========================================
# SQLite database storing the full trade history permanently.
# Path is anchored to the script directory -> always found no matter the CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'trade_history.db')

# KAPAN SCANNING DIJALANKAN (PILIHAN JAM DALAM WIB)
# Jam 7 = 07:00 WIB (Daily Close Binance - Rekomendasi Utama EBTA)
# Jam 22 = 22:00 WIB (2 Jam setelah US Market Open - Rekomendasi Taktis Volatilitas)
SCHEDULED_SCAN_HOUR = 22  # Ganti ke 22 jika ingin 2 jam setelah bursa US buka

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

        # WebSocket for market data (to avoid REST rate limits)
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

        # Turso configuration (cloud SQLite)
        self.turso_url = os.getenv("TURSO_DB_URL")
        self.turso_token = os.getenv("TURSO_AUTH_TOKEN")
        self.use_turso = TURSO_AVAILABLE and self.turso_url and self.turso_token
        self.db_conn = None
        
        self.positions = {}  # Tracks real active positions
        self.active_assets = []  # Curated koin to trade
        self.scanner_results = {} 
        self.last_scan_time = None
        self.next_scan_time = None
        
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
        Checks and initializes the database (Turso if configured, else local SQLite).
        Creates the `trades` table if it does not exist.
        """
        try:
            if self.use_turso:
                # Turso connection
                self.db_conn = turso_connect(self.turso_url, auth_token=self.turso_token)
                self.db_conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id             INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp      TEXT,
                        symbol         TEXT,
                        side           TEXT,
                        quantity       REAL,
                        entry_price    REAL,
                        exit_price     REAL,
                        net_return_pct REAL,
                        exit_reason    TEXT
                    )
                """)
                self.log_event(f"🗄️ Turso database siap: {self.turso_url}")
            else:
                # Fallback to local SQLite
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                with closing(sqlite3.connect(DB_PATH)) as conn, conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS trades (
                            id             INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp      TEXT,
                            symbol         TEXT,
                            side           TEXT,
                            quantity       REAL,
                            entry_price    REAL,
                            exit_price     REAL,
                            net_return_pct REAL,
                            exit_reason    TEXT
                        )
                    """)
                self.log_event(f"🗄️ SQLite database siap: {DB_PATH}")
        except Exception as e:
            self.log_event(f"❌ GAGAL INISIALISASI database: {str(e)}")

    def save_trade_record(self, symbol, side, quantity, entry_price, exit_price, net_return_pct, exit_reason):
        """
        Permanently inserts a closed trade into the database (Turso or SQLite).
        """
        try:
            if self.use_turso and self.db_conn:
                self.db_conn.execute(
                    """
                    INSERT INTO trades
                        (timestamp, symbol, side, quantity, entry_price,
                         exit_price, net_return_pct, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        symbol,
                        side,
                        quantity,
                        entry_price,
                        exit_price,
                        round(net_return_pct, 4),
                        exit_reason
                    )
                )
                self.db_conn.commit()
            else:
                with closing(sqlite3.connect(DB_PATH)) as conn, conn:
                    conn.execute(
                        """
                        INSERT INTO trades
                            (timestamp, symbol, side, quantity, entry_price,
                             exit_price, net_return_pct, exit_reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            symbol,
                            side,
                            quantity,
                            entry_price,
                            exit_price,
                            round(net_return_pct, 4),
                            exit_reason
                        )
                    )
            self.log_event("💾 Trade record permanently saved.")
        except Exception as e:
            self.log_event(f"❌ GAGAL MENYIMPAN trade record: {str(e)}")

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

        # Active Position Exit Checks
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos['type'] == 'LONG':
                if current_price > pos['peak_price']:
                    self.positions[symbol]['peak_price'] = current_price
                trailing_sl = self.positions[symbol]['peak_price'] - (2.5 * atr)
                
                if current_price <= trailing_sl:
                    self.close_position(symbol, current_price, "TRAILING_STOP_HIT (ATR)")
                elif current_price >= pos['tp']:
                    self.close_position(symbol, current_price, "TAKE_PROFIT_HIT (5.0x ATR)")
                elif not adx_active:  # Tri-State ADX Exit
                    self.close_position(symbol, current_price, "REGIME_EXIT_ADX_LOW (<25)")
            
            elif pos['type'] == 'SHORT':
                if current_price < pos['peak_price']:
                    self.positions[symbol]['peak_price'] = current_price
                trailing_sl = self.positions[symbol]['peak_price'] + (2.5 * atr)
                
                if current_price >= trailing_sl:
                    self.close_position(symbol, current_price, "TRAILING_STOP_HIT (ATR)")
                elif current_price <= pos['tp']:
                    self.close_position(symbol, current_price, "TAKE_PROFIT_HIT (5.0x ATR)")
                elif not adx_active:  # Tri-State ADX Exit
                    self.close_position(symbol, current_price, "REGIME_EXIT_ADX_LOW (<25)")
            
            return "HOLDING"

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
        Executes a real MARKET order on Binance USD-M Testnet using COLLATERAL_PER_TRADE = 50 USDT.
        """
        try:
            # 1. Calculate actual size based on 50 USDT collateral & 10x leverage = 500 USDT notional
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

            # Permanently save the closed trade (leveraged net return incl. 0.14% friction)
            self.save_trade_record(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry,
                exit_price=executed_price,
                net_return_pct=net_ret * 100,
                exit_reason=reason
            )

            self.log_event(f"🏁 POSISI TERTUTUP: {side} {symbol} @ {executed_price:.5f} | Net Return: {net_ret*100:+.2f}% (Friction Applied)")
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
            usdt_free = balance['free'].get('USDT', 5000.0)
            usdt_total = balance['total'].get('USDT', 5000.0)
            balance_info = {'free': usdt_free, 'total': usdt_total}
        except Exception as e:
            self.log_event(f"Error fetching balance: {self._exchange_error(e)}")
            balance_info = {'free': 5000.0, 'total': 5000.0}

        try:
            for symbol in self.active_assets:
                df = self.fetch_market_data(symbol)
                if df is not None:
                    last_row = df.iloc[-1]
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
                        'signal': 'ERROR_DATA'
                    }
        except Exception as e:
            self.log_event(f"Error in market tick: {self._exchange_error(e)}")

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
            balance_info = {'free': 5000.0, 'total': 5000.0}
            
            while True:
                # Every 15 seconds, we fetch market data and check signals
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
                        usdt_free = balance['free'].get('USDT', 5000.0)
                        usdt_total = balance['total'].get('USDT', 5000.0)
                        balance_info = {'free': usdt_free, 'total': usdt_total}
                    except Exception as e:
                        bot.log_event(f"Error fetching balance: {bot._exchange_error(e)}")
                        balance_info = {'free': 5000.0, 'total': 5000.0}

                    try:
                        for symbol in bot.active_assets:
                            df = bot.fetch_market_data(symbol)
                            if df is not None:
                                last_row = df.iloc[-1]
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
                        # Back off on error to avoid hammering API
                        bot.seconds_until_refresh = 60
                    
                    # Reset counter to 30 seconds to reduce rate limit pressure
                    bot.seconds_until_refresh = 30
                    
                    # Render the dashboard EVERY second to show countdowns ticking
                    bot.render_dashboard(markets_state, balance_info)
                    
                    # Wait exactly 1 second
                    time.sleep(1)
                    bot.seconds_until_refresh -= 1
                
        except KeyboardInterrupt:
            print("\nExiting Live Monitor gracefully...")
